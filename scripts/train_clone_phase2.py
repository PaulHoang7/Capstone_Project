"""
Phase 2: Dual Conditioning — integrate ECAPA-TDNN speaker encoder into VITS2 Variant D.

Extends train_variant.py by adding:
  - L_spk = MSE(emb_g(sid), speaker_encoder(ref_wav))
  - Training schedule:
      0-50K:   freeze speaker encoder, λ_spk = 1.0
      50K+:    unfreeze speaker encoder, λ_spk = 0.5
      100K+:   random dropout emb_g 50% of batches

Usage:
    cd /home/bes/Desktop/Tin
    python Capstone_project/scripts/train_clone_phase2.py \
        -c Capstone_project/configs/vits2_vieneu_clone_phase2.json \
        -m vieneu_clone_phase2
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torchaudio
import tqdm
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve()
_TIN_ROOT = _HERE.parents[2]
_VITS2    = _TIN_ROOT / "vits2_pytorch"

for p in (str(_TIN_ROOT), str(_VITS2)):
    if p not in sys.path:
        sys.path.insert(0, p)

import commons
import utils
from data_utils import DistributedBucketSampler, TextAudioSpeakerCollate, TextAudioSpeakerLoader
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import MultiPeriodDiscriminator, DurationDiscriminatorV1
from text.symbols import symbols

from Capstone_project.models.models_tone import SynthesizerTrnTone
from Capstone_project.data_pipeline.data_utils_tone import (
    TextAudioSpeakerToneLoader, TextAudioSpeakerToneCollate,
)
from Capstone_project.voice_cloning.speaker_encoder import ECAPASpeakerEncoder

torch.backends.cudnn.benchmark = True
global_step = 0

SAMPLE_RATE_ECAPA = 16_000


# ── Speaker encoder ───────────────────────────────────────────────────────────

def load_speaker_encoder(ckpt_path: str, device: str) -> ECAPASpeakerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Infer n_speakers from saved classifier weights
    n_speakers = ckpt["classifier_state"]["weight"].shape[0]
    enc = ECAPASpeakerEncoder(n_speakers=n_speakers, projection_dim=256, device=device)
    enc.classifier.load_state_dict(ckpt["classifier_state"])
    enc.projection.load_state_dict(ckpt["projection_state"])
    enc = enc.to(device)
    print(f"[Phase2] Speaker encoder loaded: {ckpt_path} (n_speakers={n_speakers})")
    return enc


def extract_spk_emb(enc: ECAPASpeakerEncoder, wav_24k: torch.Tensor, device: str) -> torch.Tensor:
    """wav_24k: [B, T] at 24 kHz → [B, 256] projected embedding."""
    wav_16k = torchaudio.functional.resample(wav_24k, 24000, SAMPLE_RATE_ECAPA)
    peak = wav_16k.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
    wav_16k = wav_16k / peak
    wav_lens = torch.ones(wav_16k.shape[0], device=device)
    with torch.no_grad():
        emb_192 = enc.encode(wav_16k, wav_lens)   # [B, 192]
    return enc.projection(emb_192)                  # [B, 256]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    assert torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "6061")
    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
    global global_step
    device = f"cuda:{rank}"

    if rank == 0:
        logger      = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer      = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group("nccl", init_method="env://", world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    hps.data.use_mel_posterior_encoder = True

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataset = TextAudioSpeakerToneLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus, rank=rank, shuffle=True,
    )
    collate_fn   = TextAudioSpeakerToneCollate()
    train_loader = DataLoader(
        train_dataset, num_workers=4, shuffle=False,
        pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler,
    )
    eval_loader = None
    if rank == 0:
        eval_ds    = TextAudioSpeakerToneLoader(hps.data.validation_files, hps.data)
        eval_coll  = TextAudioSpeakerToneCollate()
        eval_loader = DataLoader(
            eval_ds, num_workers=4, shuffle=False,
            batch_size=hps.train.batch_size, pin_memory=True,
            drop_last=False, collate_fn=eval_coll,
        )

    # ── Models ───────────────────────────────────────────────────────────────
    net_g = SynthesizerTrnTone(
        len(symbols), 80,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=0.01, noise_scale_delta=2e-6,
        **hps.model,
    ).cuda(rank)
    net_d        = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    net_dur_disc = DurationDiscriminatorV1(
        hps.model.hidden_channels, hps.model.hidden_channels,
        3, 0.1, gin_channels=hps.model.gin_channels,
    ).cuda(rank)

    # Speaker encoder (frozen initially)
    spk_enc = load_speaker_encoder(hps.train.speaker_encoder_ckpt, device)
    for p in spk_enc.parameters():
        p.requires_grad = False
    spk_enc_frozen = True

    # ── Warm-start from Variant D ─────────────────────────────────────────────
    warmstart = getattr(hps.train, "warmstart_ckpt", None)
    if warmstart and os.path.exists(warmstart) and rank == 0:
        utils.load_checkpoint(warmstart, net_g, None)
        print(f"[Phase2] Warm-started net_g from {warmstart}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    optim_g   = torch.optim.AdamW(net_g.parameters(),        hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d   = torch.optim.AdamW(net_d.parameters(),        hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_dur = torch.optim.AdamW(net_dur_disc.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_spk = torch.optim.AdamW(spk_enc.parameters(),      lr=1e-5,                 betas=hps.train.betas, eps=hps.train.eps)

    # ── DDP ───────────────────────────────────────────────────────────────────
    net_g        = DDP(net_g,        device_ids=[rank], find_unused_parameters=True)
    net_d        = DDP(net_d,        device_ids=[rank], find_unused_parameters=True)
    net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        epoch_str   = 1
        global_step = 0

    # Ensure initial_lr exists for all optimizers (required when resuming without full optimizer state)
    for opt in [optim_g, optim_d, optim_dur]:
        for pg in opt.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']

    scheduler_g   = torch.optim.lr_scheduler.ExponentialLR(optim_g,   hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d   = torch.optim.lr_scheduler.ExponentialLR(optim_d,   hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_dur = torch.optim.lr_scheduler.ExponentialLR(optim_dur, hps.train.lr_decay, last_epoch=epoch_str - 2)
    scaler        = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        spk_enc_frozen = _train_epoch(
            rank, epoch, hps, device,
            net_g, net_d, net_dur_disc, spk_enc,
            optim_g, optim_d, optim_dur, optim_spk,
            scaler, train_loader,
            eval_loader if rank == 0 else None,
            logger if rank == 0 else None,
            (writer, writer_eval) if rank == 0 else None,
            spk_enc_frozen,
        )
        scheduler_g.step(); scheduler_d.step(); scheduler_dur.step()


def _train_epoch(
    rank, epoch, hps, device,
    net_g, net_d, net_dur_disc, spk_enc,
    optim_g, optim_d, optim_dur, optim_spk,
    scaler, train_loader, eval_loader, logger, writers,
    spk_enc_frozen,
):
    global global_step

    net_g.train(); net_d.train(); net_dur_disc.train(); spk_enc.train()
    train_loader.batch_sampler.set_epoch(epoch)

    c_spk_base      = hps.train.c_spk               # 1.0
    spk_unfreeze_at = hps.train.spk_unfreeze_step    # 50000
    emb_g_drop_at   = hps.train.emb_g_dropout_step  # 100000
    emb_g_drop_p    = hps.train.emb_g_dropout_prob  # 0.5

    loader = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}") if rank == 0 else train_loader

    for batch in loader:
        # ── Schedule ──────────────────────────────────────────────────────────
        if spk_enc_frozen and global_step >= spk_unfreeze_at:
            for p in spk_enc.parameters():
                p.requires_grad = True
            spk_enc_frozen = False
            if rank == 0:
                print(f"\n[Phase2] step {global_step}: Unfreeze speaker encoder, λ_spk→0.5")

        c_spk    = c_spk_base if spk_enc_frozen else c_spk_base * 0.5
        use_emb_g = True
        if global_step >= emb_g_drop_at:
            use_emb_g = torch.rand(1).item() > emb_g_drop_p

        # ── Unpack batch (tone-aware) ─────────────────────────────────────────
        x, x_lengths, tone, spec, spec_lengths, y, y_lengths, speakers = batch
        x         = x.cuda(rank, non_blocking=True);         x_lengths   = x_lengths.cuda(rank, non_blocking=True)
        tone      = tone.cuda(rank, non_blocking=True)
        spec      = spec.cuda(rank, non_blocking=True);      spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
        y         = y.cuda(rank, non_blocking=True);         y_lengths   = y_lengths.cuda(rank, non_blocking=True)
        speakers  = speakers.cuda(rank, non_blocking=True)

        # ── L_spk ─────────────────────────────────────────────────────────────
        # emb_g(sid): [B, 256]
        g_lookup = net_g.module.emb_g(speakers)              # [B, 256]
        # ECAPA: encode reference waveform (y) → [B, 256]
        g_ecapa  = extract_spk_emb(spk_enc, y.squeeze(1), device)  # [B, 256]
        # MSE between the two speaker spaces — pulls them together
        loss_spk = F.mse_loss(g_lookup, g_ecapa.detach())

        # ── Noise-scaled MAS ──────────────────────────────────────────────────
        if net_g.module.use_noise_scaled_mas:
            net_g.module.current_mas_noise_scale = max(
                net_g.module.mas_noise_scale_initial - net_g.module.noise_scale_delta * global_step, 0.0
            )

        # ── TTS forward ───────────────────────────────────────────────────────
        with autocast(enabled=hps.train.fp16_run):
            model_out = net_g(x, x_lengths, spec, spec_lengths, speakers, tone=tone)
            # SynthesizerTrnTone returns 8 values
            (y_hat, l_length, attn, ids_slice, x_mask, z_mask,
             (z, z_p, m_p, logs_p, m_q, logs_q),
             (hidden_x, logw, logw_)) = model_out[:8]

            mel      = spec
            y_mel    = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y_sliced = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # Discriminators
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

            y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
            )
            with autocast(enabled=False):
                loss_dur_disc, _, _ = discriminator_loss(y_dur_hat_r, y_dur_hat_g)

        # Discriminator backward
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        optim_dur.zero_grad()
        scaler.scale(loss_dur_disc).backward(retain_graph=True)
        scaler.unscale_(optim_dur)
        commons.clip_grad_value_(net_dur_disc.parameters(), None)
        scaler.step(optim_dur)

        # Generator
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)
            y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(enabled=False):
                loss_mel      = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl       = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_dur      = torch.sum(l_length.float())
                loss_fm       = feature_loss(fmap_r, fmap_g)
                loss_gen, _   = generator_loss(y_d_hat_g)
                loss_dur_gen, _ = generator_loss(y_dur_hat_g)
                loss_gen_all  = (loss_gen + loss_fm + loss_mel + loss_dur
                                 + loss_kl + loss_dur_gen + c_spk * loss_spk)

        optim_g.zero_grad()
        if not spk_enc_frozen:
            optim_spk.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        if not spk_enc_frozen:
            scaler.unscale_(optim_spk)
            commons.clip_grad_value_(spk_enc.parameters(), None)
            scaler.step(optim_spk)
        scaler.update()

        # ── Logging ───────────────────────────────────────────────────────────
        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info(
                f"step {global_step} | mel={loss_mel.item():.3f} kl={loss_kl.item():.3f} "
                f"spk={loss_spk.item():.4f} gen={loss_gen.item():.3f} "
                f"c_spk={c_spk} frozen={spk_enc_frozen} emb_g={use_emb_g}"
            )
            writers[0].add_scalars("", {
                "loss/g/total": loss_gen_all, "loss/g/mel": loss_mel,
                "loss/g/kl":    loss_kl,      "loss/g/spk": loss_spk,
                "loss/g/dur":   loss_dur,     "loss/g/fm":  loss_fm,
                "loss/g/gen":   loss_gen,     "loss/d/total": loss_disc,
                "train/c_spk":  c_spk,        "train/use_emb_g": float(use_emb_g),
                "learning_rate": lr,
            }, global_step)

        if rank == 0 and global_step % hps.train.eval_interval == 0 and global_step > 0:
            try:
                _evaluate(hps, net_g, eval_loader, writers[1])
            except Exception as e:
                logger.warning(f"Evaluate failed at step {global_step}: {e}")
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                   os.path.join(hps.model_dir, f"G_{global_step}.pth"))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                   os.path.join(hps.model_dir, f"D_{global_step}.pth"))
            torch.save({
                "step": global_step, "spk_enc_frozen": spk_enc_frozen,
                "classifier_state": spk_enc.classifier.state_dict(),
                "projection_state": spk_enc.projection.state_dict(),
            }, os.path.join(hps.model_dir, f"spk_enc_{global_step}.pth"))
            utils.remove_old_checkpoints(hps.model_dir,
                                          prefixes=["G_*.pth", "D_*.pth", "spk_enc_*.pth"])

        global_step += 1

    if rank == 0:
        logger.info(f"====> Epoch: {epoch}")
    return spk_enc_frozen


def _evaluate(hps, net_g, eval_loader, writer_eval):
    net_g.eval()
    with torch.no_grad():
        for batch in eval_loader:
            x, x_lengths, tone, spec, spec_lengths, y, y_lengths, speakers = batch
            x = x[:1].cuda(0); x_lengths = x_lengths[:1].cuda(0)
            tone = tone[:1].cuda(0); speakers = speakers[:1].cuda(0)
            break
        y_hat, attn, mask, *_ = net_g.module.infer(x, x_lengths, speakers, tone=tone, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            hps.data.mel_fmin, hps.data.mel_fmax,
        )
    writer_eval.add_image("gen/mel", utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                          global_step, dataformats="HWC")
    writer_eval.add_audio("gen/audio", y_hat[0, :, :y_hat_lengths[0]],
                          global_step, sample_rate=hps.data.sampling_rate)
    net_g.train()


if __name__ == "__main__":
    main()
