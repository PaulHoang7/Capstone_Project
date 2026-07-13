"""
Phase 3: Zero-Shot Voice Cloning — drop emb_g, condition only on speaker encoder.

Changes from Phase 2:
  - g always from speaker_encoder(ref_wav), never from emb_g(sid)
  - Cross-speaker training: ref_wav != target utterance (same speaker)
  - L_sim = 1 - cosine_similarity(spk_enc(ref_wav), spk_enc(y_hat))
  - Hold out 20 speakers for zero-shot testing
  - Lower learning rate (1e-5)

Usage:
    cd /home/bes/Desktop/Tin
    python Capstone_project/scripts/train_clone_phase3.py \
        -c Capstone_project/configs/vits2_vieneu_clone_phase3.json \
        -m vieneu_clone_phase3
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_TIN_ROOT = _HERE.parents[2]
_VITS2 = _TIN_ROOT / "vits2_pytorch"

for p in (str(_TIN_ROOT), str(_VITS2)):
    if p not in sys.path:
        sys.path.insert(0, p)

import commons
import utils
from data_utils import DistributedBucketSampler
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch
from models import MultiPeriodDiscriminator, DurationDiscriminatorV1
from text.symbols import symbols

from Capstone_project.models.models_tone import SynthesizerTrnTone
from Capstone_project.data_pipeline.data_utils_tone import (
    TextAudioSpeakerToneLoader,
    TextAudioSpeakerToneCollate,
)
from Capstone_project.voice_cloning.speaker_encoder import ECAPASpeakerEncoder

torch.backends.cudnn.benchmark = True
global_step = 0
SAMPLE_RATE_ECAPA = 16_000


# ── Dataset with cross-speaker reference ──────────────────────────────────────

class Phase3Dataset(TextAudioSpeakerToneLoader):
    """Extends TextAudioSpeakerToneLoader to return cross-speaker reference wav.

    Returns: (text, tone, spec, wav, sid, ref_wav)
    where ref_wav is a different utterance from the same speaker,
    cropped/padded to REF_SECONDS.
    """

    REF_SECONDS = 3  # seconds of reference audio

    def __init__(self, filelist, hps_data, held_out_sids=None):
        super().__init__(filelist, hps_data)
        held_out = set(held_out_sids or [])

        # Filter held-out speakers
        if held_out:
            old_len = len(self.audiopaths_sid_text)
            keep = [
                i
                for i, item in enumerate(self.audiopaths_sid_text)
                if int(item[1]) not in held_out
            ]
            self.audiopaths_sid_text = [self.audiopaths_sid_text[i] for i in keep]
            if hasattr(self, "lengths"):
                self.lengths = [self.lengths[i] for i in keep]
            print(
                f"[Phase3] Filtered {old_len} -> {len(self.audiopaths_sid_text)} "
                f"(held out {len(held_out)} speakers)"
            )

        # Build per-speaker index for cross-speaker sampling
        self.sid_to_indices = defaultdict(list)
        for i, item in enumerate(self.audiopaths_sid_text):
            self.sid_to_indices[int(item[1])].append(i)

        self.ref_max_samples = int(self.REF_SECONDS * hps_data.sampling_rate)

    def __getitem__(self, index):
        item = self.audiopaths_sid_text[index]
        text, tone, spec, wav, sid = self.get_audio_text_speaker_pair(item)

        # Sample cross-speaker reference (different utterance, same speaker)
        sid_int = int(item[1])
        candidates = self.sid_to_indices[sid_int]
        ref_idx = random.choice(candidates)
        if len(candidates) > 1:
            for _ in range(5):
                if ref_idx != index:
                    break
                ref_idx = random.choice(candidates)

        # Load reference wav (only wav, no spec needed)
        ref_path = self.audiopaths_sid_text[ref_idx][0]
        ref_data, sr = sf.read(ref_path, dtype="float32")
        ref_wav = torch.from_numpy(ref_data)
        if sr != self.sampling_rate:
            ref_wav = torchaudio.functional.resample(ref_wav, sr, self.sampling_rate)

        # Crop or pad to fixed length
        if ref_wav.shape[0] > self.ref_max_samples:
            start = random.randint(0, ref_wav.shape[0] - self.ref_max_samples)
            ref_wav = ref_wav[start : start + self.ref_max_samples]
        elif ref_wav.shape[0] < self.ref_max_samples:
            ref_wav = F.pad(ref_wav, (0, self.ref_max_samples - ref_wav.shape[0]))

        return (text, tone, spec, wav, sid, ref_wav)

    def __len__(self):
        return len(self.audiopaths_sid_text)


class Phase3Collate:
    """Collate for Phase3Dataset — handles the extra ref_wav field."""

    def __init__(self):
        self._base = TextAudioSpeakerToneCollate()

    def __call__(self, batch):
        # Compute sort order (same as base collate uses)
        _, ids_sorted = torch.sort(
            torch.LongTensor([x[2].size(1) for x in batch]),
            dim=0,
            descending=True,
        )

        # Standard collation on first 5 fields
        base_batch = [item[:5] for item in batch]
        result = self._base(base_batch)

        # ref_wav: all fixed-length, reorder to match sort and stack
        ref_wavs = torch.stack(
            [batch[ids_sorted[i]][5] for i in range(len(batch))]
        )

        return (*result, ref_wavs)


# ── Speaker encoder helpers ───────────────────────────────────────────────────


def load_speaker_encoder(ckpt_path: str, device: str) -> ECAPASpeakerEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_speakers = ckpt["classifier_state"]["weight"].shape[0]
    enc = ECAPASpeakerEncoder(
        n_speakers=n_speakers, projection_dim=256, device=device
    )
    enc.classifier.load_state_dict(ckpt["classifier_state"])
    enc.projection.load_state_dict(ckpt["projection_state"])
    enc = enc.to(device)
    print(f"[Phase3] Speaker encoder loaded: {ckpt_path} (n_speakers={n_speakers})")
    return enc


def extract_spk_emb(enc, wav_24k, device):
    """Reference audio: [B, T] at 24kHz -> [B, 256] projected embedding. No gradient."""
    wav_16k = torchaudio.functional.resample(wav_24k, 24000, SAMPLE_RATE_ECAPA)
    peak = wav_16k.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
    wav_16k = wav_16k / peak
    wav_lens = torch.ones(wav_16k.shape[0], device=device)
    with torch.no_grad():
        emb = enc.encode(wav_16k, wav_lens)  # [B, 192]
    return enc.projection(emb)  # [B, 256]


def extract_gen_emb(enc, wav_24k, device):
    """Generated audio: [B, T] at 24kHz -> [B, 256]. WITH gradient for L_sim backprop."""
    wav_16k = torchaudio.functional.resample(wav_24k, 24000, SAMPLE_RATE_ECAPA)
    peak = wav_16k.abs().max(dim=1, keepdim=True).values.clamp(min=1e-6)
    wav_16k = wav_16k / peak
    wav_lens = torch.ones(wav_16k.shape[0], device=device)
    # NO torch.no_grad — allows gradient flow through y_hat -> generator
    emb = enc.encode(wav_16k, wav_lens)  # [B, 192]
    return enc.projection(emb)  # [B, 256]


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    assert torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "6062")
    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
    global global_step
    device = f"cuda:{rank}"

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        "nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    hps.data.use_mel_posterior_encoder = True
    held_out = list(getattr(hps.train, "held_out_speakers", []))

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataset = Phase3Dataset(
        hps.data.training_files, hps.data, held_out_sids=held_out
    )
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=Phase3Collate(),
        batch_sampler=train_sampler,
    )
    eval_loader = None
    if rank == 0:
        eval_ds = Phase3Dataset(
            hps.data.validation_files, hps.data, held_out_sids=held_out
        )
        eval_loader = DataLoader(
            eval_ds,
            num_workers=2,
            shuffle=False,
            batch_size=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=Phase3Collate(),
        )

    # ── Models ───────────────────────────────────────────────────────────────
    net_g = SynthesizerTrnTone(
        len(symbols),
        80,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=0.01,
        noise_scale_delta=2e-6,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    net_dur_disc = DurationDiscriminatorV1(
        hps.model.hidden_channels,
        hps.model.hidden_channels,
        3,
        0.1,
        gin_channels=hps.model.gin_channels,
    ).cuda(rank)

    # Speaker encoder: frozen backbone, trainable projection only
    spk_enc = load_speaker_encoder(hps.train.speaker_encoder_ckpt, device)
    for p in spk_enc._sb.parameters():
        p.requires_grad = False
    for p in spk_enc.classifier.parameters():
        p.requires_grad = False
    spk_enc.projection.weight.requires_grad = True
    spk_enc.projection.bias.requires_grad = True

    # ── Warm-start from Phase 2 ──────────────────────────────────────────────
    warmstart = getattr(hps.train, "warmstart_ckpt", None)
    if warmstart and os.path.exists(warmstart):
        utils.load_checkpoint(warmstart, net_g, None)
        if rank == 0:
            print(f"[Phase3] Warm-started net_g from {warmstart}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_dur = torch.optim.AdamW(
        net_dur_disc.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_proj = torch.optim.AdamW(
        [spk_enc.projection.weight, spk_enc.projection.bias],
        lr=1e-5,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # ── DDP ───────────────────────────────────────────────────────────────────
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        epoch_str = 1
        global_step = 0

    for opt in [optim_g, optim_d, optim_dur]:
        for pg in opt.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_dur = torch.optim.lr_scheduler.ExponentialLR(
        optim_dur, hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        _train_epoch(
            rank,
            epoch,
            hps,
            device,
            net_g,
            net_d,
            net_dur_disc,
            spk_enc,
            optim_g,
            optim_d,
            optim_dur,
            optim_proj,
            scaler,
            train_loader,
            eval_loader if rank == 0 else None,
            logger if rank == 0 else None,
            (writer, writer_eval) if rank == 0 else None,
        )
        scheduler_g.step()
        scheduler_d.step()
        scheduler_dur.step()


def _train_epoch(
    rank,
    epoch,
    hps,
    device,
    net_g,
    net_d,
    net_dur_disc,
    spk_enc,
    optim_g,
    optim_d,
    optim_dur,
    optim_proj,
    scaler,
    train_loader,
    eval_loader,
    logger,
    writers,
):
    global global_step

    net_g.train()
    net_d.train()
    net_dur_disc.train()
    spk_enc.train()
    train_loader.batch_sampler.set_epoch(epoch)

    c_sim = getattr(hps.train, "c_sim", 0.5)

    loader = (
        tqdm.tqdm(train_loader, desc=f"Epoch {epoch}") if rank == 0 else train_loader
    )

    for batch in loader:
        # ── Unpack batch ──────────────────────────────────────────────────────
        x, x_lengths, tone, spec, spec_lengths, y, y_lengths, speakers, ref_wavs = (
            batch
        )
        x = x.cuda(rank, non_blocking=True)
        x_lengths = x_lengths.cuda(rank, non_blocking=True)
        tone = tone.cuda(rank, non_blocking=True)
        spec = spec.cuda(rank, non_blocking=True)
        spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
        y = y.cuda(rank, non_blocking=True)
        y_lengths = y_lengths.cuda(rank, non_blocking=True)
        ref_wavs = ref_wavs.cuda(rank, non_blocking=True)  # [B, T_ref]
        # speakers NOT used — g comes from speaker encoder

        # ── Extract g from reference wav (no emb_g) ──────────────────────────
        ref_emb = extract_spk_emb(spk_enc, ref_wavs, device)  # [B, 256]
        g = ref_emb.unsqueeze(-1)  # [B, 256, 1] — same shape as emb_g output

        # ── Noise-scaled MAS ──────────────────────────────────────────────────
        if net_g.module.use_noise_scaled_mas:
            net_g.module.current_mas_noise_scale = max(
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step,
                0.0,
            )

        # ── TTS forward (g from speaker encoder, NOT emb_g) ──────────────────
        with autocast(enabled=hps.train.fp16_run):
            model_out = net_g(
                x, x_lengths, spec, spec_lengths, sid=None, tone=tone, g=g
            )
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = model_out[:8]

            mel = spec
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_sliced = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            # Discriminators
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y_sliced, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                hidden_x.detach(),
                x_mask.detach(),
                logw_.detach(),
                logw.detach(),
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

        # ── Generator losses ──────────────────────────────────────────────────
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_sliced, y_hat)
            y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = (
                    kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                )
                loss_dur = torch.sum(l_length.float())
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_dur_gen, _ = generator_loss(y_dur_hat_g)

        # ── L_sim: speaker similarity between ref and generated ───────────────
        try:
            gen_emb = extract_gen_emb(
                spk_enc, y_hat.squeeze(1), device
            )  # [B, 256], with grad
            loss_sim = (
                1 - F.cosine_similarity(ref_emb.detach(), gen_emb, dim=1)
            ).mean()
        except Exception:
            loss_sim = torch.tensor(0.0, device=device)

        loss_gen_all = (
            loss_gen
            + loss_fm
            + loss_mel
            + loss_dur
            + loss_kl
            + loss_dur_gen
            + c_sim * loss_sim
        )

        # ── Generator + projection backward ───────────────────────────────────
        optim_g.zero_grad()
        optim_proj.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.unscale_(optim_proj)
        scaler.step(optim_proj)
        scaler.update()

        # ── Logging ───────────────────────────────────────────────────────────
        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info(
                f"step {global_step} | mel={loss_mel.item():.3f} "
                f"kl={loss_kl.item():.3f} sim={loss_sim.item():.4f} "
                f"gen={loss_gen.item():.3f}"
            )
            writers[0].add_scalars(
                "",
                {
                    "loss/g/total": loss_gen_all,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl,
                    "loss/g/sim": loss_sim,
                    "loss/g/dur": loss_dur,
                    "loss/g/fm": loss_fm,
                    "loss/g/gen": loss_gen,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                },
                global_step,
            )

        if (
            rank == 0
            and global_step % hps.train.eval_interval == 0
            and global_step > 0
        ):
            try:
                _evaluate(hps, net_g, spk_enc, eval_loader, writers[1], device)
            except Exception as e:
                logger.warning(f"Evaluate failed at step {global_step}: {e}")
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, f"G_{global_step}.pth"),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth"),
            )
            torch.save(
                {
                    "step": global_step,
                    "projection_state": spk_enc.projection.state_dict(),
                },
                os.path.join(hps.model_dir, f"spk_enc_{global_step}.pth"),
            )
            utils.remove_old_checkpoints(
                hps.model_dir,
                prefixes=["G_*.pth", "D_*.pth", "spk_enc_*.pth"],
            )

        global_step += 1

    if rank == 0:
        logger.info(f"====> Epoch: {epoch}")


def _evaluate(hps, net_g, spk_enc, eval_loader, writer_eval, device):
    net_g.eval()
    spk_enc.eval()
    with torch.no_grad():
        for batch in eval_loader:
            x, x_lengths, tone, spec, spec_lengths, y, y_lengths, speakers, ref_wavs = (
                batch
            )
            x = x[:1].cuda(0)
            x_lengths = x_lengths[:1].cuda(0)
            tone = tone[:1].cuda(0)
            ref_wavs = ref_wavs[:1].cuda(0)
            break

        # Extract g from reference wav
        ref_emb = extract_spk_emb(spk_enc, ref_wavs, device)  # [1, 256]
        g = ref_emb.unsqueeze(-1)  # [1, 256, 1]

        y_hat, attn, mask, *_ = net_g.module.infer(
            x, x_lengths, sid=None, tone=tone, g=g, max_len=500
        )
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
    writer_eval.add_image(
        "gen/mel",
        utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        global_step,
        dataformats="HWC",
    )
    writer_eval.add_audio(
        "gen/audio",
        y_hat[0, :, : y_hat_lengths[0]],
        global_step,
        sample_rate=hps.data.sampling_rate,
    )
    net_g.train()
    spk_enc.train()


if __name__ == "__main__":
    main()
