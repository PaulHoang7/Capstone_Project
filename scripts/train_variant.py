"""
Training script for VITS2 tone-aware variants (B through F).

Based on vits2_pytorch/train_ms.py but uses:
  - SynthesizerTrnTone instead of SynthesizerTrn
  - TextAudioSpeakerToneLoader/Collate for tone-aware data loading
  - Passes tone sequences through the pipeline
  - Identity-init for warmstart merge_proj

Usage:
    # Warmstart from baseline checkpoint:
    cd /home/bes/Desktop/TTS/vits2_pytorch
    python ../Capstone_project/scripts/train_variant.py \
        -c ../Capstone_project/configs/vits2_vieneu_variant_b.json \
        -m vieneu_variant_b \
        --warmstart logs/vieneu_base/G_438000.pth

    # Resume training (auto-loads latest checkpoint from logs/vieneu_variant_b/):
    cd /home/bes/Desktop/TTS/vits2_pytorch
    python ../Capstone_project/scripts/train_variant.py \
        -c ../Capstone_project/configs/vits2_vieneu_variant_b.json \
        -m vieneu_variant_b
"""

import argparse
import itertools
import json
import math
import os
import sys

# Add paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))
_VITS2_DIR = os.path.join(_PROJECT_ROOT, 'vits2_pytorch')

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _VITS2_DIR not in sys.path:
    sys.path.insert(0, _VITS2_DIR)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import commons
import utils
from data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import (
    AVAILABLE_DURATION_DISCRIMINATOR_TYPES,
    AVAILABLE_FLOW_TYPES,
    DurationDiscriminatorV1,
    DurationDiscriminatorV2,
    MultiPeriodDiscriminator,
)
from text.symbols import symbols

# Capstone extensions
from Capstone_project.models import build_synthesizer
from Capstone_project.models.models_tone import SynthesizerTrnTone
from Capstone_project.data_pipeline.data_utils_tone import (
    TextAudioSpeakerToneLoader,
    TextAudioSpeakerToneCollate,
)

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    # Parse --warmstart before get_hparams (which uses its own argparse)
    warmstart_path = None
    filtered_argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--warmstart" and i + 1 < len(sys.argv):
            warmstart_path = sys.argv[i + 1]
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + filtered_argv

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6061"  # different port from base train_ms.py

    hps = utils.get_hparams()

    # Override model_dir to always be under vits2_pytorch/logs/
    # so checkpoints are co-located with the baseline regardless of CWD
    model_name = os.path.basename(hps.model_dir)
    hps.model_dir = os.path.join(_VITS2_DIR, "logs", model_name)
    os.makedirs(hps.model_dir, exist_ok=True)
    # Re-save config to the correct model_dir
    import json
    config_save_path = os.path.join(hps.model_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(hps.__dict__, f, indent=2, default=str)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, warmstart_path))


def run(rank, n_gpus, hps, warmstart_path=None):
    net_dur_disc = None
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # Posterior encoder channels
    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    # Check if tone embedding is enabled
    use_tone = getattr(hps.data, "use_tone_embedding", False)
    if use_tone:
        print("Using tone-aware data loader (Variant B+)")
        train_dataset = TextAudioSpeakerToneLoader(hps.data.training_files, hps.data)
        collate_fn = TextAudioSpeakerToneCollate()
    else:
        train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
        collate_fn = TextAudioSpeakerCollate()

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
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        if use_tone:
            eval_dataset = TextAudioSpeakerToneLoader(
                hps.data.validation_files, hps.data
            )
            eval_collate = TextAudioSpeakerToneCollate()
        else:
            eval_dataset = TextAudioSpeakerLoader(
                hps.data.validation_files, hps.data
            )
            eval_collate = TextAudioSpeakerCollate()
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=4,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=eval_collate,
        )

    # Feature flags
    use_transformer_flows = getattr(hps.model, "use_transformer_flows", False)
    if use_transformer_flows:
        transformer_flow_type = hps.model.transformer_flow_type
        print(f"Using transformer flows {transformer_flow_type} for VITS2")

    use_noise_scaled_mas = getattr(hps.model, "use_noise_scaled_mas", False)
    if use_noise_scaled_mas:
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    use_duration_discriminator = getattr(
        hps.model, "use_duration_discriminator", False
    )
    if use_duration_discriminator:
        duration_discriminator_type = getattr(
            hps.model, "duration_discriminator_type", "dur_disc_1"
        )
        print(f"Using duration_discriminator {duration_discriminator_type} for VITS2")
        duration_discriminator_type = AVAILABLE_DURATION_DISCRIMINATOR_TYPES
        if duration_discriminator_type == "dur_disc_1":
            net_dur_disc = DurationDiscriminatorV1(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels
                if hps.data.n_speakers != 0
                else 0,
            ).cuda(rank)
        elif duration_discriminator_type == "dur_disc_2":
            net_dur_disc = DurationDiscriminatorV2(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels
                if hps.data.n_speakers != 0
                else 0,
            ).cuda(rank)
    else:
        print("NOT using any duration discriminator like VITS1")
        net_dur_disc = None

    # Build model: factory picks correct encoder based on hps.model.variant
    variant = getattr(hps.model, "variant", None)
    if variant and variant in ("C", "D", "E", "F"):
        # New shared synthesizer + pluggable encoder
        net_g = build_synthesizer(hps, n_vocab=len(symbols)).cuda(rank)
        print(f"Using build_synthesizer for Variant {variant}")
    else:
        # Variant A/B: legacy SynthesizerTrnTone (backward compat)
        net_g = SynthesizerTrnTone(
            len(symbols),
            posterior_channels,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            mas_noise_scale_initial=mas_noise_scale_initial,
            noise_scale_delta=noise_scale_delta,
            **hps.model,
        ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

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
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc, device_ids=[rank], find_unused_parameters=True
        )

    is_warmstart = False
    epoch_str = 1
    global_step = 0

    # Check for existing checkpoints in model_dir
    existing_g = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")

    if warmstart_path is not None and os.path.isfile(warmstart_path):
        # --warmstart flag: load ONLY model weights (no optimizer, no epoch)
        print(f"Warmstarting from: {warmstart_path}")
        utils.load_checkpoint(warmstart_path, net_g, None)
        warmstart_dir = os.path.dirname(warmstart_path)
        d_ckpt = utils.latest_checkpoint_path(warmstart_dir, "D_*.pth")
        if d_ckpt:
            utils.load_checkpoint(d_ckpt, net_d, None)
        is_warmstart = True

    elif existing_g is not None:
        # Load checkpoint from model_dir
        print(f"Found checkpoint: {existing_g}")
        _, _, _, saved_epoch = utils.load_checkpoint(existing_g, net_g, None)

        # Detect warmstart vs resume:
        # If checkpoint has missing keys (e.g., tone_emb), it's a warmstart
        # from a baseline checkpoint, not a training resume.
        ckpt_dict = torch.load(existing_g, map_location="cpu")
        saved_keys = set(ckpt_dict["model"].keys())
        model_keys = set(net_g.module.state_dict().keys())
        missing_keys = model_keys - saved_keys

        if missing_keys:
            # Warmstart: baseline checkpoint copied into model_dir
            print(f"Detected warmstart (missing keys: {sorted(missing_keys)})")
            # Already loaded model weights above (with None optimizer)
            # Load discriminator too
            d_ckpt = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
            if d_ckpt:
                utils.load_checkpoint(d_ckpt, net_d, None)
            is_warmstart = True
        else:
            # True resume: reload with optimizer state
            print(f"Resuming training from epoch {saved_epoch}")
            utils.load_checkpoint(existing_g, net_g, optim_g)
            d_ckpt = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
            if d_ckpt:
                utils.load_checkpoint(d_ckpt, net_d, optim_d)
            if net_dur_disc is not None:
                dur_ckpt = utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth")
                if dur_ckpt:
                    try:
                        utils.load_checkpoint(dur_ckpt, net_dur_disc, optim_dur_disc)
                    except Exception:
                        pass
            epoch_str = saved_epoch
            global_step = (epoch_str - 1) * len(train_loader)

    else:
        print("No checkpoint found. Training from scratch.")

    # Identity-init merge_proj for warm-started tone embedding
    if is_warmstart and use_tone and hasattr(net_g.module.enc_p, 'merge_proj'):
        print("Applying identity-init to merge_proj for warmstart")
        net_g.module.enc_p.init_merge_proj_identity()

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                use_tone=use_tone,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, None],
                None,
                None,
                use_tone=use_tone,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers,
    use_tone=False,
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc="Loading train data")
    else:
        loader = train_loader

    for batch_idx, batch in enumerate(loader):
        # Unpack batch (tone-aware or standard)
        if use_tone:
            x, x_lengths, tone, spec, spec_lengths, y, y_lengths, speakers = batch
            tone = tone.cuda(rank, non_blocking=True)
        else:
            x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch
            tone = None

        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speakers = speakers.cuda(rank, non_blocking=True)

        # Variant E+: check if model returns f0_pred (9-tuple)
        variant = getattr(hps.model, "variant", None)
        use_f0_loss = variant in ("E", "F") and getattr(hps.train, "c_f0", 0) > 0

        with autocast(enabled=hps.train.fp16_run):
            model_out = net_g(x, x_lengths, spec, spec_lengths, speakers, tone=tone)

            # Unpack: variants C+ return 9 values (with f0_pred), A/B return 8
            if len(model_out) == 9:
                (
                    y_hat, l_length, attn, ids_slice, x_mask, z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    (hidden_x, logw, logw_),
                    f0_pred,
                ) = model_out
            else:
                (
                    y_hat, l_length, attn, ids_slice, x_mask, z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    (hidden_x, logw, logw_),
                ) = model_out
                f0_pred = None

            if (
                hps.model.use_mel_posterior_encoder
                or hps.data.use_mel_posterior_encoder
            ):
                mel = spec
            else:
                mel = spec_to_mel_torch(
                    spec.float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
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
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                )
                with autocast(enabled=False):
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x, x_mask, logw_, logw
                )
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = (
                    kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                )
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

                # Variant E+: auxiliary F0 loss
                loss_f0 = torch.tensor(0.0)
                if use_f0_loss and f0_pred is not None:
                    from Capstone_project.data_pipeline.f0_utils import (
                        extract_f0_from_audio,
                        align_f0_to_text,
                    )
                    from Capstone_project.models.modules.f0_predictor import compute_f0_loss
                    # Extract GT F0 from raw audio (y before slicing)
                    with torch.no_grad():
                        f0_frames_list = []
                        for i in range(y.size(0)):
                            wav_np = y[i, 0].cpu().numpy()
                            lf0 = extract_f0_from_audio(
                                wav_np, sr=hps.data.sampling_rate,
                                hop_length=hps.data.hop_length,
                            )
                            f0_frames_list.append(torch.from_numpy(lf0))
                        # Pad to same length
                        max_len_f0 = max(f.size(0) for f in f0_frames_list)
                        f0_frames = torch.zeros(y.size(0), 1, max_len_f0, device=y.device)
                        for i, f in enumerate(f0_frames_list):
                            l = min(f.size(0), max_len_f0)
                            f0_frames[i, 0, :l] = f[:l].to(y.device)
                        # Align frame-level F0 to text level via attention
                        f0_gt = align_f0_to_text(f0_frames, attn)
                    loss_f0 = compute_f0_loss(f0_pred, f0_gt, x_mask)
                    c_f0 = getattr(hps.train, "c_f0", 0.2)
                    loss_gen_all = loss_gen_all + loss_f0 * c_f0

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                if net_dur_disc is not None:
                    scalar_dict.update(
                        {
                            "loss/dur_disc/total": loss_dur_disc_all,
                            "grad_norm_dur_disc": grad_norm_dur_disc,
                        }
                    )
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                if use_f0_loss:
                    scalar_dict["loss/g/f0"] = loss_f0
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, use_tone=use_tone)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(
                            hps.model_dir, "DUR_{}.pth".format(global_step)
                        ),
                    )
                utils.remove_old_checkpoints(
                    hps.model_dir,
                    prefixes=["G_*.pth", "D_*.pth", "DUR_*.pth"],
                )
        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval, use_tone=False):
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if use_tone:
                x, x_lengths, tone, spec, spec_lengths, y, y_lengths, speakers = batch
                tone = tone.cuda(0)
            else:
                x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch
                tone = None

            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)

            # Take only first sample for evaluation
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            if tone is not None:
                tone = tone[:1]
            break

        y_hat, attn, mask, *_ = generator.module.infer(
            x, x_lengths, speakers, tone=tone, max_len=1000
        )
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
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
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    main()
