"""
Phase 1: Fine-tune ECAPA-TDNN Speaker Encoder on VieNeu-TTS.

Step A: Freeze encoder, train classifier only (verify embeddings work).
Step B: (future) Unfreeze encoder, fine-tune end-to-end.

Usage:
    cd /home/bes/Desktop/Tin
    python Capstone_project/voice_cloning/train_speaker_encoder.py \
        --train-filelist vits2_pytorch/filelists/vieneu_train_filelist.txt \
        --val-filelist   vits2_pytorch/filelists/vieneu_val_filelist.txt \
        --output-dir     /mnt/nfs-data/tin_dataset/checkpoints/speaker_encoder/ \
        --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from Capstone_project.voice_cloning.speaker_dataset import build_dataloaders
from Capstone_project.voice_cloning.speaker_encoder import (
    ECAPASpeakerEncoder, save_encoder, load_encoder,
)


# ── EER ───────────────────────────────────────────────────────────────────────

def compute_eer(model, held_out_ds, device, n_pairs=2000):
    model.eval()
    speakers = held_out_ds.speakers
    if len(speakers) < 2:
        return 1.0

    spk_embs: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for sid in speakers:
            idxs = held_out_ds.speaker_to_items[sid][:10]
            embs = []
            for i in idxs:
                wav, _, rel_len = held_out_ds[i]
                wav = wav.unsqueeze(0).to(device)
                wl = torch.FloatTensor([rel_len]).to(device)
                emb = model.encode(wav, wl).squeeze(0)
                if not emb.isnan().any():
                    embs.append(F.normalize(emb, dim=0))
            if embs:
                spk_embs[sid] = embs

    eligible = [s for s in spk_embs if len(spk_embs[s]) >= 2]
    if len(eligible) < 2:
        return 1.0

    scores, labels = [], []
    for _ in range(n_pairs):
        spk = random.choice(eligible)
        a, b = random.sample(spk_embs[spk], 2)
        scores.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        labels.append(1)

        sa, sb = random.sample(eligible, 2)
        ea, eb = random.choice(spk_embs[sa]), random.choice(spk_embs[sb])
        scores.append(F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item())
        labels.append(0)

    scores, labels = np.array(scores), np.array(labels)
    best = 1.0
    for t in np.linspace(scores.min(), scores.max(), 200):
        fpr = ((scores >= t) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        fnr = ((scores < t) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        best = min(best, (fpr + fnr) / 2)

    model.train()
    return float(best)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Held-out speakers
    all_spk = sorted(set(s for _, s in _read_filelist(args.train_filelist)))
    n_hold = max(10, min(20, len(all_spk) // 10))
    held_out = set(all_spk[::len(all_spk) // n_hold][:n_hold])
    print(f"Held-out speakers ({len(held_out)}): {sorted(held_out)}")
    with open(os.path.join(args.output_dir, "held_out_speakers.json"), "w") as f:
        json.dump(sorted(held_out), f)

    # Dataloaders
    train_loader, val_loader, held_out_ds = build_dataloaders(
        train_filelist=args.train_filelist,
        val_filelist=args.val_filelist,
        held_out_speakers=held_out,
        n_speakers=args.n_speakers_batch,
        m_utterances=args.m_utterances,
        n_batches_train=args.batches_per_epoch,
        n_batches_val=50,
        num_workers=args.num_workers,
    )

    n_train_spk = len(set(
        s for _, s in _read_filelist(args.train_filelist) if s not in held_out
    ))
    print(f"Training speakers: {n_train_spk}")

    # Model
    model = ECAPASpeakerEncoder(
        n_speakers=n_train_spk, projection_dim=256, device=device,
    ).to(device)

    # Only train classifier + projection (encoder is frozen via detach)
    optimizer = AdamW(
        list(model.classifier.parameters()) + list(model.projection.parameters()),
        lr=1e-2, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    start_epoch, best_eer = 0, 1.0
    if args.resume:
        ckpt = load_encoder(model, args.resume)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("eer", 1.0)

    log_path = os.path.join(args.output_dir, "train.log")
    metrics = []

    # ── Diagnostic ────────────────────────────────────────────────────────────
    print("\n--- Diagnostic: pre-trained embedding quality ---")
    eer_pretrained = compute_eer(model, held_out_ds, device)
    print(f"Pre-trained EER (before any training): {eer_pretrained:.4f}")
    if eer_pretrained > 0.4:
        print("WARNING: Pre-trained embeddings are near-random!")
        print("         Check audio loading / resampling.")
    else:
        print(f"Pre-trained embeddings OK (EER={eer_pretrained:.4f})")
    print("---\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_ok, n_nan = 0.0, 0, 0

        for wavs, sids, wav_lens in train_loader:
            wavs = wavs.to(device)
            sids = sids.to(device)
            wav_lens = wav_lens.to(device)

            optimizer.zero_grad()
            _, loss = model(wavs, speaker_ids=sids, wav_lens=wav_lens)

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                n_nan += 1
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_ok += 1

        scheduler.step()
        avg_loss = total_loss / max(n_ok, 1)

        # Validation
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for wavs, sids, wav_lens in val_loader:
                wavs = wavs.to(device)
                sids = sids.to(device)
                wav_lens = wav_lens.to(device)
                _, loss = model(wavs, speaker_ids=sids, wav_lens=wav_lens)
                if loss is not None and not torch.isnan(loss):
                    val_loss += loss.item()
                    n_val += 1
        val_loss /= max(n_val, 1)

        # EER every 2 epochs
        eer = best_eer
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            eer = compute_eer(model, held_out_ds, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs-1} | "
            f"loss={avg_loss:.4f} | val={val_loss:.4f} | "
            f"EER={eer:.4f} | ok={n_ok} nan={n_nan} | {elapsed:.0f}s"
        )

        entry = {"epoch": epoch, "loss": avg_loss, "val_loss": val_loss,
                 "eer": eer, "n_ok": n_ok, "n_nan": n_nan}
        metrics.append(entry)

        if eer < best_eer:
            best_eer = eer
            save_encoder(model, os.path.join(args.output_dir, "speaker_encoder_best.pth"),
                         epoch, eer)
            print(f"  ✓ New best EER={best_eer:.4f}")

        save_encoder(model, os.path.join(args.output_dir, "speaker_encoder_latest.pth"),
                     epoch, eer)

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if best_eer < 0.05:
            print("EER < 5% — early stopping")
            break

    print(f"\nDone. Best EER={best_eer:.4f}")
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"best_eer": best_eer, "history": metrics}, f, indent=2)


def _read_filelist(path):
    items = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                items.append((parts[0], int(parts[1])))
    return items


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-filelist", required=True)
    p.add_argument("--val-filelist", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-speakers-batch", type=int, default=16)
    p.add_argument("--m-utterances", type=int, default=4)
    p.add_argument("--batches-per-epoch", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
