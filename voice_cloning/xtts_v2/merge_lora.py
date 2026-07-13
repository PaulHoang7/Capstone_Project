"""Merge LoRA delta into base XTTS weights → standalone inference ckpt.

Reads a LoRA training checkpoint (`best_model.pth` from `train_xtts_lora.py`)
where state_dict has:
  - xtts.gpt.gpt.base_model.model.<...>.base_layer.weight   (frozen base)
  - xtts.gpt.gpt.base_model.model.<...>.lora_A.default.weight
  - xtts.gpt.gpt.base_model.model.<...>.lora_B.default.weight

For each (base_layer, lora_A, lora_B) triplet, compute:
    merged = base_layer + (lora_B @ lora_A) * (alpha / rank)

Then strip the peft wrapper paths so the keys match vanilla Xtts.state_dict()
layout, save as `model.pth` in a fresh ckpt-dir for inference.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import torch

VIXTTS_DIR = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts"
# LoRA training hyperparameters — MUST match train_xtts_lora.py
LORA_ALPHA = 32
LORA_RANK  = 16


def merge_lora_into_base(sd: dict, alpha: float, rank: int) -> dict:
    """Walk sd, fold LoRA delta into base_layer for each layer triplet,
    then strip peft wrappers so keys match vanilla Xtts state_dict."""
    scale = alpha / rank
    # Group by parameter prefix (everything before `.base_layer` / `.lora_A` / `.lora_B`)
    # e.g. 'xtts.gpt.gpt.base_model.model.h.0.attn.c_attn'
    merged = {}
    consumed = set()

    for key, tensor in sd.items():
        if "base_layer" in key:
            # Build companion key names
            prefix = key.rsplit(".base_layer", 1)[0]
            suffix = key.rsplit(".base_layer", 1)[1]  # ".weight" or ".bias"
            if suffix == ".weight":
                a_key = f"{prefix}.lora_A.default.weight"
                b_key = f"{prefix}.lora_B.default.weight"
                base_w = tensor.clone()
                if a_key in sd and b_key in sd:
                    A = sd[a_key]      # [rank, in_features]
                    B = sd[b_key]      # [out_features, rank]
                    # HF GPT2 Conv1D stores weight as [in_features, out_features],
                    # so the LoRA delta layout follows: delta = (B @ A).T * scale
                    # Detect orientation by shape — Conv1D weight is (in, out).
                    if base_w.shape == (B.shape[0], A.shape[1]):
                        # nn.Linear orientation: weight = (out, in)
                        delta = (B @ A) * scale
                    else:
                        # Conv1D orientation: weight = (in, out) → delta transposed
                        delta = (A.T @ B.T) * scale
                    base_w = base_w + delta
                    consumed.update({a_key, b_key})
                # Strip peft wrapper from key
                # 'xtts.gpt.gpt.base_model.model.h.0.attn.c_attn.base_layer.weight'
                # → 'xtts.gpt.gpt.h.0.attn.c_attn.weight'
                new_key = (key
                           .replace(".base_model.model.", ".")
                           .replace(".base_layer", ""))
                merged[new_key] = base_w
                consumed.add(key)
            elif suffix == ".bias":
                new_key = (key
                           .replace(".base_model.model.", ".")
                           .replace(".base_layer", ""))
                merged[new_key] = tensor
                consumed.add(key)

    # Copy every other key as-is (or with peft-prefix stripped if it has it)
    for key, tensor in sd.items():
        if key in consumed:
            continue
        if "lora_A" in key or "lora_B" in key:
            # Should have been consumed; defensive skip
            continue
        # Strip peft wrappers if present, otherwise keep key untouched
        new_key = key.replace(".base_model.model.", ".")
        merged[new_key] = tensor

    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-ckpt", required=True, help="best_model.pth from LoRA training")
    p.add_argument("--out-dir", required=True, help="Output inference ckpt-dir")
    p.add_argument("--alpha", type=float, default=LORA_ALPHA)
    p.add_argument("--rank", type=int, default=LORA_RANK)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading {args.lora_ckpt}")
    raw = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    print(f"  total keys: {len(sd)}")

    print(f"[merge] folding LoRA delta (alpha={args.alpha}, rank={args.rank}, scale={args.alpha/args.rank})")
    merged = merge_lora_into_base(sd, args.alpha, args.rank)
    print(f"  merged keys: {len(merged)}")

    # Save in {"model": sd} format — matches what Xtts.load_checkpoint expects
    out_model = out / "model.pth"
    torch.save({"model": merged}, out_model)
    print(f"[merge] saved → {out_model} ({out_model.stat().st_size / 1e9:.2f} GB)")

    # Copy config + vocab from viXTTS
    for fname in ("config.json", "vocab.json"):
        shutil.copy2(Path(VIXTTS_DIR) / fname, out / fname)
        print(f"  copied → {out / fname}")


if __name__ == "__main__":
    main()
