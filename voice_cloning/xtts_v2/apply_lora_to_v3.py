"""Apply LoRA delta (trained on baseline FT) to v3 base weights.

CAVEAT: LoRA was trained where base was xtts_vieneu_ft. v3 has different
base weights (CTC-tuned). Applying delta on top is HACK — not theoretically
grounded, but quick test to see if it helps or hurts.

Math:
    For each (c_attn, c_proj) layer in GPT2 of v3:
        merged_w = v3_w + (B @ A) * (alpha / rank)
                                       ^^^^^^^^^^^^
                                       LoRA delta computed from training ckpt

Output: standalone inference ckpt-dir with merged model.pth.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import torch

VIXTTS_DIR = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts"
LORA_ALPHA = 32
LORA_RANK  = 16


def apply_lora_delta(v3_sd: dict, lora_sd: dict, alpha: float, rank: int) -> dict:
    """Compute LoRA deltas from lora_sd, add to corresponding v3 weights."""
    scale = alpha / rank
    merged = {k: v.clone() for k, v in v3_sd.items()}
    n_applied = 0
    n_skipped = 0

    # Walk lora_sd to find LoRA A/B pairs
    lora_a_keys = sorted(k for k in lora_sd if k.endswith(".lora_A.default.weight"))
    for a_key in lora_a_keys:
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if b_key not in lora_sd:
            continue
        # Get base_layer path → map to v3 key
        # 'xtts.gpt.gpt.base_model.model.h.0.attn.c_attn.lora_A.default.weight'
        # → 'xtts.gpt.gpt.h.0.attn.c_attn.weight'  (v3 layout)
        prefix = a_key.rsplit(".lora_A.default.weight", 1)[0]
        v3_key = (prefix
                  .replace(".base_model.model.", ".")  # strip peft path
                  + ".weight")
        # v3_inference state_dict has keys WITHOUT 'xtts.' prefix
        # (Xtts.state_dict() doesn't have it), so strip it
        v3_key_stripped = v3_key.removeprefix("xtts.")
        if v3_key_stripped not in merged:
            n_skipped += 1
            continue

        A = lora_sd[a_key]            # [rank, in_features] or [rank, ?]
        B = lora_sd[b_key]            # [out_features, rank] or [?, rank]
        v3_w = merged[v3_key_stripped]

        # Conv1D in HF GPT2 stores weight as (in, out) — different from nn.Linear
        # Detect orientation: A shape gives us rank; B shape gives us (?, rank)
        # delta should match v3_w shape.
        if v3_w.shape == (B.shape[0], A.shape[1]):
            # Linear-style: (out, in)
            delta = (B @ A) * scale
        elif v3_w.shape == (A.shape[1], B.shape[0]):
            # Conv1D-style: (in, out)
            delta = (A.T @ B.T) * scale
        else:
            print(f"  [skip] shape mismatch {v3_key_stripped} = {tuple(v3_w.shape)} "
                  f"vs A {tuple(A.shape)}, B {tuple(B.shape)}")
            n_skipped += 1
            continue

        merged[v3_key_stripped] = v3_w + delta
        n_applied += 1

    return merged, n_applied, n_skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v3-ckpt", default="/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ctc_v3_inference/model.pth")
    p.add_argument("--lora-ckpt", required=True, help="best_model.pth from LoRA training")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alpha", type=float, default=LORA_ALPHA)
    p.add_argument("--rank", type=int, default=LORA_RANK)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[v3+lora] loading v3 base: {args.v3_ckpt}")
    v3_raw = torch.load(args.v3_ckpt, map_location="cpu", weights_only=False)
    v3_sd = v3_raw["model"] if isinstance(v3_raw, dict) and "model" in v3_raw else v3_raw
    print(f"  v3 keys: {len(v3_sd)}")

    print(f"[v3+lora] loading LoRA ckpt: {args.lora_ckpt}")
    lora_raw = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    lora_sd = lora_raw["model"] if isinstance(lora_raw, dict) and "model" in lora_raw else lora_raw
    lora_keys = [k for k in lora_sd if "lora_A" in k or "lora_B" in k]
    print(f"  LoRA matrix keys: {len(lora_keys)}")

    print(f"[v3+lora] applying LoRA delta (alpha={args.alpha}, rank={args.rank}, scale={args.alpha/args.rank})")
    merged, n_applied, n_skipped = apply_lora_delta(v3_sd, lora_sd, args.alpha, args.rank)
    print(f"  applied: {n_applied} layers, skipped: {n_skipped}")

    out_model = out / "model.pth"
    torch.save({"model": merged}, out_model)
    print(f"[v3+lora] saved → {out_model} ({out_model.stat().st_size/1e9:.2f} GB)")

    for fname in ("config.json", "vocab.json"):
        shutil.copy2(Path(VIXTTS_DIR) / fname, out / fname)
        print(f"  copied → {out / fname}")


if __name__ == "__main__":
    main()
