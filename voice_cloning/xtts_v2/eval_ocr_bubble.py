"""Held-out eval for WHOLE-BUBBLE Qwen-VL OCR — honest before/after measurement.

Loads base Qwen2.5-VL + optional LoRA adapter, runs the SAME prompt as
finetune_qwen_lora.py (imported, so train==eval prompt is guaranteed), and reports
case-insensitive CER + exact-match, broken down by error_type
(correct / diacritic / structural). Matches v2 inference convention
(whole bubble, do_sample=False, repetition_penalty=1.2).

Usage (BEFORE = v2):
    conda run -n comic_ocr python eval_ocr_bubble.py \\
        --csv /home/bes/Desktop/Tin/csv/ocr_ft_eval.csv \\
        --crops-dir /home/bes/Desktop/Tin/labeling_task_v4/bubble_crops \\
        --adapter /mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_7b_vncomic_uppercase_lora_v2/best \\
        --out-csv /home/bes/Desktop/Tin/csv/eval_pred_v2.csv

Omit --adapter to eval the base model. Use the v3 adapter for the AFTER run.
"""
import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

# Reuse the exact prompt the model was fine-tuned with (same directory module).
from finetune_qwen_lora import SYSTEM_PROMPT, USER_PROMPT


def norm(s: str) -> str:
    """NFC + collapse whitespace + uppercase (case-insensitive, like project eval)."""
    s = unicodedata.normalize("NFC", (s or "").strip())
    return re.sub(r"\s+", " ", s).upper()


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Eval CSV with id,image,corrected_text[,error_type]")
    p.add_argument("--crops-dir", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model")
    p.add_argument("--max-pixels", type=int, default=512 * 512)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--out-csv", default=None, help="Optional: save per-row predictions")
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded adapter: {args.adapter}")
    else:
        print("Base model (no adapter)")
    model.eval()

    crops = Path(args.crops_dir)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    num, den, cnt, exact = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    out_rows = []

    for i, r in enumerate(rows):
        img_str = r["image"]
        img_path = Path(img_str) if img_str.startswith("/") else crops / Path(img_str).name
        if not img_path.exists():
            continue
        gt = (r.get("corrected_text") or "").strip()
        et = r.get("error_type") or "all"

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if w * h > args.max_pixels * 4:  # whole bubble allowed bigger than train crops
            sc = (args.max_pixels * 4 / (w * h)) ** 0.5
            image = image.resize((int(w * sc), int(h * sc)), Image.LANCZOS)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inp, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=img_inp, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=args.repetition_penalty,
            )
        pred = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        npd, ngt = norm(pred), norm(gt)
        d, L = lev(npd, ngt), max(len(ngt), 1)
        for k in ("all", et):
            num[k] += d
            den[k] += L
            cnt[k] += 1
            exact[k] += int(npd == ngt)
        out_rows.append({"id": r["id"], "error_type": et, "gt": gt, "pred": pred})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}")

    print("\n=== CER by group (case-insensitive, NFC, whitespace-normalized) ===")
    for k in ("all", "correct", "diacritic", "structural"):
        if cnt.get(k):
            print(f"  {k:11s}: CER={num[k] / den[k] * 100:5.2f}%  "
                  f"exact={exact[k] / cnt[k] * 100:5.1f}%  (n={cnt[k]})")

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "error_type", "gt", "pred"])
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nPredictions saved: {args.out_csv}")


if __name__ == "__main__":
    main()
