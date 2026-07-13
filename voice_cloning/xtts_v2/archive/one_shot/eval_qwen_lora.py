"""Evaluate fine-tuned Qwen2.5-VL-3B (LoRA adapter) on validation set.

Loads base model + LoRA adapter, runs OCR on validation rows from data_split.json,
computes CER + hallucination stats. Comparable to baseline eval_qwen_cer.py.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from collections import defaultdict


SYSTEM_PROMPT = """Bạn là OCR engine đọc bong bóng thoại tiếng Việt trong truyện tranh.

QUY TẮC TUYỆT ĐỐI:
1. CHỈ trả về văn bản đọc được trong bong bóng. KHÔNG giải thích, KHÔNG mô tả.
2. Nếu không đọc được rõ HOẶC bubble không có chữ → trả về CHUỖI RỖNG (không gõ gì).
3. KHÔNG bịa thêm text. KHÔNG echo prompt. KHÔNG nói "Tôi không thể...".
4. Giữ chính xác dấu thanh điệu (sắc/huyền/hỏi/ngã/nặng) và dấu chữ (ă â đ ê ô ơ ư).
5. Output viết chữ thường (lowercase), trừ tên riêng và đầu câu."""

USER_PROMPT = "Đọc văn bản trong bong bóng thoại này. Trả về chỉ văn bản đọc được, hoặc chuỗi rỗng nếu không có."


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j+1]+1, cur[j]+1, prev[j]+(c1 != c2)))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def normalize(s):
    return " ".join(s.lower().strip().split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/bes/Desktop/Tin/labels_edited.csv")
    p.add_argument("--crops-dir", default="/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops")
    p.add_argument("--lora-dir", default="/mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_3b_vncomic_lora_v1/best")
    p.add_argument("--data-split", default="/mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_3b_vncomic_lora_v1/data_split.json")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--out-dir", default="/mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_3b_vncomic_lora_v1/eval_full")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV + filter to valid rows from split
    with open(args.csv, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    with open(args.data_split) as f:
        split = json.load(f)
    valid_ids = set(split["valid_ids"])
    valid_rows = [r for r in all_rows if r["id"] in valid_ids]
    if args.limit > 0:
        valid_rows = valid_rows[: args.limit]
    print(f"Validation rows: {len(valid_rows)}")

    # Load model + LoRA
    print(f"\nLoading {args.base_model} + LoRA from {args.lora_dir}...")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.eval()
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    @torch.no_grad()
    def predict(pil):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inp, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=img_inp, padding=True, return_tensors="pt").to("cuda:0")
        out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        ans = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        return ans

    # Eval loop
    results = []
    series_buckets = defaultdict(list)
    crops_dir = Path(args.crops_dir)
    n_halluc = 0
    n_long = 0
    n_exact = 0
    t0 = time.time()

    for idx, r in enumerate(valid_rows):
        crop_path = crops_dir / r["image"].split("/")[-1]
        if not crop_path.exists():
            continue
        gt = r.get("corrected_text", "").strip()
        if r.get("skip", "").strip() == "x":
            gt = ""

        pil = Image.open(crop_path).convert("RGB")
        try:
            pred = predict(pil)
        except Exception as e:
            pred = f"[ERR: {type(e).__name__}]"

        c_norm = cer(normalize(pred), normalize(gt))
        # Hallucination check (post-fine-tune patterns)
        is_halluc = ("tôi không thể" in pred.lower() or "đây là một" in pred.lower() or
                     "để trả lời" in pred.lower() or "trả về" in pred.lower())
        is_long = len(pred) > max(len(gt) * 5, 200)
        if is_halluc:
            n_halluc += 1
        if is_long:
            n_long += 1
        if c_norm == 0:
            n_exact += 1

        results.append({
            "id": r["id"], "series": r.get("series", ""),
            "gt": gt, "pred": pred, "cer": round(c_norm, 4),
            "halluc": is_halluc, "long": is_long,
        })
        series_buckets[r.get("series", "?")].append(c_norm)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            avg = sum(x["cer"] for x in results) / len(results)
            print(f"  [{idx+1:4d}/{len(valid_rows)}]  "
                  f"avg_cer={avg:.4f}  exact={n_exact}  halluc={n_halluc}  long={n_long}  "
                  f"elapsed={elapsed/60:.1f}m")

    # Stats
    cers = [r["cer"] for r in results]
    cers_sorted = sorted(cers)
    overall = {
        "n": len(cers),
        "mean_cer": round(sum(cers) / max(len(cers), 1), 4),
        "median_cer": round(cers_sorted[len(cers)//2], 4) if cers else 0,
        "exact_match_pct": round(100 * n_exact / max(len(cers), 1), 2),
        "lt_0_1_pct": round(100 * sum(1 for c in cers if c < 0.1) / max(len(cers), 1), 2),
        "gt_0_5_pct": round(100 * sum(1 for c in cers if c > 0.5) / max(len(cers), 1), 2),
        "halluc_count": n_halluc,
        "long_count": n_long,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }

    # Per series
    per_series = {}
    for s, cs in series_buckets.items():
        per_series[s] = {
            "n": len(cs),
            "mean": round(sum(cs) / len(cs), 4),
            "median": round(sorted(cs)[len(cs)//2], 4),
            "lt_0_1_pct": round(100 * sum(1 for c in cs if c < 0.1) / len(cs), 2),
        }

    # Markdown
    md = f"""# Qwen-VL-3B Fine-tuned (LoRA) — Full Eval

**Adapter:** {args.lora_dir}
**Validation rows:** {len(results)}
**Elapsed:** {overall['elapsed_min']} minutes

## Comparison with Baseline (Qwen-3B no FT)

| Metric | Baseline (no FT) | Fine-tuned (LoRA) |
|--------|------------------|-------------------|
| Mean CER | 0.294 | **{overall['mean_cer']:.4f}** |
| Mean CER (after validation filter) | 0.149 | (no filter needed) |
| Median CER | 0.091 | **{overall['median_cer']:.4f}** |
| Exact match | 4.0% | **{overall['exact_match_pct']:.2f}%** |
| CER < 0.1 | 53.1% | **{overall['lt_0_1_pct']:.2f}%** |
| CER > 0.5 | 6.4% | **{overall['gt_0_5_pct']:.2f}%** |
| Hallucination | 1.6% (90/5694) | **{n_halluc}/{len(results)} ({100*n_halluc/max(len(results),1):.2f}%)** |

## Per Series

| Series | N | Mean CER | Median | CER < 0.1 |
|--------|---|----------|--------|-----------|
"""
    for s, st in sorted(per_series.items()):
        md += f"| {s} | {st['n']} | {st['mean']:.4f} | {st['median']:.4f} | {st['lt_0_1_pct']:.1f}% |\n"

    # Worst cases
    md += "\n## 10 Worst Cases\n\n"
    worst = sorted(results, key=lambda x: x["cer"], reverse=True)[:10]
    for r in worst:
        md += f"- **[{r['id']}]** CER={r['cer']:.3f}  series={r['series']}\n"
        md += f"  - GT:   `{r['gt'][:100]}`\n"
        md += f"  - PRED: `{r['pred'][:100]}`\n"

    (out_dir / "eval_full_lora.md").write_text(md, encoding="utf-8")
    with open(out_dir / "eval_full_lora.json", "w", encoding="utf-8") as f:
        json.dump({"summary": {**overall, "per_series": per_series}, "results": results},
                  f, ensure_ascii=False, indent=2)

    print(f"\n=== DONE ===")
    print(f"Mean CER: {overall['mean_cer']:.4f}")
    print(f"Median CER: {overall['median_cer']:.4f}")
    print(f"Exact match: {overall['exact_match_pct']:.2f}%")
    print(f"Hallucination: {n_halluc}/{len(results)}")
    print(f"\nResults: {out_dir}/eval_full_lora.md + .json")


if __name__ == "__main__":
    main()
