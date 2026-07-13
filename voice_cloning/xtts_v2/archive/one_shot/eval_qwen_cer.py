"""Evaluate Qwen-VL OCR baseline CER against corrected ground truth.

Input:
    labels_edited.csv (col: id, series, page, image, prefill, corrected_text, skip)
    bubble_crops/*.png

Process:
    For each row with corrected_text (skip!='x'):
        Load crop → Qwen-VL → predicted text
        Compute CER = Levenshtein(pred, gt) / len(gt)

Output:
    eval_qwen_cer.json — per-row predictions + per-series stats + overall
    eval_qwen_cer.md   — human-readable summary

Usage:
    python eval_qwen_cer.py --csv /home/bes/Desktop/Tin/labels_edited.csv \\
        --crops-dir /home/bes/Desktop/Tin/labeling_task_v4/bubble_crops \\
        --out-dir /home/bes/Desktop/Tin/labeling_task_v4/eval_qwen_3b \\
        --model-id Qwen/Qwen2.5-VL-3B-Instruct
"""
import argparse
import csv
import json
import time
from pathlib import Path
from collections import defaultdict


BUBBLE_OCR_PROMPT = """Đọc CHÍNH XÁC văn bản tiếng Việt trong bong bóng thoại này.

ĐẶC BIỆT CHÚ Ý từng dấu thanh điệu trên chữ hoa:
- Sắc (Á É Í Ó Ú Ý): nét xéo lên trên bên phải
- Huyền (À È Ì Ò Ù Ỳ): nét xéo xuống bên phải
- Hỏi (Ả Ẻ Ỉ Ỏ Ủ Ỷ): dấu móc (?) phía trên
- Ngã (Ã Ẽ Ĩ Õ Ũ Ỹ): dấu sóng (~) phía trên
- Nặng (Ạ Ẹ Ị Ọ Ụ Ỵ): dấu chấm phía DƯỚI chữ
- Dấu chữ: Ă (á trên a) Â (mũ) Đ (gạch ngang D) Ê Ô Ơ Ư

Trả về DUY NHẤT văn bản đọc được, không giải thích, không ngoặc kép.
Nếu không đọc được hoặc bubble rỗng, trả về chuỗi rỗng."""


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = cur_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            cur_row.append(min(ins, dele, sub))
        prev_row = cur_row
    return prev_row[-1]


def cer(pred: str, gt: str) -> float:
    pred = pred.strip()
    gt = gt.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def normalize(s: str) -> str:
    """Light normalization: lower + strip + collapse whitespace."""
    return " ".join(s.lower().strip().split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/bes/Desktop/Tin/labels_edited.csv")
    p.add_argument("--crops-dir", default="/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops")
    p.add_argument("--out-dir", default="/home/bes/Desktop/Tin/labeling_task_v4/eval_qwen_3b")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--limit", type=int, default=0,
                   help="Eval only N rows (0=all). Useful for sanity check.")
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Filter: has corrected_text and not skipped
    eval_rows = [
        r for r in rows
        if r.get("corrected_text", "").strip() and r.get("skip", "").strip() != "x"
    ]
    if args.limit > 0:
        eval_rows = eval_rows[: args.limit]

    print(f"Total rows in CSV: {len(rows)}")
    print(f"Rows to evaluate (has GT, not skipped): {len(eval_rows)}")

    # Load Qwen
    print(f"\nLoading {args.model_id}...")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    proc = AutoProcessor.from_pretrained(args.model_id)
    print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    @torch.no_grad()
    def qwen_predict(pil_crop):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_crop},
                {"type": "text",  "text": BUBBLE_OCR_PROMPT},
            ],
        }]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = proc(
            text=[text], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda:0")
        out_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        answer = proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        answer = answer.strip('"').strip("'").strip()
        if ":" in answer[:20]:
            head = answer.split(":", 1)[0].lower()
            if any(kw in head for kw in ("kết quả", "text", "đọc được", "văn bản")):
                answer = answer.split(":", 1)[1].strip()
        return answer

    # Eval loop
    crops_dir = Path(args.crops_dir)
    results = []
    series_buckets = defaultdict(list)
    t_start = time.time()
    n_oom = 0
    n_err = 0

    for idx, r in enumerate(eval_rows):
        crop_name = r["image"].split("/")[-1]
        crop_path = crops_dir / crop_name
        if not crop_path.exists():
            continue

        gt = r["corrected_text"].strip()
        try:
            pil = Image.open(crop_path).convert("RGB")
            t0 = time.time()
            pred = qwen_predict(pil)
            dt = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pred = "[OOM]"
            dt = 0.0
            n_oom += 1
        except Exception as e:
            pred = f"[ERR: {type(e).__name__}]"
            dt = 0.0
            n_err += 1

        cer_val = cer(pred, gt)
        cer_norm = cer(normalize(pred), normalize(gt))
        result = {
            "id": r["id"],
            "series": r.get("series", ""),
            "image": r["image"],
            "gt": gt,
            "pred": pred,
            "cer": round(cer_val, 4),
            "cer_norm": round(cer_norm, 4),
            "dt": round(dt, 2),
        }
        results.append(result)
        series_buckets[r.get("series", "unknown")].append(cer_norm)

        if (idx + 1) % 100 == 0 or idx == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (len(eval_rows) - idx - 1) / 60
            avg_cer = sum(x["cer_norm"] for x in results) / len(results)
            print(f"  [{idx+1:5d}/{len(eval_rows)}] elapsed={elapsed/60:.1f}m  eta={eta:.1f}m  "
                  f"avg_cer_norm={avg_cer:.3f}  oom={n_oom}  err={n_err}")
            # Sample bad cases
            if (idx + 1) % 500 == 0:
                worst = max(results[-100:], key=lambda x: x["cer_norm"])
                print(f"    worst recent: gt={worst['gt'][:50]!r} pred={worst['pred'][:50]!r}")

    # Per-series stats
    series_stats = {}
    for s, cers in series_buckets.items():
        if not cers:
            continue
        series_stats[s] = {
            "n": len(cers),
            "cer_mean": round(sum(cers) / len(cers), 4),
            "cer_median": round(sorted(cers)[len(cers)//2], 4),
            "exact_match_pct": round(100 * sum(1 for c in cers if c == 0) / len(cers), 2),
            "low_cer_pct (<0.1)": round(100 * sum(1 for c in cers if c < 0.1) / len(cers), 2),
            "high_cer_pct (>0.5)": round(100 * sum(1 for c in cers if c > 0.5) / len(cers), 2),
        }

    overall_cer = [r["cer_norm"] for r in results]
    overall = {
        "n": len(overall_cer),
        "cer_mean": round(sum(overall_cer) / max(len(overall_cer), 1), 4),
        "cer_median": round(sorted(overall_cer)[len(overall_cer)//2], 4) if overall_cer else 0,
        "exact_match_pct": round(100 * sum(1 for c in overall_cer if c == 0) / max(len(overall_cer), 1), 2),
        "low_cer_pct (<0.1)": round(100 * sum(1 for c in overall_cer if c < 0.1) / max(len(overall_cer), 1), 2),
        "high_cer_pct (>0.5)": round(100 * sum(1 for c in overall_cer if c > 0.5) / max(len(overall_cer), 1), 2),
        "n_oom": n_oom,
        "n_err": n_err,
    }

    # Save
    summary = {
        "model": args.model_id,
        "n_rows": len(eval_rows),
        "elapsed_sec": round(time.time() - t_start, 1),
        "overall": overall,
        "per_series": series_stats,
    }
    with open(out_dir / "eval_qwen_cer.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    # Markdown
    md = f"""# Qwen-VL OCR Baseline Evaluation

**Model:** {args.model_id}
**Rows evaluated:** {len(eval_rows)}
**Elapsed:** {summary['elapsed_sec']/60:.1f} minutes
**OOM/Errors:** {n_oom} / {n_err}

## Overall

| Metric | Value |
|--------|-------|
| Mean CER (normalized) | **{overall['cer_mean']:.3f}** |
| Median CER | {overall['cer_median']:.3f} |
| Exact match | {overall['exact_match_pct']:.1f}% |
| CER < 0.1 (close) | {overall['low_cer_pct (<0.1)']:.1f}% |
| CER > 0.5 (very wrong) | {overall['high_cer_pct (>0.5)']:.1f}% |

## Per Series

| Series | N | Mean CER | Exact match | CER<0.1 | CER>0.5 |
|--------|---|----------|-------------|---------|---------|
"""
    for s, st in sorted(series_stats.items()):
        md += (f"| {s} | {st['n']} | {st['cer_mean']:.3f} | "
               f"{st['exact_match_pct']:.1f}% | "
               f"{st['low_cer_pct (<0.1)']:.1f}% | "
               f"{st['high_cer_pct (>0.5)']:.1f}% |\n")

    # Worst cases
    md += "\n## 10 Worst Cases (highest CER)\n\n"
    worst = sorted([r for r in results if not r["pred"].startswith("[")],
                   key=lambda x: x["cer_norm"], reverse=True)[:10]
    for r in worst:
        md += f"- **[{r['id']}]** CER={r['cer_norm']:.3f}  series={r['series']}\n"
        md += f"  - GT:   `{r['gt'][:120]}`\n"
        md += f"  - PRED: `{r['pred'][:120]}`\n"

    (out_dir / "eval_qwen_cer.md").write_text(md, encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"Mean CER (normalized): {overall['cer_mean']:.3f}")
    print(f"Exact match: {overall['exact_match_pct']:.1f}%")
    print(f"\nResults: {out_dir}/eval_qwen_cer.json")
    print(f"Summary: {out_dir}/eval_qwen_cer.md")


if __name__ == "__main__":
    main()
