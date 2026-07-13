"""Apply validation layer on existing Qwen eval results to detect hallucinations.

Reads eval_qwen_cer.json (with raw Qwen predictions + ground truth).
Applies validation rules to flag hallucinated outputs.
Computes:
  - Hallucination detection precision/recall
  - Effective CER (only on rows that pass validation)
  - Per-series breakdown

Outputs:
  - validation_report.md
  - validated_results.json (per-row: pred, gt, cer, hallucinated, reason)

Decision rules from observed patterns in eval_qwen_cer.md:
  Type 1: Echo prompt-like keywords  ("để trả lời", "chúng ta cần", "đặt biệt chú ý", ...)
  Type 2: Char/substring repetition  (>10 same char, or >5 repeats of 2-4 char unit)
  Type 3: Excessive length             (pred way longer than reasonable bubble text)
"""
import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


# Phrases that indicate Qwen "talking about" the bubble instead of reading it.
# These almost never appear at the START of legitimate comic dialogue.
PHRASES_AT_START = [
    "để trả lời",
    "chúng ta cần",
    "đây là một",
    "đây là một câu hỏi",
    "đây là một hình ảnh",
    "đây là một văn bản",
    "tôi không thể",
    "tôi có thể",
    "vâng, tôi",
    "bạn muốn tôi",
    "trong hình ảnh",
    "trong trường hợp này",
    "kết quả:",
    "đặt biệt chú ý",
    "đặc biệt chú ý",
    "1. sắc",
    "tiếng việt:",
    "dấu thanh điệu",
]

# Phrases that are clear refusals/prompt-echoes anywhere in the text
PHRASES_ANYWHERE = [
    "không thể đọc được văn bản",
    "không thể xác định nội dung",
    "không có văn bản trong",
    "trả về chuỗi rỗng",
    "bong bóng thoại này cho",
    "bong bóng thoại không",
    "đọc lại văn bản",
    "có thể giúp bạn",
    "bạn cần đọc kỹ",
    "đọc kỹ và",
]


def detect_hallucination(pred, gt_len=None):
    """Return (is_hallucination, reason) for a prediction string."""
    pred_low = pred.lower().strip()

    if not pred_low:
        return False, ""  # empty is not hallucination, just blank

    # Type 1a: Echoing prompt at the START (Qwen explaining instead of reading)
    head = pred_low[:60]
    for phrase in PHRASES_AT_START:
        if head.startswith(phrase):
            return True, f"start_echo: '{phrase}'"

    # Type 1b: Refusal / chat-style anywhere in text
    for phrase in PHRASES_ANYWHERE:
        if phrase in pred_low:
            return True, f"refusal_echo: '{phrase}'"

    # Type 2a: Single char repeated >= 10 times consecutively
    m = re.search(r'(.)\1{9,}', pred)
    if m:
        return True, f"char_repeat: '{m.group(0)[:20]}'"

    # Type 2b: Substring 2-4 chars repeated >= 5 times consecutively
    m = re.search(r'(.{2,4})\1{4,}', pred)
    if m:
        return True, f"substr_repeat: '{m.group(0)[:30]}'"

    # Type 3: Excessive length (compared to gt)
    # If gt provided: pred shouldn't be more than 5x gt length (or > 200 abs chars)
    if gt_len is not None:
        if len(pred) > 200 and len(pred) > 5 * max(gt_len, 10):
            return True, f"too_long: {len(pred)} chars (gt={gt_len})"
    elif len(pred) > 300:
        return True, f"too_long_abs: {len(pred)} chars"

    # Type 4: Mostly non-Vietnamese (suspicious for VN comic)
    vn_chars = "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵaàảãáạăâbcdđeèẻẽéẹêfghiìỉĩíịjklmnoòỏõóọôơpqrstuùủũúụưvwxyýỳỷỹỵz "
    vn_chars_set = set(vn_chars + vn_chars.upper())
    text_letters = [c for c in pred if c.isalpha()]
    if text_letters:
        vn_ratio = sum(1 for c in text_letters if c in vn_chars_set) / len(text_letters)
        if vn_ratio < 0.5 and len(text_letters) > 5:
            return True, f"non_vn: {vn_ratio:.2f} VN char ratio"

    return False, ""


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
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
    p.add_argument("--in-json", default="/home/bes/Desktop/Tin/labeling_task_v4/eval_qwen_3b/eval_qwen_cer.json")
    p.add_argument("--out-dir", default="/home/bes/Desktop/Tin/labeling_task_v4/eval_qwen_3b")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.in_json) as f:
        data = json.load(f)
    results = data["results"]
    print(f"Loaded {len(results)} eval results")

    # Apply validation
    validated = []
    series_stats = defaultdict(lambda: {
        "n": 0, "n_halluc": 0, "n_pass": 0,
        "cer_all": [], "cer_pass": [],
        "halluc_reasons": defaultdict(int),
    })

    for r in results:
        pred = r.get("pred", "")
        gt = r.get("gt", "")
        is_h, reason = detect_hallucination(pred, gt_len=len(gt))
        cer_all = cer(normalize(pred), normalize(gt))
        cer_pass = cer_all if not is_h else None  # exclude from "effective CER"

        out_row = {
            **r,
            "hallucination": is_h,
            "halluc_reason": reason,
            "cer_norm": round(cer_all, 4),
        }
        validated.append(out_row)

        s = r.get("series", "unknown")
        series_stats[s]["n"] += 1
        series_stats[s]["cer_all"].append(cer_all)
        if is_h:
            series_stats[s]["n_halluc"] += 1
            reason_key = reason.split(":")[0]
            series_stats[s]["halluc_reasons"][reason_key] += 1
        else:
            series_stats[s]["n_pass"] += 1
            series_stats[s]["cer_pass"].append(cer_all)

    # Aggregate
    def stats_summary(cers):
        if not cers:
            return {"n": 0, "mean": 0, "median": 0, "lt0_1": 0, "gt0_5": 0}
        cers_sorted = sorted(cers)
        return {
            "n": len(cers),
            "mean": round(sum(cers) / len(cers), 4),
            "median": round(cers_sorted[len(cers)//2], 4),
            "lt0_1": round(100 * sum(1 for c in cers if c < 0.1) / len(cers), 1),
            "gt0_5": round(100 * sum(1 for c in cers if c > 0.5) / len(cers), 1),
        }

    overall_all = [r["cer_norm"] for r in validated]
    overall_pass = [r["cer_norm"] for r in validated if not r["hallucination"]]
    halluc_total = sum(1 for r in validated if r["hallucination"])

    # All hallucination reason counts
    all_reasons = defaultdict(int)
    for r in validated:
        if r["hallucination"]:
            reason_key = r["halluc_reason"].split(":")[0]
            all_reasons[reason_key] += 1

    summary = {
        "total_rows": len(validated),
        "n_hallucinated": halluc_total,
        "halluc_pct": round(100 * halluc_total / len(validated), 2),
        "before_validation": {
            "cer_mean": stats_summary(overall_all)["mean"],
            "cer_median": stats_summary(overall_all)["median"],
        },
        "after_validation": {
            "n_pass": len(overall_pass),
            "cer_mean": stats_summary(overall_pass)["mean"],
            "cer_median": stats_summary(overall_pass)["median"],
            "cer_lt0_1_pct": stats_summary(overall_pass)["lt0_1"],
            "cer_gt0_5_pct": stats_summary(overall_pass)["gt0_5"],
        },
        "halluc_reasons": dict(all_reasons),
    }

    # Write Markdown
    md = f"""# Qwen-3B + Validation Layer — Effective CER

## Summary

| Metric | Before validation | After validation (filter halluc) |
|--------|-------------------|----------------------------------|
| N rows | {summary['total_rows']} | {summary['after_validation']['n_pass']} (filtered out {halluc_total}) |
| Mean CER | {summary['before_validation']['cer_mean']:.3f} | **{summary['after_validation']['cer_mean']:.3f}** |
| Median CER | {summary['before_validation']['cer_median']:.3f} | {summary['after_validation']['cer_median']:.3f} |
| CER < 0.1 | — | {summary['after_validation']['cer_lt0_1_pct']:.1f}% |
| CER > 0.5 | — | {summary['after_validation']['cer_gt0_5_pct']:.1f}% |

**Hallucination rate:** {halluc_total} / {len(validated)} ({summary['halluc_pct']:.1f}%)

## Hallucination breakdown

| Reason | Count |
|--------|-------|
"""
    for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1]):
        md += f"| {reason} | {count} |\n"

    md += "\n## Per-series\n\n"
    md += "| Series | N | Halluc% | CER all | CER pass-only |\n"
    md += "|--------|---|---------|---------|---------------|\n"
    for s, st in sorted(series_stats.items()):
        all_st = stats_summary(st["cer_all"])
        pass_st = stats_summary(st["cer_pass"])
        halluc_pct = 100 * st["n_halluc"] / st["n"]
        md += (f"| {s} | {st['n']} | {halluc_pct:.1f}% | "
               f"{all_st['mean']:.3f} | **{pass_st['mean']:.3f}** |\n")

    # Sample correctly-flagged hallucinations
    md += "\n## Examples of caught hallucinations\n\n"
    halluc_examples = [r for r in validated if r["hallucination"]][:8]
    for r in halluc_examples:
        md += f"- **[{r['id']}]** {r['halluc_reason']}\n"
        md += f"  - GT: `{r['gt'][:80]}`\n"
        md += f"  - PRED: `{r['pred'][:120]}`\n"

    (out_dir / "validation_report.md").write_text(md, encoding="utf-8")

    with open(out_dir / "validated_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": validated}, f, ensure_ascii=False, indent=2)

    print(f"\n=== Validation Layer Results ===")
    print(f"Total rows:              {summary['total_rows']}")
    print(f"Hallucinated (flagged):  {halluc_total} ({summary['halluc_pct']:.1f}%)")
    print(f"Pass validation:         {len(overall_pass)}")
    print(f"")
    print(f"  Before validation:")
    print(f"    Mean CER:   {summary['before_validation']['cer_mean']:.3f}")
    print(f"    Median CER: {summary['before_validation']['cer_median']:.3f}")
    print(f"  After validation (only pass):")
    print(f"    Mean CER:   {summary['after_validation']['cer_mean']:.3f}")
    print(f"    Median CER: {summary['after_validation']['cer_median']:.3f}")
    print(f"")
    print(f"Reports: {out_dir}/validation_report.md + validated_results.json")


if __name__ == "__main__":
    main()
