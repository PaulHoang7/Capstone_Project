"""Compare two predicted CSVs (e.g., baseline Qwen-VL vs LoRA-FT) against ground truth.

Computes per-row CER and aggregates with breakdowns relevant to this project:
    - Overall mean CER
    - Upper-only bubbles  (all alphabetic chars are uppercase) — the issue we're targeting
    - Lower/mixed bubbles
    - Empty-ground-truth handling (skip='1' rows)
    - Avg seconds per bubble (speed)
    - Per-series breakdown

Inputs must share id+image columns and a predicted_text column produced by infer_line_ocr_vn.py.
Ground truth read from --gt-csv (labels_vn.csv) — corrected_text column.

Usage:
    conda run -n comic_ocr python eval_line_ocr_vn.py \\
        --gt-csv /home/bes/Desktop/Tin/labeling_task_v4/labels_vn.csv \\
        --pred-baseline labels_vn_baseline.csv \\
        --pred-ft       labels_vn_predicted.csv \\
        --out-report    eval_line_ocr_report.json
"""
import argparse
import csv
import json
from collections import defaultdict


def cer(pred, gt, case_insensitive=True):
    pred = (pred or "").strip()
    gt = (gt or "").strip()
    if case_insensitive:
        pred = pred.lower()
        gt = gt.lower()
    if not gt:
        return 0.0 if not pred else 1.0
    s1, s2 = pred, gt
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1) / max(len(gt), 1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1] / len(gt)


def is_upper_only(s):
    s = s or ""
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    return all(c == c.upper() for c in letters)


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(rows, label, gt_map):
    """rows: dict id → {predicted_text, infer_sec, series, ...}; gt_map: dict id → gt row."""
    all_cers = []
    upper_cers = []
    lower_cers = []
    by_series = defaultdict(list)
    secs = []
    n_skip_gt_empty = 0
    n_pred_empty = 0
    n_compared = 0

    for rid, pr in rows.items():
        gt_row = gt_map.get(rid)
        if not gt_row:
            continue
        gt_text = (gt_row.get("corrected_text") or "").strip()
        is_skip = (gt_row.get("skip") or "").strip() in ("1", "true", "x")
        if is_skip and not gt_text:
            gt_text = ""

        pred_text = (pr.get("predicted_text") or "").strip()
        if not gt_text:
            n_skip_gt_empty += 1
            # Score: if pred non-empty on skip-row, CER=1; else 0
            c = 1.0 if pred_text else 0.0
        else:
            c = cer(pred_text, gt_text)

        all_cers.append(c)
        if is_upper_only(gt_text):
            upper_cers.append(c)
        elif gt_text:
            lower_cers.append(c)

        by_series[gt_row.get("series", "?")].append(c)

        try:
            secs.append(float(pr.get("infer_sec", "0") or 0))
        except ValueError:
            pass

        if not pred_text:
            n_pred_empty += 1
        n_compared += 1

    return {
        "label": label,
        "n_compared": n_compared,
        "n_gt_empty": n_skip_gt_empty,
        "n_pred_empty": n_pred_empty,
        "mean_cer": safe_mean(all_cers),
        "median_cer": sorted(all_cers)[len(all_cers) // 2] if all_cers else None,
        "upper_only_n": len(upper_cers),
        "upper_only_cer": safe_mean(upper_cers),
        "lower_mixed_n": len(lower_cers),
        "lower_mixed_cer": safe_mean(lower_cers),
        "avg_sec_per_bubble": safe_mean(secs),
        "per_series_cer": {s: safe_mean(v) for s, v in by_series.items()},
    }


def print_report(report):
    def f(x):
        return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) and abs(x) < 100 else f"{x:.2f}")

    print()
    print("=" * 70)
    print("EVAL REPORT")
    print("=" * 70)
    for s in report["summaries"]:
        print(f"\n[{s['label']}]")
        print(f"  Compared       : {s['n_compared']}  (gt_empty={s['n_gt_empty']}, pred_empty={s['n_pred_empty']})")
        print(f"  Mean CER       : {f(s['mean_cer'])}")
        print(f"  Median CER     : {f(s['median_cer'])}")
        print(f"  Upper-only CER : {f(s['upper_only_cer'])}  (n={s['upper_only_n']})")
        print(f"  Lower/mixed CER: {f(s['lower_mixed_cer'])}  (n={s['lower_mixed_n']})")
        print(f"  Avg sec/bubble : {f(s['avg_sec_per_bubble'])}")
        for k, v in s["per_series_cer"].items():
            print(f"    series {k!r}: CER {f(v)}")

    if "delta" in report:
        d = report["delta"]
        print("\n[DELTA = FT - baseline (negative = FT improves)]")
        for k, v in d.items():
            print(f"  {k}: {f(v)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--pred-baseline", required=True)
    p.add_argument("--pred-ft", required=True)
    p.add_argument("--out-report", required=True)
    args = p.parse_args()

    gt_map = load_csv(args.gt_csv)
    baseline_map = load_csv(args.pred_baseline)
    ft_map = load_csv(args.pred_ft)

    s_base = summarize(baseline_map, "baseline (no LoRA)", gt_map)
    s_ft = summarize(ft_map, "fine-tuned (line LoRA)", gt_map)

    delta = {}
    for k in ("mean_cer", "upper_only_cer", "lower_mixed_cer", "avg_sec_per_bubble"):
        if s_base.get(k) is not None and s_ft.get(k) is not None:
            delta[k] = s_ft[k] - s_base[k]

    report = {"summaries": [s_base, s_ft], "delta": delta}

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nFull report → {args.out_report}")


if __name__ == "__main__":
    main()
