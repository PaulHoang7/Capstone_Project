"""Extract (line_image, line_text) pairs from Vietnamese comic bubbles for line-level OCR fine-tune.

Pipeline:
    bubble_crops_vn/{id}.png  +  labels_vn.csv corrected_text (with \\n line breaks)
        │
        ├── PaddleOCR DBNet → detect text line boxes inside bubble
        ├── Sort boxes top→bottom by Y centroid
        ├── Split corrected_text by \\n
        ├── If #detected_boxes == #text_lines → crop each line, pair with text line
        └── Else → log mismatch and skip

Output:
    line_crops_vn/{id}_L{N}.png       — per-line crops
    line_pairs_vn.jsonl               — {image, text, source_id, line_idx, n_lines}
    line_pairs_stats.json             — coverage stats + mismatch breakdown

Usage:
    conda run -n comic_ocr python prepare_line_pairs_vn.py \\
        --csv /home/bes/Desktop/Tin/labeling_task_v4/labels_vn.csv \\
        --crops-dir /home/bes/Desktop/Tin/labeling_task_v4/bubble_crops_vn \\
        --out-crops-dir /home/bes/Desktop/Tin/labeling_task_v4/line_crops_vn \\
        --out-jsonl /home/bes/Desktop/Tin/labeling_task_v4/line_pairs_vn.jsonl
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def aabb_from_poly(poly):
    """Quadrilateral [[x,y]*4] → (x1, y1, x2, y2) axis-aligned bbox."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def merge_overlapping(boxes, iou_thr=0.5):
    """Greedy merge boxes with IoU > threshold (same physical line detected twice)."""
    merged = []
    used = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        cur = list(b)
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou(cur, boxes[j]) > iou_thr:
                cur = [
                    min(cur[0], boxes[j][0]),
                    min(cur[1], boxes[j][1]),
                    max(cur[2], boxes[j][2]),
                    max(cur[3], boxes[j][3]),
                ]
                used[j] = True
        merged.append(tuple(cur))
        used[i] = True
    return merged


def merge_same_line(boxes, y_tol_factor=0.5):
    """Cluster boxes whose Y-centers are within (median_height * y_tol_factor) → one wide box.

    Fixes PaddleOCR splitting a single line of comic dialogue into multiple horizontal
    fragments (e.g., before punctuation or wide inter-word spacing).
    """
    if not boxes:
        return []
    heights = [b[3] - b[1] for b in boxes]
    heights_sorted = sorted(heights)
    median_h = heights_sorted[len(heights_sorted) // 2]
    eps = max(4.0, median_h * y_tol_factor)

    sorted_boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    clusters = [[sorted_boxes[0]]]
    for b in sorted_boxes[1:]:
        last = clusters[-1]
        last_yc = sum((bb[1] + bb[3]) / 2 for bb in last) / len(last)
        cur_yc = (b[1] + b[3]) / 2
        if abs(cur_yc - last_yc) < eps:
            last.append(b)
        else:
            clusters.append([b])

    merged = []
    for cluster in clusters:
        x1 = min(b[0] for b in cluster)
        y1 = min(b[1] for b in cluster)
        x2 = max(b[2] for b in cluster)
        y2 = max(b[3] for b in cluster)
        merged.append((x1, y1, x2, y2))
    return merged


def detect_lines(detector, img_path):
    """Return axis-aligned boxes sorted top→bottom by Y center.

    Calls detector.text_detector(img) directly — bypasses PaddleOCR.ocr() which has
    a `not numpy_array` bug at paddleocr.py:681 in version 2.7.3.
    """
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    polys, _ = detector.text_detector(img)
    if polys is None or len(polys) == 0:
        return []
    boxes = [aabb_from_poly(p) for p in polys]
    boxes = [b for b in boxes if (b[2] - b[0]) > 4 and (b[3] - b[1]) > 4]
    boxes = merge_overlapping(boxes, iou_thr=0.5)
    boxes = merge_same_line(boxes, y_tol_factor=0.5)
    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
    return boxes


def split_text_lines(s):
    """Split corrected_text into non-empty stripped lines."""
    lines = [ln.strip() for ln in (s or "").splitlines()]
    return [ln for ln in lines if ln]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--crops-dir", required=True, help="Directory with bubble crops (PNG)")
    p.add_argument("--out-crops-dir", required=True, help="Output dir for per-line crops")
    p.add_argument("--out-jsonl", required=True, help="Output JSONL of (image, text) pairs")
    p.add_argument("--out-stats", default=None, help="Output stats JSON")
    p.add_argument("--pad", type=int, default=3, help="Pixel padding around each line crop")
    p.add_argument("--min-line-height", type=int, default=8,
                   help="Drop detected lines shorter than this (likely noise)")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process first N rows (sanity check); 0 = all")
    p.add_argument("--offset", type=int, default=0,
                   help="Skip first N rows (for sampling middle of dataset)")
    p.add_argument("--det-lang", default="vi",
                   help="PaddleOCR lang for the detection model")
    p.add_argument("--paddle-gpu", action="store_true",
                   help="Use GPU for PaddleOCR. Default CPU because PaddlePaddle "
                        "2.7.3 doesn't support Blackwell (sm_120) yet.")
    args = p.parse_args()

    out_crops = Path(args.out_crops_dir)
    out_crops.mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_stats is None:
        args.out_stats = str(out_jsonl.with_suffix(".stats.json"))

    print(f"[init] Loading PaddleOCR (lang={args.det_lang}, det only, gpu={args.paddle_gpu})...")
    from paddleocr import PaddleOCR
    detector = PaddleOCR(
        lang=args.det_lang, rec=False, use_angle_cls=False, show_log=False,
        use_gpu=args.paddle_gpu,
    )

    crops_dir = Path(args.crops_dir)
    with open(args.csv, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    if args.offset > 0:
        all_rows = all_rows[args.offset:]
    if args.limit > 0:
        all_rows = all_rows[: args.limit]
    print(f"[init] Loaded {len(all_rows)} rows from {args.csv} (offset={args.offset})")

    stats = Counter()
    mismatch_by_diff = Counter()  # (detected - text) diff distribution
    pairs_kept = 0
    rows_kept = 0
    t0 = time.time()

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, r in enumerate(all_rows):
            if i and i % 200 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1)
                eta_min = (len(all_rows) - i) / max(rate, 1) / 60
                print(f"  {i}/{len(all_rows)}  rows_kept={rows_kept}  pairs={pairs_kept}  "
                      f"rate={rate:.1f}/s  eta={eta_min:.1f}m")

            text_lines = split_text_lines(r.get("corrected_text", ""))
            if not text_lines:
                stats["skip_no_corrected_text"] += 1
                continue
            if (r.get("skip", "").strip() in ("1", "true", "x")):
                stats["skip_marked_skip"] += 1
                continue

            img_rel = r.get("image", "")
            img_path = crops_dir / Path(img_rel).name
            if not img_path.exists():
                stats["skip_image_missing"] += 1
                continue

            try:
                boxes = detect_lines(detector, img_path)
            except Exception as e:
                stats["err_paddle"] += 1
                if stats["err_paddle"] <= 5:
                    print(f"  paddle error on {img_path.name}: {e}")
                continue

            # Drop too-short boxes
            boxes = [b for b in boxes if (b[3] - b[1]) >= args.min_line_height]

            n_det, n_text = len(boxes), len(text_lines)
            diff = n_det - n_text
            mismatch_by_diff[diff] += 1

            if n_det == 0:
                stats["skip_no_lines_detected"] += 1
                continue
            if n_det != n_text:
                stats["skip_line_count_mismatch"] += 1
                continue

            # Crop and save each line
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception:
                stats["err_open_image"] += 1
                continue
            W, H = im.size

            kept_this_row = 0
            for idx, (b, txt) in enumerate(zip(boxes, text_lines)):
                x1, y1, x2, y2 = b
                x1 = max(0, x1 - args.pad)
                y1 = max(0, y1 - args.pad)
                x2 = min(W, x2 + args.pad)
                y2 = min(H, y2 + args.pad)
                if x2 - x1 < 8 or y2 - y1 < args.min_line_height:
                    continue
                crop = im.crop((x1, y1, x2, y2))
                fname = f"{r['id']}_L{idx}.png"
                crop.save(out_crops / fname)
                rec = {
                    "image": str(out_crops / fname),
                    "text": txt,
                    "source_id": r["id"],
                    "source_image": str(img_path),
                    "line_idx": idx,
                    "n_lines": n_text,
                    "series": r.get("series", ""),
                    "page": r.get("page", ""),
                    "bbox": [x1, y1, x2, y2],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pairs_kept += 1
                kept_this_row += 1
            if kept_this_row:
                rows_kept += 1
                stats[f"kept_n_lines={n_text}"] += 1

    elapsed = time.time() - t0
    summary = {
        "n_input_rows": len(all_rows),
        "n_rows_with_pairs": rows_kept,
        "n_pairs_kept": pairs_kept,
        "elapsed_sec": round(elapsed, 1),
        "stats": dict(stats),
        "mismatch_diff_histogram": dict(sorted(mismatch_by_diff.items())),
        "out_jsonl": str(out_jsonl),
        "out_crops_dir": str(out_crops),
    }
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Rows with pairs : {rows_kept}/{len(all_rows)}")
    print(f"Pairs kept      : {pairs_kept}")
    print(f"Time            : {elapsed/60:.1f} min")
    print(f"Skip reasons    : {dict(stats)}")
    print(f"Det-vs-text diff: {dict(sorted(mismatch_by_diff.items()))}")
    print(f"Stats saved     : {args.out_stats}")
    print(f"JSONL saved     : {out_jsonl}")


if __name__ == "__main__":
    main()
