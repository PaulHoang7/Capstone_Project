"""Inference: bubble → line detection (PaddleOCR DBNet) → per-line OCR (Qwen2.5-VL + LoRA) → concat.

Usage:
    conda run -n comic_ocr python infer_line_ocr_vn.py \\
        --csv /home/bes/Desktop/Tin/labeling_task_v4/labels_vn.csv \\
        --crops-dir /home/bes/Desktop/Tin/labeling_task_v4/bubble_crops_vn \\
        --lora-dir /mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_7b_vncomic_line_lora_v1/best \\
        --out-csv /home/bes/Desktop/Tin/labeling_task_v4/labels_vn_predicted.csv \\
        --base-model Qwen/Qwen2.5-VL-7B-Instruct

    # No-LoRA baseline (for comparison):
    conda run -n comic_ocr python infer_line_ocr_vn.py \\
        --csv ... --crops-dir ... --out-csv labels_vn_baseline.csv \\
        --base-model Qwen/Qwen2.5-VL-7B-Instruct --no-lora
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image


SYSTEM_PROMPT = """Bạn là OCR engine đọc MỘT DÒNG văn bản tiếng Việt từ truyện tranh.

QUY TẮC TUYỆT ĐỐI:
1. CHỈ trả về văn bản trên 1 dòng được cung cấp. KHÔNG giải thích, KHÔNG mô tả.
2. Nếu không đọc được → trả về CHUỖI RỖNG.
3. KHÔNG xuống dòng. KHÔNG echo prompt. KHÔNG nói "Tôi không thể...".
4. Giữ chính xác dấu thanh điệu (sắc/huyền/hỏi/ngã/nặng) và dấu chữ (ă â đ ê ô ơ ư).
5. Giữ đúng case (chữ HOA hay chữ thường) như xuất hiện trong ảnh."""

USER_PROMPT = "Đọc văn bản trên dòng này. Trả về chỉ văn bản, không xuống dòng."


def aabb_from_poly(poly):
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
                    min(cur[0], boxes[j][0]), min(cur[1], boxes[j][1]),
                    max(cur[2], boxes[j][2]), max(cur[3], boxes[j][3]),
                ]
                used[j] = True
        merged.append(tuple(cur))
        used[i] = True
    return merged


def detect_lines(detector, img_path, min_h=8):
    """Bypasses PaddleOCR.ocr() bug at paddleocr.py:681 in 2.7.3."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    polys, _ = detector.text_detector(img)
    if polys is None or len(polys) == 0:
        return []
    boxes = [aabb_from_poly(p) for p in polys]
    boxes = [b for b in boxes if (b[2] - b[0]) > 4 and (b[3] - b[1]) >= min_h]
    boxes = merge_overlapping(boxes, iou_thr=0.5)
    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
    return boxes


@torch.no_grad()
def ocr_one_line(model, processor, image, max_new_tokens=80, repetition_penalty=1.2):
    from qwen_vl_utils import process_vision_info
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
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        repetition_penalty=repetition_penalty,
    )
    pred = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return pred.strip().replace("\n", " ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input labels CSV (uses id/image columns only)")
    p.add_argument("--crops-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora-dir", default=None, help="LoRA adapter dir (omit / use --no-lora for baseline)")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--pad", type=int, default=3)
    p.add_argument("--min-line-height", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=384 * 384)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--det-lang", default="vi")
    p.add_argument("--paddle-gpu", action="store_true",
                   help="Use GPU for PaddleOCR. Default CPU (PaddlePaddle 2.7.3 lacks Blackwell sm_120).")
    p.add_argument("--qlora-4bit", action="store_true")
    p.add_argument("--whole-bubble", action="store_true",
                   help="Config A: skip line detection, feed whole bubble image to Qwen-VL "
                        "(baseline comparison vs line-level mode)")
    p.add_argument("--vn-lora", default=None,
                   help="Path to VN comic LoRA (used with --auto-style)")
    p.add_argument("--manga-lora", default=None,
                   help="Path to manga LoRA (used with --auto-style)")
    p.add_argument("--auto-style", action="store_true",
                   help="Run baseline on first N rows, detect uppercase ratio, "
                        "pick VN LoRA if >50%% upper else manga LoRA. Requires both --vn-lora and --manga-lora.")
    p.add_argument("--sniff-n", type=int, default=10,
                   help="Number of rows to sniff for --auto-style")
    args = p.parse_args()

    if args.auto_style and not (args.vn_lora and args.manga_lora):
        raise SystemExit("--auto-style requires both --vn-lora and --manga-lora")

    detector = None
    if not args.whole_bubble:
        print(f"[init] Loading PaddleOCR detection (lang={args.det_lang}, gpu={args.paddle_gpu})...")
        from paddleocr import PaddleOCR
        detector = PaddleOCR(
            lang=args.det_lang, rec=False, use_angle_cls=False, show_log=False,
            use_gpu=args.paddle_gpu,
        )
    else:
        print("[init] --whole-bubble mode: skipping PaddleOCR, feeding whole bubble to Qwen-VL")

    print(f"[init] Loading Qwen2.5-VL: {args.base_model}")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.base_model)
    load_kwargs = {"device_map": "cuda:0"}
    if args.qlora_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base_model, **load_kwargs)

    chosen_lora = args.lora_dir
    if args.auto_style:
        # Detection strategy:
        # 1) Path-based: if image col mentions "bubble_crops_vn" → VN; "bubble_crops" → manga.
        #    This works for the labeled datasets and any new pipeline that follows the same convention.
        # 2) Uppercase ratio fallback: known unreliable because baseline Qwen-VL defaults to
        #    uppercase on comic-bubble visuals regardless of actual case. Use only when path
        #    inspection is inconclusive.
        with open(args.csv, encoding="utf-8-sig") as f:
            csv_rows = list(csv.DictReader(f))[: args.sniff_n]
        path_signals = []
        for r in csv_rows:
            ip = (r.get("image") or "").lower()
            if "bubble_crops_vn" in ip:
                path_signals.append("vn")
            elif "bubble_crops" in ip:
                path_signals.append("manga")
        if path_signals:
            from collections import Counter
            cnt = Counter(path_signals)
            style = cnt.most_common(1)[0][0]
            print(f"[init] --auto-style (path-based): {dict(cnt)} → style={style!r}")
        else:
            print(f"[init] --auto-style (sniff fallback): no path signal, running baseline on {args.sniff_n} rows...")
            sniff_preds = []
            model.eval()
            for r in csv_rows:
                ip = Path(args.crops_dir) / Path(r.get("image", "")).name
                if not ip.exists():
                    continue
                try:
                    im = Image.open(ip).convert("RGB")
                    w, h = im.size
                    if w * h > args.max_pixels * 4:
                        scale = (args.max_pixels * 4 / (w * h)) ** 0.5
                        im = im.resize((max(8, int(w*scale)), max(8, int(h*scale))), Image.LANCZOS)
                    sniff_preds.append(ocr_one_line(model, processor, im, max_new_tokens=100))
                except Exception as e:
                    print(f"  sniff err on {ip.name}: {e}")
            text = "".join(sniff_preds)
            letters = [c for c in text if c.isalpha()]
            upper_ratio = sum(1 for c in letters if c == c.upper()) / max(len(letters), 1)
            style = "vn" if upper_ratio > 0.5 else "manga"
            print(f"[init] sniff: {len(letters)} letters upper_ratio={upper_ratio:.2f} → style={style!r}")
            print(f"  NOTE: baseline upper-ratio heuristic is unreliable; baseline tends to UPPERCASE "
                  f"on comic visuals. Prefer explicit --lora-dir or path-conventional crops.")
        chosen_lora = args.vn_lora if style == "vn" else args.manga_lora
        print(f"[init] → loading {chosen_lora}")

    if chosen_lora and not args.no_lora:
        print(f"[init] Attaching LoRA adapter from {chosen_lora}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, chosen_lora)
    elif args.no_lora:
        print("[init] Running base model without LoRA (baseline mode)")
    else:
        print("[init] No LoRA given → running base model")
    model.eval()

    crops_dir = Path(args.crops_dir)
    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[init] Loaded {len(rows)} rows")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse input columns + predicted_text + n_detected_lines
    in_cols = list(rows[0].keys()) if rows else []
    extra_cols = ["predicted_text", "n_detected_lines", "infer_sec"]
    out_cols = in_cols + [c for c in extra_cols if c not in in_cols]

    t0 = time.time()
    stats = {"n_processed": 0, "n_empty": 0, "n_lines_total": 0, "sec_total": 0.0}

    with open(out_path, "w", encoding="utf-8", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=out_cols)
        writer.writeheader()

        for i, r in enumerate(rows):
            if i and i % 50 == 0:
                rate = stats["n_processed"] / max(time.time() - t0, 1)
                eta = (len(rows) - i) / max(rate, 1) / 60
                print(f"  {i}/{len(rows)}  rate={rate:.1f}/s  eta={eta:.1f}m")

            img_path = crops_dir / Path(r.get("image", "")).name
            row_out = dict(r)
            t_start = time.time()
            if not img_path.exists():
                row_out["predicted_text"] = ""
                row_out["n_detected_lines"] = 0
                row_out["infer_sec"] = 0
                writer.writerow(row_out)
                continue

            try:
                if args.whole_bubble:
                    im = Image.open(img_path).convert("RGB")
                    w, h = im.size
                    if w * h > args.max_pixels * 4:  # whole bubble allowed bigger
                        scale = (args.max_pixels * 4 / (w * h)) ** 0.5
                        im = im.resize((max(8, int(w*scale)), max(8, int(h*scale))), Image.LANCZOS)
                    pred = ocr_one_line(model, processor, im, max_new_tokens=200)
                    row_out["predicted_text"] = pred
                    row_out["n_detected_lines"] = 1
                    elapsed = time.time() - t_start
                    row_out["infer_sec"] = round(elapsed, 3)
                    stats["sec_total"] += elapsed
                    stats["n_processed"] += 1
                    if not pred.strip():
                        stats["n_empty"] += 1
                    writer.writerow(row_out)
                    continue

                boxes = detect_lines(detector, img_path, min_h=args.min_line_height)
                if not boxes:
                    row_out["predicted_text"] = ""
                    row_out["n_detected_lines"] = 0
                else:
                    im = Image.open(img_path).convert("RGB")
                    W, H = im.size
                    line_texts = []
                    for b in boxes:
                        x1, y1, x2, y2 = b
                        x1 = max(0, x1 - args.pad); y1 = max(0, y1 - args.pad)
                        x2 = min(W, x2 + args.pad); y2 = min(H, y2 + args.pad)
                        if x2 - x1 < 8 or y2 - y1 < args.min_line_height:
                            continue
                        crop = im.crop((x1, y1, x2, y2))
                        w, h = crop.size
                        if w * h > args.max_pixels:
                            scale = (args.max_pixels / (w * h)) ** 0.5
                            crop = crop.resize(
                                (max(8, int(w * scale)), max(8, int(h * scale))), Image.LANCZOS,
                            )
                        pred = ocr_one_line(model, processor, crop)
                        line_texts.append(pred)
                    row_out["predicted_text"] = "\n".join(line_texts)
                    row_out["n_detected_lines"] = len(line_texts)
                    stats["n_lines_total"] += len(line_texts)
            except Exception as e:
                print(f"  err on {img_path.name}: {e}")
                row_out["predicted_text"] = ""
                row_out["n_detected_lines"] = -1

            elapsed = time.time() - t_start
            row_out["infer_sec"] = round(elapsed, 3)
            stats["sec_total"] += elapsed
            stats["n_processed"] += 1
            if not row_out["predicted_text"].strip():
                stats["n_empty"] += 1
            writer.writerow(row_out)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  processed: {stats['n_processed']}")
    print(f"  empty out: {stats['n_empty']}")
    print(f"  total lines OCR'd: {stats['n_lines_total']}")
    print(f"  avg sec/bubble: {stats['sec_total']/max(stats['n_processed'],1):.2f}")
    print(f"  output: {out_path}")
    with open(out_path.with_suffix(".infer_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
