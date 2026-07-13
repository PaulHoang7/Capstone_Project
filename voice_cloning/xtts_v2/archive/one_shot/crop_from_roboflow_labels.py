"""Crop bubbles directly from Roboflow YOLO-labeled dataset, then run Qwen-VL OCR.

Workflow:
    YOLO dataset (Roboflow export)
        train/images/*.jpg + train/labels/*.txt   (and valid/, test/)
        │
        ▼
    For each image:
        Read .txt → list of YOLO bboxes (class x_center y_center w h, normalized)
        Convert to pixel bbox + small padding
        Crop + upscale 2x for OCR clarity
        Save crop to bubble_crops/NNNNN.png
        Run Qwen-VL → prefill text
        │
        ▼
    Output:
        labels.csv with cols: id, split, source_image, bbox_idx, image, prefill, corrected_text, skip
        bubble_crops/NNNNN.png
        README.txt

Why this beats the redetect+Qwen-pool path:
    - 100% crops are FULL BUBBLE (user labeled the bbox themselves)
    - No "unmatched fallback" with weird padding
    - Same image as YOLO training input → no scale/distortion mismatch
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import cv2
from PIL import Image as PILImage


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


def yolo_bbox_to_pixel(bbox_norm, img_w, img_h):
    """Convert YOLO normalized (cx, cy, w, h) → pixel (x1, y1, x2, y2)."""
    cx, cy, w, h = bbox_norm
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/bes/Desktop/Tin/Capstone_project/data/yolo_bubble_v3",
                   help="YOLO dataset root with train/, valid/, test/ subfolders")
    p.add_argument("--out-dir", default="/home/bes/Desktop/Tin/labeling_task_v4")
    p.add_argument("--padding", type=int, default=10, help="Pixel padding around bbox before crop")
    p.add_argument("--upscale", type=float, default=2.0, help="Upscale factor for saved crops")
    p.add_argument("--min-crop-width", type=int, default=400, help="Force minimum crop width")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--no-ocr", action="store_true",
                   help="Skip Qwen OCR (just crop + write CSV with empty prefill)")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "bubble_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Optional Qwen
    qwen_ocr = None
    if not args.no_ocr:
        print(f"Loading {args.model_id}...")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        proc = AutoProcessor.from_pretrained(args.model_id)
        print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        @torch.no_grad()
        def qwen_ocr(pil_crop):
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
            out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            trimmed = out_ids[:, inputs.input_ids.shape[1]:]
            answer = proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            answer = answer.strip('"').strip("'").strip()
            if ":" in answer[:20]:
                head = answer.split(":", 1)[0].lower()
                if any(kw in head for kw in ("kết quả", "text", "đọc được", "văn bản")):
                    answer = answer.split(":", 1)[1].strip()
            return answer

    # --- Walk dataset ---
    rows = []
    row_id = 0
    skipped_tiny = 0

    for split in args.splits:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"
        if not img_dir.exists():
            print(f"  skip {split}: no images dir")
            continue

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"\n=== {split}: {len(img_files)} images ===")

        for img_idx, img_path in enumerate(img_files):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            for bbox_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                # cls cx cy w h  (normalized)
                _cls = parts[0]
                bbox_norm = tuple(float(v) for v in parts[1:5])
                x1, y1, x2, y2 = yolo_bbox_to_pixel(bbox_norm, w, h)
                # Pad
                x1 = max(0, x1 - args.padding)
                y1 = max(0, y1 - args.padding)
                x2 = min(w, x2 + args.padding)
                y2 = min(h, y2 + args.padding)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    skipped_tiny += 1
                    continue

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Upscale for OCR + readability
                scale = max(args.upscale, args.min_crop_width / crop.shape[1])
                if scale > 1.0:
                    new_w = int(crop.shape[1] * scale)
                    new_h = int(crop.shape[0] * scale)
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                row_id += 1
                crop_name = f"{row_id:05d}.png"
                cv2.imwrite(str(crops_dir / crop_name), crop)

                # OCR
                prefill = ""
                if qwen_ocr is not None:
                    pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    t0 = time.time()
                    try:
                        prefill = qwen_ocr(pil)
                    except Exception as e:
                        prefill = f"[OCR_ERR: {type(e).__name__}]"
                    dt = time.time() - t0
                    if row_id % 50 == 0:
                        print(f"  [{row_id:05d}] {dt:.2f}s  {img_path.stem[:40]}  prefill={prefill[:50]!r}")

                rows.append({
                    "id":             f"{row_id:05d}",
                    "split":          split,
                    "source_image":   img_path.name,
                    "bbox_idx":       bbox_idx,
                    "image":          f"bubble_crops/{crop_name}",
                    "prefill":        prefill,
                    "corrected_text": "",
                    "skip":           "",
                })

            if (img_idx + 1) % 50 == 0:
                print(f"  progress: {img_idx+1}/{len(img_files)} images, {row_id} crops total")

    # Write CSV
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "split", "source_image", "bbox_idx",
                           "image", "prefill", "corrected_text", "skip"])
        writer.writeheader()
        writer.writerows(rows)

    # README
    readme = f"""Vietnamese Manga OCR Labeling — Roboflow-direct version
========================================================

Source: {args.data_dir}
Crops: {len(rows)} bubble crops from Roboflow labels (all FULL BUBBLE — you labeled them)
OCR engine: {'Qwen2.5-VL-7B' if qwen_ocr else 'NONE (--no-ocr)'}

Skipped {skipped_tiny} bbox quá nhỏ (<10px).

WORKFLOW:
1. Mở labeler.html (http server) hoặc CSV trong Excel
2. Cho mỗi row:
   - Xem crop (đảm bảo full bubble vì bạn label tay)
   - Đọc text đúng → gõ vào corrected_text
   - Nếu là watermark/sfx → tick skip
3. Save / export
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"\n✓ Exported {len(rows)} rows → {csv_path}")
    print(f"✓ Crops → {crops_dir}")
    print(f"  Skipped tiny: {skipped_tiny}")


if __name__ == "__main__":
    main()
