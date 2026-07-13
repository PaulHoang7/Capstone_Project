"""Per-bubble OCR refinement: re-run Qwen-VL on each bubble CROP individually.

Input: existing CV JSON (has bbox) — from ocr_text_only.py or ocr_v2_pipeline.py
Output: JSON with refined text from per-bubble OCR pass

Why this helps:
- Whole-page OCR: Qwen-VL processes 1200x1700px with dozens of regions → attention
  diluted → misses fine details (dấu thanh trên chữ hoa).
- Per-bubble OCR: Qwen-VL sees only 1 crop (200x200px typical) → full attention
  on that small region → much better dấu thanh accuracy.
- Upscale crop 2x → even more detail visible to the model.
"""
import argparse, json, time
from pathlib import Path

import torch
import numpy as np
import cv2


BUBBLE_OCR_PROMPT = """Đọc CHÍNH XÁC văn bản tiếng Việt trong bong bóng thoại này.

ĐẶC BIỆT CHÚ Ý từng dấu thanh điệu trên chữ hoa:
- Sắc (Á É Í Ó Ú Ý): nét xéo lên trên bên phải
- Huyền (À È Ì Ò Ù Ỳ): nét xéo xuống bên phải
- Hỏi (Ả Ẻ Ỉ Ỏ Ủ Ỷ): dấu móc (?) phía trên
- Ngã (Ã Ẽ Ĩ Õ Ũ Ỹ): dấu sóng (~) phía trên
- Nặng (Ạ Ẹ Ị Ọ Ụ Ỵ): dấu chấm phía DƯỚI chữ
- Dấu chữ: Ă (á trên a) Â (mũ) Đ (gạch ngang D) Ê Ô Ơ Ư

Các từ thường đọc nhầm — phân biệt kỹ:
- CẬU (friend, nặng ở dưới) vs CẦU (beg, huyền nghiêng xuống) — manga thường là CẬU
- BẤT (sắc) vs BẬT (nặng) — "BẤT NGỜ" = surprised
- TỪ (huyền) vs TƯ (không dấu) — "TỪ ĐÂU" = from where
- ĐÂU (không dấu) vs ĐẦU (huyền) — "TỪ ĐÂU ĐẾN" = from where came
- ĐUỔI (hook trên O) vs ĐUÔI (không) — "ĐEN ĐUỔI" = chase bad luck
- CHỨ (sắc) vs CHƯ (không) — "BIẾT CHỨ" = of course
- ĐIÊN (không nặng trên N) vs ĐIỆN (nặng) — "ĐỒ ĐIÊN" = crazy thing
- PHÁN (sắc) vs PHẢN (hỏi) — "PHÁN CHỨ" = judge
- ĐỘ (nặng) vs ĐỒ (huyền) — "CẤP ĐỘ" = level
- ÁC (sắc) vs ẢC (hỏi) — "ÁC MỘNG" = nightmare
- CHƯA (không) vs CHUA (không, nghĩa chua) — context dependent
- RẮC RỐI (có ă) vs RÁC RỐI (sai) — "RẮC RỐI" = trouble
- LỪNG DANH (huyền) vs LŨNG DANH (ngã) — "LỪNG DANH" = famous
- HỘC BÀN (nặng O) vs HỌC BÀN (nặng A) — "HỘC BÀN" = desk drawer
- ĐÙA (huyền) vs ĐUA (không dấu) — "ĐỪNG ĐÙA" = don't joke
- ƯỚT (sắc Ơ) vs UỐT (sai) — "ƯỚT NHẸP" = soaked
- TƯỞNG TƯỢNG (tưởng tượng) — KHÔNG phải TƯỞNG TƯỞNG

Trả về DUY NHẤT văn bản đọc được, không giải thích, không ngoặc kép.
Nếu không đọc được hoặc bubble rỗng, trả về chuỗi rỗng."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-json-dir", required=True, help="Dir with bubble bboxes")
    p.add_argument("--out", required=True)
    p.add_argument("--upscale", type=float, default=2.0, help="Upscale factor for bubble crops")
    p.add_argument("--padding", type=int, default=8, help="Pixel padding around bubble bbox")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id}...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from PIL import Image as PILImage

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    proc = AutoProcessor.from_pretrained(args.model_id)
    print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    @torch.no_grad()
    def qwen_ocr_crop(pil_crop):
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
        out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        answer = proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        # Clean up: remove wrapping quotes
        answer = answer.strip('"').strip("'").strip()
        # Some models wrap with "Kết quả:" → strip
        if ":" in answer[:20]:
            first_part = answer.split(":", 1)[0].lower()
            if any(kw in first_part for kw in ("kết quả", "text", "đọc được")):
                answer = answer.split(":", 1)[1].strip()
        return answer

    in_dir = Path(args.in_json_dir)
    with open(in_dir / "index.json") as f:
        pages_meta = json.load(f)

    new_index = []
    total_changes = 0
    for meta in pages_meta:
        page_json = Path(meta["json"])
        with open(page_json) as f:
            page = json.load(f)

        img_path = page.get("image")
        page_img = cv2.imread(str(img_path)) if img_path else None
        if page_img is None:
            print(f"  skip {page_json.name}: can't read image")
            continue
        h, w = page_img.shape[:2]

        bubbles = page.get("bubbles", [])
        print(f"\n{page_json.name}: {len(bubbles)} bubbles")
        changes = 0

        for b in bubbles:
            bbox = b.get("bbox", [0, 0, 0, 0])
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Pad
            x1 = max(0, x1 - args.padding)
            y1 = max(0, y1 - args.padding)
            x2 = min(w, x2 + args.padding)
            y2 = min(h, y2 + args.padding)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = page_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Upscale for better OCR
            if args.upscale != 1.0:
                new_w = int(crop.shape[1] * args.upscale)
                new_h = int(crop.shape[0] * args.upscale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            pil_crop = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            t0 = time.time()
            new_text = qwen_ocr_crop(pil_crop)
            dt = time.time() - t0
            old_text = (b.get("text") or "").strip()

            if new_text and new_text != old_text:
                b["text_whole_page"] = old_text   # preserve old reading for audit
                b["text"] = new_text
                changes += 1
                print(f"  [{b.get('order'):02d}] ({dt:.1f}s) CHANGED")
                print(f"       before: {old_text!r}")
                print(f"       after:  {new_text!r}")

        total_changes += changes
        print(f"  → {changes} bubbles updated ({len(bubbles)-changes} kept)")

        out_json = out_dir / page_json.name
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)
        new_meta = dict(meta)
        new_meta["json"] = str(out_json)
        new_index.append(new_meta)

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {total_changes} total corrections across {len(new_index)} pages.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
