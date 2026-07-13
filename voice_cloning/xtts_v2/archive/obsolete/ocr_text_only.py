"""Option 2: Qwen-VL OCR chỉ tập trung vào text + bbox + reading order.

KHÔNG có speaker attribution / character_desc → Qwen-VL focus 100% vào accuracy text.
Mạch truyện vẫn đúng nhờ reading order (trái→phải, trên→dưới).
Downstream: tất cả bubbles → "default" voice (1 giọng xuyên suốt).
"""
import argparse, json, time, re
from pathlib import Path

import torch
import cv2


# Minimal single-task prompt — CHỈ OCR (không speaker, không emotion)
SIMPLE_PROMPT = """Đọc văn bản trong TẤT CẢ bong bóng thoại của trang truyện tranh tiếng Việt này.

ĐÂY LÀ MANGA DỊCH UPPERCASE — toàn bộ thoại là CHỮ HOA, KHÔNG phải lỗi.

YÊU CẦU CỐT LÕI:
- Đọc CHÍNH XÁC từng chữ, từng dấu thanh điệu
- Xuất theo ĐÚNG THỨ TỰ ĐỌC (trái→phải, trên→dưới, theo panel)

CHÚ Ý DẤU THANH ĐIỆU TRÊN CHỮ HOA (rất dễ miss):
- Dấu sắc: Á É Í Ó Ú Ý (nét xéo ↗)
- Dấu huyền: À È Ì Ò Ù Ỳ (nét xéo ↘)
- Dấu hỏi: Ả Ẻ Ỉ Ỏ Ủ Ỷ (móc ?)
- Dấu ngã: Ã Ẽ Ĩ Õ Ũ Ỹ (sóng ~)
- Dấu nặng: Ạ Ẹ Ị Ọ Ụ Ỵ (chấm bên dưới)
- Dấu chữ: Ă Â Đ Ê Ô Ơ Ư

CÁC LỖI THƯỜNG GẶP — hãy tránh:
- CHUA CHẮC → đúng là CHƯA CHẮC (hoặc CHÚC MỪNG theo ngữ cảnh)
- ẢC MỘNG → đúng là ÁC MỘNG
- RÁC RỐI → đúng là RẮC RỐI
- LŨNG DANH → đúng là LỪNG DANH
- HỌC BÀN → đúng là HỘC BÀN
- CẦU (you) vs CẬU (friend) — phân biệt dấu huyền vs nặng
- ĐỪNG ĐUA → đúng là ĐỪNG ĐÙA
- UỐT NHẸP → đúng là ƯỚT NHẸP
- BẬT NGỜ → đúng là BẤT NGỜ
- ĐEN ĐUÎ → đúng là ĐEN ĐUỔI
- LÁO TOẸT (không phải TOỆT)

OUTPUT FORMAT — JSON array theo thứ tự đọc:
[
  {"bbox_2d": [x1, y1, x2, y2], "text": "văn bản bubble 1"},
  {"bbox_2d": [x1, y1, x2, y2], "text": "văn bản bubble 2"},
  ...
]

QUY TẮC:
- Sound effect (RẦM, BÙM, TA-DA, BƯỚC) vẫn đọc, đưa vào mảng cùng thoại
- KHÔNG bịa thêm thoại không có trong ảnh
- KHÔNG bỏ qua bubble nào
- Thứ tự: panel trên trước panel dưới, trong panel thì trái trước phải sau

Chỉ trả về JSON thuần, không giải thích."""


VERIFY_PROMPT_TEMPLATE = """Trang truyện tranh tiếng Việt. Pass 1 đã đọc các bubble:
{pass1_json}

PASS 2 — Kiểm tra kỹ từng bubble bằng cách nhìn ảnh:
1. Đọc lại text trong bubble tại bbox đó
2. So sánh với text pass 1
3. Nếu SAI (thiếu dấu / sai dấu / sai từ) → SỬA ở field text
4. Nếu ĐÚNG → giữ nguyên

CHÚ Ý:
- Dấu trên chữ hoa rất dễ bỏ qua → kiểm tra từng chữ
- Phân biệt dấu hỏi (Ả) vs ngã (Ã) vs nặng (Ạ)
- Dấu chữ Ă Â Đ Ê Ô Ơ Ư đủ chưa
- Tên riêng: Nobita, Doraemon, Shizuka, Suneo, Jaian — chính xác

Xuất JSON cùng format pass 1 (giữ bbox_2d, chỉ đổi text nếu cần).
Chỉ trả về JSON thuần."""


def extract_json_array(raw: str) -> list:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-pages", type=int, default=10)
    p.add_argument("--skip-pages", type=int, default=0)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--no-tier2", action="store_true")
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
    def qwen_call(pil_image, prompt_text, max_new_tokens=2048):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": prompt_text},
            ],
        }]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = proc(
            text=[text], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda:0")
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    pages_dir = Path(args.pages_dir)
    all_pages = sorted(list(pages_dir.glob("*.jpg")) +
                       list(pages_dir.glob("*.webp")) +
                       list(pages_dir.glob("*.png")))
    page_files = all_pages[args.skip_pages : args.skip_pages + args.max_pages]
    print(f"Processing {len(page_files)} pages (text-only mode, single voice)")

    index = []
    for page_idx, img_path in enumerate(page_files):
        page_name = img_path.stem
        print(f"\n[{page_idx+1}/{len(page_files)}] {page_name}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        h, w = img_bgr.shape[:2]

        t0 = time.time()
        raw1 = qwen_call(pil, SIMPLE_PROMPT)
        bubbles_p1 = extract_json_array(raw1)
        print(f"  Pass 1: {len(bubbles_p1)} bubbles in {time.time()-t0:.1f}s")

        bubbles_final = bubbles_p1
        if not args.no_tier2 and bubbles_p1:
            p1_summary = json.dumps([
                {"bbox_2d": b.get("bbox_2d"), "text": b.get("text", "")}
                for b in bubbles_p1
            ], ensure_ascii=False, indent=2)
            t1 = time.time()
            raw2 = qwen_call(pil, VERIFY_PROMPT_TEMPLATE.format(pass1_json=p1_summary))
            bubbles_p2 = extract_json_array(raw2)
            print(f"  Pass 2: {len(bubbles_p2)} bubbles in {time.time()-t1:.1f}s")

            if len(bubbles_p2) == len(bubbles_p1):
                bubbles_final = []
                corrections = 0
                for b1, b2 in zip(bubbles_p1, bubbles_p2):
                    merged = dict(b1)
                    new_text = (b2.get("text") or b1.get("text") or "").strip()
                    if new_text and new_text != b1.get("text", ""):
                        merged["text_pass1"] = b1.get("text")
                        merged["text"] = new_text
                        corrections += 1
                    bubbles_final.append(merged)
                print(f"  Pass 2: {corrections} corrections applied")

        bubbles_out = []
        for order, b in enumerate(bubbles_final, start=1):
            bbox = b.get("bbox_2d", [0, 0, 0, 0])
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(x1, w)); y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w)); y2 = max(0, min(y2, h))
            except Exception:
                x1, y1, x2, y2 = 0, 0, 0, 0
            text = (b.get("text") or "").strip()
            if not text:
                continue
            bubbles_out.append({
                "order": order,
                "text": text,
                "text_pass1": b.get("text_pass1"),
                "bbox": [x1, y1, x2, y2],
                "qwen_speaker": "default",          # single voice
                "qwen_speaker_desc": "",
                "speaker_id": None,
                "attribution": "qwen-text-only",
                "ocr_conf": 1.0,
                "yolo_conf": 1.0,
            })

        out_json = out_dir / f"{page_idx+1:03d}_{page_name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "image": str(img_path),
                "size": [w, h],
                "bubbles": bubbles_out,
            }, f, ensure_ascii=False, indent=2)
        index.append({"page_idx": page_idx+1, "name": page_name,
                      "json": str(out_json), "n_bubbles": len(bubbles_out)})
        print(f"  → {out_json.name}")

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(index)} pages → {out_dir}")


if __name__ == "__main__":
    main()
