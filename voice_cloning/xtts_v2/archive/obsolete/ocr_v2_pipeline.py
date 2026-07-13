"""3-Tier OCR pipeline for UPPERCASE Vietnamese manga.

Tier 1: Enhanced Qwen-VL prompt with domain grounding + few-shot examples
Tier 2: Two-pass self-correction (feed back image + first-pass OCR for verification)
Tier 3: Manual review exports via interactive HTML editor (see editor.html builder)

Usage (Tier 1 + 2, both runs in comic_ocr env):
    python ocr_v2_pipeline.py --pages-dir <dir> --out <dir> --max-pages N
"""
import argparse, json, sys, time, types, re
from pathlib import Path

import torch
import numpy as np
import cv2

# -----------------------------------------------------------------------------
# TIER 1: Enhanced DETECT prompt for UPPERCASE Vietnamese manga
# -----------------------------------------------------------------------------
ENHANCED_DETECT_PROMPT = """Phân tích trang truyện tranh tiếng Việt này. Đây là manga Nhật dịch sang tiếng Việt — toàn bộ thoại VIẾT HOA (UPPERCASE) là style dịch thuật Việt Nam, KHÔNG phải lỗi.

# NHIỆM VỤ
Với MỖI khung thoại chứa văn bản, xuất ra:
- bbox_2d: [x1,y1,x2,y2] toạ độ pixel
- text: văn bản chính xác tuyệt đối, giữ UPPERCASE nguyên bản
- speaker: ID nhân vật ('character_1', 'character_2'..., hoặc 'sound_effect')
- speaker_desc: mô tả ngắn nhân vật

# ĐẶC BIỆT QUAN TRỌNG cho text tiếng Việt UPPERCASE
Chú ý kỹ các dấu thanh điệu trên chữ hoa (thường bị miss):
- Dấu sắc (Á É Í Ó Ú Ý) — rất dễ nhầm thành không dấu
- Dấu huyền (À È Ì Ò Ù Ỳ)
- Dấu hỏi (Ả Ẻ Ỉ Ỏ Ủ Ỷ) — rất dễ nhầm với nặng
- Dấu ngã (Ã Ẽ Ĩ Õ Ũ Ỹ) — rất dễ nhầm với hỏi
- Dấu nặng (Ạ Ẹ Ị Ọ Ụ Ỵ)
- Dấu chữ (Ă Â Đ Ê Ô Ơ Ư) — KHÔNG được bỏ qua

# BẢNG THAM CHIẾU TỪ VỰNG THƯỜNG BỊ SAI
Các cụm từ comic phổ biến — đọc kỹ tránh nhầm:
- ÁC MỘNG (nightmare — KHÔNG phải ẢC MỘNG)
- RẮC RỐI (trouble — KHÔNG phải RÁC RỐI)
- LỪNG DANH (famous — KHÔNG phải LŨNG DANH)
- CHÚC MỪNG (congrats — KHÔNG phải CHUA CHẮC)
- ĐƯƠNG NHIÊN (obviously — KHÔNG phải ĐƯỜNG NHIỆN)
- HỘC BÀN (desk drawer — KHÔNG phải HỌC BÀN)
- ĐUỔI THEO (chase — KHÔNG phải ĐUÔI THEO)
- ƯỚT NHẸP (soaked — KHÔNG phải UỐT NHẸP)
- BẤT NGỜ (surprised — KHÔNG phải BẬT NGỜ)
- DĨ NHIÊN, TẤT NHIÊN, HIỂN NHIÊN — check dấu
- CẬU (you/friend), CỤ (elder), CÂU (sentence) — phân biệt dấu
- CHỨ (right?) vs CHÚ (uncle) — phân biệt dấu
- CHÀO (hello) vs CHÁO (porridge) — phân biệt dấu

# TÊN NHÂN VẬT (manga Doraemon/Conan/Dragon Ball dịch)
Chuyển sang phiên âm VN cho TTS:
- Doraemon → Đô-rê-mon
- Nobita → Nô-bi-ta
- Shizuka → Si-zu-ka (hoặc Xu-ka)
- Suneo → Xê-cô
- Jaian → Cha-in
- Conan → Cô-nan
- Kudo/Kudou → Cu-đô
- Shinichi → Shi-ni-chi
- Ran → Ran
- Son Goku → Son Go-ku
- Bulma → Bun-ma
- Vegeta → Vê-gê-ta

# NGUYÊN TẮC KIỂM TRA
1. Nếu không chắc 1 dấu → đọc lại 2 lần, ưu tiên từ tiếng Việt **phổ biến nhất** theo ngữ cảnh.
2. Nếu bubble là sound effect ngắn (BƯỚC!, BÙM!, CẠCH!) → speaker='sound_effect'.
3. Giữ CÁC DẤU CÂU chính xác (. , ! ? ...).
4. Giữ UPPERCASE nếu comic đang UPPERCASE.
5. Xuất JSON array theo thứ tự đọc (trái→phải, trên→dưới).

# FEW-SHOT EXAMPLES (các lỗi điển hình và cách sửa)
Ví dụ 1: Thoại gốc trên trang "CHÚC MỪNG ĐẦU NĂM!"
❌ Đọc sai: "CHUA CHẮC ĐẦU NĂM"
✅ Đọc đúng: "CHÚC MỪNG ĐẦU NĂM!"

Ví dụ 2: Thoại "LÀM SAO CÓ NGƯỜI CHUI RA TỪ HỘC BÀN ĐƯỢC CHỨ?"
❌ Đọc sai: "LÀM SAO CÓ NGƯỜI CHUI RA TỪ HỌC BÀN ĐƯỢC CHÚ."
✅ Đọc đúng: "LÀM SAO CÓ NGƯỜI CHUI RA TỪ HỘC BÀN ĐƯỢC CHỨ?"

Ví dụ 3: Thoại "CHẮC LÀ CON GẶP ÁC MỘNG THÔI MÀ!"
❌ Đọc sai: "CHẮC LÀ CON GẶP ẢC MỘNG THÔI MÀ."
✅ Đọc đúng: "CHẮC LÀ CON GẶP ÁC MỘNG THÔI MÀ!"

Chỉ trả về JSON thuần, không thêm giải thích."""


# -----------------------------------------------------------------------------
# TIER 2: Self-correction prompt
# -----------------------------------------------------------------------------
VERIFY_PROMPT_TEMPLATE = """Đây là trang truyện tranh Việt Nam (manga dịch, UPPERCASE).

Ở pass 1, tôi đã OCR được các bubble sau:
{pass1_json}

NHIỆM VỤ PASS 2: Kiểm tra lại TỪNG bubble bằng cách nhìn trực tiếp ảnh. Với mỗi bubble:
1. Đọc lại text trong ảnh tại bbox đó
2. So sánh với text pass 1
3. Nếu pass 1 ĐÚNG → giữ nguyên
4. Nếu pass 1 SAI (thiếu dấu/nhầm dấu/sai từ) → sửa thành text đúng

CHÚ Ý DẤU THANH ĐIỆU TRÊN CHỮ HOA:
- Dấu sắc (Á Ó Ú) dễ bị miss → kiểm tra kỹ
- Phân biệt hỏi (Ả) vs ngã (Ã): hỏi là móc, ngã là sóng
- Kiểm tra dấu chữ Ă Â Đ Ê Ô Ơ Ư có đủ chưa

LỖI THƯỜNG THẤY:
- CHUA → CHƯA (thiếu ư+huyền) hoặc CHÚC (thiếu sắc)
- ẢC → ÁC (nhầm hỏi vs sắc)
- RÁC RỐI → RẮC RỐI (thiếu dấu ă)
- HỌC BÀN → HỘC BÀN (nhầm hỏi vs nặng)
- CẦU → CẬU (nhầm huyền vs nặng)
- LŨNG DANH → LỪNG DANH (nhầm ngã vs huyền)

Xuất JSON array với format giống pass 1 (giữ nguyên bbox_2d, speaker, speaker_desc — chỉ sửa `text` nếu cần).
Chỉ trả về JSON thuần."""


def extract_json_array(raw: str) -> list:
    """Extract JSON array from LLM output, handling markdown code fences."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Find the array
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
    p.add_argument("--no-tier2", action="store_true", help="Skip two-pass self-correction")
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
    print(f"Processing {len(page_files)} pages")

    index = []
    for page_idx, img_path in enumerate(page_files):
        page_name = img_path.stem
        print(f"\n[{page_idx+1}/{len(page_files)}] {page_name}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Cannot read {img_path}")
            continue
        pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        h, w = img_bgr.shape[:2]

        # ── Pass 1: enhanced prompt ──
        t0 = time.time()
        raw1 = qwen_call(pil, ENHANCED_DETECT_PROMPT, max_new_tokens=2048)
        bubbles_p1 = extract_json_array(raw1)
        print(f"  Pass 1: {len(bubbles_p1)} bubbles in {time.time()-t0:.1f}s")

        # ── Pass 2: self-correction ──
        bubbles_final = bubbles_p1
        if not args.no_tier2 and bubbles_p1:
            # Feed pass 1 JSON back with image
            p1_summary = json.dumps([
                {"bbox_2d": b.get("bbox_2d"),
                 "text": b.get("text", ""),
                 "speaker": b.get("speaker", "")}
                for b in bubbles_p1
            ], ensure_ascii=False, indent=2)
            verify_prompt = VERIFY_PROMPT_TEMPLATE.format(pass1_json=p1_summary)

            t1 = time.time()
            raw2 = qwen_call(pil, verify_prompt, max_new_tokens=2048)
            bubbles_p2 = extract_json_array(raw2)
            print(f"  Pass 2: {len(bubbles_p2)} bubbles in {time.time()-t1:.1f}s")

            # Merge: use pass 2 text, keep pass 1 bbox/speaker if pass 2 missing
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

        # ── Convert to pipeline format ──
        bubbles_out = []
        for order, b in enumerate(bubbles_final, start=1):
            bbox = b.get("bbox_2d", [0, 0, 0, 0])
            # Clamp bbox to page
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
            except Exception:
                x1, y1, x2, y2 = 0, 0, 0, 0
            bubbles_out.append({
                "order": order,
                "text": (b.get("text") or "").strip(),
                "text_pass1": b.get("text_pass1"),
                "bbox": [x1, y1, x2, y2],
                "qwen_speaker": b.get("speaker", "unknown"),
                "qwen_speaker_desc": b.get("speaker_desc", ""),
                "speaker_id": None,
                "attribution": "qwen",
                "ocr_conf": 1.0,
                "yolo_conf": 1.0,
            })

        page_result = {
            "image": str(img_path),
            "size": [w, h],
            "bubbles": bubbles_out,
        }

        out_json = out_dir / f"{page_idx+1:03d}_{page_name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(page_result, f, ensure_ascii=False, indent=2)
        index.append({"page_idx": page_idx+1, "name": page_name,
                      "json": str(out_json), "n_bubbles": len(bubbles_out)})
        print(f"  → {out_json.name}")

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(index)} pages → {out_dir}")


if __name__ == "__main__":
    main()
