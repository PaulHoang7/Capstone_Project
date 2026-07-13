"""Option B: Use Qwen 2.5-VL-7B (text-only mode) to correct Vietnamese OCR errors.

Runs in comic_ocr env (has transformers 5.x + Qwen-VL already loaded).
Reads CV JSONs → corrects each bubble text → writes fixed JSONs.
Also dedupes consecutive/same-page duplicate bubbles.
"""
import argparse, json, re, time
from pathlib import Path

import torch


SYSTEM_PROMPT = """Bạn là chuyên gia tiếng Việt sửa lỗi chính tả. Nhiệm vụ của bạn:

1. Sửa lỗi dấu thanh điệu (sắc, huyền, hỏi, ngã, nặng) cho đúng chuẩn tiếng Việt.
2. Sửa lỗi dấu chữ (ă, â, đ, ê, ô, ơ, ư) nếu bị sai.
3. Sửa từ sai chính tả phổ biến (vd: "chua" → "chưa", "cầu" → "cậu" nếu ngữ cảnh đúng, "hoc bàn" → "hộc bàn").
4. Giữ NGUYÊN cấu trúc, độ dài câu, dấu câu. KHÔNG thêm/bớt từ trừ khi rõ ràng bị lỗi OCR.
5. Giữ NGUYÊN trường hợp (UPPERCASE thì xuất UPPERCASE, lowercase thì xuất lowercase).
6. Nếu text là sound effect (RẦM, BƯỚC, TA-DA...) hoặc chỉ 1-2 từ → giữ nguyên.
7. Chỉ trả về câu đã sửa, không giải thích. Nếu câu gốc đã đúng, trả về y nguyên.

Đây là text từ truyện tranh Doraemon (trẻ em, có hội thoại giữa Nobita, Doraemon, Shizuka, Jaian, Suneo)."""


def build_prompt(text: str) -> str:
    return f"Hãy sửa lỗi chính tả tiếng Việt cho câu sau (giữ nguyên trường hợp chữ):\n\n{text}\n\nKết quả:"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--no-dedupe", action="store_true")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id}...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    @torch.no_grad()
    def correct(text: str) -> str:
        text = text.strip()
        if not text or len(text) < 2:
            return text
        # Keep very short 1-2 word / all-punct / sound effects alone
        words = re.findall(r"\w+", text)
        if len(words) <= 1:
            return text

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": build_prompt(text)}]},
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=None, videos=None, return_tensors="pt").to("cuda:0")
        out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1)
        generated = out_ids[:, inputs.input_ids.shape[-1]:]
        out_text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        # Strip common wrappers like markdown code fences
        out_text = re.sub(r'^["`\']|["`\']$', '', out_text).strip()
        # If model returned a line like "Kết quả: ..." pick content after colon
        if ":" in out_text and out_text.split(":")[0].lower().strip() in ("kết quả", "sửa", "đáp án"):
            out_text = out_text.split(":", 1)[1].strip()
        # Sanity check — reject if wildly different length
        if len(out_text) > len(text) * 2 + 20 or len(out_text) < 1:
            return text
        return out_text

    with open(in_dir / "index.json") as f:
        pages_meta = json.load(f)

    new_pages_meta = []
    total_corrected = 0
    for meta in pages_meta:
        in_json = Path(meta["json"])
        out_json = out_dir / in_json.name
        with open(in_json) as f:
            page = json.load(f)

        bubbles = sorted(page.get("bubbles", []), key=lambda x: x.get("order", 0))
        fixed = []
        seen_norm = set()
        page_changes = 0

        for b in bubbles:
            text = (b.get("text") or "").strip()
            if not text:
                fixed.append(b)
                continue

            # Correction
            t0 = time.time()
            corrected = correct(text)
            dt = time.time() - t0
            if corrected != text:
                page_changes += 1
                total_corrected += 1
                print(f"  [{b.get('order'):02d}] ({dt:.1f}s) {text!r}\n        →   {corrected!r}")

            # Dedupe
            if not args.no_dedupe:
                norm = re.sub(r'[^\w\s]', '', corrected.lower()).strip()
                if norm in seen_norm and len(norm) >= 3:
                    print(f"  [{b.get('order'):02d}] [skip dup] {corrected!r}")
                    continue
                if len(norm) >= 3:
                    seen_norm.add(norm)

            new_b = dict(b)
            new_b["text"] = corrected
            if corrected != text:
                new_b["text_original"] = text
            fixed.append(new_b)

        page["bubbles"] = fixed
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)

        new_meta = dict(meta)
        new_meta["json"] = str(out_json)
        new_meta["n_bubbles"] = len(fixed)
        new_pages_meta.append(new_meta)

        print(f"{in_json.name}: {len(bubbles)} → {len(fixed)} bubbles, {page_changes} corrected")

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_pages_meta, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Total corrections: {total_corrected}")
    print(f"Fixed JSONs at {out_dir}")


if __name__ == "__main__":
    main()
