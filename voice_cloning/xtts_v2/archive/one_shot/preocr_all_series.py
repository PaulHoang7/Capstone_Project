"""Pre-OCR all crawled manga pages across 5 series → JSON with bboxes + pre-filled text.

Runs Qwen-VL text-only prompt on each page to extract bubble bboxes + initial text.
Output feeds into labeling UI — user only needs to FIX text, not type from scratch.
"""
import argparse, json, sys, time
from pathlib import Path

# Reuse text-only OCR logic
sys.path.insert(0, str(Path(__file__).parent))
from ocr_text_only import SIMPLE_PROMPT, VERIFY_PROMPT_TEMPLATE, extract_json_array

import torch
import cv2


SERIES_DIRS = {
    "doraemon_ch1":   "/mnt/nfs-data/tin_dataset/comic/vietnamese/doraemon_ch1",
    "doraemon_ch2":   "/mnt/nfs-data/tin_dataset/comic/vietnamese/doraemon_ch2",
    "conan_ch1":      "/mnt/nfs-data/tin_dataset/comic/vietnamese/conan_ch1_new",
    "conan_ch2":      "/mnt/nfs-data/tin_dataset/comic/vietnamese/conan_ch2",
    "dragonball_ch1": "/mnt/nfs-data/tin_dataset/comic/vietnamese/dragonball_ch1",
    "dragonball_ch2": "/mnt/nfs-data/tin_dataset/comic/vietnamese/dragonball_ch2",
    "onepiece_ch1":   "/mnt/nfs-data/tin_dataset/comic/vietnamese/onepiece_ch1",
    "onepiece_ch2":   "/mnt/nfs-data/tin_dataset/comic/vietnamese/onepiece_ch2",
    "naruto_ch1":     "/mnt/nfs-data/tin_dataset/comic/vietnamese/naruto_ch1",
    "naruto_ch2":     "/mnt/nfs-data/tin_dataset/comic/vietnamese/naruto_ch2",
    "shinchan":       "/mnt/nfs-data/tin_dataset/comic/vietnamese/shinchan_ch1",
    "slamdunk":       "/mnt/nfs-data/tin_dataset/comic/vietnamese/slamdunk_ch1",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/mnt/nfs-data/tin_dataset/comic/labeling_pool")
    p.add_argument("--max-pages-per-series", type=int, default=40)
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

    all_index = []   # master index across all series
    total_bubbles = 0
    t_start = time.time()

    for series_name, series_dir in SERIES_DIRS.items():
        series_out = out_dir / series_name
        series_out.mkdir(exist_ok=True)

        src_dir = Path(series_dir)
        pages = sorted(list(src_dir.glob("*.webp")) + list(src_dir.glob("*.jpg")))[:args.max_pages_per_series]
        print(f"\n=== {series_name}: {len(pages)} pages ===")

        for page_idx, img_path in enumerate(pages):
            page_name = img_path.stem
            out_json = series_out / f"{page_idx+1:03d}_{page_name}.json"
            if out_json.exists():
                # Skip already done (resume support)
                with open(out_json) as f:
                    existing = json.load(f)
                all_index.append({
                    "series": series_name, "page_idx": page_idx+1,
                    "name": page_name, "json": str(out_json),
                    "n_bubbles": len(existing.get("bubbles", [])),
                })
                total_bubbles += len(existing.get("bubbles", []))
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            import cv2 as _cv
            pil = PILImage.fromarray(_cv.cvtColor(img_bgr, _cv.COLOR_BGR2RGB))
            h, w = img_bgr.shape[:2]

            t0 = time.time()
            try:
                raw = qwen_call(pil, SIMPLE_PROMPT)
                bubbles_p1 = extract_json_array(raw)
            except Exception as e:
                print(f"  [{page_idx+1}] {page_name}: OCR failed: {e}")
                continue
            dt = time.time() - t0

            bubbles_out = []
            for order, b in enumerate(bubbles_p1, start=1):
                bbox = b.get("bbox_2d", [0, 0, 0, 0])
                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1 = max(0, min(x1, w)); y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w)); y2 = max(0, min(y2, h))
                except Exception:
                    x1, y1, x2, y2 = 0, 0, 0, 0
                text = (b.get("text") or "").strip()
                if not text or (x2 <= x1) or (y2 <= y1):
                    continue
                bubbles_out.append({
                    "order": order, "bbox": [x1, y1, x2, y2],
                    "text_prefill": text,       # Qwen-VL OCR (to be edited)
                    "text_labeled": None,       # user-corrected (fill in UI)
                    "skipped": False,
                })

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "image": str(img_path),
                    "series": series_name,
                    "size": [w, h],
                    "bubbles": bubbles_out,
                }, f, ensure_ascii=False, indent=2)
            total_bubbles += len(bubbles_out)

            all_index.append({
                "series": series_name, "page_idx": page_idx+1,
                "name": page_name, "json": str(out_json),
                "n_bubbles": len(bubbles_out),
            })
            print(f"  [{page_idx+1:03d}] {page_name}: {len(bubbles_out)} bubbles ({dt:.1f}s)")

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(all_index, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n=== SUMMARY ===")
    print(f"Total pages: {len(all_index)}")
    print(f"Total bubbles: {total_bubbles}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
