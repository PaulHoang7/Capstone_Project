"""Export labeling task as CSV + bubble crops folder.

Output structure:
    <out_dir>/
      bubble_crops/
        00001.png  (series=doraemon, page=001, order=1)
        00002.png
        ...
      labels.csv       ← user edits this
      README.txt       ← instructions

CSV columns:
  id | series | page | order | image | prefill | corrected_text | skip
  └─ don't edit ─┘                   └──── user fills ─────┘

Workflow:
  1. User opens labels.csv in Excel/Google Sheets/LibreOffice
  2. For each row:
     - View bubble_crops/<id>.png
     - Read prefill column (Qwen-VL output)
     - If prefill correct → copy to corrected_text column (or leave empty = use prefill)
     - If prefill wrong → type correct text in corrected_text
     - If not-a-bubble → put 'x' in skip column
  3. Save CSV → send back
"""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="/mnt/nfs-data/tin_dataset/comic/labeling_pool_yolo",
                   help="Pool dir — use labeling_pool_yolo for YOLO-enhanced bboxes")
    p.add_argument("--out", default="/home/bes/Desktop/Tin/labeling_task")
    p.add_argument("--crop-padding-yolo", type=int, default=15,
                   help="Padding when using YOLO bubble bbox (already full bubble)")
    p.add_argument("--crop-padding-fallback", type=int, default=120,
                   help="Padding when Qwen bbox (tight text) used — need lots more")
    p.add_argument("--crop-expand-pct", type=float, default=0.25,
                   help="Extra expand fraction (safety margin)")
    p.add_argument("--min-crop-width", type=int, default=500,
                   help="Upscale crops narrower than this for readability")
    args = p.parse_args()

    pool_dir = Path(args.pool)
    out_dir = Path(args.out)
    crops_dir = out_dir / "bubble_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with open(pool_dir / "index.json") as f:
        pages_meta = json.load(f)

    csv_path = out_dir / "labels.csv"
    rows = []
    row_id = 0
    skipped_short = 0

    for page_meta in pages_meta:
        series = page_meta["series"]
        with open(page_meta["json"]) as f:
            page = json.load(f)

        image_path = page["image"]
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"  skip {image_path}: {e}")
            continue

        for b in page.get("bubbles", []):
            order = b["order"]
            bbox = b["bbox"]
            prefill = b.get("text_prefill", "").strip()

            # Skip bubbles that are clearly watermarks/single chars
            if len(prefill) <= 2 and not any(c in prefill for c in "!?"):
                skipped_short += 1
                continue

            row_id += 1
            crop_name = f"{row_id:05d}.png"
            crop_path = crops_dir / crop_name

            # Crop + save
            # If YOLO matched: bbox is already full bubble outline → small padding is enough
            # If unmatched (Qwen tight text bbox): need BIG padding to capture full bubble
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1
            is_fallback = b.get("no_yolo_match", False)
            base_pad = args.crop_padding_fallback if is_fallback else args.crop_padding_yolo
            pad_x = max(base_pad, int(bw * args.crop_expand_pct))
            pad_y = max(base_pad, int(bh * args.crop_expand_pct))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.width, x2 + pad_x)
            y2 = min(img.height, y2 + pad_y)
            crop = img.crop((x1, y1, x2, y2))

            if crop.width < args.min_crop_width:
                scale = args.min_crop_width / crop.width
                new_w = int(crop.width * scale)
                new_h = int(crop.height * scale)
                crop = crop.resize((new_w, new_h), Image.LANCZOS)

            crop.save(crop_path)

            rows.append({
                "id":             f"{row_id:05d}",
                "series":         series,
                "page":           page_meta["name"],
                "order":          order,
                "image":          f"bubble_crops/{crop_name}",
                "prefill":        prefill,
                "corrected_text": "",     # user fills
                "skip":           "",     # user marks 'x' if not a dialogue bubble
            })

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "series", "page", "order", "image",
                           "prefill", "corrected_text", "skip"])
        writer.writeheader()
        writer.writerows(rows)

    # README
    readme = """Vietnamese UPPERCASE Manga OCR Labeling Task
==============================================

GOAL: Fine-tune Qwen-VL to read Vietnamese UPPERCASE manga text accurately.

WORKFLOW:
1. Open labels.csv in Excel / Google Sheets / LibreOffice Calc
2. For each row:
   - Open the image in bubble_crops/ (click image cell link OR find file manually)
   - READ the correct text in the bubble
   - Compare with 'prefill' column (Qwen-VL output — often wrong tone marks)

3. Fill 'corrected_text' column:
   - If prefill is 100% correct → copy it into corrected_text (or leave blank = auto-use prefill)
   - If prefill is wrong → type the correct Vietnamese text with proper tone marks
   - If the image is NOT a dialogue bubble (watermark / sfx / border noise) → put 'x' in 'skip' column

4. Save file → send back

TIPS:
- Target 500 bubbles minimum for MVP fine-tune (more = better)
- Rate: ~15 seconds per bubble with pre-fill
- Partial OK — skip rows you're unsure about

COLUMNS:
  id         = unique row ID (don't edit)
  series     = manga series (doraemon/conan/dragonball/onepiece/naruto)
  page       = page filename (don't edit)
  order      = bubble order in page (don't edit)
  image      = path to bubble crop image (don't edit)
  prefill    = Qwen-VL OCR output (may have errors)
  corrected_text = YOU FILL — the correct Vietnamese text
  skip       = YOU MARK 'x' if image is not a text bubble (watermark, etc.)

VIETNAMESE TONE MARKS cheat sheet:
  SẮC: á é í ó ú ý  (upper: Á É Í Ó Ú Ý)
  HUYỀN: à è ì ò ù ỳ (À È Ì Ò Ù Ỳ)
  HỎI: ả ẻ ỉ ỏ ủ ỷ (Ả Ẻ Ỉ Ỏ Ủ Ỷ)
  NGÃ: ã ẽ ĩ õ ũ ỹ (Ã Ẽ Ĩ Õ Ũ Ỹ)
  NẶNG: ạ ẹ ị ọ ụ ỵ (Ạ Ẹ Ị Ọ Ụ Ỵ)
  SPECIAL: ă â đ ê ô ơ ư (Ă Â Đ Ê Ô Ơ Ư)
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"\n✓ Exported {len(rows)} rows → {csv_path}")
    print(f"✓ Bubble crops → {crops_dir}")
    print(f"✓ Instructions → {out_dir}/README.txt")
    print(f"\n  Skipped {skipped_short} very-short prefills (≤2 chars, likely noise)")
    print(f"  Total rows to label: {len(rows)}")
    print(f"\n  Target MVP: 500 rows labeled")
    print(f"\nOpen:  {csv_path}")


if __name__ == "__main__":
    main()
