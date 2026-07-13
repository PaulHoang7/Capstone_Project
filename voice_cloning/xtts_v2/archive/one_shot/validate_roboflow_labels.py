"""Validate Roboflow YOLOv8 export + generate preview images with bboxes drawn.

Checks:
- ZIP structure: images/ + labels/ folders
- Label format: valid YOLOv8 (class cx cy w h normalized 0-1)
- Image-label pairing: every image has a label and vice versa
- Bbox sanity: coordinates in [0,1], width/height > 0
- Sample visualization: draw bboxes on 10 random images → preview PNG
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_yolo_label(label_path: Path):
    """Parse YOLO txt: each line = class cx cy w h normalized."""
    bboxes = []
    with open(label_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                return None, f"line {line_no}: expected 5 fields, got {len(parts)}"
            try:
                cls = int(parts[0])
                cx, cy, w, h = [float(v) for v in parts[1:]]
            except ValueError as e:
                return None, f"line {line_no}: parse error {e}"
            # Sanity
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                return None, f"line {line_no}: cx/cy out of [0,1]: {cx},{cy}"
            if not (0 < w <= 1 and 0 < h <= 1):
                return None, f"line {line_no}: w/h invalid: {w},{h}"
            bboxes.append({"class": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return bboxes, None


def draw_bboxes(image_path: Path, bboxes: list, out_path: Path):
    """Draw bboxes on image and save to out_path."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    for i, b in enumerate(bboxes):
        x1 = (b["cx"] - b["w"]/2) * W
        y1 = (b["cy"] - b["h"]/2) * H
        x2 = (b["cx"] + b["w"]/2) * W
        y2 = (b["cy"] + b["h"]/2) * H
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="#ff3333", width=4)
        # Label index
        draw.text((x1+4, y1+4), f"#{i+1}", fill="#ffff00", font=font,
                  stroke_width=2, stroke_fill="#000000")
    img.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract-dir", default="/home/bes/Desktop/Tin/roboflow_extracted")
    p.add_argument("--out", default="/home/bes/Desktop/Tin/roboflow_validation")
    p.add_argument("--sample-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    extract_dir = Path(args.extract_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(exist_ok=True)

    # Find images + labels
    splits = [d for d in extract_dir.iterdir() if d.is_dir()]
    print(f"Splits found: {[s.name for s in splits]}")

    all_images = []
    all_labels = []
    for split in splits:
        img_dir = split / "images"
        lbl_dir = split / "labels"
        if img_dir.exists():
            all_images.extend(list(img_dir.glob("*.*")))
        if lbl_dir.exists():
            all_labels.extend(list(lbl_dir.glob("*.txt")))

    print(f"Total images: {len(all_images)}")
    print(f"Total label files: {len(all_labels)}")

    # Pair images + labels
    image_map = {img.stem: img for img in all_images}
    label_map = {lbl.stem: lbl for lbl in all_labels}

    unpaired_images = set(image_map) - set(label_map)
    unpaired_labels = set(label_map) - set(image_map)
    print(f"\nImages without label: {len(unpaired_images)}")
    print(f"Labels without image: {len(unpaired_labels)}")

    # Parse all labels
    total_bboxes = 0
    errors = []
    per_image_count = []
    for stem, lbl_path in label_map.items():
        bboxes, err = parse_yolo_label(lbl_path)
        if err:
            errors.append(f"{lbl_path.name}: {err}")
            continue
        total_bboxes += len(bboxes)
        per_image_count.append(len(bboxes))

    print(f"\nTotal bboxes: {total_bboxes}")
    print(f"Avg bboxes/image: {total_bboxes/max(1,len(label_map)):.1f}")
    if per_image_count:
        print(f"Min/Max bboxes per image: {min(per_image_count)}/{max(per_image_count)}")
    if errors:
        print(f"\n❌ Format errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print("\n✅ All label files valid format")

    # Visualize random sample
    random.seed(args.seed)
    paired = [stem for stem in image_map if stem in label_map]
    sample = random.sample(paired, min(args.sample_size, len(paired)))
    print(f"\nGenerating {len(sample)} preview images...")

    preview_data = []
    for stem in sample:
        img_path = image_map[stem]
        lbl_path = label_map[stem]
        bboxes, _ = parse_yolo_label(lbl_path)
        out_png = out_dir / "previews" / f"{stem}.png"
        draw_bboxes(img_path, bboxes, out_png)
        preview_data.append({
            "stem": stem,
            "preview": str(out_png),
            "n_bboxes": len(bboxes),
        })
        print(f"  {stem}: {len(bboxes)} bboxes → {out_png.name}")

    # Save JSON summary
    summary = {
        "splits": [s.name for s in splits],
        "total_images": len(all_images),
        "total_labels": len(all_labels),
        "total_bboxes": total_bboxes,
        "avg_bboxes_per_image": round(total_bboxes/max(1,len(label_map)), 2),
        "min_bboxes": min(per_image_count) if per_image_count else 0,
        "max_bboxes": max(per_image_count) if per_image_count else 0,
        "unpaired_images": len(unpaired_images),
        "unpaired_labels": len(unpaired_labels),
        "errors": errors,
        "preview_sample": preview_data,
    }
    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Validation report: {out_dir}/validation_report.json")
    print(f"✓ Preview images: {out_dir}/previews/")


if __name__ == "__main__":
    main()
