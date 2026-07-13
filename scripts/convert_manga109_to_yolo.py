"""
Convert Manga109 XML annotations to YOLOv8 format.

Manga109 annotation mapping:
  <text>  → class 0 (bubble)
  <body>  → class 1 (character)
  <frame> → class 2 (panel)

Note: <face> is NOT used — mixing face-scale and body-scale boxes produces
inconsistent training data. Pages with no <body> simply have no character labels.

Output: YOLO format — one .txt per image, each line: class_id cx cy w h (normalized 0-1)
Split: by volume (not page) to prevent data leakage.

Usage:
    # Full dataset
    python Capstone_project/scripts/convert_manga109_to_yolo.py \\
        --manga109-dir /mnt/nfs-data/tin_dataset/comic/manga109/raw \\
        --output-dir /mnt/nfs-data/tin_dataset/comic/manga109/yolo \\
        --val-ratio 0.2 --seed 42

    # Subset (fast prototyping, ~1500 images)
    python Capstone_project/scripts/convert_manga109_to_yolo.py \\
        --manga109-dir /mnt/nfs-data/tin_dataset/comic/manga109/raw \\
        --output-dir /mnt/nfs-data/tin_dataset/comic/manga109_subset/yolo \\
        --val-ratio 0.2 --seed 42 --max-volumes 13

    # Visualize spot-check (after conversion)
    python Capstone_project/scripts/convert_manga109_to_yolo.py \\
        --manga109-dir /mnt/nfs-data/tin_dataset/comic/manga109/raw \\
        --output-dir /mnt/nfs-data/tin_dataset/comic/manga109/yolo \\
        --visualize 10
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

CLASS_MAP = {
    "text": 0,   # bubble
    "body": 1,   # character
    "frame": 2,  # panel
}

CLASS_NAMES = {0: "bubble", 1: "character", 2: "panel"}


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert absolute pixel bbox to YOLO normalized (cx, cy, w, h)."""
    # Clamp to image bounds
    xmin = max(0, min(xmin, img_w))
    ymin = max(0, min(ymin, img_h))
    xmax = max(0, min(xmax, img_w))
    ymax = max(0, min(ymax, img_h))

    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h
    return (cx, cy, w, h)


def parse_volume_xml(xml_path):
    """Parse a Manga109 volume XML and return per-page annotations.

    Returns:
        list of (page_index, page_width, page_height, annotations)
        where annotations is list of (class_id, cx, cy, w, h)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pages_data = []
    pages_elem = root.find("pages")
    if pages_elem is None:
        return pages_data

    for page in pages_elem.findall("page"):
        idx = int(page.get("index"))
        pw = int(page.get("width"))
        ph = int(page.get("height"))

        annotations = []
        for elem_tag, class_id in CLASS_MAP.items():
            for elem in page.findall(elem_tag):
                try:
                    xmin = int(elem.get("xmin"))
                    ymin = int(elem.get("ymin"))
                    xmax = int(elem.get("xmax"))
                    ymax = int(elem.get("ymax"))
                except (TypeError, ValueError):
                    continue

                bbox = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, pw, ph)
                if bbox is not None:
                    annotations.append((class_id, *bbox))

        pages_data.append((idx, pw, ph, annotations))

    return pages_data


def discover_volumes(manga109_dir):
    """Discover available volumes from the annotations directory."""
    annotations_dir = Path(manga109_dir) / "annotations"
    if not annotations_dir.exists():
        # Try alternative structure: annotations might be at top level
        annotations_dir = Path(manga109_dir)

    volumes = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        vol_name = xml_path.stem
        volumes.append((vol_name, xml_path))

    return volumes


def find_image(images_dir, volume_name, page_index):
    """Find the image file for a given volume and page index."""
    vol_dir = Path(images_dir) / volume_name
    # Manga109 uses zero-padded 3-digit filenames
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = vol_dir / f"{page_index:03d}{ext}"
        if path.exists():
            return path
    return None


def convert_dataset(args):
    """Main conversion: XML annotations → YOLO format with train/val split."""
    manga109_dir = Path(args.manga109_dir)
    output_dir = Path(args.output_dir)
    images_dir = manga109_dir / "images"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        print("Expected structure: {manga109_dir}/images/{VolumeName}/000.jpg")
        return

    # Discover volumes
    volumes = discover_volumes(manga109_dir)
    if not volumes:
        print(f"ERROR: No XML annotations found in {manga109_dir}/annotations/")
        return

    print(f"Found {len(volumes)} volumes")

    # Optionally limit volumes (for subset)
    if args.max_volumes and args.max_volumes < len(volumes):
        volumes = volumes[:args.max_volumes]
        print(f"Using subset: {len(volumes)} volumes")

    # Split by volume
    random.seed(args.seed)
    vol_names = [v[0] for v in volumes]
    shuffled = vol_names.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - args.val_ratio))
    train_vols = set(shuffled[:split_idx])
    val_vols = set(shuffled[split_idx:])

    print(f"Split: {len(train_vols)} train volumes, {len(val_vols)} val volumes")

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "train": {"images": 0, "annotations": defaultdict(int)},
        "val": {"images": 0, "annotations": defaultdict(int)},
    }
    skipped_images = 0
    skipped_boxes = 0

    # Process each volume
    for vol_name, xml_path in volumes:
        split = "train" if vol_name in train_vols else "val"
        pages_data = parse_volume_xml(xml_path)

        for page_idx, pw, ph, annotations in pages_data:
            # Find source image
            src_img = find_image(images_dir, vol_name, page_idx)
            if src_img is None:
                skipped_images += 1
                continue

            # Output filenames
            stem = f"{vol_name}_{page_idx:03d}"
            dst_img = output_dir / "images" / split / f"{stem}{src_img.suffix}"
            dst_lbl = output_dir / "labels" / split / f"{stem}.txt"

            # Copy image
            shutil.copy2(src_img, dst_img)

            # Write YOLO label file
            with open(dst_lbl, "w") as f:
                for ann in annotations:
                    cls_id, cx, cy, w, h = ann
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    stats[split]["annotations"][cls_id] += 1

            stats[split]["images"] += 1

    # Generate data.yaml
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "bubble", 1: "character", 2: "panel"},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Conversion complete: {output_dir}")
    print(f"{'='*60}")
    print(f"data.yaml: {yaml_path}")
    print(f"Skipped images (not found): {skipped_images}")
    print()

    total_imgs = 0
    total_anns = 0
    for split in ["train", "val"]:
        s = stats[split]
        n_imgs = s["images"]
        total_imgs += n_imgs
        print(f"  {split}:")
        print(f"    Images: {n_imgs}")
        for cls_id in sorted(CLASS_NAMES.keys()):
            count = s["annotations"][cls_id]
            total_anns += count
            print(f"    {CLASS_NAMES[cls_id]:>12s}: {count:>6d} annotations")
        print()

    print(f"  Total: {total_imgs} images, {total_anns} annotations")


def visualize_samples(args):
    """Draw YOLO labels on random images for spot-checking."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required for visualization. Install: pip install opencv-python")
        return

    output_dir = Path(args.output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # BGR

    # Collect all image-label pairs
    pairs = []
    for split in ["train", "val"]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path, split))

    if not pairs:
        print("No image-label pairs found for visualization.")
        return

    n = min(args.visualize, len(pairs))
    random.seed(args.seed)
    samples = random.sample(pairs, n)

    for img_path, lbl_path, split in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = CLASS_NAMES.get(cls_id, str(cls_id))
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = viz_dir / f"{split}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)

    print(f"Saved {n} visualizations to {viz_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Manga109 annotations to YOLOv8 format"
    )
    parser.add_argument(
        "--manga109-dir", required=True,
        help="Path to extracted Manga109 root (contains images/ and annotations/)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for YOLO dataset (images/{train,val}, labels/{train,val})",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-volumes", type=int, default=None,
        help="Limit to first N volumes (alphabetical) for subset creation",
    )
    parser.add_argument(
        "--visualize", type=int, default=0,
        help="If >0, draw labels on N random images for spot-checking (skip conversion)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.visualize > 0:
        visualize_samples(args)
    else:
        convert_dataset(args)
