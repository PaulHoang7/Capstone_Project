"""
Run YOLOv8 inference on comic pages — detect bubbles, characters, panels.

Usage:
    # Single image
    python Capstone_project/scripts/infer_yolo_comic.py \
        --image /path/to/comic_page.jpg

    # Directory of images
    python Capstone_project/scripts/infer_yolo_comic.py \
        --image /path/to/pages/ --save-dir /tmp/yolo_output

    # Custom model + confidence threshold
    python Capstone_project/scripts/infer_yolo_comic.py \
        --image page.jpg --weights /path/to/best.pt --conf 0.3
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = "/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt"
CLASS_NAMES = {0: "bubble", 1: "character", 2: "panel"}
# BGR colors for OpenCV drawing
CLASS_COLORS = {0: (255, 80, 80), 1: (80, 255, 80), 2: (80, 80, 255)}


def run_inference(args):
    model = YOLO(args.weights)
    print(f"Model: {args.weights}")
    print(f"Input: {args.image}")
    print(f"Conf:  {args.conf}, IoU: {args.iou}")

    results = model.predict(
        source=args.image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        img_name = Path(result.path).stem
        boxes = result.boxes

        # ── Print detections ──
        print(f"\n{'─'*50}")
        print(f"  {Path(result.path).name}: {len(boxes)} detections")
        print(f"{'─'*50}")

        counts = {0: 0, 1: 0, 2: 0}
        det_list = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            counts[cls_id] = counts.get(cls_id, 0) + 1

            print(f"  {cls_name:>12s}  conf={conf:.2f}  "
                  f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

            det_list.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "bbox_xyxy": [round(v, 1) for v in [x1, y1, x2, y2]],
            })

        print(f"\n  Summary: {counts[0]} bubbles, "
              f"{counts[1]} characters, {counts[2]} panels")

        # ── Save annotated image ──
        annotated = result.plot(
            line_width=2,
            font_size=12,
            labels=True,
            conf=True,
        )
        out_img = save_dir / f"{img_name}_detected.jpg"

        import cv2
        cv2.imwrite(str(out_img), annotated)
        print(f"  Saved: {out_img}")

        # ── Save JSON detections ──
        if args.save_json:
            out_json = save_dir / f"{img_name}_detections.json"
            with open(out_json, "w") as f:
                json.dump({
                    "image": str(result.path),
                    "detections": det_list,
                    "counts": {CLASS_NAMES[k]: v for k, v in counts.items()},
                }, f, indent=2)
            print(f"  JSON:  {out_json}")


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 comic page inference")
    p.add_argument("--image", required=True,
                   help="Path to image or directory of images")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help=f"Model weights (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--save-dir", default="/tmp/yolo_comic_output",
                   help="Directory to save annotated images")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="Inference image size (default: 1024)")
    p.add_argument("--save-json", action="store_true",
                   help="Also save detections as JSON")
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
