"""
Evaluate trained YOLO model on Manga109 val set.

Usage:
    python Capstone_project/scripts/eval_yolo_manga109.py \\
        --weights /mnt/nfs-data/tin_dataset/comic/yolo_runs/manga109_full_v1/weights/best.pt \\
        --data /mnt/nfs-data/tin_dataset/comic/manga109/yolo/data.yaml

    # With prediction visualization:
    python Capstone_project/scripts/eval_yolo_manga109.py \\
        --weights best.pt --data data.yaml --predict 20
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on Manga109")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--predict", type=int, default=0,
                        help="Run prediction on N val images and save visualizations")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Validation metrics
    print("Running validation...")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    class_names = ["bubble", "character", "panel"]

    print(f"\n{'='*50}")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"{'='*50}")
    print(f"  Per-class mAP@0.5:")
    for i, name in enumerate(class_names):
        if i < len(metrics.box.maps):
            print(f"    {name:>12s}: {metrics.box.maps[i]:.4f}")
    print(f"{'='*50}")

    target = 0.8
    passed = metrics.box.map50 >= target
    print(f"\n  Target mAP@0.5 >= {target}: {'PASS' if passed else 'FAIL'}")

    # Optional: run prediction on some val images
    if args.predict > 0:
        import yaml
        from pathlib import Path

        with open(args.data) as f:
            data_cfg = yaml.safe_load(f)

        val_dir = Path(data_cfg["path"]) / data_cfg["val"]
        val_images = sorted(val_dir.glob("*.jpg"))[:args.predict]

        if val_images:
            results = model.predict(
                source=val_images,
                save=True,
                project=str(Path(args.weights).parent.parent / "predictions"),
                name="val_samples",
                imgsz=args.imgsz,
                device=args.device,
                exist_ok=True,
            )
            print(f"\nSaved {len(val_images)} prediction visualizations")


if __name__ == "__main__":
    main()
