"""
Train YOLOv8 on Manga109 for comic bubble/character/panel detection.

Usage:
    # Subset sanity check (~30 min on RTX 5090):
    python Capstone_project/scripts/train_yolo_manga109.py --subset --epochs 50

    # Full training (~2-4 hours):
    python Capstone_project/scripts/train_yolo_manga109.py

    # Resume from checkpoint:
    python Capstone_project/scripts/train_yolo_manga109.py --resume

    # Custom settings:
    python Capstone_project/scripts/train_yolo_manga109.py \\
        --model yolov8l.pt --imgsz 1280 --batch 4 --epochs 150

Targets (per cv-pipeline.md / evaluation.md):
    bubble    mAP@0.5 > 0.80
    character mAP@0.5 > 0.75
    panel     mAP@0.5 > 0.85
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────
SUBSET_YAML = "/mnt/nfs-data/tin_dataset/comic/manga109_subset/yolo/data.yaml"
FULL_YAML = "/mnt/nfs-data/tin_dataset/comic/manga109/yolo/data.yaml"
RUNS_DIR = "/mnt/nfs-data/tin_dataset/comic/yolo_runs"
BEST_SAVE_PATH = "/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt"

CLASS_NAMES = ["bubble", "character", "panel"]
CLASS_TARGETS = [0.80, 0.75, 0.85]  # mAP@0.5 targets


def print_results(metrics):
    """Print per-class mAP results with pass/fail against targets."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"  {'Class':>12s}  {'mAP@0.5':>8s}  {'mAP@50-95':>10s}  {'Target':>8s}  {'Status'}")
    print(f"  {'-'*54}")

    all_pass = True
    for i, (name, target) in enumerate(zip(CLASS_NAMES, CLASS_TARGETS)):
        try:
            ap50 = metrics.box.class_result(i)[2]   # AP@0.5
            ap5095 = metrics.box.class_result(i)[3]  # AP@0.5:0.95
            status = "PASS" if ap50 >= target else "MISS"
            if ap50 < target:
                all_pass = False
            print(f"  {name:>12s}  {ap50:>8.3f}  {ap5095:>10.3f}  {target:>8.2f}  {status}")
        except (IndexError, AttributeError):
            all_pass = False
            print(f"  {name:>12s}  {'N/A':>8s}  {'N/A':>10s}  {target:>8.2f}  ???")

    print(f"  {'-'*54}")
    print(f"  {'ALL':>12s}  {metrics.box.map50:>8.3f}  {metrics.box.map:>10.3f}")
    print()
    if all_pass:
        print("  All targets met!")
    else:
        print("  Some targets missed — consider more epochs or larger model.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Manga109")
    parser.add_argument("--subset", action="store_true",
                        help="Use 5-volume subset (~450 images) for fast prototyping")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (8 for imgsz=1024 on 32GB; 4 for imgsz=1280)")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Image size (1024 recommended — manga pages are large)")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Base model: yolov8n/s/m/l/x.pt (default: medium)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_yaml = SUBSET_YAML if args.subset else FULL_YAML
    run_name = "manga109_subset" if args.subset else "manga109_full"

    if args.resume:
        last_weights = Path(RUNS_DIR) / run_name / "weights" / "last.pt"
        if not last_weights.exists():
            print(f"ERROR: No checkpoint at {last_weights}")
            return
        model = YOLO(str(last_weights))
        print(f"Resuming from {last_weights}")
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        print(f"Model: {args.model}")
        print(f"Data:  {data_yaml}")
        print(f"Size:  {args.imgsz}, Batch: {args.batch}, Epochs: {args.epochs}")

        model.train(
            data=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=RUNS_DIR,
            name=run_name,
            device=args.device,
            workers=args.workers,
            # ── Optimizer ──
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            # ── Augmentation (tuned for manga/comics) ──
            hsv_h=0.005,    # manga is mostly grayscale — minimal color shift
            hsv_s=0.3,
            hsv_v=0.3,
            degrees=5.0,     # slight rotation only
            translate=0.1,
            scale=0.3,
            flipud=0.0,      # never flip vertically (text becomes unreadable)
            fliplr=0.3,      # horizontal flip is OK
            mosaic=0.8,
            mixup=0.1,
            # ── Control ──
            patience=args.patience,
            save_period=10,
            plots=True,
            exist_ok=True,
            val=True,
            verbose=True,
        )

    # ── Save best weights to standard checkpoint path ──
    best_weights = Path(RUNS_DIR) / run_name / "weights" / "best.pt"
    if best_weights.exists():
        Path(BEST_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, BEST_SAVE_PATH)
        print(f"\nBest model copied to: {BEST_SAVE_PATH}")

    # ── Final evaluation ──
    print(f"\n{'='*60}")
    print("Final evaluation on validation set")
    print(f"{'='*60}")
    eval_model = YOLO(str(best_weights) if best_weights.exists() else args.model)
    metrics = eval_model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        project=RUNS_DIR,
        name=f"{run_name}_eval",
        device=args.device,
        plots=True,
        exist_ok=True,
    )
    print_results(metrics)


if __name__ == "__main__":
    main()
