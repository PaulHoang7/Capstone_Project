"""Replace Qwen-VL tight text bbox với YOLO full bubble bbox.

Input: /mnt/nfs-data/tin_dataset/comic/labeling_pool/{series}/*.json
       (chứa text_prefill + Qwen tight text bbox)

Output: /mnt/nfs-data/tin_dataset/comic/labeling_pool_yolo/{series}/*.json
        (same text_prefill nhưng bbox = YOLO full bubble outline)

Logic:
- Chạy YOLO trên mỗi page → lấy bubbles class=0
- Với mỗi Qwen text bbox, tìm YOLO bubble có IoU cao nhất
- Nếu IoU > 0.05 (overlap dù nhỏ) hoặc Qwen-center nằm trong YOLO bubble → match
- Match → thay bbox bằng YOLO bubble bbox (FULL outline)
- Không match → giữ Qwen bbox + flag `no_yolo_match=True` (fallback padding lớn trong export)
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2


def iou(box_a, box_b):
    """IoU giữa 2 bbox [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_inside(small, big):
    """Check if center of `small` is inside `big`."""
    cx = (small[0] + small[2]) / 2
    cy = (small[1] + small[3]) / 2
    return big[0] <= cx <= big[2] and big[1] <= cy <= big[3]


def containment(small, big):
    """Fraction of `small` area inside `big`."""
    sx1, sy1, sx2, sy2 = small
    bx1, by1, bx2, by2 = big
    ix1 = max(sx1, bx1); iy1 = max(sy1, by1)
    ix2 = min(sx2, bx2); iy2 = min(sy2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_s = max(1, (sx2 - sx1) * (sy2 - sy1))
    return inter / area_s


def run_yolo_on_page(yolo_model, image_path: str, conf: float = 0.08, imgsz: int = 1536):
    """Return (dims, bubbles, characters, panels) — all 3 YOLO classes in 1 inference.

    Permissive conf + high imgsz for small/stylized VN comic bubbles.
    Characters and panels saved for downstream face clustering + reading order.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, [], [], []
    h, w = image.shape[:2]
    results = yolo_model.predict(
        source=image, conf=conf, iou=0.5, imgsz=imgsz,
        verbose=False, device=yolo_model.device,
    )
    bubbles, characters, panels = [], [], []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        conf_score = round(float(box.conf[0]), 3)
        entry = {"bbox": bbox, "conf": conf_score}
        if cls_id == 0:
            bubbles.append(bbox)   # keep as-is for backward compat
        elif cls_id == 1:
            characters.append(entry)
        elif cls_id == 2:
            panels.append(entry)
    return (h, w), bubbles, characters, panels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-pool", default="/mnt/nfs-data/tin_dataset/comic/labeling_pool")
    p.add_argument("--output-pool", default="/mnt/nfs-data/tin_dataset/comic/labeling_pool_yolo")
    p.add_argument("--yolo-ckpt", default="/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt")
    p.add_argument("--conf", type=float, default=0.08)
    p.add_argument("--min-containment", type=float, default=0.20,
                   help="Min fraction of Qwen bbox inside YOLO bubble to count as match")
    p.add_argument("--imgsz", type=int, default=1536,
                   help="YOLO inference image size (larger = catch smaller bubbles)")
    args = p.parse_args()

    in_pool = Path(args.input_pool)
    out_pool = Path(args.output_pool)
    out_pool.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO from {args.yolo_ckpt}...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_ckpt)
    print(f"  YOLO loaded (classes: {yolo.names})")

    # Load master index
    with open(in_pool / "index.json") as f:
        pages_meta = json.load(f)
    print(f"Processing {len(pages_meta)} pages")

    stats = {"total_bubbles": 0, "matched": 0, "unmatched": 0, "pages_processed": 0}
    new_pages_meta = []

    for page_meta in pages_meta:
        series = page_meta["series"]
        series_out = out_pool / series
        series_out.mkdir(exist_ok=True)

        with open(page_meta["json"]) as f:
            page = json.load(f)

        image_path = page.get("image")
        if not image_path:
            continue

        dims, yolo_bubbles, yolo_characters, yolo_panels = run_yolo_on_page(
            yolo, image_path, args.conf, args.imgsz)
        if dims is None:
            print(f"  skip {page_meta['name']}: can't read image")
            continue

        # Match each Qwen text bbox to best YOLO bubble
        new_bubbles = []
        for b in page.get("bubbles", []):
            qwen_bbox = b["bbox"]
            stats["total_bubbles"] += 1

            best_yolo = None
            best_score = 0.0
            for yb in yolo_bubbles:
                # Containment (% of Qwen bbox inside YOLO bubble) — main metric
                c = containment(qwen_bbox, yb)
                if c > best_score:
                    best_score = c
                    best_yolo = yb

            new_b = dict(b)
            if best_yolo and best_score >= args.min_containment:
                new_b["bbox_qwen"] = qwen_bbox      # preserve original
                new_b["bbox"] = best_yolo            # replace with YOLO bubble
                new_b["yolo_match_score"] = round(best_score, 3)
                new_b["no_yolo_match"] = False
                stats["matched"] += 1
            else:
                new_b["bbox_qwen"] = qwen_bbox
                new_b["yolo_match_score"] = round(best_score, 3)
                new_b["no_yolo_match"] = True        # flag for aggressive padding
                stats["unmatched"] += 1

            new_bubbles.append(new_b)

        page["bubbles"] = new_bubbles
        page["yolo_bubbles_total"] = len(yolo_bubbles)
        # Save character + panel detections for downstream clustering / reading order
        page["yolo_characters"] = yolo_characters
        page["yolo_panels"] = yolo_panels

        # Save new JSON
        out_json = series_out / f"{page_meta['page_idx']:03d}_{page_meta['name']}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)

        new_meta = dict(page_meta)
        new_meta["json"] = str(out_json)
        new_pages_meta.append(new_meta)

        stats["pages_processed"] += 1
        if stats["pages_processed"] % 20 == 0:
            print(f"  {stats['pages_processed']}/{len(pages_meta)} pages "
                  f"({stats['matched']}/{stats['total_bubbles']} matched)")

    # Save index
    with open(out_pool / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_pages_meta, f, ensure_ascii=False, indent=2)

    match_rate = stats["matched"] / max(1, stats["total_bubbles"]) * 100
    print(f"\n=== Summary ===")
    print(f"Pages processed: {stats['pages_processed']}")
    print(f"Total bubbles: {stats['total_bubbles']}")
    print(f"Matched to YOLO: {stats['matched']} ({match_rate:.1f}%)")
    print(f"Unmatched (Qwen-only + fallback padding): {stats['unmatched']}")
    print(f"Output: {out_pool}")


if __name__ == "__main__":
    main()
