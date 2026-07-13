"""
Full CV pipeline — comic page to structured output.

Takes a comic page image and runs the complete pipeline:
  1. YOLO          → detect bubbles, characters, panels
  2. OCR           → read text from each bubble crop
  3. Face Cluster  → extract face embeddings, match to character_db
  4. Speaker Attr  → assign each bubble to a character (MLP or rule-based)
  5. Reading Order → sort bubbles top→bottom, left→right (LTR) or RTL

Output per page (JSON):
  {
    "image": "page.jpg",
    "width": 1654, "height": 1170,
    "characters": [
      {"char_id": "char_0", "bbox": [...], "face_bbox": [...]}
    ],
    "bubbles": [
      {
        "order":      1,
        "text":       "セリフ",
        "bbox":       [x1, y1, x2, y2],
        "speaker_id": "char_0",
        "confidence": 0.92,
        "ocr_conf":   0.98
      }
    ]
  }

Usage:
    # Single page (Japanese manga)
    python Capstone_project/scripts/comic_pipeline.py \\
        --image page.jpg --lang ja

    # Chapter directory (Vietnamese, with existing character_db)
    python Capstone_project/scripts/comic_pipeline.py \\
        --image /path/to/chapter/ --lang vi \\
        --char-db /tmp/char_db/character_db.json \\
        --save-dir /tmp/pipeline_out

    # Vietnamese, paddle-only OCR, no face clustering
    python Capstone_project/scripts/comic_pipeline.py \\
        --image page.jpg --lang vi --paddle-only --no-face

    # Rule-based speaker attribution (no MLP model needed)
    python Capstone_project/scripts/comic_pipeline.py \\
        --image page.jpg --rule-based
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root (Desktop/Tin) is on sys.path regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_YOLO_WEIGHTS   = "/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt"
DEFAULT_SPEAKER_MODEL  = "/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_mlp.pt"
DEFAULT_SPEAKER_SCALER = "/mnt/nfs-data/tin_dataset/comic/speaker_attribution/scaler.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: YOLO detection
# ─────────────────────────────────────────────────────────────────────────────

def run_yolo(
    image: np.ndarray,
    model,
    conf: float = 0.25,
    iou: float  = 0.45,
    imgsz: int  = 1024,
) -> dict[str, list[dict]]:
    """Run YOLO on a BGR image. Returns detections grouped by class.

    Returns:
        {
          "bubbles":    [{"bbox": [x1,y1,x2,y2], "confidence": float}],
          "characters": [{"bbox": [x1,y1,x2,y2], "confidence": float}],
          "panels":     [{"bbox": [x1,y1,x2,y2], "confidence": float}],
        }
    """
    results = model.predict(
        source=image, conf=conf, iou=iou, imgsz=imgsz,
        verbose=False, device=model.device,
    )

    detections: dict[str, list[dict]] = {"bubbles": [], "characters": [], "panels": []}
    class_map = {0: "bubbles", 1: "characters", 2: "panels"}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        key    = class_map.get(cls_id)
        if key is None:
            continue
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections[key].append({
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(float(box.conf[0]), 4),
        })

    log.debug(
        f"  YOLO: {len(detections['bubbles'])} bubbles, "
        f"{len(detections['characters'])} chars, "
        f"{len(detections['panels'])} panels"
    )
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: OCR on bubble crops
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr_on_bubbles(
    image: np.ndarray,
    bubbles: list[dict],
    ocr_pipeline,
) -> list[dict]:
    """Run OCR on each detected bubble crop.

    Uses ocr_bubble_crop() from ocr_pipeline.py for intra-bubble text detection.
    Falls back to whole-bubble recognition if no lines detected.

    Returns bubbles list with added 'text' and 'ocr_conf' fields.
    """
    from Capstone_project.scripts.ocr_pipeline import (
        ocr_bubble_crop, PaddleDetector, VietOCRRecognizer, PaddleRecognizer,
    )

    h, w = image.shape[:2]
    enriched = []

    for bubble in bubbles:
        x1, y1, x2, y2 = [int(v) for v in bubble["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            enriched.append({**bubble, "text": "", "ocr_conf": 0.0})
            continue

        # Use 2-layer OCR if available (det + rec separately)
        if hasattr(ocr_pipeline, "_detector") and ocr_pipeline._detector is not None:
            regions = ocr_bubble_crop(
                crop,
                ocr_pipeline._recognizer,
                ocr_pipeline._detector,
            )
        elif hasattr(ocr_pipeline, "_full_paddle") and ocr_pipeline._full_paddle is not None:
            regions = ocr_pipeline._full_paddle.run(crop)
        else:
            regions = []

        if regions:
            # Join multi-line text, average confidence
            text     = " ".join(r["text"] for r in regions if r["text"])
            ocr_conf = float(np.mean([r["confidence"] for r in regions]))
        else:
            text, ocr_conf = "", 0.0

        enriched.append({**bubble, "text": text.strip(), "ocr_conf": round(ocr_conf, 4)})

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Face clustering / character matching
# ─────────────────────────────────────────────────────────────────────────────

def run_face_matching(
    image: np.ndarray,
    characters: list[dict],
    extractor,
    char_db,
    page_name: str,
) -> list[dict]:
    """Extract face embeddings from character crops, match to character_db.

    Returns characters list with added 'char_id' and 'face_bbox' fields.
    """
    h, w = image.shape[:2]
    enriched = []

    for char in characters:
        x1, y1, x2, y2 = [int(v) for v in char["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            enriched.append({**char, "char_id": None, "face_bbox": None})
            continue

        faces = extractor.extract(crop)

        if not faces:
            enriched.append({**char, "char_id": None, "face_bbox": None})
            continue

        # Use highest-confidence face
        best_face = max(faces, key=lambda f: f["det_score"])
        emb       = best_face["embedding"]

        # Translate face bbox from crop coords to page coords
        fx1, fy1, fx2, fy2 = best_face["bbox"]
        face_bbox_page = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]

        # Match against character_db
        char_id, score = char_db.match(emb)
        if char_id is not None:
            char_db.add_appearance(char_id, emb, page_name, char["bbox"])
        else:
            char_id = char_db.add_new_character(emb, page_name, char["bbox"])

        enriched.append({
            **char,
            "char_id":  char_id,
            "face_bbox": face_bbox_page,
            "face_sim":  round(score, 4),
        })

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Speaker attribution
# ─────────────────────────────────────────────────────────────────────────────

def run_speaker_attribution(
    bubbles: list[dict],
    characters: list[dict],
    page_w: int,
    page_h: int,
    panels: list[dict],
    speaker_model_path: Optional[Path],
    speaker_scaler_path: Optional[Path],
    rule_based: bool = False,
    use_gpu: bool = True,
) -> list[dict]:
    """Assign a speaker (character) to each bubble.

    Uses MLP model if available, falls back to rule-based proximity.
    """
    from Capstone_project.scripts.train_speaker_attribution import (
        assign_speaker_rule_based,
        build_features,
        predict_speaker,
    )

    # Filter characters that have been matched to a char_id
    known_chars = [c for c in characters if c.get("char_id") is not None]
    if not known_chars:
        known_chars = characters  # use all if none matched

    if not known_chars:
        return [{**b, "speaker_id": None, "speaker_conf": 0.0} for b in bubbles]

    # Convert character bboxes to format expected by feature builder
    char_bboxes = [
        {
            "id":   c.get("char_id", f"char_{i}"),
            "xmin": int(c["bbox"][0]), "ymin": int(c["bbox"][1]),
            "xmax": int(c["bbox"][2]), "ymax": int(c["bbox"][3]),
        }
        for i, c in enumerate(known_chars)
    ]

    # Convert bubble bboxes
    bubble_bboxes = [
        {
            "id":   str(i),
            "xmin": int(b["bbox"][0]), "ymin": int(b["bbox"][1]),
            "xmax": int(b["bbox"][2]), "ymax": int(b["bbox"][3]),
        }
        for i, b in enumerate(bubbles)
    ]

    # Use MLP if model available and not forced rule-based
    if not rule_based and speaker_model_path and speaker_model_path.exists():
        try:
            predictions = predict_speaker(
                model_path   = speaker_model_path,
                scaler_path  = speaker_scaler_path,
                bubbles      = bubble_bboxes,
                characters   = char_bboxes,
                page_w       = page_w,
                page_h       = page_h,
                use_gpu      = use_gpu,
            )
            enriched = []
            for bubble, pred in zip(bubbles, predictions):
                enriched.append({
                    **bubble,
                    "speaker_id":   pred["speaker_id"],
                    "speaker_conf": pred["confidence"],
                    "attribution":  "mlp",
                })
            return enriched
        except Exception as e:
            log.warning(f"MLP speaker attribution failed ({e}) → falling back to rule-based")

    # Rule-based fallback
    enriched = []
    for bubble, bbox in zip(bubbles, bubble_bboxes):
        idx = assign_speaker_rule_based(bbox, char_bboxes, page_w, page_h)
        speaker_id = char_bboxes[idx].get("id") if idx is not None else None
        enriched.append({
            **bubble,
            "speaker_id":   speaker_id,
            "speaker_conf": 0.0,
            "attribution":  "rule_based",
        })
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Reading order
# ─────────────────────────────────────────────────────────────────────────────

def sort_reading_order(
    bubbles: list[dict],
    panels: list[dict],
    page_w: int,
    direction: str = "ltr",
) -> list[dict]:
    """Sort bubbles into reading order.

    Algorithm:
      1. Assign each bubble to a panel (by overlap)
      2. Sort panels (top→bottom, then left→right or right→left)
      3. Sort bubbles within each panel by same rule
      4. Number bubbles sequentially
    """
    ROW_TOLERANCE = 0.15  # fraction of page height — bubbles within this band = same row

    def bbox_center(bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def overlap_area(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    def sort_key(bbox, direction, page_w):
        cx, cy = bbox_center(bbox)
        row_band = int(cy / (page_w * ROW_TOLERANCE))
        x_key = cx if direction == "ltr" else (page_w - cx)
        return (row_band, x_key)

    # Assign bubbles to panels
    if panels:
        sorted_panels = sorted(
            panels,
            key=lambda p: sort_key(p["bbox"], direction, page_w)
        )
        panel_bubbles: dict[int, list[dict]] = {i: [] for i in range(len(sorted_panels))}
        unassigned = []

        for bubble in bubbles:
            best_panel, best_overlap = -1, 0
            for i, panel in enumerate(sorted_panels):
                ov = overlap_area(bubble["bbox"], panel["bbox"])
                if ov > best_overlap:
                    best_overlap, best_panel = ov, i
            if best_panel >= 0 and best_overlap > 0:
                panel_bubbles[best_panel].append(bubble)
            else:
                unassigned.append(bubble)

        ordered = []
        for i in range(len(sorted_panels)):
            panel_sorted = sorted(
                panel_bubbles[i],
                key=lambda b: sort_key(b["bbox"], direction, page_w)
            )
            ordered.extend(panel_sorted)
        ordered.extend(sorted(
            unassigned,
            key=lambda b: sort_key(b["bbox"], direction, page_w)
        ))
    else:
        # No panel info → sort whole page
        ordered = sorted(
            bubbles,
            key=lambda b: sort_key(b["bbox"], direction, page_w)
        )

    # Assign reading order numbers
    for i, bubble in enumerate(ordered, 1):
        bubble["order"] = i

    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Full single-page pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_page(
    image_path: Path,
    yolo_model,
    ocr_pipeline,
    face_extractor,
    char_db,
    speaker_model_path: Optional[Path],
    speaker_scaler_path: Optional[Path],
    direction: str = "ltr",
    rule_based: bool = False,
    use_gpu: bool = True,
    yolo_conf: float = 0.25,
) -> dict:
    """Run full pipeline on a single page. Returns structured result dict."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    page_h, page_w = image.shape[:2]
    page_name = image_path.name
    log.info(f"{'─'*55}")
    log.info(f"  Page: {page_name}  ({page_w}×{page_h})")

    # ── 1. YOLO ───────────────────────────────────────────────────────────────
    detections = run_yolo(image, yolo_model, conf=yolo_conf)
    bubbles    = detections["bubbles"]
    characters = detections["characters"]
    panels     = detections["panels"]
    log.info(
        f"  YOLO: {len(bubbles)} bubbles, "
        f"{len(characters)} characters, {len(panels)} panels"
    )

    # ── 2. OCR ────────────────────────────────────────────────────────────────
    if ocr_pipeline is not None and bubbles:
        bubbles = run_ocr_on_bubbles(image, bubbles, ocr_pipeline)
        n_text = sum(1 for b in bubbles if b.get("text"))
        log.info(f"  OCR: {n_text}/{len(bubbles)} bubbles have text")
    else:
        bubbles = [{**b, "text": "", "ocr_conf": 0.0} for b in bubbles]

    # ── 3. Face matching ──────────────────────────────────────────────────────
    if face_extractor is not None and char_db is not None and characters:
        characters = run_face_matching(
            image, characters, face_extractor, char_db, page_name
        )
        n_matched = sum(1 for c in characters if c.get("char_id"))
        log.info(f"  Face: {n_matched}/{len(characters)} characters matched")
    else:
        characters = [{**c, "char_id": None, "face_bbox": None} for c in characters]

    # ── 4. Speaker attribution ────────────────────────────────────────────────
    if bubbles and characters:
        bubbles = run_speaker_attribution(
            bubbles, characters, page_w, page_h, panels,
            speaker_model_path, speaker_scaler_path,
            rule_based=rule_based, use_gpu=use_gpu,
        )
        n_attributed = sum(1 for b in bubbles if b.get("speaker_id"))
        log.info(f"  Speaker: {n_attributed}/{len(bubbles)} bubbles attributed")
    else:
        bubbles = [{**b, "speaker_id": None, "speaker_conf": 0.0} for b in bubbles]

    # ── 5. Reading order ──────────────────────────────────────────────────────
    bubbles = sort_reading_order(bubbles, panels, page_w, direction=direction)

    # ── Assemble result ───────────────────────────────────────────────────────
    result = {
        "image":      str(image_path),
        "width":      page_w,
        "height":     page_h,
        "direction":  direction,
        "characters": [
            {
                "char_id":   c.get("char_id"),
                "bbox":      c["bbox"],
                "face_bbox": c.get("face_bbox"),
                "yolo_conf": c["confidence"],
            }
            for c in characters
        ],
        "bubbles": [
            {
                "order":       b.get("order", 0),
                "text":        b.get("text", ""),
                "bbox":        b["bbox"],
                "speaker_id":  b.get("speaker_id"),
                "speaker_conf": b.get("speaker_conf", 0.0),
                "attribution": b.get("attribution", "none"),
                "ocr_conf":    b.get("ocr_conf", 0.0),
                "yolo_conf":   b["confidence"],
            }
            for b in sorted(bubbles, key=lambda x: x.get("order", 0))
        ],
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline initialisation (load all models once, reuse across pages)
# ─────────────────────────────────────────────────────────────────────────────

def init_pipeline(args) -> dict:
    """Load all models and return a components dict."""
    components = {}

    # YOLO
    log.info("Loading YOLO…")
    from ultralytics import YOLO
    components["yolo"] = YOLO(args.weights)

    # OCR
    if not args.no_ocr:
        log.info("Loading OCR pipeline…")
        from Capstone_project.scripts.ocr_pipeline import OCRPipeline
        components["ocr"] = OCRPipeline(
            lang        = args.lang,
            use_gpu     = not args.no_gpu,
            paddle_only = args.paddle_only,
        )
    else:
        components["ocr"] = None

    # Face extractor + character_db
    if not args.no_face:
        log.info("Loading ArcFace extractor…")
        from Capstone_project.scripts.face_clustering import ArcFaceExtractor, CharacterDB
        components["extractor"] = ArcFaceExtractor(use_gpu=not args.no_gpu)
        db = CharacterDB()
        if args.char_db and Path(args.char_db).exists():
            db.load(args.char_db)
        components["char_db"] = db
    else:
        components["extractor"] = None
        components["char_db"]   = None

    # Speaker attribution paths (lazy-loaded inside run_speaker_attribution)
    components["speaker_model"]  = Path(args.speaker_model)
    components["speaker_scaler"] = Path(args.speaker_scaler)

    return components


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full comic CV pipeline: YOLO → OCR → Face → Speaker → Order",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",   required=True,
                   help="Path to a single image or directory of images")
    p.add_argument("--weights", default=DEFAULT_YOLO_WEIGHTS,
                   help="YOLO model weights")
    p.add_argument("--lang",    default="vi", choices=["vi", "ja"],
                   help="OCR language: vi=Vietnamese, ja=Japanese")
    p.add_argument("--direction", default="ltr", choices=["ltr", "rtl"],
                   help="Reading direction")
    p.add_argument("--save-dir", default="/tmp/comic_pipeline_out",
                   help="Output directory for JSON results")
    p.add_argument("--char-db",  default=None,
                   help="Path to existing character_db.json (for chapter continuity)")
    p.add_argument("--speaker-model",  default=DEFAULT_SPEAKER_MODEL)
    p.add_argument("--speaker-scaler", default=DEFAULT_SPEAKER_SCALER)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    # Flags
    p.add_argument("--no-ocr",      action="store_true", help="Skip OCR")
    p.add_argument("--no-face",     action="store_true", help="Skip face clustering")
    p.add_argument("--rule-based",  action="store_true",
                   help="Use rule-based speaker attribution (no MLP)")
    p.add_argument("--paddle-only", action="store_true",
                   help="Use PaddleOCR only (no VietOCR)")
    p.add_argument("--no-gpu",      action="store_true")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load models once
    components = init_pipeline(args)

    # Collect images
    input_path = Path(args.image)
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    if input_path.is_dir():
        pages = sorted(p for p in input_path.iterdir()
                       if p.suffix.lower() in extensions)
    elif input_path.is_file():
        pages = [input_path]
    else:
        log.error(f"Path not found: {input_path}")
        sys.exit(1)

    log.info(f"Processing {len(pages)} page(s)…")

    all_results = []
    for page_path in pages:
        try:
            result = process_page(
                image_path          = page_path,
                yolo_model          = components["yolo"],
                ocr_pipeline        = components["ocr"],
                face_extractor      = components["extractor"],
                char_db             = components["char_db"],
                speaker_model_path  = components["speaker_model"],
                speaker_scaler_path = components["speaker_scaler"],
                direction           = args.direction,
                rule_based          = args.rule_based,
                use_gpu             = not args.no_gpu,
                yolo_conf           = args.yolo_conf,
            )

            # Save per-page JSON
            out_path = save_dir / f"{page_path.stem}_pipeline.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            log.info(f"  Saved: {out_path}")

            all_results.append(result)

            # Print bubble summary
            print(f"\n  {'─'*55}")
            print(f"  {page_path.name}")
            print(f"  {'─'*55}")
            for b in result["bubbles"]:
                spk = b["speaker_id"] or "?"
                txt = b["text"][:40] + "…" if len(b["text"]) > 40 else b["text"]
                print(f"  [{b['order']:2d}] {spk:>8s}  \"{txt}\"")

        except Exception as e:
            log.error(f"Failed on {page_path.name}: {e}", exc_info=args.verbose)

    # Save character_db if face clustering was used
    if components["char_db"] is not None:
        db_path = save_dir / "character_db.json"
        components["char_db"].save(db_path)
        components["char_db"].summary()

    # Save chapter-level summary
    chapter_out = save_dir / "chapter_summary.json"
    with open(chapter_out, "w", encoding="utf-8") as f:
        json.dump({"pages": len(all_results), "results": all_results}, f,
                  ensure_ascii=False, indent=2)
    log.info(f"\nChapter summary saved: {chapter_out}")


if __name__ == "__main__":
    main()
