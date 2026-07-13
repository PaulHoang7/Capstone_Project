"""
OCR pipeline for comic pages — detect text regions and recognize text.

Architecture (2-layer):
  Layer 1 — Detection:    PaddleOCR  (find text bounding boxes)
  Layer 2 — Recognition:  VietOCR    (read Vietnamese text from crops)
                          fallback → PaddleOCR lang='vi' (if VietOCR unavailable)

For Japanese manga (Manga109 testing), PaddleOCR lang='japan' is used end-to-end.

Output per page:
  [{"text": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.95, ...}, ...]
  Saved as JSON at: <save_dir>/<page_stem>_ocr.json

Usage:
    # Single page (Vietnamese)
    python Capstone_project/scripts/ocr_pipeline.py \\
        --image page.jpg --lang vi

    # Directory of pages
    python Capstone_project/scripts/ocr_pipeline.py \\
        --image /path/to/pages/ --lang vi --save-dir /tmp/ocr_out

    # Manga109 (Japanese)
    python Capstone_project/scripts/ocr_pipeline.py \\
        --image manga_page.jpg --lang ja

    # Force PaddleOCR-only (no VietOCR)
    python Capstone_project/scripts/ocr_pipeline.py \\
        --image page.jpg --lang vi --paddle-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
VIETOCR_CONFIG = "vgg_transformer"   # VietOCR model config name
PADDLE_DET_LANG = "en"               # PaddleOCR detection is language-agnostic;
                                     # 'en' is the lightest model for pure detection
MIN_BOX_AREA = 100                   # pixels² — discard tiny noise boxes
MIN_CONF_THRESHOLD = 0.3             # discard very low-confidence recognitions


# ─────────────────────────────────────────────────────────────────────────────
# Helper: crop a polygon/quad region from an image
# ─────────────────────────────────────────────────────────────────────────────

def _quad_to_xyxy(points: list[list[float]]) -> tuple[int, int, int, int]:
    """Convert a 4-point polygon to axis-aligned (x1, y1, x2, y2)."""
    pts = np.array(points, dtype=np.float32)
    x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
    return x1, y1, x2, y2


def _crop_region(image: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """Crop and deskew a text region defined by a 4-point polygon.

    If the polygon is already axis-aligned, this is just a rectangle crop.
    For skewed text (common in speech bubbles), we apply a perspective transform
    so the crop fed to the recognizer is always upright.
    """
    pts = np.array(points, dtype=np.float32)
    x1, y1, x2, y2 = _quad_to_xyxy(points)

    # Clamp to image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # Check if polygon is sufficiently non-rectangular to warrant warping
    rect_area = (x2 - x1) * (y2 - y1)
    poly_area = cv2.contourArea(pts)
    if rect_area > 0 and (poly_area / rect_area) > 0.85:
        # Nearly axis-aligned — plain crop is fine
        return image[y1:y2, x1:x2]

    # Skewed — apply perspective warp
    dst_w = x2 - x1
    dst_h = y2 - y1
    dst_pts = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    return cv2.warpPerspective(image, M, (dst_w, dst_h))


# ─────────────────────────────────────────────────────────────────────────────
# Detector — PaddleOCR (detection only)
# ─────────────────────────────────────────────────────────────────────────────

class PaddleDetector:
    """Wraps PaddleOCR in detection-only mode to find text bounding boxes."""

    def __init__(self, use_gpu: bool = True) -> None:
        from paddleocr import PaddleOCR

        log.info("Loading PaddleOCR detector (detection-only)…")
        self._ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            rec=False,
            use_gpu=use_gpu,
            show_log=False,
        )
        # Keep a full (det+rec) instance as backup — the rec=False path
        # sometimes misses text in low-resolution or comic-style images.
        self._ocr_full = PaddleOCR(
            lang="vi",
            use_angle_cls=True,
            use_gpu=use_gpu,
            show_log=False,
        )
        log.info("PaddleOCR detector ready.")

    def detect(self, image: np.ndarray) -> list[list[list[float]]]:
        """Return list of 4-point polygons [[x,y], ...] for each text box."""
        boxes = self._detect_core(image, rec=False)
        if boxes:
            return boxes
        # Fallback: run full pipeline and extract only the box coordinates.
        # This handles cases where rec=False mode misses detections (common
        # with low-res comic scans and unusual fonts).
        return self._detect_via_full(image)

    def _detect_core(self, image: np.ndarray, rec: bool = False) -> list:
        try:
            result = self._ocr.ocr(image, rec=rec)
        except ValueError:
            # PaddleOCR bug: 'if not dt_boxes' fails on numpy array with >1 element
            # Happens with some PaddleOCR versions + numpy 2.x combinations
            return []

        if result is None or result[0] is None:
            return []

        boxes = result[0]
        valid = []
        for box in boxes:
            pts = box
            x1, y1, x2, y2 = _quad_to_xyxy(pts)
            if (x2 - x1) * (y2 - y1) >= MIN_BOX_AREA:
                valid.append(pts)
        return valid

    def _detect_via_full(self, image: np.ndarray) -> list:
        """Extract text box polygons from the full (det+rec) pipeline."""
        try:
            result = self._ocr_full.ocr(image)
        except (ValueError, Exception):
            return []
        if result is None or result[0] is None:
            return []
        valid = []
        for item in result[0]:
            pts = item[0]  # full pipeline format: [pts, (text, conf)]
            x1, y1, x2, y2 = _quad_to_xyxy(pts)
            if (x2 - x1) * (y2 - y1) >= MIN_BOX_AREA:
                valid.append(pts)
        return valid


# ─────────────────────────────────────────────────────────────────────────────
# Recognizer — VietOCR (Vietnamese, primary)
# ─────────────────────────────────────────────────────────────────────────────

class VietOCRRecognizer:
    """VietOCR (ProtonX) transformer-based Vietnamese text recognizer.

    See: https://github.com/pbcquoc/vietocr
    Config 'vgg_transformer' is the recommended balance of speed/accuracy.
    """

    def __init__(self, config_name: str = VIETOCR_CONFIG, device: str = "cuda") -> None:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        log.info(f"Loading VietOCR ({config_name}) on {device}…")
        config = Cfg.load_config_from_name(config_name)
        config["device"] = device
        config["predictor"]["beamsearch"] = False  # greedy for speed
        self._predictor = Predictor(config)
        log.info("VietOCR ready.")

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """Recognize text in a crop. Returns (text, confidence)."""
        from PIL import Image as PILImage

        if crop.size == 0:
            return "", 0.0

        # VietOCR expects a PIL RGB image
        pil_img = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        text, prob = self._predictor.predict(pil_img, return_prob=True)
        return text.strip(), float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# Recognizer — PaddleOCR (fallback or Japanese)
# ─────────────────────────────────────────────────────────────────────────────

class PaddleRecognizer:
    """PaddleOCR used as a recognizer.

    For Vietnamese: lang='vi'  (fallback when VietOCR unavailable)
    For Japanese:   lang='japan'
    """

    def __init__(self, lang: str = "vi", use_gpu: bool = True) -> None:
        from paddleocr import PaddleOCR

        log.info(f"Loading PaddleOCR recognizer (lang={lang})…")
        self._ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            det=False,
            use_gpu=use_gpu,
            show_log=False,
        )
        self._lang = lang
        log.info(f"PaddleOCR recognizer (lang={lang}) ready.")

    def recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """Recognize text in a crop. Returns (text, confidence)."""
        if crop.size == 0:
            return "", 0.0

        result = self._ocr.ocr(crop, det=False)
        if not result or not result[0]:
            return "", 0.0

        # result[0] = [(text, score), ...]  — join multi-line crops
        texts, scores = [], []
        for item in result[0]:
            if item and len(item) == 2:
                texts.append(item[0])
                scores.append(item[1])

        if not texts:
            return "", 0.0

        combined_text = " ".join(texts).strip()
        avg_conf = float(np.mean(scores))
        return combined_text, avg_conf


# ─────────────────────────────────────────────────────────────────────────────
# Full PaddleOCR pipeline (detection + recognition in one call)
# Used for Japanese, or when --paddle-only flag is set.
# ─────────────────────────────────────────────────────────────────────────────

class PaddleFullPipeline:
    """PaddleOCR end-to-end (det + rec) for Japanese or paddle-only mode."""

    def __init__(self, lang: str = "japan", use_gpu: bool = True) -> None:
        from paddleocr import PaddleOCR

        log.info(f"Loading PaddleOCR full pipeline (lang={lang})…")
        self._ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True,
            use_gpu=use_gpu,
            show_log=False,
        )
        log.info(f"PaddleOCR full pipeline (lang={lang}) ready.")

    def run(self, image: np.ndarray) -> list[dict]:
        """Run det+rec on image. Returns list of OCR result dicts."""
        result = self._ocr.ocr(image)
        if not result or result[0] is None:
            return []

        records = []
        for item in result[0]:
            pts, (text, conf) = item
            x1, y1, x2, y2 = _quad_to_xyxy(pts)
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue
            if conf < MIN_CONF_THRESHOLD:
                continue
            records.append({
                "text": text.strip(),
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "quad": [[float(p[0]), float(p[1])] for p in pts],
                "recognizer": "paddleocr",
            })
        return records


# ─────────────────────────────────────────────────────────────────────────────
# Main OCR pipeline
# ─────────────────────────────────────────────────────────────────────────────

class OCRPipeline:
    """Two-layer OCR pipeline.

    Detection: PaddleOCR  — language-agnostic DB text detector
    Recognition:
      - Vietnamese: VietOCR (ProtonX) → best accuracy for tone marks
      - Vietnamese fallback: PaddleOCR lang='vi'
      - Japanese: PaddleOCR lang='japan' (full pipeline)
    """

    def __init__(
        self,
        lang: str = "vi",
        use_gpu: bool = True,
        paddle_only: bool = False,
        vietocr_config: str = VIETOCR_CONFIG,
    ) -> None:
        self.lang = lang
        self.paddle_only = paddle_only

        if lang == "ja":
            # Japanese — use PaddleOCR end-to-end (det + rec)
            log.info("Language: Japanese (Manga109 mode) — PaddleOCR full pipeline")
            self._full_paddle = PaddleFullPipeline(lang="japan", use_gpu=use_gpu)
            self._detector = None
            self._recognizer = None

        elif paddle_only:
            # Force PaddleOCR for Vietnamese (no VietOCR dependency)
            log.info("Language: Vietnamese — PaddleOCR full pipeline (paddle-only mode)")
            self._full_paddle = PaddleFullPipeline(lang="vi", use_gpu=use_gpu)
            self._detector = None
            self._recognizer = None

        else:
            # Vietnamese — 2-layer: PaddleOCR det + VietOCR rec
            self._full_paddle = None
            self._detector = PaddleDetector(use_gpu=use_gpu)
            self._recognizer = self._load_vietocr(vietocr_config, use_gpu)

    def _load_vietocr(
        self, config_name: str, use_gpu: bool
    ) -> VietOCRRecognizer | PaddleRecognizer:
        """Try loading VietOCR; fall back to PaddleOCR lang='vi' if not installed."""
        try:
            device = "cuda" if use_gpu else "cpu"
            return VietOCRRecognizer(config_name=config_name, device=device)
        except ImportError:
            log.warning(
                "VietOCR not installed (pip install vietocr). "
                "Falling back to PaddleOCR lang='vi'."
            )
            return PaddleRecognizer(lang="vi", use_gpu=use_gpu)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_image(self, image_path: str | Path) -> list[dict]:
        """Run OCR on a single comic page image.

        Returns:
            List of dicts, each with:
              - text        (str)            recognized text
              - bbox        ([x1,y1,x2,y2]) axis-aligned bounding box in pixels
              - confidence  (float)          recognition confidence 0–1
              - quad        ([[x,y], ...])   original 4-point polygon
              - recognizer  (str)            which recognizer produced this result
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        log.info(f"Processing: {image_path.name}  ({image.shape[1]}×{image.shape[0]}px)")

        if self._full_paddle is not None:
            # Japanese or paddle-only: single-call end-to-end
            results = self._full_paddle.run(image)
        else:
            results = self._two_layer_ocr(image)

        # Sort results top-to-bottom, then left-to-right (LTR reading order)
        results.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        log.info(f"  → {len(results)} text regions found")
        return results

    def _two_layer_ocr(self, image: np.ndarray) -> list[dict]:
        """Detection (PaddleOCR) + Recognition (VietOCR or PaddleOCR) pipeline."""
        # Step 1: Detect text regions
        boxes = self._detector.detect(image)
        if not boxes:
            log.debug("  No text boxes detected.")
            return []

        log.debug(f"  Detected {len(boxes)} text boxes → recognizing…")

        results = []
        for pts in boxes:
            # Step 2: Crop the text region
            crop = _crop_region(image, pts)

            # Skip degenerate crops
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                continue

            # Step 3: Recognize text
            text, conf = self._recognizer.recognize(crop)

            if not text or conf < MIN_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = _quad_to_xyxy(pts)
            rec_name = (
                "vietocr"
                if isinstance(self._recognizer, VietOCRRecognizer)
                else "paddleocr_vi"
            )
            results.append({
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "quad": [[float(p[0]), float(p[1])] for p in pts],
                "recognizer": rec_name,
            })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing + JSON saving
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(
    pipeline: OCRPipeline,
    image_dir: Path,
    save_dir: Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> None:
    """Run OCR on all images in a directory, save one JSON per page."""
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )
    if not images:
        log.warning(f"No images found in {image_dir}")
        return

    log.info(f"Found {len(images)} images in {image_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        try:
            results = pipeline.process_image(img_path)
            _save_json(results, img_path, save_dir)
        except Exception as e:
            log.error(f"Failed on {img_path.name}: {e}")


def _save_json(
    results: list[dict],
    image_path: Path,
    save_dir: Path,
) -> Path:
    """Save OCR results as JSON next to the image (or in save_dir)."""
    out_path = save_dir / f"{image_path.stem}_ocr.json"
    payload = {
        "image": str(image_path),
        "width": None,   # filled below if readable
        "height": None,
        "num_regions": len(results),
        "regions": results,
    }

    # Optionally record image dimensions for downstream use
    img = cv2.imread(str(image_path))
    if img is not None:
        payload["height"], payload["width"] = img.shape[:2]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"  Saved JSON: {out_path}  ({len(results)} regions)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for use as a library (called by infer_yolo_comic, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def ocr_bubble_crop(
    crop: np.ndarray,
    recognizer: "VietOCRRecognizer | PaddleRecognizer",
    detector: Optional[PaddleDetector] = None,
) -> list[dict]:
    """Run OCR on a pre-cropped speech bubble image (output from YOLO).

    If a detector is provided, first detect text lines within the bubble crop,
    then recognize each line. Otherwise, recognize the entire crop as one region.

    This is the integration point called from the full comic pipeline after YOLO
    detects bubble bounding boxes.

    Args:
        crop:       BGR image crop of a single speech bubble.
        recognizer: VietOCRRecognizer or PaddleRecognizer instance.
        detector:   Optional PaddleDetector for intra-bubble line detection.

    Returns:
        List of (text, bbox_within_crop, confidence) dicts.
    """
    if crop is None or crop.size == 0:
        return []

    # Upscale tiny crops — PaddleOCR detection and VietOCR both struggle
    # below ~200px width.  Bicubic 2× is cheap and significantly improves
    # accuracy on low-res comic scans.
    MIN_OCR_DIM = 200
    h_orig, w_orig = crop.shape[:2]
    scale = 1
    if min(h_orig, w_orig) < MIN_OCR_DIM:
        scale = max(2, MIN_OCR_DIM // min(h_orig, w_orig) + 1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    if detector is not None:
        boxes = detector.detect(crop)
        results = []
        for pts in boxes:
            line_crop = _crop_region(crop, pts)
            if line_crop.shape[0] < 4 or line_crop.shape[1] < 4:
                continue
            text, conf = recognizer.recognize(line_crop)
            if text and conf >= MIN_CONF_THRESHOLD:
                x1, y1, x2, y2 = _quad_to_xyxy(pts)
                # Map bbox back to original scale
                results.append({
                    "text": text,
                    "bbox": [x1 // scale, y1 // scale,
                             x2 // scale, y2 // scale],
                    "confidence": round(conf, 4),
                })
        # Fallback: if detector found no text boxes, try direct recognition
        # on the whole crop (common with small bubble crops from low-res pages)
        if not results:
            text, conf = recognizer.recognize(crop)
            if text and conf >= MIN_CONF_THRESHOLD:
                results.append({
                    "text": text,
                    "bbox": [0, 0, w_orig, h_orig],
                    "confidence": round(conf, 4),
                })
        return results
    else:
        # Treat whole crop as one text region
        text, conf = recognizer.recognize(crop)
        if text and conf >= MIN_CONF_THRESHOLD:
            return [{"text": text, "bbox": [0, 0, w_orig, h_orig],
                     "confidence": round(conf, 4)}]
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-layer OCR pipeline: PaddleOCR detection + VietOCR/PaddleOCR recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image", required=True,
        help="Path to a single image or directory of images",
    )
    p.add_argument(
        "--lang", default="vi", choices=["vi", "ja"],
        help="Language: 'vi' for Vietnamese, 'ja' for Japanese (Manga109)",
    )
    p.add_argument(
        "--save-dir", default="/tmp/ocr_output",
        help="Directory to save JSON results",
    )
    p.add_argument(
        "--paddle-only", action="store_true",
        help="Use PaddleOCR for both detection and recognition (no VietOCR dependency)",
    )
    p.add_argument(
        "--vietocr-config", default=VIETOCR_CONFIG,
        help="VietOCR model config name",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Run on CPU (slower but no CUDA required)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    use_gpu = not args.no_gpu
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = OCRPipeline(
        lang=args.lang,
        use_gpu=use_gpu,
        paddle_only=args.paddle_only,
        vietocr_config=args.vietocr_config,
    )

    input_path = Path(args.image)

    if input_path.is_dir():
        process_directory(pipeline, input_path, save_dir)
    elif input_path.is_file():
        results = pipeline.process_image(input_path)
        _save_json(results, input_path, save_dir)

        # Print summary to stdout
        print(f"\n{'─'*60}")
        print(f"  {input_path.name}: {len(results)} text regions")
        print(f"{'─'*60}")
        for i, r in enumerate(results, 1):
            x1, y1, x2, y2 = r["bbox"]
            print(
                f"  [{i:2d}] conf={r['confidence']:.2f}  "
                f"[{x1},{y1},{x2},{y2}]  \"{r['text']}\""
            )
    else:
        log.error(f"Path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
