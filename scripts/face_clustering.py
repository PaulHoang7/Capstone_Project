"""
Face clustering pipeline for comic characters — ArcFace embeddings + DBSCAN.

Architecture:
  1. Anime face detector  → find faces within YOLO character crops (or full page)
  2. ArcFace (pre-trained) → extract 512-dim embedding per face
  3. DBSCAN clustering    → group embeddings → each cluster = 1 unique character
  4. character_db         → persistent dict saved as JSON across a chapter

character_db structure:
  {
    "char_0": {
      "char_id":        "char_0",
      "label":          "char_0",          # user-assigned name later
      "avg_embedding":  [512 floats],      # centroid for fast matching
      "all_embeddings": [[512], ...],      # all appearances (for re-clustering)
      "appearances":    [                  # which page + bbox
        {"page": "page_001.jpg", "bbox": [x1,y1,x2,y2]},
        ...
      ],
      "voice_embedding": null              # filled later by Voice Cloning module
    },
    ...
  }

Usage:
    # Cluster a full chapter directory (YOLO not available → detect on full pages)
    python Capstone_project/scripts/face_clustering.py \\
        --pages /path/to/chapter/ --save-dir /tmp/char_db

    # With YOLO character crops (preferred — cleaner faces)
    python Capstone_project/scripts/face_clustering.py \\
        --pages /path/to/chapter/ \\
        --yolo-json /tmp/yolo_out/  \\
        --save-dir /tmp/char_db

    # Match new faces against existing character_db (e.g. chapter 2 → chapter 1 DB)
    python Capstone_project/scripts/face_clustering.py \\
        --pages /path/to/chapter2/ \\
        --load-db /tmp/char_db/character_db.json \\
        --save-dir /tmp/char_db
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ARCFACE_MODEL_NAME = "buffalo_l"      # insightface model pack (ArcFace backbone)
MATCH_THRESHOLD    = 0.55             # cosine similarity threshold for known-character match
                                      # tuned for anime: real-face ArcFace is ~0.7, anime lower
DBSCAN_EPS         = 0.55             # DBSCAN neighbourhood radius (cosine distance = 1 - sim)
DBSCAN_MIN_SAMPLES = 2               # minimum faces to form a cluster
MIN_FACE_SIZE      = 24               # pixels — discard tiny face detections


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace extractor (insightface)
# ─────────────────────────────────────────────────────────────────────────────

class ArcFaceExtractor:
    """Wraps insightface FaceAnalysis for detection + ArcFace embedding extraction.

    insightface's 'buffalo_l' pack bundles:
      - RetinaFace detector   (works on real faces; acceptable on manga)
      - ArcFace-R100 backbone (512-dim embedding)

    For better manga/anime face detection, pass use_anime_detector=True to
    additionally run the lightweight anime-face-detector (hysts/anime-face-detector)
    and merge results. Falls back gracefully if the package is not installed.
    """

    def __init__(
        self,
        model_name: str = ARCFACE_MODEL_NAME,
        use_gpu: bool = True,
        use_anime_detector: bool = True,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        import insightface
        from insightface.app import FaceAnalysis

        ctx_id = 0 if use_gpu else -1
        log.info(f"Loading insightface ({model_name}) on {'GPU' if use_gpu else 'CPU'}…")
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        log.info("ArcFace extractor ready.")

        # Optional: anime-specific face detector
        self._anime_det = None
        if use_anime_detector:
            self._anime_det = self._load_anime_detector()

    def _load_anime_detector(self):
        """Load hysts/anime-face-detector if available."""
        try:
            import anime_face_detector
            detector = anime_face_detector.create_detector("yolov3")
            log.info("Anime face detector loaded.")
            return detector
        except ImportError:
            log.warning(
                "anime-face-detector not installed "
                "(pip install anime-face-detector). "
                "Using RetinaFace only — may miss some manga faces."
            )
            return None

    def extract(self, image: np.ndarray) -> list[dict]:
        """Detect faces and extract ArcFace embeddings from a BGR image.

        Returns list of:
          {"bbox": [x1,y1,x2,y2], "embedding": np.ndarray(512,), "det_score": float}
        """
        results = []
        seen_boxes: list[list[int]] = []  # to deduplicate across detectors

        # ── insightface RetinaFace + ArcFace ──────────────────────────────────
        faces = self._app.get(image)
        for face in faces:
            if face.embedding is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                continue
            emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
            results.append({
                "bbox": [x1, y1, x2, y2],
                "embedding": emb.astype(np.float32),
                "det_score": float(face.det_score),
            })
            seen_boxes.append([x1, y1, x2, y2])

        # ── Anime face detector (supplement — fills gaps RetinaFace misses) ──
        if self._anime_det is not None:
            anime_faces = self._anime_det(image)
            for aface in anime_faces:
                # anime-face-detector returns (x1, y1, x2, y2, score)
                ax1, ay1, ax2, ay2 = [int(v) for v in aface[:4]]
                if (ax2 - ax1) < MIN_FACE_SIZE or (ay2 - ay1) < MIN_FACE_SIZE:
                    continue

                # Skip if heavily overlapping with an already-detected box
                if _iou_overlap([ax1, ay1, ax2, ay2], seen_boxes) > 0.4:
                    continue

                crop = image[ay1:ay2, ax1:ax2]
                if crop.size == 0:
                    continue

                # Feed crop back through ArcFace recognition module only
                emb = self._embed_crop(crop)
                if emb is None:
                    continue

                results.append({
                    "bbox": [ax1, ay1, ax2, ay2],
                    "embedding": emb,
                    "det_score": float(aface[4]) if len(aface) > 4 else 0.9,
                })
                seen_boxes.append([ax1, ay1, ax2, ay2])

        return results

    def _embed_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Run ArcFace recognition on a pre-cropped face image."""
        # Resize to standard ArcFace input
        resized = cv2.resize(crop, (112, 112))
        faces = self._app.get(resized)
        if not faces or faces[0].embedding is None:
            return None
        emb = faces[0].embedding
        return (emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32)


def _iou_overlap(box: list[int], boxes: list[list[int]]) -> float:
    """Return max IoU between box and any box in boxes."""
    if not boxes:
        return 0.0
    x1, y1, x2, y2 = box
    best = 0.0
    for b in boxes:
        ix1 = max(x1, b[0]); iy1 = max(y1, b[1])
        ix2 = min(x2, b[2]); iy2 = min(y2, b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2-x1)*(y2-y1) + (b[2]-b[0])*(b[3]-b[1]) - inter
        best = max(best, inter / union if union > 0 else 0.0)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "dbscan",
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    n_clusters: Optional[int] = None,
) -> np.ndarray:
    """Cluster face embeddings into character groups.

    Args:
        embeddings:  (N, 512) float32 array of L2-normalised ArcFace embeddings.
        method:      'dbscan' or 'agglomerative'.
        eps:         DBSCAN cosine-distance threshold (1 - cosine_similarity).
        min_samples: DBSCAN minimum cluster size.
        n_clusters:  Required for 'agglomerative'; ignored for 'dbscan'.

    Returns:
        labels: (N,) int array. -1 = noise/unassigned (DBSCAN only).
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering

    if len(embeddings) == 0:
        return np.array([], dtype=int)

    if method == "dbscan":
        # cosine distance = 1 - cosine_similarity; embeddings already L2-normalised
        # so dot product = cosine similarity → distance = 1 - dot
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        labels = clusterer.fit_predict(embeddings)

    elif method == "agglomerative":
        if n_clusters is None:
            raise ValueError("n_clusters required for agglomerative clustering")
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(embeddings)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    n_clusters_found = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    log.info(
        f"Clustering ({method}): {n_clusters_found} characters found, "
        f"{n_noise} noise faces"
    )
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# character_db
# ─────────────────────────────────────────────────────────────────────────────

class CharacterDB:
    """Persistent character database for a chapter.

    Stores per-character:
      - avg_embedding  : centroid of all known face embeddings (for fast matching)
      - all_embeddings : full list (for re-clustering or fine-tuning later)
      - appearances    : list of {page, bbox} dicts
      - voice_embedding: placeholder, filled by Voice Cloning module
      - label          : user-assigned name (default: char_0, char_1, ...)
    """

    def __init__(self) -> None:
        self._db: dict[str, dict] = {}
        self._next_id: int = 0

    # ── Build from clustering results ─────────────────────────────────────────

    def build_from_clusters(
        self,
        face_records: list[dict],
        labels: np.ndarray,
    ) -> None:
        """Populate DB from a clustering result.

        Args:
            face_records: list of {"page", "bbox", "embedding"} dicts (one per face).
            labels:       cluster label per face (parallel to face_records).
        """
        self._db.clear()
        self._next_id = 0

        # Group records by cluster label
        groups: dict[int, list[dict]] = {}
        for record, label in zip(face_records, labels):
            if label == -1:
                continue  # noise — skip
            groups.setdefault(label, []).append(record)

        for label in sorted(groups):
            records = groups[label]
            char_id = f"char_{self._next_id}"
            self._next_id += 1

            embeddings = np.stack([r["embedding"] for r in records])
            avg_emb = embeddings.mean(axis=0)
            avg_emb /= np.linalg.norm(avg_emb) + 1e-8  # re-normalise centroid

            self._db[char_id] = {
                "char_id": char_id,
                "label": char_id,                         # user renames later
                "avg_embedding": avg_emb.tolist(),
                "all_embeddings": embeddings.tolist(),
                "appearances": [
                    {"page": r["page"], "bbox": r["bbox"]}
                    for r in records
                ],
                "voice_embedding": None,                  # filled by Voice Cloning
            }

        log.info(f"character_db built: {len(self._db)} characters")

    # ── Match a new face against existing characters ──────────────────────────

    def match(
        self,
        embedding: np.ndarray,
        threshold: float = MATCH_THRESHOLD,
    ) -> tuple[Optional[str], float]:
        """Find the best matching character for a new face embedding.

        Returns:
            (char_id, similarity) if match found, else (None, best_score).
        """
        if not self._db:
            return None, 0.0

        best_id, best_score = None, -1.0
        for char_id, data in self._db.items():
            avg = np.array(data["avg_embedding"], dtype=np.float32)
            score = float(np.dot(embedding, avg))  # cosine sim (both L2-normalised)
            if score > best_score:
                best_id, best_score = char_id, score

        if best_score >= threshold:
            return best_id, best_score
        return None, best_score

    def add_appearance(
        self,
        char_id: str,
        embedding: np.ndarray,
        page: str,
        bbox: list[int],
    ) -> None:
        """Add a new face appearance to an existing character entry and update centroid."""
        entry = self._db[char_id]
        entry["appearances"].append({"page": page, "bbox": bbox})
        entry["all_embeddings"].append(embedding.tolist())

        # Update running average centroid
        all_embs = np.array(entry["all_embeddings"], dtype=np.float32)
        avg = all_embs.mean(axis=0)
        avg /= np.linalg.norm(avg) + 1e-8
        entry["avg_embedding"] = avg.tolist()

    def add_new_character(
        self,
        embedding: np.ndarray,
        page: str,
        bbox: list[int],
    ) -> str:
        """Create a new character entry. Returns the new char_id."""
        char_id = f"char_{self._next_id}"
        self._next_id += 1

        self._db[char_id] = {
            "char_id": char_id,
            "label": char_id,
            "avg_embedding": embedding.tolist(),
            "all_embeddings": [embedding.tolist()],
            "appearances": [{"page": page, "bbox": bbox}],
            "voice_embedding": None,
        }
        log.info(f"New character created: {char_id} (first seen on {page})")
        return char_id

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save DB to JSON. Embeddings are stored as lists of floats."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise — embeddings are already lists from .tolist() calls above
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"next_id": self._next_id, "characters": self._db},
                f,
                ensure_ascii=False,
                indent=2,
            )
        log.info(f"character_db saved: {path}  ({len(self._db)} characters)")

    def load(self, path: str | Path) -> None:
        """Load DB from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"character_db not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._next_id = data.get("next_id", 0)
        self._db = data.get("characters", {})
        log.info(f"character_db loaded: {path}  ({len(self._db)} characters)")

    # ── Accessors ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._db)

    def summary(self) -> None:
        """Print a human-readable summary of all characters."""
        print(f"\n{'─'*60}")
        print(f"  character_db: {len(self._db)} characters")
        print(f"{'─'*60}")
        for char_id, data in self._db.items():
            pages = sorted({a["page"] for a in data["appearances"]})
            n = len(data["appearances"])
            print(
                f"  {data['label']:>12s}  |  {n:3d} appearances  |  "
                f"pages: {', '.join(pages)}"
            )
        print(f"{'─'*60}\n")

    @property
    def characters(self) -> dict:
        return self._db


# ─────────────────────────────────────────────────────────────────────────────
# Chapter-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_chapter(
    extractor: ArcFaceExtractor,
    page_paths: list[Path],
    yolo_jsons: Optional[dict[str, Path]],  # page_stem → YOLO JSON path
    existing_db: Optional[CharacterDB] = None,
    cluster_method: str = "dbscan",
) -> CharacterDB:
    """Run the full face clustering pipeline for a chapter.

    Modes:
      - existing_db=None  → cluster all faces from scratch (chapter 1)
      - existing_db given → match faces against existing DB, add new characters
                            (chapter 2+ or incremental update)

    Args:
        extractor:   ArcFaceExtractor instance.
        page_paths:  Ordered list of page images.
        yolo_jsons:  Optional mapping page_stem → YOLO output JSON.
                     If provided, only process character-class bboxes.
                     If None, run face detection on the full page.
        existing_db: CharacterDB from a previous chapter (or None).
        cluster_method: 'dbscan' or 'agglomerative'.

    Returns:
        Populated (or updated) CharacterDB.
    """
    all_face_records: list[dict] = []  # {page, bbox, embedding}

    for page_path in page_paths:
        image = cv2.imread(str(page_path))
        if image is None:
            log.warning(f"Could not read: {page_path.name}")
            continue

        page_name = page_path.name
        log.info(f"Processing page: {page_name}")

        # ── Determine regions to search for faces ─────────────────────────────
        regions: list[tuple[np.ndarray, list[int]]] = []  # (crop, [x_off, y_off, ...])

        if yolo_jsons and page_path.stem in yolo_jsons:
            # Use YOLO character bboxes — cleaner, faster
            yolo_data = _load_yolo_json(yolo_jsons[page_path.stem])
            for det in yolo_data:
                if det.get("class_name") != "character":
                    continue
                x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    regions.append((crop, [x1, y1, x2, y2]))
        else:
            # No YOLO → search whole page
            regions = [(image, [0, 0, image.shape[1], image.shape[0]])]

        # ── Extract faces from each region ────────────────────────────────────
        for region_img, region_bbox in regions:
            faces = extractor.extract(region_img)
            rx1, ry1 = region_bbox[0], region_bbox[1]

            for face in faces:
                # Translate bbox from crop coordinates to page coordinates
                fx1, fy1, fx2, fy2 = face["bbox"]
                abs_bbox = [rx1 + fx1, ry1 + fy1, rx1 + fx2, ry1 + fy2]

                all_face_records.append({
                    "page": page_name,
                    "bbox": abs_bbox,
                    "embedding": face["embedding"],
                })

        log.info(f"  {page_name}: {len(all_face_records)} total faces so far")

    if not all_face_records:
        log.warning("No faces found across chapter.")
        return existing_db or CharacterDB()

    embeddings = np.stack([r["embedding"] for r in all_face_records])

    # ── Mode 1: fresh clustering (chapter 1 or standalone) ────────────────────
    if existing_db is None:
        labels = cluster_embeddings(embeddings, method=cluster_method)
        db = CharacterDB()
        db.build_from_clusters(all_face_records, labels)
        return db

    # ── Mode 2: match against existing DB (chapter 2+) ────────────────────────
    db = existing_db
    for record in all_face_records:
        emb = record["embedding"]
        char_id, score = db.match(emb)

        if char_id is not None:
            db.add_appearance(char_id, emb, record["page"], record["bbox"])
            log.debug(
                f"  Matched {record['page']} bbox{record['bbox']} "
                f"→ {char_id} (sim={score:.3f})"
            )
        else:
            new_id = db.add_new_character(emb, record["page"], record["bbox"])
            log.info(
                f"  New character {new_id} on {record['page']} "
                f"(best sim was {score:.3f})"
            )

    return db


def _load_yolo_json(path: Path) -> list[dict]:
    """Load YOLO detection JSON (format from infer_yolo_comic.py)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("detections", [])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ArcFace face clustering → character_db for a comic chapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pages", required=True,
                   help="Directory of page images (or glob pattern)")
    p.add_argument("--save-dir", default="/tmp/char_db",
                   help="Where to save character_db.json and debug output")
    p.add_argument("--yolo-json", default=None,
                   help="Directory containing YOLO output JSONs (optional)")
    p.add_argument("--load-db", default=None,
                   help="Load existing character_db.json and match against it")
    p.add_argument("--cluster-method", default="dbscan",
                   choices=["dbscan", "agglomerative"],
                   help="Clustering algorithm")
    p.add_argument("--no-anime-det", action="store_true",
                   help="Disable anime face detector (use RetinaFace only)")
    p.add_argument("--no-gpu", action="store_true",
                   help="Run on CPU")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    use_gpu = not args.no_gpu
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect page images
    pages_dir = Path(args.pages)
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    page_paths = sorted(p for p in pages_dir.iterdir()
                        if p.suffix.lower() in extensions)
    if not page_paths:
        log.error(f"No images found in {pages_dir}")
        sys.exit(1)
    log.info(f"Found {len(page_paths)} pages")

    # Build page_stem → YOLO JSON mapping (if provided)
    yolo_jsons: Optional[dict[str, Path]] = None
    if args.yolo_json:
        yolo_dir = Path(args.yolo_json)
        yolo_jsons = {
            p.stem.replace("_detections", ""): p
            for p in yolo_dir.glob("*_detections.json")
        }
        log.info(f"YOLO JSONs found: {len(yolo_jsons)}")

    # Load existing DB if continuing from previous chapter
    existing_db: Optional[CharacterDB] = None
    if args.load_db:
        existing_db = CharacterDB()
        existing_db.load(args.load_db)

    # Initialise extractor
    extractor = ArcFaceExtractor(
        use_gpu=use_gpu,
        use_anime_detector=not args.no_anime_det,
    )

    # Run pipeline
    db = process_chapter(
        extractor=extractor,
        page_paths=page_paths,
        yolo_jsons=yolo_jsons,
        existing_db=existing_db,
        cluster_method=args.cluster_method,
    )

    # Save + print summary
    db.save(save_dir / "character_db.json")
    db.summary()


if __name__ == "__main__":
    main()
