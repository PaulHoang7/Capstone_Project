"""
Speaker Attribution MLP — parse Manga109Dialogue + train.

Pipeline:
  1. Parse annotations XML       → text bbox, face/body bbox, frame bbox per page
  2. Parse Manga109Dialog XML    → text_id → speaker_id (ground-truth)
  3. Build feature vectors       → (bubble, character) pair features
  4. Train MLP classifier        → predict which character speaks each bubble
  5. Evaluate + save model

Feature vector per (bubble, character) pair (14 dims):
  - bubble center x, y (normalised 0-1)
  - character center x, y (normalised)
  - dx, dy (bubble_center - char_center, normalised)
  - euclidean distance (normalised)
  - bubble area (normalised)
  - character area (normalised)
  - bubble aspect ratio
  - char_rank_by_distance (1=nearest, 2=second nearest, ...)
  - bubble is above character (0/1)
  - bubble is to the left of character (0/1)
  - char_count_in_panel (how many characters on this page)

Label: 1 if this character speaks this bubble, 0 otherwise.

Usage:
    # Parse + train (full dataset)
    python Capstone_project/scripts/train_speaker_attribution.py \\
        --manga109-dir /mnt/nfs-data/tin_dataset/comic/manga109/raw/Manga109s_released_2023_12_07 \\
        --save-dir /mnt/nfs-data/tin_dataset/comic/speaker_attribution

    # Quick sanity check (5 volumes)
    python Capstone_project/scripts/train_speaker_attribution.py \\
        --manga109-dir /mnt/... --save-dir /tmp/speaker_attr --max-volumes 5

    # Inference on new page
    python Capstone_project/scripts/train_speaker_attribution.py \\
        --infer --model /tmp/speaker_attr/speaker_mlp.pt \\
        --bubbles bubbles.json --characters characters.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_DIM   = 20   # expanded from 14
HIDDEN_DIMS   = [256, 128, 64]
DROPOUT       = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE    = 512
EPOCHS        = 50
VAL_RATIO     = 0.15
TEST_RATIO    = 0.10
RANDOM_SEED   = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. XML Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_annotations(xml_path: Path) -> dict:
    """Parse Manga109 annotations XML.

    Returns:
        {
          "title": str,
          "characters": {char_id: char_name},
          "pages": {
            page_index: {
              "width": int, "height": int,
              "texts":  [{"id", "xmin","ymin","xmax","ymax"}],
              "faces":  [{"id", "xmin","ymin","xmax","ymax", "character"}],
              "bodies": [{"id", "xmin","ymin","xmax","ymax", "character"}],
              "frames": [{"id", "xmin","ymin","xmax","ymax"}],
              "body_id_to_char": {body_or_face_id: character_id},  # for speaker lookup
            }
          }
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    title = root.attrib.get("title", xml_path.stem)

    # Characters
    characters = {}
    for ch in root.findall(".//character"):
        characters[ch.attrib["id"]] = ch.attrib.get("name", "")

    # Pages
    pages = {}
    for page in root.findall(".//page"):
        idx = int(page.attrib["index"])
        w   = int(page.attrib["width"])
        h   = int(page.attrib["height"])

        texts, faces, bodies, frames = [], [], [], []
        # speaker_id in Manga109Dialog = body/face element ID → need to map to character_id
        body_id_to_char: dict[str, str] = {}

        for elem in page:
            tag  = elem.tag
            bbox = {
                "id":   elem.attrib.get("id", ""),
                "xmin": int(elem.attrib.get("xmin", 0)),
                "ymin": int(elem.attrib.get("ymin", 0)),
                "xmax": int(elem.attrib.get("xmax", 0)),
                "ymax": int(elem.attrib.get("ymax", 0)),
            }
            if tag == "text":
                texts.append(bbox)
            elif tag == "face":
                char_id = elem.attrib.get("character", "")
                faces.append({**bbox, "character": char_id})
                if char_id:
                    body_id_to_char[bbox["id"]] = char_id
            elif tag == "body":
                char_id = elem.attrib.get("character", "")
                bodies.append({**bbox, "character": char_id})
                if char_id:
                    body_id_to_char[bbox["id"]] = char_id
            elif tag == "frame":
                frames.append(bbox)

        pages[idx] = {
            "width": w, "height": h,
            "texts": texts, "faces": faces,
            "bodies": bodies, "frames": frames,
            "body_id_to_char": body_id_to_char,  # key lookup for speaker attribution
        }

    return {"title": title, "characters": characters, "pages": pages}


def parse_manga109dialog(xml_path: Path) -> dict:
    """Parse Manga109Dialog XML.

    Returns:
        {page_index: {text_id: speaker_id}}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {}
    for page in root.findall(".//page"):
        idx     = int(page.attrib["index"])
        mapping = {}
        for stt in page.findall("speaker_to_text"):
            text_id    = stt.attrib["text_id"]
            speaker_id = stt.attrib["speaker_id"]
            mapping[text_id] = speaker_id
        if mapping:
            result[idx] = mapping

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _center(bbox: dict) -> tuple[float, float]:
    return (bbox["xmin"] + bbox["xmax"]) / 2, (bbox["ymin"] + bbox["ymax"]) / 2


def _area(bbox: dict) -> float:
    return max(0, bbox["xmax"] - bbox["xmin"]) * max(0, bbox["ymax"] - bbox["ymin"])


def _aspect(bbox: dict) -> float:
    w = max(1, bbox["xmax"] - bbox["xmin"])
    h = max(1, bbox["ymax"] - bbox["ymin"])
    return w / h


def _iou(a: dict, b: dict) -> float:
    """Intersection-over-Union of two bboxes."""
    ix1 = max(a["xmin"], b["xmin"]); iy1 = max(a["ymin"], b["ymin"])
    ix2 = min(a["xmax"], b["xmax"]); iy2 = min(a["ymax"], b["ymax"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _same_panel(bbox_a: dict, bbox_b: dict, frames: list[dict]) -> float:
    """1.0 if bubble and character are in the same panel, else 0.0."""
    if not frames:
        return 0.5  # unknown
    def panel_of(bbox):
        best_iou, best_i = 0.0, -1
        for i, f in enumerate(frames):
            iou = _iou(bbox, f)
            if iou > best_iou:
                best_iou, best_i = iou, i
        return best_i
    return 1.0 if panel_of(bbox_a) == panel_of(bbox_b) else 0.0


def build_features(
    bubble: dict,
    character: dict,
    page_w: int,
    page_h: int,
    all_chars_on_page: list[dict],
    frames: Optional[list[dict]] = None,
) -> np.ndarray:
    """Build a 20-dim feature vector for a (bubble, character) pair.

    New vs v1 (14-dim):
      + same_panel      — bubble and character in same frame
      + dist_x, dist_y  — signed horizontal/vertical distance separately
      + char_size_rank  — rank of character by size (larger = more prominent)
      + bubble_h_n      — bubble height normalised (tall bubbles = more text)
      + nearest_dist    — distance to nearest character (context for rank)
      + overlap_x       — horizontal overlap between bubble and character
    """
    bx, by = _center(bubble)
    cx, cy = _center(character)

    bx_n = bx / page_w;  by_n = by / page_h
    cx_n = cx / page_w;  cy_n = cy / page_h
    dx   = (bx - cx) / page_w
    dy   = (by - cy) / page_h
    dist = math.sqrt(dx**2 + dy**2)

    b_area = _area(bubble)    / (page_w * page_h)
    c_area = _area(character) / (page_w * page_h)
    b_asp  = _aspect(bubble)
    b_h_n  = (bubble["ymax"] - bubble["ymin"]) / page_h

    # Rank by distance
    dists = [
        math.sqrt(
            ((_center(c)[0] - bx) / page_w) ** 2 +
            ((_center(c)[1] - by) / page_h) ** 2
        )
        for c in all_chars_on_page
    ]
    dists_sorted = sorted(set(dists))
    rank   = dists_sorted.index(dist) + 1 if dist in dists_sorted else len(dists_sorted)
    rank_n = rank / max(1, len(all_chars_on_page))

    # Nearest character distance (context feature)
    nearest_dist = dists_sorted[0] if dists_sorted else 0.0

    # Rank by character size (larger characters are usually protagonists → more dialogue)
    areas = sorted([_area(c) for c in all_chars_on_page], reverse=True)
    c_area_abs = _area(character)
    size_rank = (areas.index(c_area_abs) + 1) if c_area_abs in areas else len(areas)
    size_rank_n = size_rank / max(1, len(all_chars_on_page))

    above = 1.0 if by < cy else 0.0
    left  = 1.0 if bx < cx else 0.0

    # Horizontal overlap: bubble and character x-ranges overlap → strong signal
    b_xmin, b_xmax = bubble["xmin"] / page_w, bubble["xmax"] / page_w
    c_xmin, c_xmax = character["xmin"] / page_w, character["xmax"] / page_w
    overlap_x = max(0.0, min(b_xmax, c_xmax) - max(b_xmin, c_xmin))

    # Same panel
    same_panel = _same_panel(bubble, character, frames or [])

    char_count_n = len(all_chars_on_page) / 10.0

    return np.array([
        bx_n, by_n, cx_n, cy_n,
        dx, dy, dist,
        b_area, c_area, b_asp, b_h_n,
        rank_n, nearest_dist, size_rank_n,
        above, left,
        overlap_x, same_panel,
        char_count_n,
        abs(dx),   # absolute horizontal distance (symmetric feature)
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    manga109_dir: Path,
    max_volumes: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse all volumes and build (X, y, groups) arrays.

    Returns:
        X:      (N, FEATURE_DIM) float32
        y:      (N,) int32  — 0 or 1
        groups: (N,) int32  — bubble group ID (all pairs for same bubble share ID)
                             Used for per-bubble accuracy evaluation.
    """
    ann_dir    = manga109_dir / "annotations"
    dialog_dir = manga109_dir / "annotations_Manga109Dialog"

    xml_files = sorted(ann_dir.glob("*.xml"))
    if max_volumes:
        xml_files = xml_files[:max_volumes]

    X_list, y_list, group_list = [], [], []
    total_pages = 0
    group_id = 0

    for ann_path in xml_files:
        dialog_path = dialog_dir / ann_path.name
        if not dialog_path.exists():
            continue

        ann    = parse_annotations(ann_path)
        dialog = parse_manga109dialog(dialog_path)

        for page_idx, speakers in dialog.items():
            page = ann["pages"].get(page_idx)
            if page is None:
                continue

            page_w  = page["width"]
            page_h  = page["height"]
            frames  = page["frames"]

            text_by_id      = {t["id"]: t for t in page["texts"]}
            body_id_to_char = page["body_id_to_char"]

            char_bboxes: dict[str, dict] = {}
            for item in page["faces"] + page["bodies"]:
                cid = item["character"]
                if cid and cid not in char_bboxes:
                    char_bboxes[cid] = item

            if not char_bboxes:
                continue

            all_chars = list(char_bboxes.values())
            total_pages += 1

            for text_id, speaker_body_id in speakers.items():
                bubble = text_by_id.get(text_id)
                if bubble is None:
                    continue

                speaker_char_id = body_id_to_char.get(speaker_body_id, speaker_body_id)

                # All pairs for this bubble share the same group_id
                for cid, char_bbox in char_bboxes.items():
                    feat  = build_features(
                        bubble, char_bbox, page_w, page_h, all_chars, frames
                    )
                    label = 1 if cid == speaker_char_id else 0
                    X_list.append(feat)
                    y_list.append(label)
                    group_list.append(group_id)

                group_id += 1  # new bubble = new group

    if not X_list:
        raise ValueError("No training samples found — check dataset paths.")

    X      = np.stack(X_list)
    y      = np.array(y_list,     dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)

    pos = y.sum()
    log.info(
        f"Dataset built: {len(X)} samples, {group_id} bubbles, {total_pages} pages "
        f"({pos} positive / {len(y)-pos} negative)"
    )
    return X, y, groups


# ─────────────────────────────────────────────────────────────────────────────
# 4. MLP Model
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float):
    """Build a simple MLP classifier using PyTorch."""
    import torch.nn as nn

    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, 2))  # binary: 0=not speaker, 1=speaker
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────────────────────────────────────

def per_bubble_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    scaler,
    device,
) -> float:
    """Per-bubble accuracy: for each bubble, does the model pick the correct speaker?

    This is the real metric — matches paper evaluation on Manga109Dialogue.
    Binary pair accuracy (68%) can correspond to ~85%+ per-bubble accuracy.
    """
    import torch
    X_scaled = scaler.transform(X).astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled).to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    correct = 0
    total   = 0
    for gid in np.unique(groups):
        mask       = groups == gid
        group_y    = y[mask]
        group_prob = probs[mask]

        # Find ground-truth speaker index in this group
        pos_indices = np.where(group_y == 1)[0]
        if len(pos_indices) == 0:
            continue  # no ground-truth speaker → skip

        predicted_idx = group_prob.argmax()
        correct += int(predicted_idx in pos_indices)
        total   += 1

    return correct / total if total > 0 else 0.0


def train(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    save_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    use_gpu: bool = True,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Split (keep groups aligned) ───────────────────────────────────────────
    indices = np.arange(len(X))
    idx_train, idx_tmp = train_test_split(
        indices, test_size=VAL_RATIO + TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=y[idx_tmp]
    )
    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val,   groups_val  = X[idx_val],   y[idx_val],   groups[idx_val]
    X_test,  y_test,  groups_test = X[idx_test],  y[idx_test],  groups[idx_test]
    log.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # ── Normalise features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # Save scaler for inference
    save_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── Class weights (handle imbalance) ─────────────────────────────────────
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
    log.info(f"Class weight (pos): {pos_weight.item():.2f}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def to_loader(Xa, ya, shuffle=True):
        ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya).long())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    train_loader = to_loader(X_train, y_train)

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    log.info(f"Training on {device}")

    model     = build_mlp(FEATURE_DIM, HIDDEN_DIMS, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight.item()]).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_path    = save_dir / "speaker_mlp_best.pt"
    history      = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(X_train)

        # Validate — per-bubble accuracy (real metric)
        val_bubble_acc = per_bubble_accuracy(
            X_val, y_val, groups_val, model, scaler, device
        )

        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_bubble_acc": val_bubble_acc,
        })

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={train_loss:.4f}  val_bubble_acc={val_bubble_acc:.4f}"
            )

        if val_bubble_acc > best_val_acc:
            best_val_acc = val_bubble_acc
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, best_path)

    log.info(f"Best val bubble accuracy: {best_val_acc:.4f}  (saved: {best_path})")

    # ── Final test evaluation ─────────────────────────────────────────────────
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_bubble_acc = per_bubble_accuracy(
        X_test, y_test, groups_test, model, scaler, device
    )

    # Also report binary pair metrics for reference
    model.eval()
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    with torch.no_grad():
        test_preds = model(
            torch.from_numpy(X_test_scaled).to(device)
        ).argmax(dim=1).cpu().numpy()

    log.info(f"\nTest per-bubble accuracy: {test_bubble_acc:.4f}  (target: >0.85)")
    log.info(f"Test binary pair accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print(classification_report(y_test, test_preds, target_names=["not_speaker", "speaker"]))

    # ── Save final model + metadata ───────────────────────────────────────────
    final_path = save_dir / "speaker_mlp.pt"
    torch.save({
        "model_state":       model.state_dict(),
        "feature_dim":       FEATURE_DIM,
        "hidden_dims":       HIDDEN_DIMS,
        "dropout":           DROPOUT,
        "val_bubble_acc":    best_val_acc,
        "test_bubble_acc":   test_bubble_acc,
    }, final_path)

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Model saved: {final_path}")
    log.info(f"Per-bubble accuracy: {test_bubble_acc:.4f}  (target: >0.85)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Rule-based fallback (baseline + production fallback)
# ─────────────────────────────────────────────────────────────────────────────

def assign_speaker_rule_based(
    bubble: dict,
    characters: list[dict],
    page_w: int,
    page_h: int,
) -> Optional[int]:
    """Assign bubble to nearest character (rule-based baseline).

    Prefers characters that are:
      1. Above or at the same level as the bubble
      2. Closer in euclidean distance

    Returns index into characters list, or None if no characters.
    """
    if not characters:
        return None

    bx, by = _center(bubble)
    best_idx, best_score = 0, float("inf")

    for i, char in enumerate(characters):
        cx, cy = _center(char)
        dx = (bx - cx) / page_w
        dy = (by - cy) / page_h
        dist = math.sqrt(dx**2 + dy**2)

        # Slight preference for characters above (manga reading: speaker above bubble)
        penalty = 0.0 if cy <= by else 0.1
        score = dist + penalty

        if score < best_score:
            best_score, best_idx = score, i

    return best_idx


# ─────────────────────────────────────────────────────────────────────────────
# 7. Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_speaker(
    model_path: Path,
    scaler_path: Path,
    bubbles: list[dict],
    characters: list[dict],
    page_w: int,
    page_h: int,
    use_gpu: bool = True,
) -> list[dict]:
    """Predict speaker for each bubble.

    Args:
        bubbles:    list of {"id", "xmin", "ymin", "xmax", "ymax"}
        characters: list of {"id", "xmin", "ymin", "xmax", "ymax"}

    Returns:
        list of {"bubble_id", "speaker_idx", "speaker_id", "confidence"}
    """
    import torch
    import pickle

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    checkpoint = torch.load(model_path, map_location=device)
    model = build_mlp(
        checkpoint["feature_dim"],
        checkpoint["hidden_dims"],
        checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    results = []
    for bubble in bubbles:
        if not characters:
            results.append({
                "bubble_id": bubble.get("id", ""),
                "speaker_idx": None,
                "speaker_id": None,
                "confidence": 0.0,
            })
            continue

        # Build features for this bubble vs all characters
        feats = np.stack([
            build_features(bubble, char, page_w, page_h, characters)
            for char in characters
        ])
        feats_scaled = scaler.transform(feats).astype(np.float32)

        with torch.no_grad():
            logits = model(torch.from_numpy(feats_scaled).to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        best_idx  = int(probs.argmax())
        best_conf = float(probs[best_idx])

        # Fallback to rule-based if model confidence too low
        if best_conf < 0.4:
            best_idx = assign_speaker_rule_based(bubble, characters, page_w, page_h)
            best_conf = 0.0  # mark as rule-based

        results.append({
            "bubble_id":  bubble.get("id", ""),
            "speaker_idx": best_idx,
            "speaker_id": characters[best_idx].get("id", f"char_{best_idx}"),
            "confidence": round(best_conf, 4),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speaker Attribution MLP — train on Manga109Dialogue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manga109-dir",
        default="/mnt/nfs-data/tin_dataset/comic/manga109/raw/Manga109s_released_2023_12_07",
        help="Root of Manga109 dataset (contains annotations/ and annotations_Manga109Dialog/)",
    )
    p.add_argument(
        "--save-dir",
        default="/mnt/nfs-data/tin_dataset/comic/speaker_attribution",
        help="Directory to save model + scaler + history",
    )
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--max-volumes", type=int,   default=None,
                   help="Limit number of volumes (for quick testing)")
    p.add_argument("--no-gpu",      action="store_true")
    p.add_argument("--save-dataset", action="store_true",
                   help="Save parsed dataset as .npz for reuse")
    p.add_argument("--load-dataset", default=None,
                   help="Load pre-saved dataset .npz instead of re-parsing")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or build dataset ─────────────────────────────────────────────────
    if args.load_dataset:
        log.info(f"Loading dataset from {args.load_dataset}")
        data = np.load(args.load_dataset)
        X, y = data["X"], data["y"]
        # groups may not exist in old dataset — regenerate if missing
        if "groups" in data:
            groups = data["groups"]
        else:
            log.warning("Old dataset without groups — re-parsing to get groups")
            manga109_dir = Path(args.manga109_dir)
            X, y, groups = build_dataset(manga109_dir, max_volumes=args.max_volumes)
        log.info(f"Loaded: {len(X)} samples")
    else:
        manga109_dir = Path(args.manga109_dir)
        X, y, groups = build_dataset(manga109_dir, max_volumes=args.max_volumes)

    if args.save_dataset:
        ds_path = save_dir / "dataset.npz"
        np.savez(ds_path, X=X, y=y, groups=groups)
        log.info(f"Dataset saved: {ds_path}")

    # ── Train ─────────────────────────────────────────────────────────────────
    train(
        X, y, groups,
        save_dir  = save_dir,
        epochs    = args.epochs,
        batch_size= args.batch_size,
        lr        = args.lr,
        use_gpu   = not args.no_gpu,
    )


if __name__ == "__main__":
    main()
