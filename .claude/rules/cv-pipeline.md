# CV Pipeline Rules
> YOLO bubble/character detection, OCR, face clustering, speaker attribution, reading order, character_db. Auto-loaded by Claude Code.

---
### Important Environment Note
Use dedicated conda environment `comic_ocr` with Python 3.11 due to strict dependency requirements of VietOCR 0.3.13 and PaddleOCR.
Do not install OCR packages in the main TTS environment.

## Pipeline Flow

```
Comic page image
    │
    ├── YOLO → detect bubbles, characters, panels (bounding boxes)
    │
    ├── PaddleOCR → read Vietnamese text from each bubble crop
    │
    ├── ArcFace → extract face embeddings → cluster same character
    │
    ├── Speaker Attribution → assign each bubble to a character
    │
    ├── Reading Order → sort bubbles by LTR/RTL reading direction
    │
    └── character_db → maintain voice consistency across pages
```

---

## 1. YOLO Bubble/Character/Panel Detection

### Model
- **YOLOv8** (Ultralytics) — fine-tune for 3 classes: `bubble`, `character`, `panel`

### Training Data
| Dataset | Annotation | Size |
|---------|-----------|------|
| **Manga109** | Bubble + character + panel boxes | 109 volumes (~10K+ images) |
| **DCM772** | Mixed annotations | 772 images |
| **Vietnamese comics** (self-labeled) | 100-200 images for domain adaptation | Manual, 2-3 days with LabelImg/Roboflow |

### Strategy
- Pre-train on Manga109 (largest, annotated)
- Fine-tune on 100-200 Vietnamese comic pages (font/style differences)
- Target: mAP > 0.8 on Vietnamese comics

### Notes
- Bubble shapes are language-independent → model transfers well across languages
- Panel detection helps reading order (sort panels first, then bubbles within panels)
- Vietnamese comics may have different bubble styles than manga → need fine-tune data

---

## 2. OCR (VietOCR + PaddleOCR)

### Strategy: 2-layer OCR
| Layer | Tool | Vai trò |
|-------|------|---------|
| **Text Detection** | PaddleOCR (detection module) | Tìm vị trí text trong bubble (bounding box) |
| **Text Recognition** | **ProtonX VietOCR** | Đọc chữ tiếng Việt từ crop — accuracy cao hơn PaddleOCR generic |

### Why VietOCR (ProtonX)
- Chuyên biệt cho tiếng Việt — trained on Vietnamese text datasets
- Transformer-based (attention OCR) — xử lý tốt dấu thanh điệu
- Accuracy cao hơn PaddleOCR generic trên Vietnamese text
- Open-source: https://github.com/pbcquoc/vietocr

### Pipeline
```python
# Step 1: PaddleOCR detect text regions in bubble
from paddleocr import PaddleOCR
detector = PaddleOCR(lang='vi', rec=False)  # detection only
text_boxes = detector.ocr(bubble_crop, rec=False)

# Step 2: VietOCR recognize Vietnamese text from crops
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cuda'
recognizer = Predictor(config)

for box in text_boxes:
    text_crop = crop_from_box(bubble_crop, box)
    text = recognizer.predict(text_crop)  # Vietnamese text output
```

### Fallback
- Nếu VietOCR gặp vấn đề → fallback PaddleOCR `lang='vi'` (recognition + detection)
- So sánh CER của cả 2 trên Vietnamese comics → chọn tool tốt hơn

### Target
- CER (Character Error Rate) < 3% on Vietnamese comics (VietOCR expected better than PaddleOCR's ~5%)
- Handle: dấu thanh điệu, chữ đặc biệt (ă, â, đ, ê, ô, ơ, ư), mixed fonts

---

## 3. Face Clustering (ArcFace)

### Purpose
Group same character across multiple appearances/pages.

### Pipeline
```
1. Detect faces in each panel (anime face detector or YOLOv8-face)
2. Extract face embedding per face (ArcFace, pre-trained)
3. Cluster embeddings (DBSCAN or agglomerative clustering)
4. Each cluster = 1 unique character
```

### character_db structure
```python
character_db = {
    "Character_1": {
        "face_embeddings": [emb1, emb2, ...],  # all appearances
        "avg_embedding": avg_emb,                # centroid for matching
        "speaker_voice": speaker_embedding,      # Voice Cloning embedding
        "name": "Naruto",                        # user-assigned name
        "appearances": [page1, page3, page5]     # which pages
    },
    ...
}
```

### Matching new faces
```python
for face in new_faces:
    face_emb = arcface(face.crop)
    best_match, best_score = None, 0
    for name, data in character_db.items():
        score = cosine_similarity(face_emb, data["avg_embedding"])
        if score > best_score:
            best_match, best_score = name, score

    if best_score > 0.7:
        # Known character → reuse voice
        face.character = best_match
    else:
        # New character → create entry, user assigns voice later
        create_new_character(face_emb)
```

### Persistence
- character_db is maintained across all pages in a chapter
- Same character on page 1 and page 15 → same voice

---

## 4. Speaker Attribution (AI Model)

### Problem
Which character speaks which bubble? Rule-based (proximity) is ~60-70% accurate.

### AI Approach
Input features per bubble:
- `bubble_center_x`, `bubble_center_y` (normalized)
- `nearest_char_dist` (distance to nearest character)
- `char_positions` (relative positions of all characters in panel)
- `tail_direction` (angle of bubble tail, if detected)
- `panel_id` (which panel this bubble belongs to)
- `bubble_area` (relative size)
- `char_count_in_panel` (how many characters nearby)

Model: MLP (3 layers) or lightweight GNN
Output: probability distribution over characters in the panel

### Training
- **Manga109Dialogue** dataset: ground-truth bubble→speaker ID mapping
- Augment with Vietnamese comic annotations (100-200 pages)
- Accuracy target: > 85%

### Paper Benchmark Context
> Paper "Emotion-Aware Speech Generation..." (2025) results trên Manga109:
> - Rule-based (frame dist): 71.5% total
> - Manga109Speaker (trained DL): **75.7% total, 30.7% hard** ← current SOTA
> - LLM (GPT-4o): 64.8% total, 20.5% hard ← **LLM thua trained model**
>
> **Key insight**: Hard cases (bubble xa speaker) chỉ 20-30%. Paper thừa nhận:
> "the deep learning model can learn implicit spatial patterns... the LLM relies
> primarily on textual layout and proximity heuristics"
>
> → SGG/GNN có lợi thế cho hard cases vì explicit spatial graph relationships.
> → Nếu project đạt >85% total → vượt SOTA paper 10%.

### Fallback: Rule-based
```python
def assign_bubble_rule_based(bubble, characters):
    # Rule 1: Tail detection (if bubble has visible tail)
    if bubble.tail_detected:
        return nearest_character_to_tail(bubble.tail_tip, characters)

    # Rule 2: Proximity (closest character above or to the left)
    distances = [(c, distance(bubble.center, c.center)) for c in characters]
    return min(distances, key=lambda x: x[1])[0]
```

---

## 5. Reading Order

### Configuration
```python
reading_direction = "ltr"   # left-to-right (Vietnamese comics, webtoons)
# or
reading_direction = "rtl"   # right-to-left (Japanese manga)
```

### Algorithm
```python
def sort_reading_order(bubbles, panels, direction="ltr"):
    # Step 1: Sort panels (top→bottom, then left→right or right→left)
    panels = sorted(panels, key=lambda p: (p.y // row_height,
                                            p.x if direction == "ltr" else -p.x))

    # Step 2: Assign bubbles to panels (by overlap)
    for bubble in bubbles:
        bubble.panel = find_panel_with_max_overlap(bubble, panels)

    # Step 3: Within each panel, sort bubbles
    ordered = []
    for panel in panels:
        panel_bubbles = [b for b in bubbles if b.panel == panel]
        panel_bubbles.sort(key=lambda b: (b.y // row_height,
                                           b.x if direction == "ltr" else -b.x))
        ordered.extend(panel_bubbles)

    return ordered
```

---

## 6. Review Step (User Verification)

After scanning all pages, show user:

```
Found 4 characters:
  😀 Character 1: appears on pages 1,3,5,8,12
     Voice: [Upload 5s audio] or [Select speaker ▼]

  😮 Character 2: appears on pages 1,4,7,10
     Voice: [Upload 5s audio] or [Select speaker ▼]
  ...

[✓ Confirm & Generate]
```

User can:
- Rename characters
- Merge/split incorrectly clustered characters
- Choose voice for each character
- Review bubble→character assignments

This step takes ~30 seconds but significantly improves output quality.

---

## Pre-process Ahead (Real-time Feel)

```
User is listening to page 3 audio (~15s)
    │
    While audio plays:
    ├── Pipeline processes page 4 → ready
    ├── Pipeline processes page 5 → ready
    └── Pipeline starts page 6
    │
User flips to page 4 → audio plays IMMEDIATELY
```

- 1 page processing: ~3-5s
- 1 page audio: ~10-15s
- Pipeline always ahead of user → no waiting after first page

---

## Optional Extensions (KHÔNG nằm trong core pipeline)

### BLIP/LLaVA Scene Understanding
- Purpose: hiểu context ảnh (ai đang làm gì, bối cảnh, cảm xúc)
- Hỗ trợ: emotion detection, speaker attribution cải thiện
- Cost: ~7B params, tốn VRAM, chậm
- Chỉ thêm nếu: core pipeline hoàn thành + có thời gian + có giá trị rõ ràng

### Scene Graph Generation (SGG) cho Speaker Attribution
- Upgrade từ MLP: YOLO output → graph (nodes + spatial edges) → GNN predict "speaks" edges
- Nodes: character, bubble, panel. Edges: contains, near, tail_points, closest_to
- Model: GAT/GCN. Target accuracy: >90% (vs MLP ~85%)
- Strategy: MLP làm baseline trước → SGG upgrade nếu còn thời gian hoặc MLP chưa đạt target
- Chi tiết: xem `rules/model-strategy.md` → Optional Extensions → SGG section

### LLM Attribution (baseline comparison)
- Dùng GPT-4o / Llama-3.1 / Gemini cho speaker attribution
- Mục đích: **so sánh accuracy** với self-trained model, KHÔNG thay thế
- So sánh: Rule-based (~65%) vs Self-trained (>85%) vs SGG (~90%) vs LLM
- Giá trị: nếu self-trained ≈ LLM → chứng minh model nhẹ đạt chất lượng LLM lớn
