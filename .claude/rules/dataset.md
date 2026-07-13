# Dataset Strategy and Processing Rules
> Dataset sources, tone extraction, shared data format, comic dataset strategy. Auto-loaded by Claude Code.

---

## Datasets Overview

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| **VieNeu-TTS-140h** | TTS + Voice Cloning training | 140h, 193 speakers, 24kHz | Primary |
| **Manga109** | YOLO training (bubble/char/panel) | 109 volumes, annotated | CV pipeline |
| **DCM772** | Additional YOLO data | 772 images | CV pipeline |
| **Vietnamese comics** | Test/demo | 20-50 pages (crawl/scan) | Evaluation |
| **Manga109Dialogue** | Speaker Attribution training (bubble→speaker mapping) | Derived from Manga109 | Speaker Attribution |
| **KangaiSet** | Emotion labels for manga characters (facial expression) | Emotion labels on Manga109 | Emotion extension |
| **ProtonX VietOCR** | Vietnamese text recognition (pre-trained model) | Transformer-based | OCR pipeline |
| **ESD** | Emotion extension (optional) | 29h, English, 5 emotions | Optional |

---

## TTS Dataset: VieNeu-TTS-140h

### Strategy
- All TTS experiments train on same dataset for fair comparison
- Prioritize clean subset before scaling to full data
- Data usage order:
  1. Clean core subset (~20-40h) → verify convergence
  2. Full 140h → full training
  3. Additional datasets only if needed

### Text Normalization: sea-g2p (MANDATORY)
- Source: https://github.com/pnnbao97/VieNeu-TTS
- Handles: numbers, currency, English words, special symbols
- Pipeline: `Raw text → sea-g2p (normalize + phonemize) → Tone extraction → Model input`
- sea-g2p is a prerequisite, not a contribution

### Tone Extraction
Deterministic from Vietnamese diacritics:
```
Tone 0: padding/consonants
Tone 1 (ngang):  no diacritic     (a, e, i, o, u, y)
Tone 2 (sắc):   acute accent     (á, é, í, ó, ú, ý)
Tone 3 (huyền):  grave accent     (à, è, ì, ò, ù, ỳ)
Tone 4 (hỏi):   hook above       (ả, ẻ, ỉ, ỏ, ủ, ỷ)
Tone 5 (ngã):   tilde            (ã, ẽ, ĩ, õ, ũ, ỹ)
Tone 6 (nặng):  dot below        (ạ, ẹ, ị, ọ, ụ, ỵ)
```

### Shared Data Format
Each sample: `audio_path`, `text`, `phonemes`, `tones`, `speaker_id`, `duration`, `sample_rate`
Extended: `gender`, `region` (North/Central/South), `quality_score`

### Voice Cloning Data Requirements
- DataLoader must sample reference audio (different utterance, same speaker)
- Hold out 10-20 speakers for zero-shot evaluation
- Reference audio: 5-10s clean speech

---

## Comic Dataset: Manga109 + Vietnamese Comics

### Manga109 (YOLO training)
- 109 manga volumes with bounding box annotations
- Classes: `text` (bubble), `face` (character), `frame` (panel)
- Language-independent — bubble shapes transfer across languages
- Use for primary YOLO training

### Manga109Dialogue (Speaker Attribution)
- Derived from Manga109 — links dialogue bubbles to speaker character IDs
- Ground-truth bubble→character mapping → dùng train Speaker Attribution model
- Referenced in paper "Emotion-Aware Speech Generation with Character-Specific Voices for Comics" (2025)
- SOTA trên dataset này: Manga109Speaker trained model = 75.7% accuracy
- Target của project: >85% (vượt SOTA bằng SGG/GNN spatial reasoning)

### KangaiSet (Emotion — optional)
- Emotion labels cho manga character faces trên Manga109
- Simplified version: binary (neutral vs non-neutral) — paper dùng cách này
- Full version: 5 classes (neutral, anger, happiness, sadness, surprise)
- Dùng để train Emotion Intensity Estimation (ResNet-50 fine-tune)
- Paper benchmark: F1 ~42.9% cho 5-way classification (emotion trong comic rất khó)
- Lưu ý: disgust và fear bị loại vì quá ít samples

### Vietnamese Comics (test/demo)
Sources for 20-50 test pages:
- Manga dịch Việt (NXB Kim Đồng, NXB Trẻ) — professional translation, natural Vietnamese
- Webtoon Việt (Naver Webtoon VN) — original Vietnamese content
- Truyện tranh Việt Nam (Thần Đồng Đất Việt, Long Thần Tướng)

### Labeling Vietnamese comics
- Label 100-200 pages for YOLO fine-tune (domain adaptation)
- Tool: LabelImg or Roboflow
- Classes: `bubble`, `character`, `panel`
- Time: ~2-3 days
- Purpose: adapt YOLO from manga style → Vietnamese comic style

---

## Audio Processing Rules
- Sample rate: 24kHz (VieNeu-TTS standard)
- Filter: remove < 0.5s or > 15s
- Normalize loudness
- Silence trimming (VAD)
- Clean audio priority over quantity

## Text Processing Rules
- Use sea-g2p for ALL normalization (no custom normalizers)
- Verify output: spot-check 50+ samples
- No dirty transcripts in training

## Data Splits
- Same train/val/test split for all experiments
- Test set = custom evaluation set (200-500 sentences for TTS)
- Val set = ~5% of training data

---

## Storage
All datasets, checkpoints, logs, results:
**bes-ai-machine-02:/mnt/nfs-data/tin_dataset**

```
/mnt/nfs-data/tin_dataset/
├── raw/                    ← Original VieNeu-TTS-140h
├── processed/              ← Cleaned, normalized
│   ├── common/             ← Shared format (text, phonemes, tones, mel)
│   └── splits/             ← train/val/test filelists
├── comic/                  ← Comic datasets
│   ├── manga109/           ← Manga109 for YOLO
│   ├── manga109dialogue/   ← Speaker attribution labels
│   ├── kangaiset/          ← Emotion labels (optional)
│   ├── vietnamese/         ← Vietnamese comics (test/demo)
│   └── yolo_labels/        ← YOLO annotation files
├── checkpoints/            ← Model checkpoints
├── logs/                   ← Training logs
└── results/                ← Evaluation results, audio samples
```
