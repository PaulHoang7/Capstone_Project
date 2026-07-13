# Capstone Project: Comic Voice-Over System

## Context
Đề tài tốt nghiệp: hệ thống tự động lồng tiếng truyện tranh. Upload chapter truyện → hệ thống detect bubble, đọc text, nhận diện nhân vật → mỗi nhân vật nói bằng giọng riêng với đúng thanh điệu tiếng Việt.

**AI core**: VITS2 + Dual-Path Encoder + Voice Cloning + Speaker Attribution
**CV pipeline**: YOLO + ProtonX VietOCR + PaddleOCR (detect) + ArcFace (pre-trained/fine-tuned)

**Constraints**: 16 tuần, 1 GPU.

---

## Pipeline Overview

```
Trang truyện (ảnh)
  │
  ▼
┌─ CV Pipeline (single-pass) ──────────────────────────────┐
│  1. YOLOv8: bubble/character/panel detect  [fine-tune]   │
│  2. ProtonX VietOCR: đọc text tiếng Việt   [pre-trained] │
│  3. ArcFace: face embedding + clustering   [pre-trained] │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
              Structured Output JSON
       (panels, bubbles, texts, faces, reading order)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─ Speaker Attribution ──┐  ┌─ Character DB ─────────────┐
│  AI model (self-trained)│  │  ArcFace clustering        │
│  Input: bbox, tail,    │  │  + voice assignment         │
│  distance, panel_id    │  │  (upload ref / select spk)  │
│  [TRAIN, ✅ AI]         │  │  Persistent across pages   │
└────────────┬───────────┘  └────────────┬───────────────┘
             └────────────┬──────────────┘
                          ▼
┌─ TTS + Voice Cloning ────────────────────────────────────┐
│  VITS2 + Dual-Path Encoder + ECAPA-TDNN                  │
│  Text + Tone Seq + Speaker Emb → Audio (end-to-end)      │
│  Pre-process ahead (page N+2 while playing N)            │
│  [TRAIN, ✅ AI core]                                      │
└──────────────────────────┬───────────────────────────────┘
                           ▼
                   Output: Audio per bubble
                   + full chapter track
                   Character voice consistent

- - - - OPTIONAL (nếu còn thời gian) - - - -

  Emotion-Aware Comic Speech Synthesis (Week 16+):
    Step 1: Train emotion TTS trên ESD English (GST hoặc Emotion Embedding)
    Step 2: Evaluate cross-lingual transfer EN → VI (tone preservation?)
    Step 3: Build Emotion Detection (face ViT + text sentiment + visual context → fusion)
    Step 4: Integrate emotion label vào TTS pipeline
    Step 5: End-to-end eval: comic page → emotion-aware voiced audio

  Scene Graph Generation (SGG) cho Speaker Attribution:
    - YOLO output → graph nodes + spatial edges → GNN predict "speaks" edges
    - Upgrade từ MLP baseline nếu accuracy chưa đạt >85% hoặc còn thời gian
    - Target: >90% accuracy (vs MLP ~85%)

  Emotion Intensity Estimation:
    - Thêm regression head (0.0→1.0) vào Emotion Detection
    - "angry 0.3" (hơi bực) vs "angry 0.9" (phẫn nộ) → TTS scale proportionally
    - Phụ thuộc: Emotion-Aware extension phải xong trước

  VoiceDesign (text-described voice assignment):
    - User mô tả giọng bằng text thay vì upload audio
    - Match description → VieNeu-TTS 193 speaker profiles
    - Alternative cho Voice Clone khi không có reference audio
    - ~3-5 ngày implement

  Other optional:
    - BLIP/LLaVA scene understanding (hỗ trợ emotion/context)
    - LLM attribution (GPT-4o baseline comparison)
```

---

## Timeline (16 tuần)

| Tuần | Task | Milestone |
|------|------|-----------|
| **1-2** | Data pipeline + CV tools setup | ✓ VieNeu-TTS pipeline done, YOLO fine-tuned |
| | ├── VieNeu-TTS: sea-g2p + tone extraction + mel-spectrogram | |
| | ├── Install YOLOv8, PaddleOCR, ArcFace | |
| | └── Fine-tune YOLOv8 on Manga109 (bubble/character/panel) | |
| **3-4** | VITS2 baseline training (curriculum: 1→10→193 speakers) | ✓ Baseline converged |
| | └── Song song: test OCR + face detection trên truyện Việt | |
| **5-6** | Dual-Path Encoder ablation B, C, D | ✓ 3 ablation variants done |
| **7-8** | Dual-Path ablation E, F + evaluation + tone confusion matrices | ✓ **TTS COMPLETE** |
| | ← **CHECKPOINT 1** | |
| **9** | ECAPA-TDNN pre-train on VieNeu-TTS + integrate vào VITS2 | ✓ Speaker encoder works |
| **10-11** | Voice Cloning training Phase 2 (dual conditioning) + Phase 3 (zero-shot) | ✓ Clone works |
| **12** | Voice Cloning eval + Clone ablation + Speaker Attribution model | ✓ **VOICE CLONING COMPLETE** |
| | ← **CHECKPOINT 2** | |
| **13** | CV pipeline integration: character clustering + bubble-char assignment | ✓ Full pipeline works |
| **14** | Streaming web demo (upload chapter → audio) | ✓ Demo ready |
| **15** | Report + slides | ✓ Documentation done |
| **16** | Buffer / Emotion extension: ESD train → cross-lingual eval → emotion detection module → integrate | Optional |

---

## Checkpoints

### Checkpoint 1 (tuần 8): TTS complete
- **On track**: proceed to Voice Cloning (tuần 9-12)
- **Slightly behind**: proceed to Voice Cloning, reduce ablation variants
- **Very behind**: skip Voice Cloning, focus evaluation + simple demo

### Checkpoint 2 (tuần 12): Voice Cloning complete
- **On track**: build full CV pipeline + Speaker Attribution + streaming demo
- **Slightly behind**: use rule-based assignment (no AI Speaker Attribution)
- **Very behind**: skip CV pipeline, demo with manual text input + voice cloning

---

## Fallback Levels

| Level | Scope | Đủ tốt nghiệp? |
|-------|-------|-----------------|
| **Full success** | TTS + Dual-Path + Voice Cloning + CV pipeline + Speaker Attribution + streaming demo | Excellent |
| **Good success** | TTS + Dual-Path + Voice Cloning + CV pipeline (rule-based assign) | Strong |
| **Minimum** | TTS + Dual-Path + Voice Cloning (no CV, manual text input) | Sufficient |
| **Emergency** | TTS + Dual-Path only (no Voice Cloning, no CV) | Minimum pass |

---

## AI Contribution: Dual-Path Encoder

```
phoneme_ids → [Linguistic Encoder (Transformer x N)] ─┐
tone_ids    → [Tonal Encoder (Transformer x M)]    ───┤
                                                       ↓
                                              Cross-Attention Fusion
                                                       ↓
                                               Fused Representation
                                                       ↑
                                               speaker_embedding (Voice Clone)
                                                       ↓
                                               Flow → Decoder → Audio
```

### Ablation Study (6 variants)

| Variant | Mô tả | Warmstart |
|---------|-------|-----------|
| A: Baseline | VITS2 gốc, single TextEncoder | Không |
| B: + Tone Embedding | Thêm tone_emb, single encoder | Từ A |
| C: + Dual-Path (no cross-attn) | 2 encoder, concatenate | Không (arch change) |
| D: + Cross-Attention | Full Dual-Path architecture | Từ C |
| E: + F0 Loss | Auxiliary F0 contour supervision | Từ D |
| F: + Tone-Aware Duration | Condition duration predictor on tone | Từ E |

### Clone Ablation (3 variants)

| Variant | Mô tả |
|---------|-------|
| Clone-A | VITS2 baseline + ECAPA-TDNN (no tone awareness) |
| Clone-B | VITS2 + Tone Embedding + ECAPA-TDNN |
| Clone-D | VITS2 + Dual-Path + CrossAttn + ECAPA-TDNN (full) |

Expected: Clone-D giữ tone accuracy tốt nhất khi clone.

---

## Training Time Estimates (1 GPU, fp16)

| Task | RTX 3090 | A100 |
|------|----------|------|
| VITS2 6 ablation variants | ~12-21 ngày | ~6-12 ngày |
| Voice Cloning (3 phases + ablation) | ~13-19 ngày | ~6-10 ngày |
| YOLO fine-tune | ~1-2 ngày | ~0.5-1 ngày |
| Speaker Attribution | ~1-2 ngày | ~0.5-1 ngày |
| **Tổng** | **~27-44 ngày** | **~13-24 ngày** |

Calendar time > GPU time vì cần debug, evaluate, adjust giữa các runs.

---

## Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| **VieNeu-TTS-140h** | TTS + Voice Cloning training | 140h, 193 speakers, 24kHz |
| **Manga109** | YOLO fine-tune (bubble/character/panel) | 109 volumes, annotated |
| **DCM772** | Additional YOLO training data | 772 images |
| **Vietnamese comics** (crawl) | Test/demo | 20-50 pages |

---

## Cấu trúc thư mục

```
TTS/
├── Capstone_project/
│   ├── .claude/                ← Rules & docs
│   ├── shared/                 ← Shared pipeline
│   │   ├── tone_extractor.py
│   │   ├── data_pipeline.py
│   │   └── evaluation/
│   ├── voice_cloning/          ← Voice Cloning module
│   │   ├── speaker_encoder.py
│   │   ├── speaker_encoder_train.py
│   │   └── cloning_eval.py
│   ├── cv_pipeline/            ← Comic Vision pipeline
│   │   ├── bubble_detector.py
│   │   ├── ocr_reader.py
│   │   ├── face_cluster.py
│   │   ├── speaker_attribution.py
│   │   └── reading_order.py
│   ├── web_demo/               ← Streaming web demo
│   ├── configs/
│   ├── scripts/
│   └── docs/
├── vits2_pytorch/              ← VITS2 model (modify)
└── download_vieneu.py
```

---

## Verification
1. **Tone extractor**: Test 100 câu Vietnamese, verify tone labels
2. **VITS2 baseline**: Train small subset, verify loss decreasing
3. **Dual-Path ablation**: Progressive improvement A < B < C < D
4. **Voice Cloning**: Clone held-out speakers, cosine similarity > 0.75
5. **Tone preservation**: Tone accuracy delta < 5% (cloned vs non-cloned)
6. **YOLO**: Bubble detection mAP > 0.8 trên truyện Việt
7. **OCR**: CER < 3% trên truyện Việt (ProtonX VietOCR)
8. **Full pipeline**: Upload comic page → audio output end-to-end
9. **Streaming demo**: Upload chapter → hear audio per character
