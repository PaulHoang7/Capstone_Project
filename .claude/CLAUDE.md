<!-- # CLAUDE.md

## Project
Comic Voice-Over System — Vietnamese TTS with Voice Cloning for Comic Characters

## Working Title
**EN**: Automatic Comic Voice-Over System with Tone-Aware Vietnamese TTS and Zero-Shot Voice Cloning
**VN**: Hệ thống tự động lồng tiếng truyện tranh với tổng hợp giọng nói tiếng Việt nhận biết thanh điệu và nhân bản giọng nói không cần mẫu huấn luyện

---

## Project Identity
This is an **AI-first Comic Voice-Over project** that automatically generates voiced dialogue for comic/manga pages, with each character having a unique cloned voice. The primary AI contributions are:
1. **Dual-Path Encoder** for VITS2 — separates linguistic and tonal representation for accurate Vietnamese tone synthesis
2. **Tone-Preserved Voice Cloning** — zero-shot voice cloning that maintains correct Vietnamese 6-tone prosody
3. **Speaker Attribution Model** — AI-based assignment of speech bubbles to characters

The CV pipeline (bubble detection, OCR, face clustering) uses pre-trained/fine-tuned models. The web demo is a controlled extension — do not let it shift the research focus.

---

## 1. Project Goals
Build a system that automatically generates voiced audio from comic pages:
1. **TTS quality**: Reduce tone and prosody errors via Dual-Path Encoder (explicit tone representation)
2. **Voice Cloning**: Clone unique voices per character while preserving Vietnamese 6-tone accuracy
3. **Comic Pipeline**: Automatically detect speech bubbles, read text, identify characters, and assign dialogue
4. **Demo**: Upload a comic chapter (10-15 pages) → hear each character speak with a unique voice

This is an **AI-first research project** — the comic application showcases the AI contributions.

---

## 2. Core Direction

### AI Contributions (self-built)

#### A. TTS + Dual-Path Encoder (core)
- Primary model: **VITS2 + Dual-Path Encoder** (end-to-end, VAE + flow + GAN)
- **Dual-Path Encoder**: separate Linguistic Encoder + Tonal Encoder + Cross-Attention Fusion
- Vietnamese Tone Extraction Pipeline
- Ablation study (6 variants A-F)

#### B. Voice Cloning (core)
- **ECAPA-TDNN** speaker encoder (SpeechBrain pre-trained → fine-tune VieNeu-TTS)
- 3-phase training: pre-train encoder → dual conditioning → zero-shot
- Tone preservation evaluation: prove Dual-Path disentangles tone from speaker
- Clone ablation: Clone-A (no tone) vs Clone-B (tone emb) vs Clone-D (Dual-Path)

#### C. Speaker Attribution (core)
- AI model to assign speech bubbles to characters (replaces rule-based proximity)
- Input: bubble position, character positions, tail direction, panel layout
- Trained on Manga109 annotated data

### Pre-trained/Fine-tuned Tools (not core AI)
- **YOLOv8**: fine-tune for bubble/character/panel detection (Manga109 dataset)
- **ProtonX VietOCR**: Vietnamese text recognition — chuyên biệt tiếng Việt, Transformer-based, accuracy cao hơn PaddleOCR generic
- **PaddleOCR**: text detection (bounding box) — VietOCR handles recognition
- **ArcFace**: face embedding for character clustering (pre-trained)
- **Reading order**: rule-based sort by coordinates (LTR/RTL config)

### Optional Extensions (only if time permits)
- **Emotion-Controlled TTS**: train on ESD dataset (English, 29h, 5 emotions) → cross-lingual emotion transfer to Vietnamese. VieNeu-TTS là newspaper reading (~95% neutral) → KHÔNG train emotion trên dataset này.
- **Scene Graph Generation (SGG)**: graph-based Speaker Attribution — YOLO output → nodes (characters, bubbles, panels) + spatial edges → GNN (GAT/GCN) predict "speaks" edges. Upgrade từ MLP nếu accuracy chưa đạt target >85%.
- **Emotion Intensity Estimation**: regression head (0.0→1.0) trên emotion detection — "angry 0.3" (hơi bực) vs "angry 0.9" (phẫn nộ) → TTS điều chỉnh pitch/speed/energy theo mức độ. Phụ thuộc Emotion-Controlled TTS hoàn thành trước.
- **BLIP/LLaVA**: scene understanding for context/emotion detection (tốn VRAM, chỉ thêm nếu có giá trị)
- **VoiceDesign**: text-described voice assignment — user mô tả giọng bằng text ("giọng nam trầm, nghiêm túc") thay vì upload audio. Match description với VieNeu-TTS 193 speaker profiles. Alternative cho Voice Clone khi không có reference audio.
- **LLM Speaker Attribution baseline**: dùng GPT-4o/Llama-3.1 làm baseline so sánh accuracy với self-trained model (KHÔNG thay thế AI contribution)
- Tone-aware prosody supervision (auxiliary F0 loss)
- Region-aware embedding (North/Central/South)

### Why Dual-Path Encoder (architectural justification)
Vietnamese has 6 tones — the same phoneme `/ma/` with different tones produces completely different meanings (ma, má, mà, mả, mã, mạ). In standard VITS2, phoneme and tone information are mixed into a single embedding. The model must implicitly learn to disentangle them → leads to tone confusion (especially hỏi/ngã, sắc/nặng).

**Hypothesis**: Separating linguistic and tonal representations into two dedicated encoder paths, then fusing via cross-attention, will:
1. Reduce tone errors compared to a single mixed encoder
2. Enable better voice cloning — because tone and speaker are processed separately, swapping speaker identity won't corrupt tone accuracy

---

## 3. Scope

### In scope
- Vietnamese TTS dataset pipeline with tone extraction (VieNeu-TTS 140h)
- VITS2 + Dual-Path Encoder with ablation study (6 variants)
- Voice Cloning (ECAPA-TDNN) with tone preservation evaluation
- CV pipeline: YOLO fine-tune (bubble/character/panel), PaddleOCR, ArcFace clustering
- Speaker Attribution model (AI-based bubble-character assignment)
- Streaming web demo: upload comic chapter → audio per character
- Character database persistence across pages (consistent voices)
- Evaluation: TTS metrics + Voice Cloning metrics + CV pipeline metrics

### Out of scope
- FastSpeech2 / Matcha-TTS / 3-model benchmark (removed — focus depth over breadth)
- Full production-grade voice cloning with fine-tuning per speaker
- Fully automatic emotion TTS from Vietnamese dataset (VieNeu-TTS lacks emotion labels)
- Real-time conversation / voicebot
- Mobile app deployment
- Full auto CV without any manual review step

---

## 4. Project Success Criteria

### Minimum success (TTS + Voice Cloning, no CV)
- Functional dataset pipeline with tone extraction
- Stable VITS2 + Dual-Path Encoder with at least 2 ablation variants
- Voice Cloning working (zero-shot, clone unseen speakers)
- Tone preservation evaluation (confusion matrix: cloned vs non-cloned)
- Simple demo: user types text + uploads voice → audio in cloned voice
- Evaluation with metrics and qualitative comments

### Good success (+ CV pipeline)
- All above +
- YOLO detects bubbles/characters/panels on Vietnamese comics
- PaddleOCR reads Vietnamese text from bubbles
- Face clustering groups same character across pages
- Rule-based or AI speaker attribution
- Demo: upload comic page → audio per character

### Excellent success (full system)
- All above +
- Speaker Attribution AI model (not just rule-based)
- Streaming web demo: upload 10-15 page chapter → hear entire chapter
- Character voices consistent across all pages
- Pre-process ahead for real-time feel
- Clone ablation proving Dual-Path helps tone preservation
- Well-designed evaluation covering TTS + Voice Cloning + CV metrics
- (Optional) Emotion extension with ESD English dataset

---

## 5. Final Priority Order
When choosing tasks, prioritize in this order:
1. Clean dataset with tone extraction pipeline (VieNeu-TTS)
2. Stable baseline VITS2
3. Dual-Path Encoder (main AI contribution)
4. Ablation study for VITS2 (6 variants)
5. Voice Cloning (ECAPA-TDNN, 3-phase training)
6. Voice Cloning evaluation (tone preservation, clone ablation)
7. CV pipeline setup (YOLO fine-tune, PaddleOCR, ArcFace)
8. Speaker Attribution model
9. Pipeline integration (comic page → audio)
10. Streaming web demo
11. Report + slides
12. (Optional) Emotion extension (ESD English)

---

## 6. Timeline and Constraints
- **Total time**: 16 weeks (starting 2026-03-19)
- **Hardware**: 1 GPU
- **Strategy**: TTS + Dual-Path must complete by week 8. Voice Cloning weeks 9-12. CV pipeline + demo weeks 12-14.

### Checkpoint (week 8)
At week 8, VITS2 + Dual-Path must be complete. Evaluate:
- **On track**: proceed to Voice Cloning + CV pipeline (full plan)
- **Slightly behind**: proceed to Voice Cloning, simplify CV (rule-based only)
- **Very behind**: skip Voice Cloning, focus on evaluation + simple demo

### Checkpoint (week 12)
At week 12, Voice Cloning must be complete. Evaluate:
- **On track**: build full CV pipeline + Speaker Attribution + streaming demo
- **Slightly behind**: use rule-based assignment instead of AI Speaker Attribution
- **Very behind**: skip CV pipeline, demo with manual text input

---

## Storage Rule
All downloaded datasets, processed subsets, checkpoints, logs, results, and related files must be placed in:
**bes-ai-machine-02:/mnt/nfs-data/tin_dataset**

---

## Modular Rules
Detailed operational guidelines are modularized in `.claude/rules/`:

| File | Covers |
|------|--------|
| `rules/dataset.md` | Dataset strategy (VieNeu-TTS + Manga109), tone extraction, shared data format |
| `rules/model-strategy.md` | Dual-Path Encoder design, Voice Cloning architecture, Speaker Attribution, ablation |
| `rules/experiment-and-research.md` | Experiment philosophy, ablation design, logging rules, verification checklists |
| `rules/evaluation.md` | TTS metrics, Voice Cloning metrics, CV pipeline metrics, evaluation protocol |
| `rules/deployment.md` | Streaming web demo, server-side CV, pre-process ahead strategy |
| `rules/cv-pipeline.md` | YOLO, OCR, face clustering, speaker attribution, reading order, character_db |
| `rules/voice-cloning.md` | ECAPA-TDNN, 3-phase training, clone ablation, tone preservation |
| `rules/project-artifacts.md` | Required deliverables for Comic Voice-Over system |
| `rules/gotchas.md` | Lessons learned, mistakes to avoid, debugging notes |

These files are automatically loaded by Claude Code when working in this project. -->

# CLAUDE.md

## Project Identity
Comic Voice-Over System — Vietnamese TTS with Tone-Preserved Zero-Shot Voice Cloning for Comic Characters  
Hệ thống tự động lồng tiếng truyện tranh với TTS tiếng Việt nhận biết thanh điệu và nhân bản giọng nói zero-shot.

**Core AI contributions (self-built)**:
1. VITS2 + Dual-Path Encoder (Linguistic + Tonal paths + Cross-Attention) → disentangle tone for Vietnamese 6-tone accuracy.
2. Tone-Preserved Voice Cloning (ECAPA-TDNN + 3-phase training).
3. AI-based Speaker Attribution (bubble → character assignment).

This is an **AI-first research project**. Comic pipeline (YOLO, PaddleOCR, ArcFace) is the application showcase.

## Goals & Scope Summary
- Build end-to-end: comic page → voiced audio with character-specific cloned voices.
- Hypothesis: Dual-Path Encoder preserves tone accuracy during voice cloning.
- In-scope: TTS + Cloning + CV pipeline + streaming demo.
- Out-of-scope: full emotion TTS from VN data, mobile app, real-time bot.

Xem chi tiết: docs/specs/00-overview.md

## Priority Order (strict)
1. Clean dataset + tone extraction pipeline (VieNeu-TTS)
2. Baseline VITS2
3. Dual-Path Encoder + ablation (6 variants)
4. Voice Cloning (ECAPA-TDNN, tone preservation eval)
5. CV pipeline (YOLO fine-tune, clustering, reading order)
6. Speaker Attribution model
7. Full integration & streaming demo
8. Report + slides

## Timeline Checkpoints
- Week 8: VITS2 + Dual-Path complete & evaluated
- Week 12: Voice Cloning complete
Xem chi tiết: docs/specs/04-timeline-milestones.md

## Key Rules & Conventions
All detailed guidelines are in .claude/rules/*.md — Claude loads them automatically.

- Dataset & tone extraction: dataset.md
- Model & cloning strategy: model-strategy.md
- CV pipeline: cv-pipeline.md
- Experiment philosophy & checklists: experiment-and-research.md
- Deliverables & success criteria: project-artifacts.md
- Gotchas (bugs >30min): gotchas.md
- Deployment: deployment.md

Storage: All data/checkpoints → bes-ai-machine-02:/mnt/nfs-data/tin_dataset

When generating code/experiments:
- Modular, type-hinted, well-logged
- Use experiment tracking (wandb or simple folders)
- Always reference rules/experiment-and-research.md & gotchas.md