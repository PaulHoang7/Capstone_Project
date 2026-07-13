<!-- # Memory — Locked Decisions & Current Status
> Records finalized decisions and current project status.
> Update when a new decision is locked or the project transitions to a new phase.

---

## Locked Decisions

Decisions that are **finalized and should not change** unless there is a very strong reason:

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | Primary model: **VITS2** with **Dual-Path Encoder** | End-to-end TTS, best suited for the main AI contribution | 2026-03-19 |
| 2 | Comparison models: **FastSpeech2 + HiFi-GAN**, **Matcha-TTS** | 3 different paradigms for cross-architecture benchmarking | 2026-03-19 |
| 3 | Primary dataset: **VieNeu-TTS-140h** | Largest available Vietnamese TTS dataset, 193 speakers | 2026-03-19 |
| 4 | AI contribution: **Dual-Path Encoder** (Linguistic + Tonal + Cross-Attention) | Explicit tone representation to reduce hỏi/ngã, sắc/nặng confusion | 2026-03-19 |
| 5 | Deployment target: **ONNX Runtime Web** (browser, client-side) | Offline, no server needed | 2026-03-19 |
| 6 | Storage: **bes-ai-machine-02:/mnt/nfs-data/tin_dataset** | NFS shared storage | 2026-03-19 |
| 7 | AI-first priority — do not prioritize demo over research | This is a graduation research project, not a product | 2026-03-19 |
| 8 | All 3 models train on **same dataset and same splits** | Fair cross-model comparison | 2026-03-19 |

---

## Current Status

- **Current phase**: Setup & Planning
- **Next milestone**: Dataset pipeline with tone extraction
- **Blockers**: _(none)_

---

## Key Dates

| Milestone | Date | Notes |
|-----------|------|-------|
| Project start | 2026-03-19 | CLAUDE.md and rules setup |
| Week 8 checkpoint | _(TBD)_ | Decide whether to proceed with FS2/Matcha or focus on VITS2 evaluation |
| _(add deadlines as they come)_ | | |

---

## Notes

_(Additional notes if needed)_ -->

# Memory — Locked Decisions & Current Status
> Records finalized decisions, current project status, and high-signal learnings.
> Update when: a decision is locked, phase changes, major milestone reached, or important lesson learned.
> Keep first 200 lines high-priority (Claude auto-loads only this part at session start).

Last Updated: 2026-03-22

## Locked Decisions (Final – Do Not Change Without Strong Justification)

| # | Decision | Rationale | Locked Date |
|---|----------|-----------|-------------|
| 1 | Primary TTS model: **VITS2** + **Dual-Path Encoder** (Linguistic + Tonal + Cross-Attention Fusion) | End-to-end, best suited for explicit tone disentanglement in Vietnamese 6-tone system | 2026-03-19 |
| 2 | Comparison models for benchmarking: **FastSpeech2 + HiFi-GAN** và **Matcha-TTS** | Đại diện 3 paradigm khác nhau (non-autoregressive + diffusion) để so sánh fair | 2026-03-19 |
| 3 | Primary dataset: **VieNeu-TTS-140h** (193 speakers, 24kHz) | Dataset lớn nhất tiếng Việt hiện có, đa dạng speaker | 2026-03-19 |
| 4 | Text normalization: **sea-g2p** (MANDATORY, no custom normalizer) | Xử lý số, tiền tệ, từ tiếng Anh, đảm bảo phoneme + tone chính xác | 2026-03-19 |
| 5 | Tone extraction: Deterministic từ diacritics (0–6) | Không phụ thuộc model → reproducible, tránh error propagation | 2026-03-19 |
| 6 | Voice Cloning: **ECAPA-TDNN** + 3-phase training (pretrain → dual cond → zero-shot) | SOTA speaker encoder, hỗ trợ zero-shot với reference ngắn (5–10s) | 2026-03-19 |
| 7 | Deployment target (demo): **ONNX Runtime Web** (browser-side TTS) + server-side CV pipeline | Offline TTS, giảm latency, vẫn dùng GPU server cho heavy inference | 2026-03-19 |
| 8 | Storage path: **bes-ai-machine-02:/mnt/nfs-data/tin_dataset** | NFS shared, dễ access từ nhiều máy | 2026-03-19 |
| 9 | Priority rule: **AI-first** — research contribution (Dual-Path + tone-preserved cloning) quan trọng hơn demo | Đây là đồ án tốt nghiệp, không phải sản phẩm thương mại | 2026-03-19 |
| 10 | All models phải train trên **cùng dataset + cùng splits** | Đảm bảo so sánh công bằng giữa VITS2, FS2, Matcha | 2026-03-19 |

## Current Status (as of 2026-03-22)

- **Current phase**: Setup & Planning → Dataset Pipeline  
- **Progress**:
  - CLAUDE.md, rules/, docs/specs/ đã setup hoàn chỉnh
  - VieNeu-TTS-140h đã download & mount tại /mnt/nfs-data/tin_dataset/raw/
  - sea-g2p installed & test 50 samples: phoneme + tone extraction ổn
  - Manga109 & Vietnamese comic samples (20–30 pages) đã chuẩn bị cho YOLO labeling
- **Next milestone**: Hoàn thành dataset pipeline với tone extraction + shared format (audio_path, text, phonemes, tones, speaker_id, duration) → Week 2 end
- **Blockers**:
  - Chưa có Vietnamese comic annotation cho YOLO fine-tune (dự kiến 100–200 pages, cần 2–3 ngày LabelImg/Roboflow)
  - GPU usage: 1 GPU duy nhất → cần schedule curriculum training (1→10→193 speakers)
- **Pending decisions**:
  - Chọn tool labeling comic: LabelImg hay Roboflow?
  - Có dùng LLM (Gemini/GPT-4o) hỗ trợ speaker attribution ban đầu không?

## Key Dates & Milestones (Locked)

| Milestone | Target Date | Status | Notes |
|-----------|-------------|--------|-------|
| Project start | 2026-03-19 | Done | CLAUDE.md + rules setup |
| Dataset pipeline + tone extraction complete | ~2026-04-02 (Week 2) | In progress | sea-g2p verified, shared format JSON/CSV |
| VITS2 baseline training start | Week 3 (Apr 2026) | Pending | Curriculum 1 speaker first |
| Week 8 checkpoint | ~2026-05-14 | Pending | VITS2 + Dual-Path complete → decide keep FS2/Matcha or focus VITS2 |
| Week 12 checkpoint | ~2026-06-25 | Pending | Voice Cloning complete |

## High-Signal Learnings & Reminders

- **Tone extraction**: Tone 4 (hỏi) và Tone 5 (ngã) dễ lẫn nhất → cần confusion matrix riêng cho ablation.
- **Curriculum training**: Bắt đầu 1 speaker → 10 → full 193 → tránh divergence sớm.
- **Reference audio cho cloning**: Luôn normalize -3dB peak + silence trim → cosine sim tăng đáng kể.
- **VietOCR vs PaddleOCR**: VietOCR tốt hơn dấu thanh điệu, nhưng fallback PaddleOCR nếu font comic lạ.
- **Avoid**: Train full dataset ngay → dễ OOM hoặc loss không hội tụ.

## Topic References (for deeper recall)

- @.claude/rules/experiment-and-research.md → ablation & checklist
- @docs/specs/03-ablation-plan.md → 6 variants VITS2 chi tiết
- @.claude/rules/gotchas.md → bug-specific fixes
- @.claude/rules/model-strategy.md → Dual-Path + ECAPA-TDNN architecture

Last major update: 2026-03-22 (setup phase complete)