# Evaluation Rules
> TTS metrics, Voice Cloning metrics, CV pipeline metrics, evaluation protocol. Auto-loaded by Claude Code.

---

## TTS Quality Metrics

### Objective metrics
- **MOS** (Mean Opinion Score) — subjective naturalness (1-5)
- **MCD** (Mel-Cepstral Distortion) — spectral distance to ground truth
- **F0 RMSE** — pitch accuracy, critical for tones

### Tone-specific metrics
- **Tone Accuracy via STT** — synthesize → Whisper/wav2vec2-vi → compare transcript
- **Tone Confusion Matrix (MANDATORY)** — 6x6 per variant
  - Row = actual tone, Column = predicted tone
  - Must have for ALL ablation variants + clone variants
  - Proves Dual-Path reduces confusion at hỏi/ngã, sắc/nặng

---

## Voice Cloning Metrics

### Speaker similarity
| Metric | Target |
|--------|--------|
| Cosine similarity (ECAPA-TDNN) | > 0.75 |
| Speaker Verification EER | < 15% |
| SMOS (Similarity MOS, 1-5) | > 3.5 |

### Tone preservation (critical for research)
| Metric | Description | Target |
|--------|-------------|--------|
| Tone Confusion Matrix (cloned) | 6x6 on cloned voices | Compare with non-cloned |
| Tone Accuracy Delta | Acc(cloned) - Acc(non-cloned) | < 5% degradation |
| Per-tone degradation | Which tones degrade most | Document all |
| F0 RMSE (cloned) | Pitch accuracy on cloned speech | Compare with non-cloned |

### Clone ablation comparison
| Metric | Clone-A (no tone) | Clone-B (+tone emb) | Clone-D (+Dual-Path) |
|--------|-------------------|---------------------|---------------------|
| Speaker similarity | | | |
| Tone accuracy | | | |
| Tone accuracy delta | | | |
| MOS | | | |

---

## CV Pipeline Metrics

| Component | Metric | Target |
|-----------|--------|--------|
| YOLO bubble detection | mAP@0.5 | > 0.80 |
| YOLO character detection | mAP@0.5 | > 0.75 |
| YOLO panel detection | mAP@0.5 | > 0.85 |
| ProtonX VietOCR | CER (Character Error Rate) | < 3% |
| Face clustering | Cluster purity | > 90% |
| Speaker Attribution (AI) | Accuracy | > 85% |
| Speaker Attribution (rule-based) | Accuracy | > 65% (baseline) |
| Reading order | Order accuracy | > 95% |

---

## TTS Ablation Comparison Table

| Metric | A: Base | B: +Tone | C: +DualPath | D: +CrossAttn | E: +F0Loss | F: +Duration |
|--------|---------|----------|-------------|---------------|-----------|-------------|
| MOS | | | | | | |
| MCD | | | | | | |
| F0 RMSE | | | | | | |
| Tone Accuracy | | | | | | |

Each variant MUST include a 6x6 Tone Confusion Matrix.

---

## End-to-End Evaluation

### Comic Voice-Over quality
- Upload 5 test comic pages (different styles/genres)
- Evaluate: correct text extraction, correct character assignment, natural audio
- Measure: end-to-end accuracy (% bubbles correctly voiced by correct character)

### Processing speed
| Metric | Target |
|--------|--------|
| Time per page | < 5s |
| Time to first audio | < 7s |
| Pre-process ahead buffer | Always ≥ 2 pages ahead |

---

## Custom Evaluation Test Set (200-500 sentences)

### Tone-specific tests
- hỏi/ngã minimal pairs
- sắc/nặng minimal pairs
- All 6 tones in sequence
- Tone sandhi

### Structural tests
- Short (<5 words), long (>20 words)
- Sentences with pauses
- Questions vs statements

### Comic dialogue tests
- Short exclamations ("Đi đi!", "Không!")
- Emotional expressions
- Character-specific speech patterns
- Mixed Vietnamese-English text

---

## Optional: Emotion Extension Evaluation (Week 16+)

### Emotion Detection Accuracy
| Metric | Description | Target |
|--------|-------------|--------|
| Emotion classification accuracy | Face + text + context fusion trên comic test set | > 70% |
| Per-source accuracy | Face-only vs text-only vs context-only | Document all |
| Fusion improvement | Multimodal > single source | Prove fusion helps |

### Emotion TTS Quality
| Metric | Description | Target |
|--------|-------------|--------|
| MOS per emotion | Naturalness rating cho mỗi emotion (1-5) | > 3.5 |
| Emotion recognition rate | Listeners nhận đúng emotion từ audio | > 60% |
| Cross-lingual transfer quality | So sánh English vs Vietnamese emotion TTS | Document gap |

### Tone Preservation khi có Emotion (CRITICAL)
| Metric | Description | Target |
|--------|-------------|--------|
| Tone accuracy delta (emotion vs neutral) | Emotion modulation có phá tone? | < 5% degradation |
| Tone Confusion Matrix (emotional speech) | 6x6 per emotion | Compare with neutral |
| F0 RMSE (emotional vs neutral) | Pitch distortion từ emotion | Document |

### Thách thức đặc biệt cần đánh giá
- hỏi/ngã confusion khi happy (pitch tăng)
- sắc/nặng confusion khi sad (pitch giảm)
- Tone accuracy phải được bảo vệ bởi Dual-Path Tonal Encoder ngay cả khi emotion modulate pitch

---

## Optional: SGG Speaker Attribution Evaluation

### SGG vs MLP Comparison
| Metric | Rule-based | MLP | SGG (GNN) |
|--------|-----------|-----|-----------|
| Accuracy | ~65% | >85% | >90% |
| F1 Score | | | |
| Per-panel accuracy | | | |
| Multi-character scenes | | | |

### SGG-specific metrics
| Metric | Description | Target |
|--------|-------------|--------|
| "speaks" edge prediction accuracy | Correct bubble→character assignment | > 90% |
| F1 score (speaks edges) | Precision + Recall for "speaks" | > 0.88 |
| Multi-character panel accuracy | Accuracy on panels with 3+ characters | > 80% |
| Inference time per page | Graph construction + GNN forward | < 200ms |

---

## Optional: Emotion Intensity Evaluation

### Intensity Estimation Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| MAE (Mean Absolute Error) | Predicted intensity vs ground-truth | < 0.15 |
| Correlation (Pearson) | Predicted vs ground-truth intensity | > 0.7 |
| Listener perception test | Listeners rate if intensity matches expected level | > 65% agreement |

### Intensity-Aware TTS Quality
| Metric | Description | Target |
|--------|-------------|--------|
| Intensity discrimination | Listeners distinguish low vs high intensity | > 70% |
| Naturalness (MOS) per intensity level | Low/mid/high intensity all sound natural | > 3.0 |
| Tone preservation at high intensity | Tone accuracy when intensity > 0.8 (max emotion modulation) | delta < 8% |

---

## Paper Benchmark Reference (Comparison Targets)

> Source: "Emotion-Aware Speech Generation with Character-Specific Voices for Comics" (2025)

### Speaker Attribution (Manga109Dialogue test set)
| Method | Easy | Hard | Total |
|--------|------|------|-------|
| Rule-based (short dist) | 71.4 | 22.7 | 63.4 |
| Rule-based (frame dist) | 81.6 | 22.1 | 71.5 |
| Manga109Speaker (trained DL) | **84.8** | **30.7** | **75.7** |
| Zero-Shot Multimodal (LLM) | 52.4 | 51.3 | 51.8 |
| Paper Setting C (LLM + pred char + emo) | 79.2 | 20.5 | 64.8 |

**Key insight**: Trained DL model (75.7%) > LLM (64.8%) > Rule-based (63.4%). Hard cases (bubble xa speaker) là bottleneck — chỉ 20-30%. SGG/GNN có thể giải quyết nhờ explicit spatial relationships.

**Target project**: Total >85% (vượt SOTA 75.7%), Hard >50% (gấp đôi paper).

### Emotion Classification (5-way, paper Setting C)
| Label | Precision | Recall | F1 | #Support |
|-------|-----------|--------|----|----------|
| Neutral | 56.6 | 34.6 | 42.6 | 159 |
| Surprise | 15.7 | 52.6 | 24.2 | 38 |
| Anger | 42.2 | 47.9 | 44.9 | 73 |
| Happiness | 76.8 | 39.9 | 52.5 | 158 |
| Sadness | 47.1 | 50.0 | 48.5 | 48 |
| **Macro avg** | **47.5** | **45.0** | **42.9** | 476 |

**Key insights**:
- Surprise over-predicted cho câu hỏi — nhưng perceptually acceptable cho TTS
- Neutral recall rất thấp (34.6%) — biểu cảm subtle bị misclassify
- Emotion trong comic là subjective — accuracy metric underestimates true quality
- **Perceptual evaluation quan trọng hơn accuracy metric**

**Target project** (optional emotion): F1 ~40-45% (realistic), perceptual acceptability >70%.

---

## Perceptual Evaluation Protocol (từ paper insights)

### Tại sao cần perceptual evaluation
Paper chứng minh: emotion accuracy ~41% nhưng audio vẫn "perceptually acceptable" vì:
- Surprise voice cho câu hỏi = tự nhiên (dù ground truth là neutral)
- Emotion trong comic vốn subjective — nhiều label đều hợp lý
- Accuracy metric không phản ánh đúng chất lượng thực tế

### Perceptual Test Design
```
Cho listener nghe audio + xem trang truyện tương ứng:

Q1: "Giọng nói có phù hợp với nhân vật này không?" (1-5 scale)
Q2: "Cảm xúc trong giọng có phù hợp với context không?" (1-5 scale)
Q3: "Giọng có tự nhiên không?" (1-5 scale)

Target: Average score > 3.5/5.0
```

### AB Testing cho Emotion
```
A: TTS không có emotion (neutral voice)
B: TTS có emotion (detected emotion applied)

Listener chọn: "Audio nào phù hợp hơn với trang truyện?"
Target: B được chọn > 60% (chứng minh emotion có giá trị)
```

---

## Evaluation Discipline

- Do not conclude improvement based on 1-2 samples
- Always include: metrics table + test set + qualitative comments
- Report negative results honestly
- Per-tone breakdown is mandatory
- Clone evaluation must compare with non-cloned baseline
- CV pipeline evaluation on Vietnamese comics (not just Manga109)
- **Perceptual evaluation bắt buộc cho emotion** — accuracy thấp không có nghĩa là kết quả tệ
- **Cite paper benchmarks** khi so sánh kết quả
