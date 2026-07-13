# Experiment Philosophy and Research Style
> Experiment methodology, ablation design, logging rules, verification checklists. Auto-loaded by Claude Code.

---

## Experiment Philosophy

Every change must be tested with:
- Clear hypothesis
- Baseline
- Metrics
- Audio samples for listening
- Decision to keep or discard

---

## Required Experiments

### TTS Ablation (tuần 3-8)
| Variant | Hypothesis |
|---------|-----------|
| A: VITS2 Baseline | Lower bound |
| B: + Tone Embedding | Tone info helps Vietnamese TTS |
| C: + Dual-Path (no cross-attn) | Separate encoding helps |
| D: + Cross-Attention | Interaction before fusion helps |
| E: + F0 Loss | Supervising Tonal Encoder improves pitch |
| F: + Tone-Aware Duration | Duration conditioning on tone improves rhythm |

### Voice Cloning (tuần 9-12)
| Experiment | Hypothesis |
|-----------|-----------|
| Phase 1: Speaker encoder EER | ECAPA-TDNN fine-tuned < 10% EER |
| Phase 2: Dual conditioning | L_spk decreases, audio quality maintained |
| Phase 3: Zero-shot | Clone unseen speakers with cosine sim > 0.75 |
| Clone-A vs Clone-D | Dual-Path preserves tone better during cloning |

### CV Pipeline (tuần 12-13)
| Experiment | Target | Paper Reference |
|-----------|--------|----------------|
| YOLO bubble detection | mAP > 0.8 on Vietnamese comics | — |
| ProtonX VietOCR accuracy | CER < 3% on Vietnamese text | — |
| Face clustering | Correctly group same character > 90% | Paper: ResNet-50 per-title = 62.9% |
| Speaker Attribution (MLP) | Accuracy > 85% (vs rule-based ~65%) | Paper SOTA: 75.7% (Manga109Speaker) |
| Speaker Attribution vs baselines | Self-trained > Rule-based (71.5%) > LLM (64.8%) | Paper: LLM thua trained DL model |

### Comparison with Paper SOTA (required in report)
| Component | Paper SOTA | Paper LLM | Project Target | Method |
|-----------|-----------|-----------|----------------|--------|
| Speaker (total) | 75.7% | 64.8% | >85% | MLP/SGG |
| Speaker (hard) | 30.7% | 20.5% | >50% | SGG spatial reasoning |
| Character ID | 62.9% | — | >90% cluster purity | ArcFace (vs ResNet-50) |
| Emotion F1 | 42.9% | — | ~40-45% | Visual + text (optional) |

### End-to-End (tuần 14)
| Experiment | Target |
|-----------|--------|
| Full pipeline | Comic page → correct audio for each character |
| Tone preservation in pipeline | Tone accuracy comparable to standalone TTS |
| Processing speed | < 5s per page |

---

## Experiment Progression

```
Phase 1 (week 1-2):   Data pipeline + tone extraction + CV tools setup
Phase 2 (week 3-4):   VITS2 baseline (curriculum: 1 → 10 → 193 speakers)
Phase 3 (week 5-6):   VITS2 ablation B + C + D
Phase 4 (week 7-8):   VITS2 ablation E + F + evaluation → CHECKPOINT 1
Phase 5 (week 9):     ECAPA-TDNN pre-train + integrate
Phase 6 (week 10-11): Voice Cloning Phase 2 + 3
Phase 7 (week 12):    Voice Cloning eval + Clone ablation + Speaker Attribution
                      → CHECKPOINT 2
Phase 8 (week 13):    CV pipeline integration
Phase 9 (week 14):    Streaming web demo
Phase 10 (week 15):   Report + slides
Phase 11 (week 16):   Buffer / emotion extension
```

---

## Ablation Study Rules

### TTS ablation
- **Same dataset** (VieNeu-TTS-140h, same split) for all variants
- **Same hyperparameters** (lr, batch size, steps)
- **Same evaluation set** (custom tone test set)
- 6x6 Tone Confusion Matrix for every variant
- Progressive: A < B < C < D expected. Document if not.

### Clone ablation
- All 3 variants (Clone-A/B/D) use same speaker encoder
- Compare: speaker similarity (should be similar) vs tone accuracy (Clone-D should be best)
- Hold out 10-20 speakers for zero-shot evaluation

---

## Logging Rules

Every experiment must log:
- Config (hyperparameters, architecture variant)
- Data used (dataset, split, subset size)
- Checkpoint path + step number
- All objective metrics
- Audio samples (at least 5 per experiment)
- Comments (what worked, what didn't, why)

Storage: `bes-ai-machine-02:/mnt/nfs-data/tin_dataset/experiments/`

---

## Research Style

### Good behavior
- Justify every architectural decision with hypothesis
- Run ablation before claiming improvement
- Log everything
- Distinguish core (Dual-Path + Voice Cloning) vs supporting (CV pipeline)
- Track time spent vs remaining — adjust scope early
- Report negative results honestly

### Bad behavior
- Replace components without justification
- Claim improvement from 1-2 cherry-picked samples
- Add modules without ablation
- Spend too long on CV pipeline before TTS is solid
- Forget to log results

---

## Verification Checklists

### Dataset Phase (week 1-2)
- [ ] VieNeu-TTS audio loads without errors, 24kHz
- [ ] sea-g2p text normalization verified (50+ samples)
- [ ] Tone extraction produces correct sequences
- [ ] Train/val/test splits created
- [ ] YOLO fine-tuned on Manga109, tested on Vietnamese comics
- [ ] PaddleOCR tested on Vietnamese comic text

### VITS2 Baseline Phase (week 3-4)
- [ ] Loss decreasing steadily
- [ ] Synthesized audio is intelligible Vietnamese
- [ ] Baseline metrics recorded
- [ ] Checkpoint saved to NFS

### Dual-Path Encoder Phase (week 5-8)
- [ ] All 6 variants trained with same conditions
- [ ] Tone Confusion Matrix for every variant
- [ ] Progressive improvement documented
- [ ] Audio samples saved

### Voice Cloning Phase (week 9-12)
- [ ] ECAPA-TDNN EER < 10% on VieNeu-TTS
- [ ] Phase 2: audio quality maintained, L_spk decreasing
- [ ] Phase 3: clone unseen speakers successfully
- [ ] Clone ablation: Clone-D tone accuracy > Clone-A
- [ ] Tone preservation delta < 5%

### CV Pipeline Phase (week 12-13)
- [ ] YOLO mAP > 0.8 on Vietnamese comics
- [ ] OCR CER < 5%
- [ ] Face clustering groups same character > 90%
- [ ] Speaker Attribution accuracy > 85% (or rule-based fallback working)
- [ ] Reading order correct for LTR comics

### Integration Phase (week 14)
- [ ] Upload comic page → audio output end-to-end
- [ ] Character voices consistent across pages
- [ ] Streaming demo working
- [ ] Pre-process ahead for real-time feel
