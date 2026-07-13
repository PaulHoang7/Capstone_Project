
### 4. docs/specs/03-ablation-plan.md
```markdown
# Ablation Study Plan

## TTS Ablation (VITS2 + Dual-Path – Week 5–8)
| Variant | Description | Warm-start | Hypothesis Tested |
|---------|-------------|------------|-------------------|
| A | Baseline VITS2 (single encoder) | - | Lower bound |
| B | + Tone Embedding (concat) | From A | Tone info có giúp? |
| C | Dual-Path no cross-attn (concat late) | - | Separate encoding có lợi? |
| D | Dual-Path + Cross-Attention | From C | Interaction bidirectional giúp? |
| E | D + Auxiliary F0 Loss | From D | Supervise tonal encoder cải thiện pitch? |
| F | E + Tone-Aware Duration Predictor | From E | Duration phụ thuộc tone cải thiện rhythm? |

Metrics: CER/WER, Tone Confusion Matrix (6×6), F0 RMSE, MOS.

## Voice Cloning Ablation (Week 10–12)
| Variant | TTS Base | Speaker Encoder | Tests |
|---------|----------|------------------|-------|
| Clone-A | VITS2 baseline | ECAPA-TDNN | Baseline clone |
| Clone-B | VITS2 + Tone Embedding | ECAPA-TDNN | Tone awareness có giúp clone? |
| Clone-D | VITS2 + Dual-Path full | ECAPA-TDNN | Dual-Path giữ tone khi clone? |

Metrics: Cosine similarity (>0.75), Tone Accuracy Delta (<5%), SMOS (>3.5), per-tone degradation.

Tất cả ablation dùng cùng dataset split, hyperparams, eval set để fair comparison.