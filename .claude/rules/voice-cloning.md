<!-- # Voice Cloning Rules
> ECAPA-TDNN architecture, 3-phase training, clone ablation, tone preservation evaluation. Auto-loaded by Claude Code.

---

## Speaker Encoder: ECAPA-TDNN

### Why ECAPA-TDNN (not GE2E or d-vector)
- SOTA speaker verification, strong on VoxCeleb benchmarks
- Works with short reference audio (3-10s)
- Pre-trained available from SpeechBrain (7000+ speakers)
- ONNX-compatible for browser deployment
- 192-d embedding → project to 256 (gin_channels)

### Architecture
```
Reference mel-spectrogram [B, 80, T]
    │
    Conv1d input layer
    │
    SE-Res2Block x 3 (channels: 512)
    │
    Conv1d + BatchNorm
    │
    Attentive Statistics Pooling
    │
    Linear → speaker embedding [B, 192]
    │
    Projection → [B, 256] (gin_channels)
    │
    Unsqueeze → g [B, 256, 1]
```

---

## 3-Phase Training

### Phase 1: Pre-train Speaker Encoder (3-4 days)
- Start: SpeechBrain ECAPA-TDNN (VoxCeleb pre-trained)
- Fine-tune on VieNeu-TTS 193 speakers
- Freeze all except last 2 SE-Res2Blocks + classification head
- 10-20 epochs
- Validation: EER < 10% on held-out speaker pairs
- Hold out 10-20 speakers for zero-shot evaluation

### Phase 2: Dual Conditioning (2 weeks)
- Start from best VITS2 + Dual-Path checkpoint (week 8)
- Add pre-trained speaker encoder (frozen initially)
- Keep `emb_g` (lookup table) active
- Loss: `L_total = L_recon + L_kl + L_dur + L_adv + λ_spk * L_spk`
  - `L_spk = MSE(emb_g(sid), speaker_encoder(ref_mel))`
- Schedule:
  - 0-50K steps: speaker encoder frozen, λ_spk = 1.0
  - 50K+: unfreeze speaker encoder, λ_spk = 0.5
  - 100K+: random dropout emb_g 50% of batches

### Phase 3: Zero-Shot (1 week)
- Drop `emb_g` entirely — only speaker encoder
- Cross-speaker training: reference utterance ≠ target utterance (same speaker)
- Add speaker similarity loss:
  ```
  spk_ref = speaker_encoder(ref_mel)
  spk_gen = speaker_encoder(generated_mel)
  L_sim = 1 - cosine_similarity(spk_ref, spk_gen).mean()
  ```
- Learning rate: reduce to 1e-5

---

## Data Requirements

### Training
- VieNeu-TTS 193 speakers — sufficient (YourTTS used 109 VCTK speakers)
- DataLoader must sample reference audio: different utterance from same speaker
- Hold out 10-20 speakers entirely for zero-shot evaluation

### Inference
- Reference audio: 5-10s clean speech, 24kHz mono
- Processing: resample, normalize, silence trim → mel-spectrogram → ECAPA-TDNN → embedding

---

## Clone Ablation

| Variant | TTS Model | Speaker Encoder | Tests |
|---------|-----------|----------------|-------|
| Clone-A | VITS2 baseline (no tone) | ECAPA-TDNN | Baseline clone quality |
| Clone-B | VITS2 + Tone Embedding | ECAPA-TDNN | Does tone awareness help cloning? |
| Clone-D | VITS2 + Dual-Path + CrossAttn | ECAPA-TDNN | Does Dual-Path preserve tone during cloning? |

Expected: Clone-D tone accuracy >> Clone-A tone accuracy (proves Dual-Path disentangles tone/speaker)

---

## Evaluation Metrics

### Speaker Similarity
| Metric | Target |
|--------|--------|
| Cosine similarity (ECAPA-TDNN embedding) | > 0.75 |
| Speaker Verification EER (separate model) | < 15% |
| SMOS (Similarity MOS, 1-5) | > 3.5 |

### Tone Preservation (critical)
| Metric | Description |
|--------|-------------|
| Tone Confusion Matrix (cloned) | Same 6x6 protocol, on cloned voices |
| Tone Accuracy Delta | Acc(cloned) - Acc(non-cloned) — target: < 5% degradation |
| Per-tone degradation | Which tones degrade most during cloning? |
| F0 RMSE (cloned) | Pitch accuracy on cloned speech |

---

## Reference audio best practices:
- 5–10s, clean, no background noise/reverb
- Same sample rate 24kHz, mono
- Silence trim (librosa.effects.trim or VAD)
- Normalize to -3 dB peak
- Nếu user upload từ comic demo: warn về chất lượng thấp → similarity drop
## ONNX Export

Two separate ONNX models:
1. `speaker_encoder.onnx` (~10-15MB)
   - Input: `ref_mel [1, 80, T]`
   - Output: `speaker_embedding [1, 192]`

2. `tts_model.onnx` (~80-120MB)
   - Input: `text [1, T]`, `text_lengths [1]`, `scales [3]`, `speaker_embedding [1, 256]`
   - Output: `audio [1, 1, T_audio]`

---

## Integration with Comic Pipeline

```
User uploads comic chapter
    │
    ▼
CV pipeline detects characters + clusters faces
    │
    ▼
User assigns voice per character:
  ├── Upload 5s reference audio per character
  └── OR select from VieNeu-TTS 193 pre-trained speakers
    │
    ▼
Speaker Encoder extracts embedding per character → store in character_db
    │
    ▼
For each bubble:
  text + character's speaker_embedding → TTS → audio
  (speaker_embedding is FIXED per character → voice consistency)
```

Character voices are stored in `character_db` and reused across all pages. -->
