# SPEC — XTTS GPT + CTC Auxiliary Head

**Goal:** Reduce XTTS-FT length-mismatch failures on Vietnamese by adding a CTC alignment loss as an auxiliary head on the GPT decoder. Tone accuracy is already near-saturation (95.53% on aligned subset); the bottleneck is alignment rate (25%).

**Architectural contribution** (defendable as capstone work):
- New module: `CTCAlignmentHead` between GPT hidden states and Vietnamese character logits
- New training objective: `L_total = L_gpt + λ_ctc · L_ctc`
- Inference unchanged (head is auxiliary; can be optionally used for confidence)

---

## 1. Background — Why CTC

XTTS GPT is an autoregressive decoder that predicts audio tokens (DAC codec tokens) from text + speaker conditioning. The text→audio alignment is **implicit**: the model learns to "stop" when audio matches text via the EOS audio token, but has no explicit per-frame supervision tying audio tokens to text characters.

Observation: 75% of VieNeu heldout samples produce wrong syllable count → model hallucinates extra phonemes or drops them.

**CTC fix idea:** add a linear head on GPT hidden states that predicts a Vietnamese **character sequence** (a-z, diacritic marks, tone diacritics, space, blank). Train with CTC loss against the target text. This forces every GPT hidden state to be "aware" of which text position it's currently emitting — which is exactly what alignment requires.

---

## 2. Architecture Changes

### 2.1 New module

```python
class CTCAlignmentHead(nn.Module):
    """Project GPT hidden states → Vietnamese character logits for CTC loss."""
    def __init__(self, gpt_hidden_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(gpt_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(gpt_hidden_dim, vocab_size),
        )

    def forward(self, gpt_hidden: torch.Tensor) -> torch.Tensor:
        """gpt_hidden: [B, T_audio, D] → logits [B, T_audio, V]"""
        return self.proj(gpt_hidden)
```

### 2.2 Where to hook

Place after the last GPT transformer block, BEFORE the audio-token classifier. Reuses the same `hidden_states` tensor.

In coqui_tts XTTS GPT (TTS/tts/layers/xtts/gpt.py):
- `GPT.forward(...)` returns `audio_logits`. The pre-classifier `hidden_states` is what we tap.
- Add `self.ctc_head = CTCAlignmentHead(D=1024, V=64)`.
- Return both `audio_logits, ctc_logits`.

### 2.3 Vocab design

Vietnamese character vocab (~50-60 tokens):
- 26 lowercase Latin letters
- 6 vowels × 5 tone variants (e.g. `a`, `á`, `à`, `ả`, `ã`, `ạ`) — covers tonal diacritics
- Special diacritic chars (`â`, `ă`, `ê`, `ô`, `ơ`, `ư`, `đ`)
- Space + punctuation cluster
- CTC blank token

Total ~64 tokens. Encode target text per-character (NFC normalized, lowercased).

---

## 3. Training Changes

### 3.1 Loss

```python
L_ctc = F.ctc_loss(
    log_probs    = ctc_logits.log_softmax(dim=-1).transpose(0, 1),  # [T, B, V]
    targets      = target_char_ids,                                  # [B, max_target_len]
    input_lengths = audio_token_lengths,                             # [B]
    target_lengths = target_char_lengths,                            # [B]
    blank=0, zero_infinity=True,
)

L_total = L_gpt + λ_ctc * L_ctc
```

### 3.2 Schedule

| Phase | λ_ctc | LR | Notes |
|---|---|---|---|
| Warm-up (0-2k steps) | 0.0 | 1e-5 | CTC head frozen; only GPT trains briefly to recover |
| Main (2k-30k) | 0.3 | 1e-5 | CTC head trains; GPT slightly adapts |
| Final (30k+) | 0.1 | 5e-6 | Reduce CTC weight, let GPT optimize jointly |

Total: ~30-50k steps on VieNeu-140h. Estimate: 3-5 days on 1× A100 / RTX 5090.

### 3.3 Data

Same XTTS FT data pipeline. Add `target_chars` field (char-tokenized target text). Per batch:
- `target_audio_tokens` (existing)
- `target_chars` (new) — for CTC supervision

### 3.4 Modifications to coqui_tts training loop

File: `Capstone_project/voice_cloning/xtts_v2/coqui_tts/recipes/ljspeech/xtts_v2/train_gpt_xtts.py` (or similar). Need to:
1. Build char-vocab + tokenizer (1-2 days)
2. Add `target_chars` to dataset class
3. Modify GPT forward to return `(audio_logits, ctc_logits)`
4. Add CTC loss to trainer step
5. Logging: separate `loss_gpt` and `loss_ctc`

---

## 4. Evaluation Protocol

Reuse `Capstone_project/scripts/eval_tone_accuracy.py` — already produces all numbers we need.

### 4.1 Comparison table (required for defense)

| System | Align rate | Tone Acc (aligned) | Effective Tone Acc (align×acc) |
|---|---|---|---|
| XTTS-FT (baseline, current) | 25.0% | 95.53% | 23.9% |
| XTTS-FT + CTC head (this work) | **target: 60-80%** | target: ≥95% | **target: 57-76%** |
| XTTS-vanilla | 53.3% | 96.18% | 51.3% |
| Gwen-TTS | 78.3% | 87.97% | 68.9% |

**Success criteria (one of):**
- ✅ Align rate ≥60% (2.4× baseline)
- ✅ Effective tone acc ≥50% (2× baseline)
- ✅ Hallucination rate (via Whisper word count delta) ≥50% reduction

### 4.2 Ablation

Train 3 checkpoints with λ_ctc ∈ {0.1, 0.3, 0.5} → pick best on dev set.

### 4.3 Qualitative test

Re-generate Doraemon bubble demo with CTC-trained XTTS → check if Whisper CTC trim is still needed (if alignment is good, trim should cut <5% audio).

---

## 5. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| CTC head doesn't converge (no signal) | Pretrain head 2k steps with GPT frozen first |
| Joint training destabilizes pretrained GPT | Low LR (1e-5), λ_ctc warmup, gradient clipping |
| Vocab choice wrong (composed vs decomposed VN chars) | Test both NFC and NFD; pick the one that aligns to phoneme count |
| 30k steps not enough | Have budget for 50k; eval every 5k |
| Tone accuracy regresses | Monitor per-tone accuracy each eval; if drop >2pp, lower λ_ctc |

---

## 6. Timeline (1-2 weeks effective work)

| Day | Task |
|---|---|
| D1 | VN char vocab + tokenizer + unit tests |
| D2-3 | CTC head module, integrate into XTTS GPT forward, smoke test |
| D4-5 | Training loop modification, 1k-step dry run to verify gradient flow |
| D6-10 | Main training (3-5 days GPU) — λ_ctc ablation in parallel if possible |
| D11-12 | Eval (tone_accuracy.py on 60 heldout) + bubble demo regen |
| D13-14 | Buffer + report |

---

## 7. Defense Narrative

> "Profile XTTS-FT trên VieNeu heldout chỉ ra **tone accuracy đã đạt 95.5% trên aligned subset, nhưng align rate chỉ 25%** — GPT decoder thiếu explicit alignment supervision, gây length mismatch / hallucination. Đóng góp architecture: **CTC auxiliary head** trên GPT hidden states predict Vietnamese character sequence, train cùng main loss. Kỳ vọng align rate tăng 25% → ≥60%. Reproducible với eval_tone_accuracy.py có sẵn."

---

## 8. Code locations to modify

| File | Change |
|---|---|
| `coqui_tts/TTS/tts/layers/xtts/gpt.py` | Add `ctc_head`, return ctc_logits |
| `coqui_tts/TTS/tts/models/xtts.py` | Wire CTC head into training forward |
| NEW: `xtts_v2/vn_char_vocab.py` | Vietnamese character tokenizer |
| NEW: `xtts_v2/train_xtts_ctc.py` | Training script with CTC loss |
| `scripts/eval_tone_accuracy.py` | (no change needed) |

---

## 9. Not in scope (out of capstone)

- Char-vocab from scratch — use deterministic NFC mapping, no learned tokenizer
- Streaming inference changes — eval only offline
- Multi-language: VN only
- Replacing GPT loss with CTC entirely — keep auxiliary only
- Distillation, knowledge transfer
