<!-- # Model Strategy and Architecture
> Covers Dual-Path Encoder design, Voice Cloning architecture, Speaker Attribution, and ablation study. Auto-loaded by Claude Code.

---

## System Architecture

```
Comic Page → [CV Pipeline] → text + character_id → [TTS + Voice Cloning] → audio per character
```

Two independent AI systems:
1. **TTS + Voice Cloning** — generates speech with correct tone + unique voice per character
2. **Speaker Attribution** — assigns bubbles to characters using AI

---

## AI Contribution 1: Dual-Path Encoder for VITS2

### The problem
Vietnamese has 6 tones. Same phoneme `/ma/` with different tones = different meaning (ma/má/mà/mả/mã/mạ). Standard VITS2 mixes phoneme and tone into a single embedding → model must implicitly disentangle → tone confusion (hỏi/ngã, sắc/nặng).

### The solution: Dual-Path Encoder
Replace single `TextEncoder` with:

1. **Linguistic Encoder** (Transformer x N) — processes phoneme sequence
   - Learns articulation patterns, coarticulation
   - Focus: **what** is being said

2. **Tonal Encoder** (Transformer x M) — processes tone sequence
   - Learns F0 contour shapes for 6 Vietnamese tones
   - Learns tone-to-tone transitions
   - Focus: **how** it should sound (pitch pattern)

3. **Cross-Attention Fusion** — bidirectional
   - L→T: phoneme context influences tone realization
   - T→L: tone context influences phoneme duration/energy
   - Concatenate + linear projection → fused representation

### Why this design
- **Why not just tone embedding?** Too shallow — model still processes mixed representation
- **Why not concatenate?** No interaction before fusion — tone realization depends on phoneme context
- **Why cross-attention?** Allows controlled interaction — coarticulation modeled explicitly

### Ablation study (6 variants)

| Variant | Tests | Warmstart |
|---------|-------|-----------|
| A: Baseline | Lower bound — no tone awareness | No |
| B: + Tone Embedding | Does tone info help? | From A |
| C: + Dual-Path (no cross-attn) | Does separate encoding help? | No (arch change) |
| D: + Cross-Attention | Does cross-attention help? | From C |
| E: + Auxiliary F0 Loss | Does supervising Tonal Encoder help? | From D |
| F: + Tone-Aware Duration | Full system with duration conditioning | From E |

Expected: F > E > D > C > B > A (each component contributes)

### Additional AI contributions (included in plan)

**Auxiliary F0 Contour Loss (Variant E)**
- Linear head on Tonal Encoder → predict F0 contour
- Supervise with ground-truth F0 (PyWorld DIO)
- Loss: L1, weighted 0.1-0.5 in total loss

**Tone-Aware Duration (Variant F)**
- Condition Duration Predictor on tone label
- Vietnamese tones have different natural durations (ngã longest, sắc shortest)

**Curriculum Training**
- Stage 1: 1 speaker (~5-10h) → basic mapping
- Stage 2: 10 speakers (~30-40h) → multi-speaker
- Stage 3: 193 speakers (140h) → generalize

---

## AI Contribution 2: Voice Cloning (ECAPA-TDNN)

### Architecture
```
Reference Audio (5-10s) → Mel-spectrogram
    │
    ▼
ECAPA-TDNN (SE-Res2Block x3 + Attentive Statistics Pooling)
    │
    ▼
Speaker Embedding [192]
    │
    ▼
Linear Projection [192 → 256] (match gin_channels)
    │
    ▼
g [b, 256, 1] → injected at 5 points in VITS2
```

### Injection points (same as existing speaker conditioning)
1. TextEncoder (Dual-Path) at `cond_layer_idx`
2. PosteriorEncoder (WaveNet conditioning)
3. Flow (gin_channels)
4. Duration Predictor (gin_channels)
5. Decoder (cond layer)

### Training strategy (3 phases)

**Phase 1: Pre-train Speaker Encoder (3-4 days)**
- Fine-tune SpeechBrain ECAPA-TDNN on VieNeu-TTS 193 speakers
- Freeze all except last 2 SE-Res2Blocks + classification head
- Target: EER < 10%

**Phase 2: Multi-speaker with dual conditioning (2 weeks)**
- Start from best VITS2 + Dual-Path checkpoint
- Keep both `emb_g` (lookup) and `speaker_encoder`
- `L_spk = MSE(emb_g(sid), speaker_encoder(ref_mel))`
- After 50K steps: unfreeze speaker encoder
- After 100K steps: random dropout emb_g 50%

**Phase 3: Zero-shot fine-tuning (1 week)**
- Drop `emb_g` entirely
- Cross-speaker training: reference ≠ target utterance (same speaker)
- Add cosine speaker similarity loss

### Integration with Dual-Path Encoder
```
Linguistic Encoder → content (what)     ← independent of speaker
Tonal Encoder → tone (pitch pattern)     ← independent of speaker
Speaker Encoder → identity (whose voice) ← independent of content/tone

Cross-Attention fuses content + tone FIRST
Then speaker conditioning injected at cond_layer_idx
→ Swapping speaker does NOT corrupt tone
```

**Hypothesis**: Dual-Path separates tone from speaker → clone voice without tone degradation.

### Clone ablation (3 variants)

| Variant | Description |
|---------|-------------|
| Clone-A | VITS2 baseline + ECAPA (no tone awareness) |
| Clone-B | VITS2 + Tone Embedding + ECAPA |
| Clone-D | VITS2 + Dual-Path + CrossAttn + ECAPA (full) |

Compare: speaker similarity (should be similar) vs tone accuracy (Clone-D should be best).

---

## AI Contribution 3: Speaker Attribution

### Problem
Given detected bubbles and characters on a comic page, which character speaks which bubble?

### Architecture
```
Input features:
  - Bubble center (x, y)
  - Nearest character center (x, y)
  - Distance bubble → each character
  - Tail direction (if detected)
  - Panel ID (which panel contains this bubble)
  - Relative position within panel

Model: MLP or lightweight GNN
Output: character_id for each bubble
```

### Training data
- Manga109 has ground-truth annotation: bubble → character mapping
- Fine-tune on 100-200 annotated Vietnamese comic pages

### Fallback
- If AI model doesn't reach acceptable accuracy → use rule-based (proximity + tail detection)
- Rule-based is always available as baseline for comparison

---

## Files to modify for Dual-Path Encoder

| File | Change |
|------|--------|
| `vits2_pytorch/models.py` | Replace `TextEncoder` → `DualPathTextEncoder`, modify `SynthesizerTrn` |
| `vits2_pytorch/attentions.py` | Add `TonalEncoder`, `CrossAttentionFusion` |
| `vits2_pytorch/data_utils.py` | Extend loader for tone sequence + reference audio |
| `vits2_pytorch/train_ms.py` | Modify batch unpacking, add L_spk loss |
| `vits2_pytorch/export_onnx.py` | Export speaker encoder separately |
| `vits2_pytorch/configs/` | Add dual-path + voice-cloning params |

### New files to create

| File | Purpose |
|------|---------|
| `Capstone_project/voice_cloning/speaker_encoder.py` | ECAPA-TDNN wrapper |
| `Capstone_project/voice_cloning/speaker_encoder_train.py` | Pre-training script |
| `Capstone_project/voice_cloning/cloning_eval.py` | Speaker similarity + tone preservation |
| `Capstone_project/cv_pipeline/speaker_attribution.py` | AI bubble→character assignment |

---

## Optional: Emotion Extension

- **NOT in core plan** — only if time permits after week 15
- VieNeu-TTS is newspaper reading → 95% neutral → cannot train emotion on this
- Use **ESD dataset (English, 29h, 5 emotions)** to learn emotion styles
- Hypothesis: emotion style (speed/pitch/energy patterns) transfers cross-lingually
- Implementation: GST (Global Style Tokens) or emotion embedding -->


# Model Strategy and Architecture
> Covers Dual-Path Encoder design, Voice Cloning architecture (ECAPA-TDNN + 3-phase training), Speaker Attribution, ablation study, tone preservation evaluation, and integration details. Auto-loaded by Claude Code.

---

## System Architecture Overview

Comic Page → [CV Pipeline] → text + bubble bbox + character_id → [TTS + Voice Cloning] → audio per character

Hai hệ thống AI độc lập chính:
1. **TTS + Voice Cloning** — sinh giọng nói đúng thanh điệu + giọng riêng cho từng nhân vật.
2. **Speaker Attribution** — gán bong bóng thoại cho nhân vật (AI-based, fallback rule-based).

---

## AI Contribution 1: Dual-Path Encoder for VITS2

### Vấn đề
Tiếng Việt có 6 thanh điệu (ngang, sắc, huyền, hỏi, ngã, nặng). Cùng âm vị `/ma/` với tone khác nhau → nghĩa khác nhau hoàn toàn.  
Standard VITS2 mix phoneme + tone vào một embedding → model phải học implicit disentanglement → dễ tone confusion (hỏi/ngã lẫn, sắc/nặng lẫn).

### Giải pháp: Dual-Path Encoder
Thay TextEncoder bằng:

1. **Linguistic Encoder** (Transformer x N layers)  
   - Input: phoneme sequence  
   - Focus: articulation, coarticulation, nội dung nói (what is said)  

2. **Tonal Encoder** (Transformer x M layers)  
   - Input: tone sequence (0–6)  
   - Focus: F0 contour shapes, tone transitions, cách nói (how it sounds)  

3. **Cross-Attention Fusion** (bidirectional)  
   - L→T: phoneme context ảnh hưởng tone realization  
   - T→L: tone context ảnh hưởng phoneme duration/energy  
   - Output: fused representation → feed vào posterior encoder, duration predictor, decoder  

### Các thành phần bổ sung
- **Auxiliary F0 Contour Loss** (Variant E): Linear head trên Tonal Encoder → predict F0 (PyWorld DIO) → L1 loss weighted 0.1–0.5.
- **Tone-Aware Duration Predictor** (Variant F): Condition duration predictor bằng tone label (tone ngã dài hơn, sắc ngắn hơn).

### Curriculum Training
- Stage 1: 1 speaker (~5–10h) → basic mapping  
- Stage 2: 10 speakers (~30–40h) → multi-speaker  
- Stage 3: 193 speakers (140h) → generalize

### Ablation study (6 variants)
| Variant | Description                              | Warm-start | Hypothesis Tested                          |
|---------|------------------------------------------|------------|--------------------------------------------|
| A       | Baseline VITS2 (single encoder)          | -          | Lower bound                                |
| B       | + Tone Embedding (concat)                | From A     | Tone info có giúp?                         |
| C       | Dual-Path no cross-attn (concat late)    | -          | Separate encoding có lợi?                  |
| D       | Dual-Path + Cross-Attention              | From C     | Interaction bidirectional giúp?            |
| E       | D + Auxiliary F0 Loss                    | From D     | Supervise tonal encoder cải thiện pitch?   |
| F       | E + Tone-Aware Duration                  | From E     | Duration phụ thuộc tone cải thiện rhythm?  |

Expected: F > E > D > C > B > A  
Metrics: Tone Confusion Matrix (6×6), F0 RMSE, CER/WER, MOS.

---

## AI Contribution 2: Tone-Preserved Zero-Shot Voice Cloning (ECAPA-TDNN)

### Tại sao ECAPA-TDNN
- SOTA speaker verification (VoxCeleb)  
- Hoạt động tốt với reference ngắn (3–10s)  
- Pre-trained SpeechBrain (7000+ speakers)  
- Embedding 192 → project to 256 (match gin_channels)  
- ONNX-compatible cho browser (nếu cần)

### Architecture
```
Reference Audio (5–10s) → Mel-spectrogram [B, 80, T]
│
Conv1d → SE-Res2Block x3 (channels 512) → Attentive Statistics Pooling
│
Linear → Speaker Embedding [B, 192]
│
Projection → [B, 256] → Unsqueeze → g [B, 256, 1]
│
Inject g tại 5 điểm trong VITS2:

TextEncoder (Dual-Path) tại cond_layer_idx
PosteriorEncoder (WaveNet conditioning)
Flow (gin_channels)
Duration Predictor (gin_channels)
Decoder (cond layer)
```

### 3-Phase Training

**Phase 1: Pre-train Speaker Encoder** (3–4 days)  
- Start: SpeechBrain ECAPA-TDNN (VoxCeleb pre-trained)  
- Fine-tune trên VieNeu-TTS 193 speakers  
- Freeze all trừ last 2 SE-Res2Blocks + classification head  
- Target: EER < 10% trên hold-out pairs  
- Hold out 10–20 speakers cho zero-shot eval

**Phase 2: Dual Conditioning** (2 weeks)  
- Start từ best VITS2 + Dual-Path checkpoint (week 8)  
- Keep emb_g (lookup) + add speaker_encoder  
- Loss: L_total + λ_spk * L_spk (MSE(emb_g(sid), speaker_encoder(ref_mel)))  
- Schedule:  
  - 0–50K steps: freeze encoder, λ_spk=1.0  
  - 50K+: unfreeze, λ_spk=0.5  
  - 100K+: random dropout emb_g 50%

**Phase 3: Zero-Shot Fine-tuning** (1 week)  
- Drop emb_g hoàn toàn  
- Cross-speaker training: ref ≠ target utterance (cùng speaker)  
- Add L_sim = 1 - cosine_similarity(spk_ref, spk_gen)  
- LR giảm xuống 1e-5

### Clone Ablation (3 variants)
| Variant   | TTS Base                  | Speaker Encoder | Tests                                      |
|-----------|---------------------------|------------------|--------------------------------------------|
| Clone-A   | VITS2 baseline            | ECAPA-TDNN       | Baseline clone quality                     |
| Clone-B   | VITS2 + Tone Embedding    | ECAPA-TDNN       | Tone awareness có giúp cloning?            |
| Clone-D   | VITS2 + Dual-Path full    | ECAPA-TDNN       | Dual-Path có giữ tone khi clone?           |

Expected: Clone-D tone accuracy >> Clone-A (chứng minh Dual-Path disentangle tone/speaker).

### Tone Preservation Evaluation
- Cosine similarity: >0.75  
- SMOS (Similarity MOS): >3.5  
- Tone Confusion Matrix (cloned vs non-cloned): delta accuracy <5%  
- F0 RMSE trên cloned speech  
- Per-tone degradation analysis (tone nào dễ lỗi nhất khi clone?)

### Reference Audio Best Practices
- 5–10s, clean, no noise/reverb  
- 24kHz mono, silence trim (VAD), normalize -3dB peak  
- Nếu user upload từ demo: warn về chất lượng thấp → similarity drop

---

## AI Contribution 3: Speaker Attribution

### Input Features
- Bubble center (x,y normalized)  
- Distance to each character  
- Nearest char dist  
- Tail direction (angle nếu detect)  
- Panel ID  
- Bubble area, char count in panel

### Model
- MLP (3 layers) hoặc lightweight GNN  
- Output: probability distribution over characters in panel  
- Target accuracy: >85% (vs rule-based ~60–70%)

### Training
- Manga109 ground-truth bubble→character
- Augment với 100–200 trang VN comics annotated
- Fallback: rule-based (proximity + tail detection)

### Baseline Comparison (optional)
- Dùng LLM (GPT-4o / Llama-3.1) làm **baseline so sánh** — KHÔNG thay thế self-trained model
- So sánh: Self-trained accuracy vs LLM accuracy vs Rule-based accuracy
- LLM cần API/internet → không phù hợp offline demo
- Nếu self-trained > rule-based → chứng minh AI contribution có giá trị
- Nếu self-trained ≈ LLM → chứng minh model nhẹ đạt được chất lượng tương đương LLM lớn

---

## Integration với Comic Pipeline
User upload comic chapter → CV pipeline detect characters → cluster faces → user assign voice (upload ref audio hoặc select pre-trained) → extract speaker embedding → lưu vào character_db → mỗi bubble: text + fixed speaker_emb → TTS → audio nhất quán.

## ONNX Export (nếu cần browser)
1. speaker_encoder.onnx: input ref_mel → output speaker_emb [192]  
2. tts_model.onnx: input text + speaker_emb [256] → output audio

## Optional Extensions

### Emotion-Aware Comic Speech Synthesis (chỉ sau week 15)

> **Status**: OPTIONAL — chỉ thực hiện khi core pipeline (TTS + Voice Cloning + CV) hoàn thành.
> **Vấn đề chính**: VieNeu-TTS là newspaper reading → ~95% neutral → KHÔNG train emotion trên dataset này.

#### 1. Emotion Detection Pipeline (module mới cần xây)

Tại inference, cần xác định emotion cho mỗi câu thoại trong trang truyện mới. Có 3 nguồn:

**a) Facial Expression Recognition (Visual)**
- Vision Transformer (ViT) phân loại biểu cảm trên mặt nhân vật vẽ
- Dataset: **KangaiSet** (emotion labels cho manga characters)
- Key insight: miệng đủ nhận diện happy, lông mày đủ nhận diện sad
- Thách thức: mặt vẽ khác ảnh thật, ít data cho comic Việt

**b) Text Sentiment Analysis (Text)**
- NLP phân tích nội dung câu thoại → emotion label
- AFINN lexicon hoặc Vietnamese sentiment model
- Hạn chế: không phân biệt mỉa mai vs chân thành

**c) Visual Context (Background/Effects)**
- Màu nền panel (HSV extraction): tối = buồn/sợ, sáng = vui
- Hiệu ứng manga: tia sáng = giận, hoa = vui, mưa = buồn

**d) Multimodal Fusion**
- Kết hợp cả 3 nguồn → emotion label cuối cùng (angry/happy/sad/surprised/neutral)
- Weighted fusion hoặc learned attention

#### 2. Acoustic Parameter Control theo Emotion

| Emotion   | Pitch (F0)     | Speed (Duration) | Energy         |
|-----------|----------------|------------------|----------------|
| Happy     | ↑ cao hơn      | ↑ nhanh hơn      | ↑ mạnh hơn     |
| Sad       | ↓ thấp hơn     | ↓ chậm hơn       | ↓ yếu hơn      |
| Angry     | ↑ cao          | ↑ nhanh           | ↑↑ rất mạnh    |
| Surprised | ↑↑ rất cao     | ↑ nhanh           | ↑ mạnh         |
| Neutral   | bình thường    | bình thường       | bình thường     |

#### 3. Phương pháp tích hợp Emotion vào TTS

**Option 1: GST (Global Style Tokens)** — Recommended
- Bank of learnable style embeddings (Tacotron-based)
- Mỗi token đại diện 1 kiểu nói, không cần label thủ công
- Reference encoder → style embedding → inject vào decoder
- Ưu: unsupervised, flexible

**Option 2: Emotion Embedding**
- Emotion label → learned vector → inject vào decoder cùng speaker embedding
- Ưu: đơn giản, dễ control
- Nhược: cần emotion labels cho mọi training sample

**Option 3: Reference Audio Conditioning**
- Dùng audio mẫu có cảm xúc tương ứng làm reference
- Model bắt chước style từ reference audio
- Ưu: không cần labels, linh hoạt
- Nhược: cần bộ audio mẫu cho mỗi emotion

#### 4. Cross-lingual Transfer Strategy

- Dùng **ESD dataset (English, 29h, 5 emotions: angry/happy/neutral/sad/surprised)**
- Hypothesis: emotion style patterns (speed/pitch/energy) transfers cross-lingually EN → VI
- **Thách thức riêng tiếng Việt**: pitch vừa mang nghĩa emotion vừa mang nghĩa từ (6 thanh điệu)
  - Nếu tăng pitch cho "happy" → "mà" (huyền) có thể bị nghe thành "má" (sắc) → SAI NGHĨA
  - Dual-Path Encoder phải protect tone accuracy khi emotion modulation được áp dụng
  - Tonal Encoder giữ F0 contour cho tone, emotion chỉ modulate phần "residual" pitch
- Chưa có research chứng minh cross-lingual transfer cho tonal languages → đây là contribution mới nếu thành công

#### 5. Implementation Steps (Week 16+)

```
Step 1: Train emotion TTS trên ESD English (GST hoặc Emotion Embedding)
Step 2: Evaluate cross-lingual transfer EN → VI (có giữ tone không?)
Step 3: Build Emotion Detection module (face + text + context fusion)
Step 4: Integrate emotion label vào TTS pipeline
Step 5: End-to-end eval: comic page → emotion-aware voiced audio
```

#### 6. Papers tham khảo

- "Emotion-Aware Speech Generation with Character-Specific Voices for Comics" (2025) — pipeline đầy đủ nhất
- "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis" (Google) — nền tảng GST
- "KangaiSet: A Dataset for Visual Emotion Recognition on Manga" — dataset emotion cho manga
- "MELS-TTS: Multi-Emotion Multi-Lingual Multi-Speaker TTS via Disentangled Style Tokens" (Samsung)
- "EmoComicNet: A multi-task model for comic emotion recognition"
- "Leveraging multimodal fusion for emotion detection in comics" (2025)

### Scene Graph Generation (SGG) cho Speaker Attribution (optional)

> **Status**: OPTIONAL — chỉ thực hiện nếu MLP Speaker Attribution chưa đạt target >85% accuracy, hoặc nếu còn thời gian muốn tăng AI contribution.

#### Concept
Thay vì MLP xử lý flat features → biến trang truyện thành **graph** → GNN predict "speaks" edges.

#### Scene Graph Construction
```
YOLO detect output:
  Characters: [C1(x,y,w,h), C2(x,y,w,h)]
  Bubbles: [B1(x,y,w,h), B2(x,y,w,h)]
  Panels: [P1(x,y,w,h)]
       │
       ▼
Graph nodes: C1, C2, B1, B2, P1
       │
       ▼
Edges (spatial relationships):
  P1 ──contains──→ C1, C2, B1, B2
  C1 ──near──→ C2 (distance < threshold)
  B1 ──tail_points──→ C1 (tail detection)
  B1 ──closest_to──→ C1 (proximity)
```

#### GNN Architecture
- Model: **GAT (Graph Attention Network)** hoặc GCN
- Node features: position (x,y), size (w,h), type (character/bubble/panel)
- Edge features: distance, angle, contains/near/tail relationship type
- Output: probability cho "speaks" edge giữa mỗi cặp (character, bubble)
- Target accuracy: >90% (vs MLP ~85%)

#### Training
- Convert Manga109 bbox annotations → scene graph format
- Ground-truth "speaks" edges từ Manga109 bubble→character mapping
- Augment với Vietnamese comic annotations

#### So sánh
| Method | Input | Accuracy target | Giữ quan hệ |
|--------|-------|----------------|-------------|
| Rule-based | positions | ~65% | Không |
| MLP (baseline) | flat features | ~85% | Implicit |
| **SGG + GNN** | graph | ~90%+ | **Explicit** |

#### Thời gian: ~2-3 tuần

---

### Emotion Intensity Estimation (optional)

> **Status**: OPTIONAL — phụ thuộc Extension 1 (Emotion Detection) hoàn thành trước.
> **Reference**: Paper "Emotion-Aware Speech Generation..." (2025) dùng exact approach này.

#### Concept
Thay vì chỉ predict emotion category (angry/happy/sad) → thêm **intensity score (0.0 → 1.0)**.

```
Emotion Detection (category only):
  Input → "angry"                    → TTS nói giận (1 kiểu)

Emotion Detection + Intensity:
  Input → "angry", intensity=0.3     → TTS hơi bực (nhẹ)
  Input → "angry", intensity=0.9     → TTS phẫn nộ (mạnh)
```

#### Implementation (theo paper approach — ResNet-50 binary)
Paper dùng cách đơn giản và hiệu quả:
```python
# Binary classifier: neutral vs emotional (gộp tất cả non-neutral → 1 class)
# Fine-tune pre-trained ResNet-50 trên KangaiSet (simplified)
model = resnet50(pretrained=True)
model.fc = nn.Linear(2048, 2)  # 2 classes: neutral, emotional

# Train trên KangaiSet simplified binary labels
# Inference: logit output = intensity score
logits = model(face_crop)
intensity = torch.sigmoid(logits[:, 1])  # 0.0 = neutral, 1.0 = strong emotion
```

Hoặc multi-task approach (nâng cao hơn paper):
```
Shared features → Classification head → emotion_category (5 classes)
                → Regression head     → intensity (0.0 → 1.0)
Multi-task loss: L = L_classification + λ * L_regression (MSE)
```

#### Cảnh báo từ paper
- **Neutral over-prediction thành emotional**: model hay predict neutral → "strong emotion" → cascade error
- **Cần threshold calibration**: không dùng raw logit, cần calibrate threshold cho neutral/emotional boundary
- **Surprise over-prediction cho câu hỏi**: LLM/model hay label câu hỏi = surprise, nhưng cho TTS thì perceptually acceptable

#### TTS Acoustic Control theo Intensity
```python
def apply_emotion_style(audio_params, emotion, intensity):
    emotion_config = {
        "angry":     {"pitch": +1.0, "speed": +1.0, "energy": +1.5},
        "happy":     {"pitch": +0.8, "speed": +0.8, "energy": +0.8},
        "sad":       {"pitch": -0.8, "speed": -0.8, "energy": -1.0},
        "surprised": {"pitch": +1.2, "speed": +0.8, "energy": +0.8},
    }
    config = emotion_config[emotion]
    # Scale by intensity (0.0 = no change, 1.0 = full change)
    audio_params.pitch   += config["pitch"] * intensity
    audio_params.speed   += config["speed"] * intensity
    audio_params.energy  += config["energy"] * intensity
```

#### Realistic Targets (dựa trên paper benchmarks)
- Emotion F1: ~40-45% (paper đạt 42.9% — emotion trong comic rất khó)
- Perceptual acceptability: >70% listeners đánh giá "phù hợp context"
- Emotion accuracy thấp KHÔNG có nghĩa kết quả tệ — paper xác nhận

#### Thời gian: ~1 tuần (trên Extension 1)

---

### VoiceDesign — Text-Described Voice Assignment (optional)

> **Status**: OPTIONAL — alternative voice assignment khi user không có reference audio.
> **Inspiration**: Qwen3-TTS VoiceDesign / CosyVoice — dùng text description thay reference audio.

#### Concept
Thay vì BẮT BUỘC upload 5s audio cho mỗi nhân vật, user có thể **mô tả giọng bằng text**:

```
Hiện tại (Voice Cloning only):
  User PHẢI upload 5s audio → ECAPA-TDNN → speaker embedding
  ❌ Không có audio → không assign được giọng

Thêm VoiceDesign:
  User CHỌN 1 trong 2:
  ├── Option A: Upload 5s audio → Voice Clone (chính xác hơn)
  └── Option B: Gõ mô tả: "giọng nam trầm, nghiêm túc, trung niên"
                → Match với speaker trong VieNeu-TTS 193 speakers
                → Không cần audio
```

#### Implementation (lightweight — không cần Codec LM)

**Không cần train Codec LM** (quá lớn, 7-14B params). Thay vào đó:

```
Bước 1: Pre-compute speaker profiles cho VieNeu-TTS 193 speakers
  Speaker 1 → {gender: "male", age: "young", pitch: "high", speed: "fast"}
  Speaker 2 → {gender: "female", age: "middle", pitch: "low", speed: "slow"}
  ...
  (Extract tự động từ audio: F0 mean → pitch, speaking rate → speed, etc.)

Bước 2: User nhập description
  "giọng nữ, trẻ, vui vẻ, nói nhanh"

Bước 3: Match description → speaker profile
  ├── Simple: keyword matching (gender + age + pitch)
  └── Advanced: encode description bằng text encoder → cosine match với profiles
```

#### Architecture
```
User description: "giọng nam trầm, nghiêm túc"
         │
         ▼
  Text Encoder (PhoBERT / simple keyword parser)
         │
         ▼
  Description Embedding [256]
         │
         ▼
  Cosine Similarity với 193 speaker profiles
         │
         ▼
  Top-K matching speakers → user chọn 1
         │
         ▼
  speaker_embedding → inject vào VITS2 (giống Voice Clone flow)
```

#### Trong Comic Demo
```
┌─ Character Voice Assignment ──────────────────┐
│                                                │
│  😀 Nhân vật 1:                               │
│  ● Upload audio [🎤 Record 5s]  ← Voice Clone │
│  ○ Mô tả giọng: [giọng nam trẻ, vui vẻ  ]   │  ← VoiceDesign
│  ○ Chọn từ danh sách: [Speaker #42 ▼]        │
│                                                │
└────────────────────────────────────────────────┘
```

#### Ưu/nhược so với Voice Clone
| | Voice Clone (core) | VoiceDesign (optional) |
|--|-------------------|----------------------|
| **Input** | 5s audio | Text description |
| **Accuracy** | Rất cao — exact voice | Trung bình — approximate match |
| **UX** | Cần thu âm / có sẵn audio | Chỉ cần gõ text |
| **Use case** | Clone giọng cụ thể | Không có audio, chỉ biết muốn giọng kiểu gì |
| **AI contribution** | ECAPA-TDNN (core) | Text-to-speaker matching (nhỏ) |

#### Thời gian: ~3-5 ngày
- Pre-compute speaker profiles: 1 ngày
- Keyword matching: 1 ngày
- (Optional) PhoBERT encoding: 2 ngày
- Integrate vào demo UI: 1 ngày

---

### BLIP/LLaVA Scene Understanding (optional)
- Vision-language model cho comic scene understanding
- Hỗ trợ: emotion detection, context understanding
- Chi phí: ~7B params, tốn VRAM → chỉ thêm nếu có giá trị rõ ràng
- KHÔNG nằm trong core pipeline

### LLM Attribution Baseline (optional)
- GPT-4o / Llama-3.1 / Gemini cho speaker attribution
- Dùng làm **baseline comparison** (không thay thế self-trained model)
- So sánh 3 chiều: Rule-based (~65%) vs Self-trained (target >85%) vs LLM

---

## Key Paper References

| Paper | Relevance | Key Results |
|-------|-----------|-------------|
| "Emotion-Aware Speech Generation with Character-Specific Voices for Comics" (2025) | **Most related** — full comic voice-over pipeline | Speaker 75.7% (trained), 64.8% (LLM); Emotion F1 42.9%; LLM thua trained model cho hard cases |
| Manga109Speaker [9] | Speaker Attribution baseline | 75.7% total, 30.7% hard — SOTA trên Manga109Dialogue |
| Zero-Shot Multimodal [10] | LLM-based speaker attribution | 51.8% — LLM kém hơn trained model |
| KangaiSet [11] | Emotion labels cho manga characters | Binary/5-way classification, imbalanced dataset |
| "Style Tokens" (Google) | GST cho emotion TTS | Unsupervised style modeling, no labels needed |
| "MELS-TTS" (Samsung) | Multi-emotion multi-speaker TTS | Disentangled style tokens |

### So sánh project vs paper SOTA
| Component | Paper SOTA | Project Target | Advantage |
|-----------|-----------|----------------|-----------|
| Speaker Attribution | 75.7% (trained DL) | **>85%** (SGG/GNN) | Spatial graph reasoning cho hard cases |
| Emotion | F1 42.9% (LLM) | F1 ~40-45% (visual+text) | Tương đương, nhưng offline |
| Voice Cloning | Basic speaker embedding | **ECAPA-TDNN + tone preservation** | Novel — paper không có |
| TTS | Pre-trained (Nhật) | **VITS2 + Dual-Path** (tự train) | Novel architecture cho Vietnamese |
| Offline | ❌ Cần GPT-4o | ✅ Hoàn toàn local | Không phụ thuộc API |

Xem thêm: experiment-and-research.md cho ablation & logging rules.
