# Project Overview & Vision

**Tên dự án**: Comic Voice-Over System — Vietnamese TTS with Tone-Preserved Zero-Shot Voice Cloning for Comic Characters  
**Tên tiếng Việt**: Hệ thống tự động lồng tiếng truyện tranh với TTS tiếng Việt nhận biết thanh điệu và nhân bản giọng nói zero-shot

### Mục tiêu nghiên cứu chính
Xây dựng hệ thống end-to-end chuyển trang truyện tranh (comic/manga/webtoon tiếng Việt) thành audio narration, với:
- Mỗi nhân vật có giọng nói riêng biệt, nhất quán xuyên suốt chapter.
- Giọng nói clone từ reference audio ngắn (5–10s), giữ nguyên tính chính xác thanh điệu 6 tones của tiếng Việt.
- Tự động detect bong bóng thoại, nhận diện nhân vật, gán người nói, và tổng hợp giọng theo cảm xúc/ ngữ cảnh (nếu mở rộng).

### AI Contributions cốt lõi (self-built)
1. **Dual-Path Encoder cho VITS2**  
   - Tách riêng Linguistic Encoder (nội dung) và Tonal Encoder (thanh điệu).  
   - Fusion bằng Cross-Attention → disentangle tone khỏi speaker identity.  
   - Giải quyết vấn đề tone confusion (hỏi/ngã, sắc/nặng) trong TTS tiếng Việt.

2. **Tone-Preserved Zero-Shot Voice Cloning**  
   - Sử dụng ECAPA-TDNN speaker encoder.  
   - 3-phase training: pre-train → dual conditioning → zero-shot.  
   - Đảm bảo cloning không làm suy giảm độ chính xác thanh điệu (tone delta <5%).

3. **AI-based Speaker Attribution**  
   - Model MLP/GNN gán bong bóng thoại → nhân vật dựa trên vị trí, tail, panel layout.  
   - Accuracy mục tiêu >85% (so với rule-based ~60–70%).

### Ứng dụng showcase
- Pipeline CV: YOLOv8 (bubble/character/panel), PaddleOCR + VietOCR (text VN), ArcFace clustering.  
- Demo: Upload chapter 10–15 trang → nghe audio với giọng nhân vật riêng, pre-process ahead cho real-time feel.

### Hypothesis cốt lõi
Dual-Path Encoder + tone disentanglement →  
- Giảm tone error trong TTS baseline.  
- Giữ tone accuracy cao khi clone voice (swapping speaker không ảnh hưởng tone).

### Scope & Out-of-scope
- In-scope: TTS + Cloning + CV pipeline + streaming demo.  
- Out-of-scope: Full emotion TTS từ dataset VN, mobile app, real-time voicebot.

Xem thêm:  
- Success criteria: specs/01-success-criteria.md  
- Architecture: specs/02-architecture.md  
- Ablation plan: specs/03-ablation-plan.md