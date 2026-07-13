# Success Criteria & Deliverables

Dựa trên timeline 16 tuần và checkpoint week 8/12.

### Minimum Success (TTS + Cloning, no CV – Week 8–12)
- Dataset pipeline sạch với tone extraction (sea-g2p + deterministic tone label).  
- VITS2 baseline ổn định.  
- Dual-Path Encoder (ít nhất 2–3 variants ablation).  
- Voice Cloning zero-shot hoạt động (cosine sim >0.75).  
- Tone preservation eval: confusion matrix cloned vs non-cloned, delta <5%.  
- Audio demo samples + metrics table.  
- Report/slides cơ bản.

### Good Success (+ CV Pipeline – Week 12–14)
Tất cả minimum +  
- YOLO fine-tune bubble/character/panel (mAP >0.8 trên comic VN).  
- OCR CER <3–5% (VietOCR ưu tiên).  
- Face clustering >90% correct grouping.  
- Speaker attribution (rule-based hoặc AI) >80%.  
- End-to-end: comic page → audio per character.  
- Character voice consistency qua các trang.

### Excellent Success (Full System – Week 14–16)
Tất cả good +  
- Speaker Attribution AI model trained & evaluated (>85%).  
- Clone ablation chứng minh Dual-Path giúp tone preservation.  
- Streaming web demo: upload chapter → audio, pre-process ahead.  
- Custom tone test set (200–500 sentences) + full eval.  
- Error analysis chi tiết (tone nào dễ lỗi nhất khi clone).  
- Report + defense slides hoàn chỉnh (20–25 slides).

### Optional Extensions (nếu dư thời gian sau week 15)
- Emotion transfer cross-lingual từ ESD dataset (English → VN).  
- Region-aware (North/Central/South accent).  
- ONNX export full cho browser TTS.

Mỗi mức success phải có audio samples qualitative + objective metrics để defend.