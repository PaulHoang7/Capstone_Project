# Project Artifacts and Deliverables
> Required deliverables for Comic Voice-Over system. Auto-loaded by Claude Code.

---

## Project Files To Maintain

| File | Purpose |
|------|---------|
| `.claude/CLAUDE.md` | Direction, scope, success criteria |
| `.claude/plan.md` | Timeline, milestones, fallback plan |
| `rules/dataset.md` | Data sources, tone extraction, comic data |
| `rules/model-strategy.md` | Dual-Path, Voice Cloning, Speaker Attribution |
| `rules/voice-cloning.md` | ECAPA-TDNN, 3-phase training, clone ablation |
| `rules/cv-pipeline.md` | YOLO, OCR, face clustering, reading order |
| `rules/experiment-and-research.md` | Experiment philosophy, logging, checklists |
| `rules/evaluation.md` | TTS + Voice Cloning + CV metrics |
| `rules/deployment.md` | Streaming web demo, server architecture |
| `rules/project-artifacts.md` | This file |
| `rules/gotchas.md` | Lessons learned |

---

## Expected Deliverables

### Minimum (TTS + Voice Cloning)
1. Clean dataset + tone extraction  
2. VITS2 baseline checkpoint  
3. Dual-Path Encoder (2–3 variants)  
4. Voice Cloning zero-shot  
5. Tone preservation eval (confusion matrix, delta <5%)  
6. TTS metrics table + audio samples  
7. Report/slides cơ bản  
**Status**: [ ] Week target: 8–12

### Good (+ CV pipeline)
All minimum +  
9. YOLO fine-tuned  
10. PaddleOCR + VietOCR  
11. Face clustering  
12. Speaker attribution (rule/AI)  
13. End-to-end comic → audio  
14. CV metrics  
**Status**: [ ] Week target: 12–14

### Full (complete system)
All good +  
15. Speaker Attribution AI (>85%)  
16. Clone ablation  
17. Streaming web demo  
18. Voice consistency across pages  
19. Pre-process ahead  
20. Custom tone test set  
21. End-to-end eval on VN comics  
**Status**: [ ] Week target: 14–16

### Excellent (+ extensions)
All full +
22. Emotion extension (ESD)
23. Cross-lingual emotion eval
24. Error analysis
25. Scene Graph Generation (SGG) cho Speaker Attribution + comparison với MLP baseline
26. Emotion Intensity Estimation (regression 0.0→1.0) + proportional TTS acoustic control
27. VoiceDesign — text-described voice assignment (match description → VieNeu-TTS speaker profiles)
**Status**: [ ] Week target: 15–16

## Defense/Report
- Introduction + motivation  
- Related work  
- Method (Dual-Path + 3-phase cloning + CV)  
- Experiments & ablation  
- Results (metrics + audio)  
- Conclusion & future work  
- Slides: 15–25 slides

Storage: /mnt/nfs-data/tin_dataset

