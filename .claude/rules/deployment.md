# Deployment Rules
> Streaming web demo, server-side CV pipeline, pre-process ahead strategy. Auto-loaded by Claude Code.

---

## Deployment Architecture

```
┌─ CLIENT (Browser) ──────────────┐     ┌─ SERVER (GPU) ──────────────────┐
│                                  │     │                                  │
│  Upload comic pages              │────→│  YOLO detect (bubble/char/panel) │
│  Review characters + voices      │     │  PaddleOCR (Vietnamese)          │
│  Audio player (streaming)        │     │  ArcFace face clustering         │
│  Page navigation                 │←────│  Speaker Attribution             │
│                                  │     │  TTS + Voice Cloning (VITS2)     │
│  Tech: React/Vite                │     │  Tech: FastAPI + PyTorch         │
└──────────────────────────────────┘     └──────────────────────────────────┘
```

### Why server-side (not pure browser)
- YOLO + PaddleOCR + ArcFace + TTS = too heavy for browser
- GPU required for TTS inference speed
- Browser handles: UI, audio playback, file upload

### Optional: ONNX browser for TTS only
- If time permits: export TTS + Speaker Encoder to ONNX
- Run TTS in browser (ONNX Runtime Web), CV stays on server
- This reduces server load but requires WebGPU support

---

## Streaming Workflow

```
1. User uploads 10-15 comic pages
2. Server scans all pages: detect characters → cluster → return character list
3. User reviews: rename characters, assign voices
4. User clicks "Generate"
5. Server processes page-by-page, streams results:
   - Page 1 ready → send to client → user can listen
   - Page 2 processing in background
   - ...
6. Pre-process ahead: always 2-3 pages ahead of user
```

### API Design
```
POST /upload          → upload comic pages
GET  /characters      → return detected characters + appearances
POST /assign-voices   → user assigns voice per character
POST /generate        → start processing (returns job_id)
GET  /status/{job_id} → polling for progress
GET  /audio/{page_id} → stream audio for specific page
```

---

## Pre-process Ahead Strategy

```
Processing time per page: ~3-5s
Audio duration per page:  ~10-15s
→ Pipeline is ALWAYS ahead of user listening

User listens to page 1 (15s) → pages 2,3,4 already done
User flips to page 2 → instant playback
```

No special optimization needed — pipeline is naturally faster than listening.

---

## Demo UI Design

```
┌─ Comic Voice-Over Demo ──────────────────────────┐
│                                                    │
│  [Upload Pages]  Reading: ● LTR  ○ RTL            │
│                                                    │
│  Characters:                                       │
│  😀 Naruto  [🎤 Upload] [Auto ▼]                 │
│  😮 Sasuke  [🎤 Upload] [Auto ▼]                 │
│  [✓ Generate]                                      │
│                                                    │
│  ┌─────────────────────┐  Dialogues:              │
│  │                     │  1. 😀 "Xin chào!" [▶]  │
│  │   (comic page)      │  2. 😮 "Hmph."    [▶]  │
│  │                     │                           │
│  └─────────────────────┘  [▶ Play All] [💾 Save] │
│                                                    │
│  [◀ Prev]  Page 3/15  [Next ▶]                   │
└────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| TTS model | PyTorch (VITS2 + Dual-Path + Voice Cloning) |
| CV models | YOLOv8 (Ultralytics) + PaddleOCR + ArcFace |
| Frontend | React/Vite |
| Audio | Web Audio API |
| Communication | REST API + Server-Sent Events (SSE) for streaming |

---

## Priority
Deployment is a **supporting part**:
1. AI must work first (TTS + Voice Cloning + CV pipeline)
2. Demo is to showcase AI — keep it simple
3. Do not over-engineer the frontend
4. A working demo with basic UI > a beautiful demo that doesn't work
