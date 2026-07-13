"""Comic pipeline using vanilla Gwen-TTS 0.6B for voice cloning.

Flow:
  1. Load Gwen-TTS (g-group-ai-lab/gwen-tts-0.6B) — best VN voice clone model
  2. Transcribe character ref wavs (once, cached as .txt)
  3. Reuse comic_pipeline (YOLO + Qwen-VL OCR + speaker attribution)
  4. Synthesize each bubble with generate_voice_clone() using character ref
  5. Concatenate into per-page audio + full chapter audio

Gwen-TTS result: cos_sim 0.7501 mean vs XTTS FT 0.7121 — stronger speaker similarity.

Usage:
    python comic_pipeline_gwen.py --pages-dir <dir> --out <out_dir>
    python comic_pipeline_gwen.py --pages-dir <dir> --out <out_dir> --refs-dir <custom_refs>
"""
import argparse, csv, json, os, re, sys, time, types
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

SAMPLE_RATE_GWEN = 24_000   # Gwen-TTS output sample rate (returned by generate_voice_clone)
DEFAULT_REFS_DIR = "/home/bes/Desktop/Tin/demo_refs_xtts"
MODEL_ID = "g-group-ai-lab/gwen-tts-0.6B"

GWEN_GEN_CONFIG = dict(
    temperature=0.3,
    top_k=20,
    top_p=0.9,
    max_new_tokens=4096,
    repetition_penalty=2.0,
    subtalker_do_sample=True,
    subtalker_temperature=0.1,
    subtalker_top_k=20,
    subtalker_top_p=1.0,
)

FOREIGN_NAME_MAP = {
    "Doraemon": "Đô-rê-mon", "doraemon": "đô-rê-mon",
    "Nobita": "Nô-bi-ta",    "nobita": "nô-bi-ta",
    "Shizuka": "Si-zu-ka",   "shizuka": "si-zu-ka",
    "Suneo": "Su-nê-ô",      "suneo": "su-nê-ô",
    "Jaian": "Cha-i-an",     "jaian": "cha-i-an",
    "Gian": "Chi-an",        "gian": "chi-an",
    "Dorami": "Đô-ra-mi",    "dorami": "đô-ra-mi",
    "Naruto": "Na-ru-tô",    "naruto": "na-ru-tô",
    "Sasuke": "Sa-su-kê",    "sasuke": "sa-su-kê",
    "Sakura": "Sa-ku-ra",    "sakura": "sa-ku-ra",
}


def vietnamize(text):
    for foreign, viet in FOREIGN_NAME_MAP.items():
        text = text.replace(foreign, viet)
    return text


def transcribe_wav(wav_path: str) -> str:
    """Transcribe a wav file with Whisper (via transformers). Cached as .txt sibling."""
    txt_path = Path(wav_path).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()

    print(f"  Transcribing {Path(wav_path).name} with Whisper …")
    from transformers import pipeline as hf_pipeline
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        generate_kwargs={"language": "vi", "task": "transcribe"},
        device=0,
    )
    result = asr(wav_path)
    text = result["text"].strip()
    txt_path.write_text(text, encoding="utf-8")
    print(f"    → \"{text}\"")
    return text


def build_voice_registry(refs_dir: Path) -> dict:
    """Return {char_id: (wav_path_str, ref_text_str)} for all .wav in refs_dir."""
    registry = {}
    for wav_path in sorted(refs_dir.glob("*.wav")):
        char_id = wav_path.stem
        ref_text = transcribe_wav(str(wav_path))
        registry[char_id] = (str(wav_path), ref_text)
        print(f"  {char_id:15s}  ref={len(ref_text)} chars")
    return registry


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, ensure sentence terminator."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def chunk_text(text: str, max_words: int = 12) -> list:
    """Split long text on sentence boundaries; short text returned as-is."""
    if len(text.split()) <= max_words:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def load_label_csv(csv_path: str) -> dict:
    """Return {(page_stem, order): corrected_text | None} from labels.csv.

    None means the bubble is marked skip=x and should be omitted from TTS.
    """
    lookup = {}
    if not csv_path:
        return lookup
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["page"].strip(), int(row["order"]))
            if row.get("skip", "").strip().lower() == "x":
                lookup[key] = None
            elif row.get("corrected_text", "").strip():
                lookup[key] = row["corrected_text"].strip()
    return lookup


def synthesize(model, text: str, ref_wav: str, ref_text: str) -> np.ndarray:
    """Call Gwen-TTS generate_voice_clone, return float32 wav array."""
    # Dynamic cap: ~60 audio tokens per word, minimum 150 — prevents hallucination
    max_tokens = max(150, len(text.split()) * 60)
    config = {**GWEN_GEN_CONFIG, "max_new_tokens": max_tokens}

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Vietnamese",
        ref_audio=ref_wav,
        ref_text=ref_text,
        **config,
    )
    wav_np = np.asarray(wavs[0], dtype=np.float32).flatten()
    if sr != SAMPLE_RATE_GWEN:
        import torchaudio
        wav_t = torch.from_numpy(wav_np).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE_GWEN)
        wav_np = wav_t.squeeze(0).numpy()
    return wav_np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--refs-dir", default=DEFAULT_REFS_DIR)
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--ocr-engine", default="qwen-vl")
    p.add_argument("--yolo-ckpt",
                   default="/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt")
    p.add_argument("--speaker-mlp",
                   default="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_mlp.pt")
    p.add_argument("--speaker-scaler",
                   default="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/scaler.pkl")
    p.add_argument("--trim-silence", action="store_true", default=True)
    p.add_argument("--gap-seconds", type=float, default=0.25)
    p.add_argument("--labels-csv", default="",
                   help="Path to labeling_task/labels.csv for OCR text override")
    p.add_argument("--qwen-lora-adapter", default=None,
                   help="Path to LoRA adapter for fine-tuned Qwen-VL OCR.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Gwen-TTS ──────────────────────────────────────────────
    print(f"[1/4] Loading Gwen-TTS ({MODEL_ID}) …")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded — VRAM: {vram:.2f} GB")

    # ── Build voice registry ───────────────────────────────────────
    print("[2/4] Building voice registry …")
    refs_dir = Path(args.refs_dir)
    voice_registry = build_voice_registry(refs_dir)
    print(f"  {len(voice_registry)} character voices registered")

    # ── Init CV pipeline ───────────────────────────────────────────
    print("[3/4] Initializing CV pipeline (YOLO + Qwen-VL) …")
    sys.path.insert(0, "/home/bes/Desktop/Tin")
    cv_args = types.SimpleNamespace(
        weights=args.yolo_ckpt,
        speaker_model=args.speaker_mlp,
        speaker_scaler=args.speaker_scaler,
        ocr_engine=args.ocr_engine,
        qwen_lora_adapter=args.qwen_lora_adapter,
        lang="vi",
        no_ocr=False,
        no_face=True,      # skip ArcFace — using voice registry instead
        no_gpu=True,       # PaddleOCR doesn't support sm_120
        paddle_only=False,
        char_db=None,
        yolo_conf=0.25,
        rule_based=False,
        direction="ltr",
        verbose=False,
    )
    from Capstone_project.scripts.comic_pipeline import init_pipeline, process_page
    cv_models = init_pipeline(cv_args)
    print("  CV pipeline ready")

    # ── Load label CSV (optional) ──────────────────────────────────
    label_lookup = load_label_csv(args.labels_csv)
    if label_lookup:
        print(f"  Loaded {len(label_lookup)} label entries from {args.labels_csv}")

    # ── Process pages ──────────────────────────────────────────────
    print("[4/4] Processing pages + Gwen-TTS synthesis …")
    pages_dir = Path(args.pages_dir)
    page_files = sorted(
        list(pages_dir.glob("*.jpg")) +
        list(pages_dir.glob("*.webp")) +
        list(pages_dir.glob("*.png"))
    )[:args.max_pages]
    print(f"  {len(page_files)} pages to process")

    all_page_audios = []
    page_dir = out_dir / "pages"
    page_dir.mkdir(exist_ok=True)
    bubble_dir = out_dir / "bubbles"
    bubble_dir.mkdir(exist_ok=True)

    for page_idx, img_path in enumerate(page_files):
        page_name = img_path.stem
        print(f"\n[Page {page_idx+1}/{len(page_files)}] {page_name}")

        t_cv = time.time()
        try:
            page_result = process_page(
                image_path          = img_path,
                yolo_model          = cv_models["yolo"],
                ocr_pipeline        = cv_models["ocr"],
                face_extractor      = cv_models["extractor"],
                char_db             = cv_models["char_db"],
                speaker_model_path  = cv_models["speaker_model"],
                speaker_scaler_path = cv_models["speaker_scaler"],
                direction           = "ltr",
                rule_based          = False,
                use_gpu             = False,
                yolo_conf           = 0.25,
            )
        except Exception as exc:
            print(f"  CV failed: {exc}")
            continue
        n_bubbles = len(page_result.get("bubbles", []))
        print(f"  CV done in {time.time()-t_cv:.1f}s — {n_bubbles} bubbles")

        page_audio_chunks = []
        bubbles_info = []

        for bubble in sorted(page_result.get("bubbles", []),
                             key=lambda b: b.get("order", 0)):
            order = bubble.get("order", 0)
            text = bubble.get("text", "").strip()
            if not text:
                continue
            text = vietnamize(text)

            # Task 1: CSV override — dùng corrected_text nếu có, bỏ qua nếu skip
            csv_key = (page_name, order)
            if csv_key in label_lookup:
                if label_lookup[csv_key] is None:
                    print(f"    [{order:02d}] skipped (CSV skip=x)")
                    continue
                text = label_lookup[csv_key]

            # Task 2: Normalize — lowercase, clean whitespace, đảm bảo dấu câu cuối
            text = normalize_text(text)
            if not text:
                continue

            # Resolve character → voice
            qwen_spk = bubble.get("qwen_speaker")
            mlp_spk = bubble.get("speaker_id")
            if qwen_spk and qwen_spk != "sound_effect":
                char_id = qwen_spk
            elif mlp_spk:
                char_id = mlp_spk
            else:
                char_id = "unknown"

            if char_id not in voice_registry:
                char_id = "default"

            ref_wav, ref_text = voice_registry[char_id]

            # Task 3: Chunk + synthesize — chia câu dài, ghép audio từng chunk
            t0 = time.time()
            try:
                chunks = chunk_text(text)
                chunk_gap = np.zeros(int(0.05 * SAMPLE_RATE_GWEN), dtype=np.float32)
                chunk_wavs = []
                for chunk in chunks:
                    chunk_wavs.append(synthesize(model, chunk, ref_wav, ref_text))
                if len(chunk_wavs) == 1:
                    wav_np = chunk_wavs[0]
                else:
                    interleaved = []
                    for i, w in enumerate(chunk_wavs):
                        interleaved.append(w)
                        if i < len(chunk_wavs) - 1:
                            interleaved.append(chunk_gap)
                    wav_np = np.concatenate(interleaved)
            except Exception as exc:
                print(f"    [{order:02d}] TTS failed: {exc}")
                continue

            # Trim silence
            if args.trim_silence and len(wav_np) > 1000:
                try:
                    import librosa
                    wav_trim, _ = librosa.effects.trim(wav_np, top_db=25)
                    if len(wav_trim) > 1000:
                        wav_np = wav_trim
                except Exception:
                    pass

            dur = len(wav_np) / SAMPLE_RATE_GWEN
            elapsed = time.time() - t0
            rtf = elapsed / dur if dur > 0 else 0
            src = "CSV" if csv_key in label_lookup else "OCR"
            print(f"    [{order:02d}] {char_id:12s} {dur:.1f}s (RTF={rtf:.2f}) [{src}]: \"{text[:50]}\"")

            bubble_path = bubble_dir / f"{page_name}_b{order:03d}_{char_id}.wav"
            sf.write(str(bubble_path), wav_np, SAMPLE_RATE_GWEN)

            page_audio_chunks.append(wav_np)
            bubbles_info.append({
                "order": order, "char_id": char_id, "text": text,
                "duration": round(dur, 2), "path": str(bubble_path),
                "text_source": src,
            })

        if not page_audio_chunks:
            print("  No audio for this page")
            continue

        gap = np.zeros(int(args.gap_seconds * SAMPLE_RATE_GWEN), dtype=np.float32)
        page_audio = []
        for i, chunk in enumerate(page_audio_chunks):
            page_audio.append(chunk)
            if i < len(page_audio_chunks) - 1:
                page_audio.append(gap)
        page_audio = np.concatenate(page_audio)

        page_wav = page_dir / f"{page_idx+1:03d}_{page_name}.wav"
        sf.write(str(page_wav), page_audio, SAMPLE_RATE_GWEN)
        total_dur = len(page_audio) / SAMPLE_RATE_GWEN
        print(f"  → {page_wav.name} ({total_dur:.1f}s, {len(bubbles_info)} bubbles)")

        all_page_audios.append(page_audio)

        with open(page_dir / f"{page_idx+1:03d}_{page_name}.json", "w", encoding="utf-8") as f:
            json.dump({"page": page_name, "bubbles": bubbles_info,
                       "total_duration": round(total_dur, 2)}, f, ensure_ascii=False, indent=2)

    # ── Full chapter concat ────────────────────────────────────────
    if all_page_audios:
        gap_page = np.zeros(int(0.5 * SAMPLE_RATE_GWEN), dtype=np.float32)
        full = []
        for i, pa in enumerate(all_page_audios):
            full.append(pa)
            if i < len(all_page_audios) - 1:
                full.append(gap_page)
        full = np.concatenate(full)
        full_path = out_dir / "full_chapter.wav"
        sf.write(str(full_path), full, SAMPLE_RATE_GWEN)
        print(f"\nFull chapter: {full_path} ({len(full)/SAMPLE_RATE_GWEN:.1f}s)")

    print(f"\nDone. All output: {out_dir}")


if __name__ == "__main__":
    main()
