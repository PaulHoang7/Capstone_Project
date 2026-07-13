"""Run comic pipeline on Vietnamese comic pages using XTTS FT for voice cloning.

Flow:
  1. Reuse comic_pipeline (YOLO + Qwen-VL OCR + speaker attribution) to extract bubbles
  2. Pre-compute XTTS conditioning latents for each character ref
  3. Synthesize each bubble with XTTS FT using its character's ref voice
  4. Concatenate into per-page audio + full chapter audio

Usage:
    python comic_pipeline_xtts.py --pages-dir <dir> --out <out_dir>
"""
import argparse, csv, json, os, re, sys, time, types
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from scipy.io.wavfile import write as wav_write

# Shared XTTS config — see xtts_gen_config.py for rationale.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtts_gen_config import (
    gen_config_for_text, SAMPLE_RATE, clean_for_xtts, cap_duration_by_text,
)

REFS_DIR = "/home/bes/Desktop/Tin/demo_refs_xtts"
FT_DIR   = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--ocr-engine", default="qwen-vl")
    p.add_argument("--yolo-ckpt", default="/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt")
    p.add_argument("--speaker-mlp", default="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_mlp.pt")
    p.add_argument("--qwen-lora-adapter",
                   default="/mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_3b_vncomic_lora_v1/best",
                   help="Fine-tuned Qwen-VL LoRA adapter (CER 0.03 vs vanilla 0.29). "
                        "Pass empty string to disable.")
    p.add_argument("--labels-csv",
                   default="/home/bes/Desktop/Tin/labels_edited.csv",
                   help="CSV with manual corrected_text per bubble. Overrides Qwen output "
                        "when (page, order) matches. Pass empty string to disable.")
    p.add_argument("--character-gallery",
                   default="/home/bes/Desktop/Tin/gallery",
                   help="Path to character gallery root (one subfolder per character with "
                        "3-5 ref images). Names match voice ref filenames in REFS_DIR. "
                        "Pass empty string to disable.")
    p.add_argument("--gallery-threshold", type=float, default=0.80,
                   help="Cosine similarity threshold for gallery match. Below this = unknown.")
    p.add_argument("--trim-silence", action="store_true", default=True)
    p.add_argument("--gap-seconds", type=float, default=0.25)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # ── Load corrected text labels (override Qwen OCR per bubble) ─
    labels_by_key: dict[tuple[str, int], str] = {}
    if args.labels_csv:
        try:
            with open(args.labels_csv, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if (row.get("skip") or "").strip():
                        continue
                    ct = (row.get("corrected_text") or "").strip()
                    if not ct:
                        continue
                    try:
                        labels_by_key[(row["page"], int(row["order"]))] = ct
                    except (KeyError, ValueError):
                        continue
            print(f"[labels] {len(labels_by_key)} corrected texts loaded from {args.labels_csv}")
        except FileNotFoundError:
            print(f"[labels] CSV not found: {args.labels_csv} — skip override")

    # ── Load XTTS FT ──────────────────────────────────────────────
    print("[1/4] Loading XTTS FT...")
    sys.path.insert(0, "/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2/coqui_tts")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    config = XttsConfig()
    config.load_json(os.path.join(FT_DIR, "config.json"))
    xtts = Xtts.init_from_config(config)
    xtts.load_checkpoint(config, checkpoint_dir=FT_DIR, use_deepspeed=False, eval=True)
    xtts.cuda()
    print("  XTTS FT loaded on GPU")

    # ── Pre-compute conditioning latents for each char ref ──────
    print("[2/4] Pre-computing XTTS conditioning for 8 character refs...")
    voice_registry = {}
    refs_dir = Path(REFS_DIR)
    for ref_wav in sorted(refs_dir.glob("*.wav")):
        char_id = ref_wav.stem
        t0 = time.time()
        gpt_latent, spk_emb = xtts.get_conditioning_latents(
            audio_path=str(ref_wav), gpt_cond_len=6, max_ref_length=30
        )
        voice_registry[char_id] = (gpt_latent, spk_emb)
        print(f"  {char_id:15s}  ({time.time()-t0:.1f}s)")

    # ── Init CV pipeline (YOLO + Qwen-VL) ──────────────────────
    print("[3/4] Initializing CV pipeline (YOLO + Qwen-VL)...")
    sys.path.insert(0, "/home/bes/Desktop/Tin")
    # Attribute names match comic_pipeline.init_pipeline (which reads parse_args() output).
    cv_args = types.SimpleNamespace(
        weights=args.yolo_ckpt,
        speaker_model=args.speaker_mlp,
        speaker_scaler=str(Path(args.speaker_mlp).parent / "scaler.pkl"),
        ocr_engine=args.ocr_engine,
        qwen_lora_adapter=(args.qwen_lora_adapter or None),
        character_gallery=(args.character_gallery or None),
        gallery_threshold=args.gallery_threshold,
        reading_direction="ltr",
        lang="vi",
        no_gpu=True,        # PaddleOCR doesn't support 5090 sm_120
        no_ocr=False,
        no_face=False,
        paddle_only=False,
        char_db=None,
        verbose=False,
    )
    if args.qwen_lora_adapter:
        print(f"  Using Qwen-VL LoRA: {args.qwen_lora_adapter}")
    if args.character_gallery:
        print(f"  Using Character Gallery: {args.character_gallery}  (threshold={args.gallery_threshold:.2f})")
    from Capstone_project.scripts.comic_pipeline import init_pipeline, process_page
    cv_models = init_pipeline(cv_args)
    print("  CV pipeline ready")

    # ── Process pages ────────────────────────────────────────────
    print("[4/4] Processing pages + XTTS synthesis...")
    pages_dir = Path(args.pages_dir)
    page_files = sorted(list(pages_dir.glob("*.jpg")) +
                        list(pages_dir.glob("*.webp")) +
                        list(pages_dir.glob("*.png")))[:args.max_pages]
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
                img_path,
                yolo_model=cv_models["yolo"],
                ocr_pipeline=cv_models["ocr"],
                face_extractor=cv_models["extractor"],
                char_db=cv_models["char_db"],
                speaker_model_path=cv_models["speaker_model"],
                speaker_scaler_path=cv_models["speaker_scaler"],
                direction=cv_args.reading_direction,
                use_gpu=not cv_args.no_gpu,
                character_gallery=cv_models.get("gallery"),
            )
        except Exception as exc:
            print(f"  CV failed: {exc}")
            continue
        print(f"  CV done in {time.time()-t_cv:.1f}s — {len(page_result.get('bubbles', []))} bubbles")

        # Override bubble text from manual labels (strip pNN_ prefix from filename to match CSV `page`)
        csv_page = re.sub(r"^p\d+_", "", page_name)
        ovr = 0
        bubble_list = page_result.get("bubbles", [])
        for b in bubble_list:
            k = (csv_page, b.get("order"))
            if k in labels_by_key:
                b["text"] = labels_by_key[k]
                b["text_source"] = "csv_override"
                ovr += 1
        if ovr:
            print(f"  Override: {ovr}/{len(bubble_list)} bubble texts from CSV (page={csv_page})")
        elif labels_by_key:
            print(f"  Override: 0 matches for page={csv_page} (check CSV `page` column)")

        page_audio_chunks = []
        bubbles_info = []
        for bubble in sorted(page_result.get("bubbles", []), key=lambda b: b.get("order", 0)):
            order = bubble.get("order", 0)
            raw_text = bubble.get("text", "").strip()
            if not raw_text:
                continue
            text = clean_for_xtts(raw_text)
            if len(text.replace(",", "").replace(".", "").strip()) < 2:
                # All punctuation after cleaning — skip
                continue

            # Resolve character
            qwen_spk = bubble.get("qwen_speaker")
            mlp_spk = bubble.get("speaker_id")
            if qwen_spk and qwen_spk != "sound_effect":
                char_id = qwen_spk
            elif mlp_spk:
                char_id = mlp_spk
            else:
                char_id = "unknown"

            # Normalize to our ref names
            if char_id not in voice_registry:
                char_id = "default"

            gpt_latent, spk_emb = voice_registry[char_id]

            # TTS
            t0 = time.time()
            try:
                out = xtts.inference(
                    text, language="vi",
                    gpt_cond_latent=gpt_latent,
                    speaker_embedding=spk_emb,
                    **gen_config_for_text(text),
                )
            except Exception as exc:
                print(f"    [{order:02d}] TTS failed: {exc}")
                continue

            wav_np = np.asarray(out["wav"], dtype=np.float32)

            # Trim silence
            if args.trim_silence and len(wav_np) > 1000:
                import librosa
                wav_trim, _ = librosa.effects.trim(wav_np, top_db=25)
                if len(wav_trim) > 1000:
                    wav_np = wav_trim

            # Cap duration by text length — cuts hallucinated tails
            wav_np, truncated = cap_duration_by_text(wav_np, text)
            dur = len(wav_np) / SAMPLE_RATE
            marker = " ✂" if truncated else ""
            print(f"    [{order:02d}] {char_id:12s} {dur:.1f}s{marker}: \"{text[:50]}\"")

            # Save bubble wav
            bubble_path = bubble_dir / f"{page_name}_b{order:03d}_{char_id}.wav"
            sf.write(str(bubble_path), wav_np, SAMPLE_RATE)

            page_audio_chunks.append(wav_np)
            bubbles_info.append({
                "order": order, "char_id": char_id, "text": text,
                "duration": round(dur, 2), "path": str(bubble_path),
            })

        if not page_audio_chunks:
            print(f"  No audio for page")
            continue

        # Concatenate page audio with small gaps
        gap = np.zeros(int(args.gap_seconds * SAMPLE_RATE), dtype=np.float32)
        page_audio = []
        for i, chunk in enumerate(page_audio_chunks):
            page_audio.append(chunk)
            if i < len(page_audio_chunks) - 1:
                page_audio.append(gap)
        page_audio = np.concatenate(page_audio)

        page_wav = page_dir / f"{page_idx+1:03d}_{page_name}.wav"
        sf.write(str(page_wav), page_audio, SAMPLE_RATE)
        total_dur = len(page_audio) / SAMPLE_RATE
        print(f"  → {page_wav.name} ({total_dur:.1f}s, {len(bubbles_info)} bubbles)")

        all_page_audios.append(page_audio)

        # Save per-page json
        with open(page_dir / f"{page_idx+1:03d}_{page_name}.json", "w", encoding="utf-8") as f:
            json.dump({"page": page_name, "bubbles": bubbles_info,
                       "total_duration": round(total_dur, 2)}, f, ensure_ascii=False, indent=2)

    # ── Full chapter concat ─────────────────────────────────────
    if all_page_audios:
        gap_page = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        full = []
        for i, pa in enumerate(all_page_audios):
            full.append(pa)
            if i < len(all_page_audios) - 1:
                full.append(gap_page)
        full = np.concatenate(full)
        full_path = out_dir / "full_chapter.wav"
        sf.write(str(full_path), full, SAMPLE_RATE)
        print(f"\nFull chapter: {full_path} ({len(full)/SAMPLE_RATE:.1f}s)")

    print(f"\nDone. All output: {out_dir}")


if __name__ == "__main__":
    main()
