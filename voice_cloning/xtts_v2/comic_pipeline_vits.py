"""Run comic pipeline on Vietnamese comic pages using VITS2 Variant D
(+ Dual-Path + Cross-Attention) for multi-speaker TTS.

Flow:
  1. Reuse comic_pipeline (YOLO + Qwen-VL OCR + speaker attribution) to extract bubbles
  2. Map each character_id to a FIXED VieNeu speaker_id (NOT zero-shot cloning)
  3. Synthesize each bubble with VITS2 Variant D using its character's speaker_id
  4. Concatenate into per-page audio + full chapter audio

Why fixed speaker mapping (not cloning):
  Phase 3 zero-shot clone plateaued at sim=0.43 — too weak for demo. Variant D
  on a fixed speaker_id from the 193 VieNeu speakers produces clean audio with
  correct tones (Dual-Path), at the cost of not matching the character's ref voice.
  This isolates the *Vietnamese TTS quality* contribution from the cloning angle
  (which XTTS already covers).

Usage:
    python comic_pipeline_vits.py --pages-dir <dir> --out <out_dir>
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_VITS2_DIR = os.path.join(_PROJECT_ROOT, "vits2_pytorch")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _VITS2_DIR)

SAMPLE_RATE = 24000

DEFAULT_CONFIG = "Capstone_project/configs/vits2_vieneu_variant_d.json"
DEFAULT_CKPT   = "/mnt/nfs-data/tin_dataset/vits2_logs/vieneu_variant_d/G_466000.pth"

# Fixed character → VieNeu speaker_id mapping. Chosen for voice diversity;
# can be re-tuned by listening to each speaker's reference samples.
CHARACTER_TO_SID = {
    # Generic Qwen labels
    "character_1": 17, "character_2": 0,  "character_3": 42,
    "character_4": 75, "character_5": 19, "character_6": 84,
    "character_7": 92, "character_8": 111,
    # Doraemon cast (Qwen may return character names directly)
    "Doraemon":  17,  "doraemon":  17,
    "Nobita":    42,  "nobita":    42,
    "Shizuka":   0,   "shizuka":   0,
    "Suneo":     75,  "suneo":     75,
    "Jaian":     84,  "jaian":     84,
    "Gian":      84,  "gian":      84,
    "Dorami":    19,  "dorami":    19,
    # Conan
    "Conan":     17,  "conan":     17,
    "Ran":       0,   "ran":       0,
    "Kogoro":    84,  "kogoro":    84,
    # Dragonball
    "Goku":      17,  "goku":      17,
    "Bulma":     0,   "bulma":     0,
    # Fallback
    "default":   10,  "unknown":   10,
}
# Pool of speaker IDs used for hash-based fallback (covers ~5 distinct voices)
FALLBACK_POOL = [17, 0, 42, 75, 19, 84, 92, 111]
SKIP_CHARACTERS = {"sound_effect"}  # do not synthesize these


def char_to_sid(char_id):
    if char_id in CHARACTER_TO_SID:
        return CHARACTER_TO_SID[char_id]
    # Stable hash → consistent voice across pages for any new character name
    h = sum(ord(c) for c in char_id)
    return FALLBACK_POOL[h % len(FALLBACK_POOL)]

# Same foreign-name vietnamization as XTTS pipeline so both demos compare fair
FOREIGN_NAME_MAP = {
    "Doraemon": "Đô-rê-mon", "doraemon": "đô-rê-mon",
    "Nobita": "Nô-bi-ta",    "nobita": "nô-bi-ta",
    "Shizuka": "Si-zu-ka",   "shizuka": "si-zu-ka",
    "Suneo": "Su-nê-ô",      "suneo": "su-nê-ô",
    "Jaian": "Cha-i-an",     "jaian": "cha-i-an",
    "Gian": "Chi-an",        "gian": "chi-an",
    "Dorami": "Đô-ra-mi",    "dorami": "đô-ra-mi",
    "Mineko": "Mi-nê-kô",    "mineko": "mi-nê-kô",
    "Tokyo": "Tô-ky-ô",      "tokyo": "tô-ky-ô",
    "Naruto": "Na-ru-tô",    "naruto": "na-ru-tô",
    "Sasuke": "Sa-su-kê",    "sasuke": "sa-su-kê",
    "Sakura": "Sa-ku-ra",    "sakura": "sa-ku-ra",
}


def vietnamize_foreign_names(text):
    out = text
    for foreign, viet in FOREIGN_NAME_MAP.items():
        out = out.replace(foreign, viet)
    return out


def get_text_and_tone(text, hps):
    """Vietnamese phoneme + tone extraction (mirrors synth_heldout_variant_d)."""
    import commons
    from text import text_to_sequence
    from Capstone_project.tone_encoder.tone_utils import text_to_tone_sequence

    if hps.data.text_cleaners[0] != "vietnamese_cleaners":
        raise NotImplementedError(f"Unknown cleaner: {hps.data.text_cleaners}")

    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    tone_norm = text_to_tone_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
        tone_norm = commons.intersperse(tone_norm, 0)
    assert len(text_norm) == len(tone_norm)
    return torch.LongTensor(text_norm), torch.LongTensor(tone_norm)


def resolve_char(bubble):
    qwen_spk = bubble.get("qwen_speaker")
    mlp_spk = bubble.get("speaker_id")
    if qwen_spk and qwen_spk != "sound_effect":
        return qwen_spk
    if mlp_spk:
        return mlp_spk
    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--skip-pages", type=int, default=0,
                   help="Skip first N pages in the sorted page list (e.g. cover pages)")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ckpt",   default=DEFAULT_CKPT)
    p.add_argument("--ocr-engine", default="qwen-vl")
    p.add_argument("--yolo-ckpt", default="/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt")
    p.add_argument("--speaker-mlp", default="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_mlp.pt")
    p.add_argument("--qwen-model",
                   default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HF model name for Qwen-VL (3B or 7B). LoRA must match.")
    p.add_argument("--qwen-lora-adapter",
                   default="/mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_3b_vncomic_lora_v1/best",
                   help="Fine-tuned Qwen-VL LoRA adapter for OCR. Pass empty to disable.")
    p.add_argument("--noise-scale",   type=float, default=0.667)
    p.add_argument("--noise-scale-w", type=float, default=0.8)
    p.add_argument("--length-scale",  type=float, default=1.0)
    p.add_argument("--trim-silence", action="store_true", default=True)
    p.add_argument("--gap-seconds", type=float, default=0.25)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # ── Load VITS2 Variant D ────────────────────────────────────
    print("[1/3] Loading VITS2 Variant D...")
    import utils
    from text.symbols import symbols
    from Capstone_project.models.Vits2 import build_synthesizer

    hps = utils.get_hparams_from_file(args.config)
    net_g = build_synthesizer(hps, len(symbols)).to(device).eval()
    utils.load_checkpoint(args.ckpt, net_g, None)
    print(f"  loaded {args.ckpt}")
    print(f"  character→sid mapping: {CHARACTER_TO_SID}")

    # ── Init CV pipeline (YOLO + Qwen-VL) ──────────────────────
    print("[2/3] Initializing CV pipeline (YOLO + Qwen-VL)...")
    cv_args = types.SimpleNamespace(
        weights=args.yolo_ckpt,
        lang="vi",
        no_ocr=False,
        no_face=False,
        paddle_only=False,
        no_gpu=True,            # PaddleOCR doesn't support 5090 sm_120
        ocr_engine=args.ocr_engine,
        qwen_model=args.qwen_model,
        qwen_lora_adapter=(args.qwen_lora_adapter or None),
        char_db=None,
        speaker_model="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_mlp.pt",
        speaker_scaler="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/scaler.pkl",
        direction="ltr",
        rule_based=False,
        yolo_conf=0.25,
        verbose=False,
    )
    if args.qwen_lora_adapter:
        print(f"  Using Qwen-VL LoRA: {args.qwen_lora_adapter}")
    from Capstone_project.scripts.comic_pipeline import init_pipeline, process_page
    cv_models = init_pipeline(cv_args)
    print("  CV pipeline ready")

    # ── Process pages ────────────────────────────────────────────
    print("[3/3] Processing pages + VITS2 synthesis...")
    pages_dir = Path(args.pages_dir)
    all_pages = sorted(list(pages_dir.glob("*.jpg")) +
                       list(pages_dir.glob("*.webp")) +
                       list(pages_dir.glob("*.png")))
    page_files = all_pages[args.skip_pages : args.skip_pages + args.max_pages]
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
                face_extractor      = cv_models.get("extractor"),
                char_db             = cv_models.get("char_db"),
                speaker_model_path  = cv_models.get("speaker_model"),
                speaker_scaler_path = cv_models.get("speaker_scaler"),
                direction           = cv_args.direction,
                rule_based          = cv_args.rule_based,
                use_gpu             = not cv_args.no_gpu,
                yolo_conf           = cv_args.yolo_conf,
            )
        except Exception as exc:
            print(f"  CV failed: {exc}")
            continue
        bubbles = page_result.get("bubbles", [])
        print(f"  CV done in {time.time()-t_cv:.1f}s — {len(bubbles)} bubbles")

        page_audio_chunks = []
        bubbles_info = []
        for bubble in sorted(bubbles, key=lambda b: b.get("order", 0)):
            order = bubble.get("order", 0)
            text = bubble.get("text", "").strip()
            if not text:
                continue

            char_id = resolve_char(bubble)
            if char_id in SKIP_CHARACTERS:
                print(f"    [{order:02d}] skip ({char_id})")
                continue
            sid = char_to_sid(char_id)

            text = vietnamize_foreign_names(text)

            # TTS
            t0 = time.time()
            try:
                stn, tone_seq = get_text_and_tone(text, hps)
            except Exception as exc:
                print(f"    [{order:02d}] g2p fail: {exc} — text={text[:40]!r}")
                continue

            with torch.no_grad():
                x = stn.to(device).unsqueeze(0)
                x_lens = torch.LongTensor([stn.size(0)]).to(device)
                tone = tone_seq.to(device).unsqueeze(0)
                sid_t = torch.LongTensor([sid]).to(device)
                try:
                    audio_t = net_g.infer(
                        x, x_lens, sid=sid_t, tone=tone,
                        noise_scale=args.noise_scale,
                        noise_scale_w=args.noise_scale_w,
                        length_scale=args.length_scale,
                    )[0][0, 0]
                except Exception as exc:
                    print(f"    [{order:02d}] VITS2 infer fail: {exc}")
                    continue

            wav_np = audio_t.cpu().float().numpy().astype(np.float32)

            # Trim silence
            if args.trim_silence and len(wav_np) > 1000:
                import librosa
                wav_trim, _ = librosa.effects.trim(wav_np, top_db=25)
                if len(wav_trim) > 1000:
                    wav_np = wav_trim

            dur = len(wav_np) / SAMPLE_RATE
            print(f"    [{order:02d}] {char_id:12s} sid={sid:3d} {dur:.1f}s ({time.time()-t0:.1f}s): \"{text[:50]}\"")

            bubble_path = bubble_dir / f"{page_idx+1:03d}_{page_name}_b{order:03d}_{char_id}_sid{sid:03d}.wav"
            sf.write(str(bubble_path), wav_np, SAMPLE_RATE)

            page_audio_chunks.append(wav_np)
            # Bbox (xyxy) from pipeline — needed for overlay rendering
            bbox = bubble.get("bbox") or bubble.get("bbox_xyxy")
            bubbles_info.append({
                "order": order, "char_id": char_id, "sid": sid, "text": text,
                "duration": round(dur, 2), "path": str(bubble_path),
                "bbox": bbox,
            })

        if not page_audio_chunks:
            print(f"  No audio for page")
            continue

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
