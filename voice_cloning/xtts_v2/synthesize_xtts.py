"""Stage 2: Read bubble JSONs (from stage 1), synthesize with XTTS FT.

Runs in `xtts_ft` conda env. Reads page JSONs produced by extract_bubbles_cv.py.
"""
import argparse, json, os, sys, time, re
from pathlib import Path
import numpy as np
import soundfile as sf

# Shared XTTS config — see xtts_gen_config.py for rationale.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtts_gen_config import (
    XTTS_GEN_CONFIG, SAMPLE_RATE, clean_for_xtts, cap_duration_by_text,
    DEFAULT_CHARS_PER_SEC, DEFAULT_DURATION_BUFFER, DEFAULT_MIN_DURATION,
)

REFS_DIR = "/home/bes/Desktop/Tin/demo_refs_xtts"
FT_DIR   = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft"


# Kept for backward-compat with callers that imported this symbol.
normalize_text_for_xtts = clean_for_xtts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-json-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gap-seconds", type=float, default=0.2)
    p.add_argument("--page-gap", type=float, default=0.4)
    p.add_argument("--trim-silence", action="store_true", default=True)
    # XTTS inference params — defaults from shared XTTS_GEN_CONFIG (see xtts_gen_config.py).
    # Override only if you need to experiment with config.
    p.add_argument("--temperature",  type=float, default=XTTS_GEN_CONFIG["temperature"])
    p.add_argument("--length-penalty", type=float, default=XTTS_GEN_CONFIG["length_penalty"])
    p.add_argument("--rep-penalty",  type=float, default=XTTS_GEN_CONFIG["repetition_penalty"],
                   help="Repetition penalty (shared default — high to prevent loops)")
    p.add_argument("--top-k", type=int, default=XTTS_GEN_CONFIG["top_k"])
    p.add_argument("--top-p", type=float, default=XTTS_GEN_CONFIG["top_p"])
    # Duration cap based on text length — cuts off hallucinated extra content
    p.add_argument("--chars-per-sec", type=float, default=DEFAULT_CHARS_PER_SEC,
                   help="Expected VN speech rate (chars/sec). Caps audio to len(text)/rate + buffer.")
    p.add_argument("--duration-buffer", type=float, default=DEFAULT_DURATION_BUFFER,
                   help="Extra seconds added to max duration cap.")
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                   help="Absolute minimum audio duration (sec).")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bubble_dir = out_dir / "bubbles"; bubble_dir.mkdir(exist_ok=True)
    page_dir = out_dir / "pages"; page_dir.mkdir(exist_ok=True)

    print("[1/4] Loading XTTS FT...")
    sys.path.insert(0, "/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2/coqui_tts")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    config = XttsConfig()
    config.load_json(os.path.join(FT_DIR, "config.json"))
    xtts = Xtts.init_from_config(config)
    xtts.load_checkpoint(config, checkpoint_dir=FT_DIR, use_deepspeed=False, eval=True)
    xtts.cuda()
    print(f"  Loaded. GPU mem: {__import__('torch').cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Params: temp={args.temperature} rep_pen={args.rep_penalty} top_k={args.top_k} top_p={args.top_p}")

    print("[2/4] Pre-computing conditioning latents for refs...")
    voice_registry = {}
    for ref_wav in sorted(Path(REFS_DIR).glob("*.wav")):
        char_id = ref_wav.stem
        gpt_latent, spk_emb = xtts.get_conditioning_latents(
            audio_path=str(ref_wav), gpt_cond_len=6, max_ref_length=30
        )
        voice_registry[char_id] = (gpt_latent, spk_emb)
    print(f"  {len(voice_registry)} voices ready")

    index_path = Path(args.cv_json_dir) / "index.json"
    with open(index_path) as f:
        pages_meta = json.load(f)
    print(f"[3/4] {len(pages_meta)} pages to synthesize")

    import librosa
    all_page_audios = []
    for page_meta in pages_meta:
        page_json = page_meta["json"]
        page_name = page_meta["name"]
        page_idx = page_meta["page_idx"]

        with open(page_json) as f:
            page_result = json.load(f)

        print(f"\n[Page {page_idx}] {page_name}")
        page_audio_chunks = []
        bubbles = sorted(page_result.get("bubbles", []), key=lambda b: b.get("order", 0))

        for bubble in bubbles:
            order = bubble.get("order", 0)
            raw = bubble.get("text", "").strip()
            if not raw:
                continue
            text = normalize_text_for_xtts(raw)

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

            gpt_latent, spk_emb = voice_registry[char_id]
            try:
                out = xtts.inference(
                    text, language="vi",
                    gpt_cond_latent=gpt_latent, speaker_embedding=spk_emb,
                    temperature=args.temperature,
                    length_penalty=args.length_penalty,
                    repetition_penalty=args.rep_penalty,
                    top_k=args.top_k, top_p=args.top_p,
                )
            except Exception as exc:
                print(f"  [{order:02d}] TTS failed: {exc}")
                continue

            wav_np = np.asarray(out["wav"], dtype=np.float32)
            if args.trim_silence and len(wav_np) > 1000:
                wav_trim, _ = librosa.effects.trim(wav_np, top_db=25)
                if len(wav_trim) > 1000:
                    wav_np = wav_trim

            # Cap duration by text length — cuts off hallucinated extra content
            wav_np, truncated = cap_duration_by_text(
                wav_np, text,
                chars_per_sec=args.chars_per_sec,
                buffer_sec=args.duration_buffer,
                min_sec=args.min_duration,
            )

            dur = len(wav_np) / SAMPLE_RATE
            marker = " ✂" if truncated else ""
            print(f"  [{order:02d}] {char_id:12s} {dur:4.1f}s{marker}  \"{text[:55]}\"")

            bubble_path = bubble_dir / f"{page_idx:03d}_{page_name}_b{order:03d}_{char_id}.wav"
            sf.write(str(bubble_path), wav_np, SAMPLE_RATE)
            page_audio_chunks.append(wav_np)

        if not page_audio_chunks:
            continue

        gap = np.zeros(int(args.gap_seconds * SAMPLE_RATE), dtype=np.float32)
        page_audio = []
        for i, c in enumerate(page_audio_chunks):
            page_audio.append(c)
            if i < len(page_audio_chunks) - 1:
                page_audio.append(gap)
        page_audio = np.concatenate(page_audio)

        page_wav = page_dir / f"{page_idx:03d}_{page_name}.wav"
        sf.write(str(page_wav), page_audio, SAMPLE_RATE)
        print(f"  → {page_wav.name} ({len(page_audio)/SAMPLE_RATE:.1f}s)")
        all_page_audios.append(page_audio)

    if all_page_audios:
        gap_page = np.zeros(int(args.page_gap * SAMPLE_RATE), dtype=np.float32)
        full = []
        for i, p in enumerate(all_page_audios):
            full.append(p)
            if i < len(all_page_audios) - 1:
                full.append(gap_page)
        full = np.concatenate(full)
        full_path = out_dir / "full_chapter.wav"
        sf.write(str(full_path), full, SAMPLE_RATE)
        print(f"\n[4/4] Full chapter: {full_path} ({len(full)/SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
