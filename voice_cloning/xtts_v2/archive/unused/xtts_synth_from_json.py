"""Stage-2 XTTS-only synthesis script.

Reads per-page JSON output of comic_pipeline_vits.py (which already has bubble
text + char_id from Qwen-VL LoRA OCR), and re-synthesizes each bubble with
XTTS FT using character reference audio.

This avoids needing Qwen-VL in the XTTS env (deps incompatible with
transformers 4.x that XTTS requires).

Usage:
    conda run -n xtts_env python xtts_synth_from_json.py \
        --src-json-dir demo_audio/doraemon_story_vits/pages \
        --out demo_audio/doraemon_story_xtts_lora \
        --refs-dir demo_refs_xtts
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000

FT_DIR_DEFAULT = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft"


# Map char_id (Vietnamese names + generic labels) → ref filename (without .wav)
# in --refs-dir. Fallback uses hash for unknown names.
CHAR_TO_REF = {
    "character_1": "character_1", "character_2": "character_2",
    "character_3": "character_3", "character_4": "character_4",
    "character_5": "character_5",
    # Doraemon cast
    "Doraemon": "character_1",   "doraemon": "character_1",   "Doremon": "character_1",
    "Nobita":   "character_2",   "nobita":   "character_2",
    "Shizuka":  "character_3",   "shizuka":  "character_3",
    "Suneo":    "character_4",   "suneo":    "character_4",
    "Jaian":    "character_5",   "jaian":    "character_5",   "Gian": "character_5",
    "Dorami":   "character_3",   "dorami":   "character_3",
    # Conan, Dragonball etc.
    "Conan":    "character_1",   "Goku":     "character_1",
    "Ran":      "character_3",   "Bulma":    "character_3",
    "Kogoro":   "character_5",
    # Fallbacks
    "default":     "default",
    "unknown":     "unknown",
    "sound_effect": None,   # skip
}
FALLBACK_REFS = ["character_1", "character_2", "character_3", "character_4", "character_5"]


def resolve_ref(char_id, refs_dir):
    if char_id in CHAR_TO_REF:
        v = CHAR_TO_REF[char_id]
        if v is None:
            return None
        return refs_dir / f"{v}.wav"
    # Hash fallback for unseen names
    h = sum(ord(c) for c in char_id)
    return refs_dir / f"{FALLBACK_REFS[h % len(FALLBACK_REFS)]}.wav"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-json-dir", required=True,
                   help="Dir containing per-page JSON (from comic_pipeline_vits.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--refs-dir", default="/home/bes/Desktop/Tin/demo_refs_xtts")
    p.add_argument("--ft-dir", default=FT_DIR_DEFAULT)
    p.add_argument("--gap-seconds", type=float, default=0.25)
    p.add_argument("--trim-silence", action="store_true", default=True)
    p.add_argument("--trim-top-db", type=float, default=20.0,
                   help="librosa.effects.trim top_db (lower = trim more aggressively)")
    p.add_argument("--temperature", type=float, default=0.4,
                   help="XTTS GPT sampling temperature. Lower = more text-faithful.")
    p.add_argument("--length-penalty", type=float, default=0.5)
    p.add_argument("--repetition-penalty", type=float, default=12.0)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--sec-per-word", type=float, default=0.6,
                   help="Hard cap: max seconds per Vietnamese word in input text")
    p.add_argument("--dur-headroom", type=float, default=1.0,
                   help="Extra seconds added to the per-word cap")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Limit pages for debug (default: all)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pages").mkdir(exist_ok=True)
    (out_dir / "bubbles").mkdir(exist_ok=True)
    refs_dir = Path(args.refs_dir)

    # ── Load XTTS FT ──────────────────────────────────────────────
    print("[1/3] Loading XTTS FT...")
    sys.path.insert(0, "/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2/coqui_tts")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import torch

    config = XttsConfig()
    config.load_json(os.path.join(args.ft_dir, "config.json"))
    xtts = Xtts.init_from_config(config)
    xtts.load_checkpoint(config, checkpoint_dir=args.ft_dir, use_deepspeed=False, eval=True)
    xtts.cuda()
    print("  XTTS FT loaded on GPU")

    # ── Pre-compute conditioning latents for each ref ────────────
    print("[2/3] Pre-computing XTTS conditioning latents...")
    voice_registry = {}
    for ref_wav in sorted(refs_dir.glob("*.wav")):
        char_id = ref_wav.stem
        t0 = time.time()
        gpt_latent, spk_emb = xtts.get_conditioning_latents(
            audio_path=str(ref_wav), gpt_cond_len=6, max_ref_length=30
        )
        voice_registry[char_id] = (gpt_latent, spk_emb)
        print(f"  {char_id:15s} ({time.time()-t0:.1f}s)")

    # ── Process each page JSON ──────────────────────────────────
    print("[3/3] Synthesizing bubbles from JSON...")
    json_files = sorted(Path(args.src_json_dir).glob("*.json"))
    if args.max_pages:
        json_files = json_files[:args.max_pages]
    print(f"  {len(json_files)} pages to process")

    all_page_audios = []
    n_bubbles = 0
    for page_idx, jf in enumerate(json_files, 1):
        with open(jf) as f:
            data = json.load(f)
        page_stem = data["page"]
        print(f"\n[Page {page_idx}/{len(json_files)}] {page_stem}")

        page_audio_chunks = []
        bubbles_info = []
        for b in data["bubbles"]:
            order   = b["order"]
            char_id = b["char_id"]
            text    = b["text"].strip()
            if not text:
                continue

            ref_path = resolve_ref(char_id, refs_dir)
            if ref_path is None:
                print(f"    [{order:02d}] skip ({char_id})")
                continue
            ref_key = ref_path.stem
            if ref_key not in voice_registry:
                ref_key = "default"
            gpt_latent, spk_emb = voice_registry[ref_key]

            # Anti-hallucination: ensure text ends with sentence terminator so
            # XTTS GPT knows where to stop. Without it the model often continues
            # generating phonemes past the actual content.
            text_norm = text.rstrip()
            if text_norm and text_norm[-1] not in ".?!…":
                text_norm = text_norm + "."

            t0 = time.time()
            try:
                out = xtts.inference(
                    text_norm, language="vi",
                    gpt_cond_latent=gpt_latent,
                    speaker_embedding=spk_emb,
                    # Lower temperature + tighter top_p → more text-faithful,
                    # less likely to add unwritten phonemes
                    temperature=args.temperature,
                    length_penalty=args.length_penalty,
                    repetition_penalty=args.repetition_penalty,
                    top_k=args.top_k, top_p=args.top_p,
                )
            except Exception as exc:
                print(f"    [{order:02d}] XTTS fail: {exc}")
                continue
            wav_np = np.asarray(out["wav"], dtype=np.float32)

            # More aggressive silence trim (top_db lower = trim quieter speech too)
            if args.trim_silence and len(wav_np) > 1000:
                import librosa
                wav_trim, _ = librosa.effects.trim(wav_np, top_db=args.trim_top_db)
                if len(wav_trim) > 1000:
                    wav_np = wav_trim

            # Hard duration cap based on text word count (Vietnamese ~3-4 syl/sec
            # average → cap = 0.6s/word + 1s headroom). If audio still longer
            # after silence trim, the tail is almost certainly hallucinated.
            n_words = max(1, len(text_norm.split()))
            max_dur_s = n_words * args.sec_per_word + args.dur_headroom
            max_samples = int(max_dur_s * SAMPLE_RATE)
            if len(wav_np) > max_samples:
                wav_np = wav_np[:max_samples]
                # Re-trim silence after cap so we don't end mid-phoneme abruptly
                if args.trim_silence and len(wav_np) > 1000:
                    wav_trim, _ = librosa.effects.trim(wav_np, top_db=args.trim_top_db)
                    if len(wav_trim) > 1000:
                        wav_np = wav_trim

            dur = len(wav_np) / SAMPLE_RATE
            print(f"    [{order:02d}] {char_id:12s} ref={ref_key:12s} {dur:.1f}s ({time.time()-t0:.1f}s)")

            bubble_path = out_dir / "bubbles" / f"{page_idx:03d}_{page_stem}_b{order:03d}_{char_id}_{ref_key}.wav"
            sf.write(str(bubble_path), wav_np, SAMPLE_RATE)

            page_audio_chunks.append(wav_np)
            bubbles_info.append({
                "order": order, "char_id": char_id, "ref": ref_key, "text": text,
                "duration": round(dur, 2), "path": str(bubble_path),
            })
            n_bubbles += 1

        if not page_audio_chunks:
            continue

        gap = np.zeros(int(args.gap_seconds * SAMPLE_RATE), dtype=np.float32)
        page_audio = []
        for i, chunk in enumerate(page_audio_chunks):
            page_audio.append(chunk)
            if i < len(page_audio_chunks) - 1:
                page_audio.append(gap)
        page_audio = np.concatenate(page_audio)

        page_wav = out_dir / "pages" / f"{page_idx:03d}_{page_stem}.wav"
        sf.write(str(page_wav), page_audio, SAMPLE_RATE)
        total_dur = len(page_audio) / SAMPLE_RATE
        print(f"  → {page_wav.name} ({total_dur:.1f}s, {len(bubbles_info)} bubbles)")
        all_page_audios.append(page_audio)

        with open(out_dir / "pages" / f"{page_idx:03d}_{page_stem}.json", "w", encoding="utf-8") as f:
            json.dump({"page": page_stem, "bubbles": bubbles_info,
                       "total_duration": round(total_dur, 2)}, f, ensure_ascii=False, indent=2)

    # Full chapter
    if all_page_audios:
        gap_page = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        full = []
        for i, pa in enumerate(all_page_audios):
            full.append(pa)
            if i < len(all_page_audios) - 1:
                full.append(gap_page)
        full = np.concatenate(full)
        sf.write(str(out_dir / "full_chapter.wav"), full, SAMPLE_RATE)
        print(f"\nFull chapter: {len(full)/SAMPLE_RATE:.1f}s ({n_bubbles} bubbles)")
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
