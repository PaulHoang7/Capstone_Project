"""Post-process XTTS bubble audio to eliminate hallucination tails using
Whisper word-level forced alignment.

Pipeline:
    For each bubble WAV:
      1. Transcribe with faster-whisper (lang=vi) → word + timestamps
      2. Match transcribed words against target text from JSON (normalized)
      3. Find the LAST target word in the transcription → take its end_time
      4. Cut audio at end_time + buffer
      5. If no match, keep original (don't risk over-trimming)

Then rebuild page-level + chapter audio + new JSON.

Usage:
    conda run -n xtts_env python xtts_ctc_trim.py \
        --src demo_audio/doraemon_story_xtts_v5 \
        --out demo_audio/doraemon_story_xtts_v6
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000


def normalize_word(w):
    """Strip diacritics, lowercase, remove punct — for matching."""
    if not w:
        return ""
    w = w.lower().strip()
    # NFD then drop combining marks
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    # Strip non-letter chars
    w = re.sub(r"[^a-z0-9]", "", w)
    return w


def tokenize_text(text):
    """Split target text into normalized comparable tokens."""
    raw = re.findall(r"\S+", text)
    out = []
    for w in raw:
        n = normalize_word(w)
        if n:
            out.append(n)
    return out


def find_last_target_end(target_tokens, whisper_words):
    """Walk through whisper_words, find the last index that matches the last
    target token (or its prefix if not exact). Return end_time of that word.

    Strategy: longest-prefix match — find the latest whisper word whose
    normalized form == any target token. Among ties, prefer the one matching
    the LAST target token to anchor the actual end.
    """
    if not target_tokens or not whisper_words:
        return None

    last_target = target_tokens[-1]
    target_set = set(target_tokens)

    # Scan whisper words from the end; take the latest one that matches the
    # final target word; if none, fall back to the latest one matching ANY
    # target word.
    end_time = None
    for ww in reversed(whisper_words):
        nw = normalize_word(ww["word"])
        if nw == last_target:
            end_time = ww["end"]
            break
    if end_time is None:
        for ww in reversed(whisper_words):
            nw = normalize_word(ww["word"])
            if nw in target_set:
                end_time = ww["end"]
                break
    return end_time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Dir from xtts_synth_from_json (has pages/, bubbles/)")
    p.add_argument("--out", required=True)
    p.add_argument("--whisper-model", default="medium",
                   help="faster-whisper model size (tiny|base|small|medium|large-v3)")
    p.add_argument("--buffer-sec", type=float, default=0.18)
    p.add_argument("--gap-seconds", type=float, default=0.25)
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "bubbles").mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading faster-whisper {args.whisper_model}...")
    from faster_whisper import WhisperModel
    asr = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
    print("  loaded")

    json_files = sorted((src / "pages").glob("*.json"))
    print(f"[2/3] Processing {len(json_files)} pages...")

    all_page_audios = []
    stats = {"total": 0, "trimmed": 0, "no_match": 0, "removed_total_s": 0.0}

    for page_idx, jf in enumerate(json_files, 1):
        with open(jf) as f:
            data = json.load(f)
        page_stem = data["page"]
        print(f"\n[Page {page_idx}/{len(json_files)}] {page_stem}")

        page_audio_chunks = []
        new_bubbles = []
        for b in data["bubbles"]:
            stats["total"] += 1
            order   = b["order"]
            text    = b["text"]
            src_wav = Path(b["path"])
            if not src_wav.exists():
                # Path may be relative — try src/bubbles/<name>
                src_wav = src / "bubbles" / Path(b["path"]).name
            if not src_wav.exists():
                print(f"    [{order:02d}] missing wav: {b['path']}")
                continue

            audio, sr = sf.read(str(src_wav))
            if sr != SAMPLE_RATE:
                print(f"    [{order:02d}] unexpected sr={sr}")
            orig_dur = len(audio) / sr

            target_tokens = tokenize_text(text)
            if not target_tokens:
                # No content to match; keep original
                kept = audio
                end_time = None
            else:
                segments, _info = asr.transcribe(
                    str(src_wav), language="vi", word_timestamps=True,
                    vad_filter=False, beam_size=1, condition_on_previous_text=False,
                )
                whisper_words = []
                for seg in segments:
                    if seg.words:
                        whisper_words.extend([
                            {"word": w.word, "start": w.start, "end": w.end}
                            for w in seg.words
                        ])
                end_time = find_last_target_end(target_tokens, whisper_words)
                if end_time is None:
                    stats["no_match"] += 1
                    kept = audio  # safer to keep
                else:
                    end_samples = int((end_time + args.buffer_sec) * sr)
                    end_samples = min(end_samples, len(audio))
                    kept = audio[:end_samples]
                    if end_samples < len(audio):
                        stats["trimmed"] += 1
                        stats["removed_total_s"] += (len(audio) - end_samples) / sr

            new_dur = len(kept) / sr
            cut_s = orig_dur - new_dur
            tag = f"cut={cut_s:.2f}s" if cut_s > 0.05 else "kept"
            print(f"    [{order:02d}] {b['char_id']:12s} orig={orig_dur:.1f}s → {new_dur:.1f}s ({tag})")

            new_wav_path = out / "bubbles" / src_wav.name
            sf.write(str(new_wav_path), kept.astype(np.float32), sr)
            page_audio_chunks.append(kept.astype(np.float32))
            new_bubbles.append({**b, "duration": round(new_dur, 2),
                                "path": str(new_wav_path),
                                "trim_cut_s": round(cut_s, 2)})

        if not page_audio_chunks:
            continue

        gap = np.zeros(int(args.gap_seconds * SAMPLE_RATE), dtype=np.float32)
        page_audio = []
        for i, chunk in enumerate(page_audio_chunks):
            page_audio.append(chunk)
            if i < len(page_audio_chunks) - 1:
                page_audio.append(gap)
        page_audio = np.concatenate(page_audio)

        page_wav = out / "pages" / f"{page_idx:03d}_{page_stem}.wav"
        sf.write(str(page_wav), page_audio, SAMPLE_RATE)
        total_dur = len(page_audio) / SAMPLE_RATE
        with open(out / "pages" / f"{page_idx:03d}_{page_stem}.json", "w", encoding="utf-8") as f:
            json.dump({"page": page_stem, "bubbles": new_bubbles,
                       "total_duration": round(total_dur, 2)}, f, ensure_ascii=False, indent=2)
        all_page_audios.append(page_audio)

    if all_page_audios:
        gap_page = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        full = []
        for i, pa in enumerate(all_page_audios):
            full.append(pa)
            if i < len(all_page_audios) - 1:
                full.append(gap_page)
        full = np.concatenate(full)
        sf.write(str(out / "full_chapter.wav"), full, SAMPLE_RATE)
        full_dur = len(full) / SAMPLE_RATE
    else:
        full_dur = 0.0

    print(f"\n[3/3] Done.")
    print(f"  total bubbles:   {stats['total']}")
    print(f"  trimmed:         {stats['trimmed']}")
    print(f"  no whisper match: {stats['no_match']} (kept original)")
    print(f"  removed audio:   {stats['removed_total_s']:.1f}s")
    print(f"  full chapter:    {full_dur:.1f}s")
    print(f"  output:          {out}")


if __name__ == "__main__":
    main()
