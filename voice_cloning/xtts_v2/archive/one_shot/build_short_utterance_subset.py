"""Phase B Track B3 — Build short-utterance subset from VieNeu-TTS-140h.

Motivation:
  VieNeu is newspaper-reading (long sentences). XTTS fine-tuned on it learns
  long-form patterns and DRIFTS on short comic bubbles ("hả?", "Đi đi!").
  We need short-utterance training data, but no labeled set exists.

Approach (no manual labeling needed):
  1. For each (audio, full_text) in train.csv:
     - Split full_text into phrases on `,`, `;`, `.` punctuation
     - Filter phrases that are 2-8 words (comic-bubble length)
  2. For each candidate phrase:
     - Whisper-transcribe the full audio with word-level timestamps
     - Match phrase tokens (normalized) against transcribed words
     - Locate start_time of first phrase-word, end_time of last phrase-word
  3. Slice audio[start:end], save to short_subset_wavs/, log to short_subset.csv
  4. Output: a new csv that train_xtts_lora.py can consume directly
     (same `audio_file|text|speaker_name` format as upstream train.csv)

Quality control:
  - Only keep slices where ≥80% of phrase words matched in Whisper output
  - Drop slices <0.4s or >5s (sanity bounds for short utterance)
  - Add small buffer (0.1s) before/after slice boundary

Runtime estimate: ~6-8h for full 60K samples on RTX 5090 with Whisper-small.
Run with tmux/nohup. Resumable: skips already-processed audio_files.

Usage:
    conda activate xtts_env  # has faster-whisper installed
    python build_short_utterance_subset.py \
        --train-csv /mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits/train.csv \
        --out-dir /mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/short_subset \
        --whisper-model small \
        --max-source-samples 0   # 0 = all; set small for testing

Output structure:
    short_subset/
      ├── wavs/                          ← sliced audio files
      │   ├── capybara1812_0046_454__s0.wav
      │   └── ...
      ├── short_subset.csv               ← LoRA-trainable manifest
      └── progress.json                  ← resume marker
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

# ─── Constants ────────────────────────────────────────────────────────
SAMPLE_RATE = 24000
WHISPER_SR = 16000
BUFFER_SEC = 0.10        # padding around start/end of slice
MIN_PHRASE_WORDS = 2
MAX_PHRASE_WORDS = 8
MIN_SLICE_SEC = 0.4
MAX_SLICE_SEC = 5.0
MIN_MATCH_RATIO = 0.8    # require ≥80% phrase words found in Whisper output


# ─── Text utilities ──────────────────────────────────────────────────
def normalize_word_for_match(w: str) -> str:
    """Strip diacritics, lowercase, drop punctuation."""
    if not w:
        return ""
    w = w.lower().strip()
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = re.sub(r"[^a-z0-9]", "", w)
    return w


def split_into_phrases(text: str) -> list[str]:
    """Split full sentence on phrase punctuation. Returns phrases trimmed."""
    parts = re.split(r"[,;.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def is_valid_phrase(phrase: str) -> bool:
    """Filter phrases to comic-bubble length."""
    word_count = len(phrase.split())
    return MIN_PHRASE_WORDS <= word_count <= MAX_PHRASE_WORDS


# ─── Whisper alignment ───────────────────────────────────────────────
def transcribe_with_timestamps(asr, audio_path: str) -> list[dict]:
    """Run Whisper with word_timestamps. Returns list of {word, start, end}."""
    segments, _ = asr.transcribe(
        audio_path, language="vi", word_timestamps=True,
        vad_filter=False, beam_size=1, condition_on_previous_text=False,
    )
    words = []
    for seg in segments:
        if seg.words:
            words.extend([
                {"word": w.word, "start": w.start, "end": w.end}
                for w in seg.words
            ])
    return words


def locate_phrase_in_words(phrase: str, whisper_words: list[dict]):
    """Find first/last word of phrase in transcribed words.

    Returns (start_time, end_time, match_ratio) or None if not enough match.
    """
    phrase_tokens = [normalize_word_for_match(w) for w in phrase.split()]
    phrase_tokens = [t for t in phrase_tokens if t]
    if not phrase_tokens:
        return None

    norm_words = [(normalize_word_for_match(w["word"]), w) for w in whisper_words]

    # Sliding window — find best contiguous match anchored on first phrase token
    best = None
    best_matches = 0
    n = len(phrase_tokens)
    for i in range(len(norm_words) - n + 1):
        window = norm_words[i:i + n]
        matches = sum(1 for (a, b), pt in zip(window, phrase_tokens) if a == pt)
        if matches > best_matches:
            best_matches = matches
            best = (window[0][1], window[-1][1])

    if best is None:
        return None
    ratio = best_matches / max(1, n)
    if ratio < MIN_MATCH_RATIO:
        return None

    start_w, end_w = best
    return start_w["start"], end_w["end"], ratio


# ─── Audio I/O ───────────────────────────────────────────────────────
def slice_audio(audio_path: Path, start_sec: float, end_sec: float,
                out_path: Path, buffer: float = BUFFER_SEC) -> float:
    """Read audio, slice, write. Returns slice duration in seconds."""
    audio, sr = sf.read(str(audio_path))
    s = max(0.0, start_sec - buffer)
    e = min(len(audio) / sr, end_sec + buffer)
    if e <= s:
        return 0.0
    si = int(s * sr)
    ei = int(e * sr)
    sliced = audio[si:ei]
    sf.write(str(out_path), sliced, sr)
    return (ei - si) / sr


# ─── Main loop ───────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True,
                   help="Source pipe-separated csv (audio_file|text|speaker_name)")
    p.add_argument("--out-dir", required=True,
                   help="Output directory — will write wavs/ + short_subset.csv")
    p.add_argument("--whisper-model", default="small",
                   help="faster-whisper model (tiny|base|small|medium|large-v3)")
    p.add_argument("--max-source-samples", type=int, default=0,
                   help="0 = process all rows; otherwise limit (for testing)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=100,
                   help="Print progress every N source samples")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "short_subset.csv"
    progress_path = out_dir / "progress.json"

    # ── Resume support ──
    processed_files: set[str] = set()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        processed_files = set(progress.get("processed", []))
        print(f"[resume] {len(processed_files)} source files already processed")

    # ── Load Whisper once ──
    from faster_whisper import WhisperModel
    print(f"[whisper] loading {args.whisper_model} on {args.device}…")
    t0 = time.time()
    asr = WhisperModel(args.whisper_model, device=args.device,
                       compute_type="float16" if args.device == "cuda" else "int8")
    print(f"[whisper] loaded in {time.time()-t0:.1f}s")

    # ── Open output CSV (append if resuming) ──
    csv_mode = "a" if out_csv.exists() else "w"
    with open(out_csv, csv_mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        if csv_mode == "w":
            writer.writerow(["audio_file", "text", "speaker_name"])

        # ── Process source rows ──
        with open(args.train_csv, encoding="utf-8") as src:
            rows = list(csv.reader(src, delimiter="|"))
        header = rows[0]
        if header != ["audio_file", "text", "speaker_name"]:
            print(f"[warn] unexpected header: {header}")
        data_rows = rows[1:]
        if args.max_source_samples > 0:
            data_rows = data_rows[:args.max_source_samples]
        print(f"[input] {len(data_rows)} source rows to process")

        stats = {"processed": 0, "phrases_kept": 0, "phrases_dropped": 0,
                 "audio_missing": 0, "transcribe_failed": 0}

        for idx, row in enumerate(data_rows):
            if len(row) < 3:
                continue
            audio_path = Path(row[0])
            full_text = row[1]
            speaker = row[2]

            stem = audio_path.stem
            if stem in processed_files:
                continue
            if not audio_path.exists():
                stats["audio_missing"] += 1
                continue

            # Candidate phrases
            phrases = [ph for ph in split_into_phrases(full_text) if is_valid_phrase(ph)]
            if not phrases:
                processed_files.add(stem)
                stats["processed"] += 1
                continue

            # Transcribe once per audio
            try:
                whisper_words = transcribe_with_timestamps(asr, str(audio_path))
            except Exception as exc:
                print(f"  [{idx}] transcribe failed for {audio_path.name}: {exc}")
                stats["transcribe_failed"] += 1
                continue

            for phrase_idx, phrase in enumerate(phrases):
                loc = locate_phrase_in_words(phrase, whisper_words)
                if loc is None:
                    stats["phrases_dropped"] += 1
                    continue
                start_sec, end_sec, ratio = loc
                duration = end_sec - start_sec
                if duration < MIN_SLICE_SEC or duration > MAX_SLICE_SEC:
                    stats["phrases_dropped"] += 1
                    continue

                out_wav = wavs_dir / f"{stem}__s{phrase_idx}.wav"
                slice_dur = slice_audio(audio_path, start_sec, end_sec, out_wav)
                if slice_dur < MIN_SLICE_SEC:
                    out_wav.unlink(missing_ok=True)
                    stats["phrases_dropped"] += 1
                    continue

                writer.writerow([str(out_wav), phrase, speaker])
                stats["phrases_kept"] += 1

            processed_files.add(stem)
            stats["processed"] += 1

            # Flush + log periodically
            if stats["processed"] % args.progress_every == 0:
                f.flush()
                progress_path.write_text(json.dumps({
                    "processed": list(processed_files),
                    "stats": stats,
                }, ensure_ascii=False))
                kept_per_src = stats["phrases_kept"] / max(1, stats["processed"])
                print(f"  [{stats['processed']}/{len(data_rows)}] "
                      f"kept={stats['phrases_kept']} ({kept_per_src:.1f}/src), "
                      f"dropped={stats['phrases_dropped']}, "
                      f"missing={stats['audio_missing']}, "
                      f"asr_fail={stats['transcribe_failed']}")

        # Final save
        progress_path.write_text(json.dumps({
            "processed": list(processed_files),
            "stats": stats,
        }, ensure_ascii=False))

    print(f"\n[done] kept {stats['phrases_kept']} short utterances "
          f"from {stats['processed']} source rows")
    print(f"  output csv: {out_csv}")
    print(f"  output wavs: {wavs_dir}")


if __name__ == "__main__":
    main()
