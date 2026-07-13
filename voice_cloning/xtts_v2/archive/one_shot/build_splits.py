"""Build train/val/heldout splits for XTTS fine-tune on VieNeu-TTS-140h.

Heldout protocol — MATCHES VITS2 Phase 3 evaluation for fair comparison:
  - Heldout speaker IDs: 0, 10, 20, ..., 190 (20 speakers → unseen for zero-shot)
  - All remaining 173 speakers → train
  - Within train: 1% random split → val (for training monitoring)

Output format (XTTS trainer expects LJSpeech-style):
  - train.csv:   wav_path|text|speaker_name
  - val.csv:     wav_path|text|speaker_name
  - heldout.csv: wav_path|text|speaker_name  (zero-shot eval, unseen speakers)
"""
import csv
import json
import random
from pathlib import Path

RAW_TEXT = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/raw_text.tsv")
SPEAKER_MAP = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/speaker_map.json")
WAVS_DIR = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs")
OUT_DIR = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HELDOUT_SIDS = set(range(0, 193, 10))  # 0, 10, 20, ..., 190 → 20 speakers
MIN_DURATION = 1.5   # seconds — XTTS needs ≥1s
MAX_DURATION = 11.5  # seconds — XTTS default cap
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

print(f"[1/4] Loading speaker map...")
with open(SPEAKER_MAP) as f:
    spk2sid = json.load(f)
sid2spk = {v: k for k, v in spk2sid.items()}
print(f"  {len(spk2sid)} speakers. Heldout sids: {sorted(HELDOUT_SIDS)[:5]}...")

print(f"[2/4] Reading raw_text.tsv...")
rows = []
with open(RAW_TEXT, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)
print(f"  {len(rows)} rows read.")

print(f"[3/4] Filtering + split by speaker...")
train, val, heldout = [], [], []
skipped_missing = skipped_dur = skipped_empty = 0

for r in rows:
    _id = r["id"]
    speaker_name = r["speaker"]
    sid = spk2sid.get(speaker_name)
    if sid is None:
        skipped_missing += 1
        continue

    dur = float(r["duration"])
    if dur < MIN_DURATION or dur > MAX_DURATION:
        skipped_dur += 1
        continue

    text = r["text"].strip()
    if not text:
        skipped_empty += 1
        continue

    wav_path = WAVS_DIR / f"{_id}.wav"
    # Speaker name for XTTS — use VieNeu speaker (e.g., "capybara1812_1003")
    entry = (str(wav_path), text, speaker_name)
    if sid in HELDOUT_SIDS:
        heldout.append(entry)
    else:
        train.append(entry)

print(f"  Train pool: {len(train)}  Heldout pool: {len(heldout)}")
print(f"  Skipped: missing={skipped_missing}, duration={skipped_dur}, empty={skipped_empty}")

# Move 1% of train to val
random.shuffle(train)
val_n = max(500, len(train) // 100)
val = train[:val_n]
train = train[val_n:]
print(f"  After val split: train={len(train)}, val={len(val)}, heldout={len(heldout)}")

print(f"[4/4] Writing CSVs...")
for name, data in [("train", train), ("val", val), ("heldout", heldout)]:
    p = OUT_DIR / f"{name}.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write("audio_file|text|speaker_name\n")
        for wav_path, text, spk in data:
            text_clean = text.replace("|", "").strip()
            f.write(f"{wav_path}|{text_clean}|{spk}\n")
    print(f"  {name}.csv: {len(data)} rows → {p}")

# Also write a small sanity val subset (100 rows) for fast eval during training
small = val[:100]
p = OUT_DIR / "val_small.csv"
with open(p, "w", encoding="utf-8") as f:
    f.write("audio_file|text|speaker_name\n")
    for wav_path, text, spk in small:
        f.write(f"{wav_path}|{text.replace('|','').strip()}|{spk}\n")
print(f"  val_small.csv: 100 rows → {p}")

# Dump heldout speaker list for eval script
heldout_spk_set = sorted({row[2] for row in heldout})
with open(OUT_DIR / "heldout_speakers.json", "w") as f:
    json.dump({"heldout_sids": sorted(HELDOUT_SIDS),
               "heldout_speakers": heldout_spk_set}, f, indent=2, ensure_ascii=False)
print(f"  heldout_speakers.json: {len(heldout_spk_set)} speakers")

print(f"\nDone. Splits at {OUT_DIR}")
