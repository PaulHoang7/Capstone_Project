"""Fast raw-text extraction — download arrow files directly, skip audio column."""
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import pyarrow as pa
from tqdm import tqdm

OUT = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/raw_text.tsv")
CACHE = Path("/mnt/nfs-data/tin_dataset/hf_cache/vieneu_140h")
CACHE.mkdir(parents=True, exist_ok=True)

print("[1/3] Downloading all arrow files in parallel...")
local_dir = snapshot_download(
    repo_id="pnnbao-ump/VieNeu-TTS-140h",
    repo_type="dataset",
    local_dir=str(CACHE),
    allow_patterns=["*.arrow", "*.json"],
    max_workers=16,
)
print(f"  Downloaded to: {local_dir}")

arrow_files = sorted(Path(local_dir).glob("*.arrow"))
print(f"[2/3] Found {len(arrow_files)} arrow files.")

print("[3/3] Extracting text columns...")
count = 0
with open(OUT, "w", encoding="utf-8") as out:
    out.write("id\tspeaker\tgender\tduration\ttext\n")
    for af in tqdm(arrow_files, desc="Arrow files"):
        with open(af, "rb") as f:
            reader = pa.ipc.open_stream(f)
            for batch in reader:
                t = batch.to_pydict()
                # Columns: _id, audio (bytes+path), text, phonemized_text, duration, speaker, gender, language
                ids = t.get("_id", [])
                texts = t.get("text", [])
                speakers = t.get("speaker", [])
                genders = t.get("gender", [])
                durations = t.get("duration", [])
                for _id, text, spk, gen, dur in zip(ids, texts, speakers, genders, durations):
                    text = (text or "").replace("\t", " ").replace("\n", " ").strip()
                    out.write(f"{_id}\t{spk}\t{gen}\t{dur:.2f}\t{text}\n")
                    count += 1
print(f"\nExtracted {count} samples → {OUT}")
print(f"Size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")
