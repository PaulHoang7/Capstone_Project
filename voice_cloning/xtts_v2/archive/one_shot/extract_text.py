"""Extract raw Vietnamese text from VieNeu-TTS-140h HF dataset.

Produces: /mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/raw_text.tsv
Format:   {_id}\t{speaker}\t{gender}\t{duration}\t{text}
"""
import os
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

OUT = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/raw_text.tsv")

# Stream mode — we only want metadata, not audio bytes (50 GB wasted otherwise)
print("Streaming dataset (metadata only)...")
ds = load_dataset("pnnbao-ump/VieNeu-TTS-140h", split="train", streaming=True)
# Remove audio column to speed up
ds = ds.remove_columns(["audio"])

count = 0
with open(OUT, "w", encoding="utf-8") as f:
    f.write("id\tspeaker\tgender\tduration\ttext\n")
    for ex in tqdm(ds, desc="Extracting text"):
        _id = ex["_id"]
        text = ex["text"].replace("\t", " ").replace("\n", " ").strip()
        row = f"{_id}\t{ex['speaker']}\t{ex['gender']}\t{ex['duration']:.2f}\t{text}\n"
        f.write(row)
        count += 1
        if count % 5000 == 0:
            f.flush()

print(f"Extracted {count} samples → {OUT}")
print(f"Size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")
