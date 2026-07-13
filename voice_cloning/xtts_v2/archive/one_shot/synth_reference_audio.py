"""Synthesize per-page audio for label_reference folder using XTTS FT + rule corrections."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REF_DIR = Path("/home/bes/Desktop/Tin/label_reference")
POOL_DIR = Path("/mnt/nfs-data/tin_dataset/comic/labeling_pool")
FT_DIR = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft"
REF_VOICE = "/home/bes/Desktop/Tin/demo_refs_xtts/default.wav"
SAMPLE_RATE = 24000

sys.path.insert(0, "/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2")
sys.path.insert(0, "/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2/coqui_tts")

from vn_correct_rules import correct as rule_correct

print("Loading XTTS FT...")
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
config = XttsConfig()
config.load_json(f"{FT_DIR}/config.json")
xtts = Xtts.init_from_config(config)
xtts.load_checkpoint(config, checkpoint_dir=FT_DIR, use_deepspeed=False, eval=True)
xtts.cuda()

print("Computing reference voice...")
gpt_latent, spk_emb = xtts.get_conditioning_latents(
    audio_path=REF_VOICE, gpt_cond_len=6, max_ref_length=30
)

# Load pool index
with open(POOL_DIR / "index.json") as f:
    pool_index = json.load(f)
pool_by_key = {(m["series"], m["name"]): m for m in pool_index}

import librosa

# For each reference image, find matching pool entry and synth
ref_pages = sorted(REF_DIR.glob("*.webp"))
print(f"\nSynthesizing {len(ref_pages)} reference pages...")

for ref_page in ref_pages:
    stem = ref_page.stem   # "01_doraemon_018_img_00013"
    parts = stem.split("_", 2)  # ['01', 'doraemon', '018_img_00013']
    if len(parts) < 3:
        continue
    idx, series, page_name = parts

    key = (series, page_name)
    pool_meta = pool_by_key.get(key)
    if not pool_meta:
        print(f"  skip {stem}: no pool entry")
        continue

    with open(pool_meta["json"]) as f:
        page = json.load(f)
    bubbles = sorted(page.get("bubbles", []), key=lambda b: b.get("order", 0))

    chunks = []
    for b in bubbles:
        text = (b.get("text_prefill") or "").strip()
        if len(text) <= 2:
            continue
        # Apply rule correction
        corrected, _ = rule_correct(text)
        if not corrected:
            continue
        # Lowercase for TTS
        corrected_lc = corrected.lower()
        try:
            out = xtts.inference(
                corrected_lc, language="vi",
                gpt_cond_latent=gpt_latent, speaker_embedding=spk_emb,
                temperature=0.45, length_penalty=1.0,
                repetition_penalty=2.0, top_k=30, top_p=0.7,
            )
        except Exception as e:
            print(f"    [{b['order']}] synth failed: {e}")
            continue
        wav = np.asarray(out["wav"], dtype=np.float32)
        # Trim + cap duration
        if len(wav) > 1000:
            wav_trim, _ = librosa.effects.trim(wav, top_db=25)
            if len(wav_trim) > 1000:
                wav = wav_trim
        max_dur = max(1.0, len(corrected_lc) / 6.0 + 0.8)
        max_samples = int(max_dur * SAMPLE_RATE)
        if len(wav) > max_samples:
            fade = min(2400, int(0.1 * SAMPLE_RATE))
            wav = wav[:max_samples]
            wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        chunks.append(wav)

    if not chunks:
        print(f"  skip {stem}: no chunks")
        continue

    gap = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
    page_audio = []
    for i, c in enumerate(chunks):
        page_audio.append(c)
        if i < len(chunks) - 1:
            page_audio.append(gap)
    page_audio = np.concatenate(page_audio)

    out_wav = REF_DIR / f"{stem}.wav"
    sf.write(str(out_wav), page_audio, SAMPLE_RATE)
    print(f"  [{idx}] {series}/{page_name}: {len(chunks)} bubbles → {len(page_audio)/SAMPLE_RATE:.1f}s")

print(f"\nDone. Audio files in {REF_DIR}")
