"""Baseline inference test with viXTTS — verify pretrained works before fine-tune."""
import os, sys, torch, time
from pathlib import Path

CKPT = "/mnt/nfs-data/tin_dataset/checkpoints/vixtts"
REF = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs/capybara1812_1003_193.wav"
OUT = Path("/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2/samples_baseline")
OUT.mkdir(exist_ok=True)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("[1/4] Loading config...")
config = XttsConfig()
config.load_json(os.path.join(CKPT, "config.json"))

print("[2/4] Init model...")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=CKPT, use_deepspeed=False, eval=True)
model.cuda()
print(f"  Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("[3/4] Compute speaker latents from ref...")
t0 = time.time()
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=REF, gpt_cond_len=6, max_ref_length=30
)
print(f"  Done in {time.time()-t0:.1f}s")

print("[4/4] Generate Vietnamese TTS...")
texts = [
    "Xin chào, đây là hệ thống lồng tiếng tự động cho truyện tranh.",
    "Tôi rất vui được gặp bạn hôm nay.",
    "Thanh điệu tiếng Việt rất phong phú và đẹp.",
]
import soundfile as sf
import numpy as np
for i, text in enumerate(texts):
    t0 = time.time()
    out = model.inference(
        text, language="vi",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7, length_penalty=1.0,
        repetition_penalty=10.0, top_k=30, top_p=0.85,
    )
    wav_np = np.asarray(out["wav"], dtype=np.float32)
    path = OUT / f"baseline_{i:02d}.wav"
    sf.write(str(path), wav_np, 24000)
    print(f"  [{i}] {text[:40]}... -> {path.name} ({time.time()-t0:.1f}s)")

print(f"\nDone. Samples: {OUT}")
