"""Generate demo samples for user to listen — evaluate XTTS FT quality subjectively.

For each of 3 reference voices (1 female, 1 male, 1 heldout), generate:
  - 6 comic-style Vietnamese sentences
  - Includes tone-heavy texts (hỏi/ngã, sắc/nặng minimal pairs)
"""
import os, sys, torch
import numpy as np
import soundfile as sf
from pathlib import Path

FT_DIR = "/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft"
OUT = Path("/home/bes/Desktop/Tin/demo_audio/xtts_ft_listen")
OUT.mkdir(parents=True, exist_ok=True)

# Reference voices — mix of heldout (zero-shot) + seen
REFS = {
    "ref_female_heldout_jellyfish_0052": "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs/jellyfish1010_0052_500.wav",
    "ref_female_heldout_capybara_1023":  "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs/capybara1812_1023_100.wav",
    "ref_male_heldout_jellyfish_0044":   "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs/jellyfish1010_0044_150.wav",
}

# Comic-style sentences with tonal challenges
DEMO_TEXTS = [
    ("greeting",       "Xin chào các bạn, tôi rất vui được gặp mọi người hôm nay."),
    ("comic_short",    "Này, cậu đi đâu đấy? Chờ tớ với!"),
    ("comic_emotion",  "Không thể tin nổi! Chuyện này thật sự đã xảy ra sao?"),
    ("tone_minimal1",  "Mã và mả, hai từ khác nhau nhưng nghe rất giống."),
    ("tone_minimal2",  "Cô ấy hỏi tôi về cái hộp đỏ trên bàn."),
    ("long_complex",   "Trong cuộc hành trình dài đầy thử thách, chúng ta đã học được nhiều bài học quý giá về tình bạn."),
]

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("[1/3] Loading XTTS FT...")
config = XttsConfig()
config.load_json(os.path.join(FT_DIR, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=FT_DIR, use_deepspeed=False, eval=True)
model.cuda()

print("[2/3] Generating for each reference voice...")
for ref_name, ref_wav in REFS.items():
    if not Path(ref_wav).exists():
        print(f"  skip {ref_name}: missing {ref_wav}")
        continue
    print(f"  voice: {ref_name}")
    ref_dir = OUT / ref_name
    ref_dir.mkdir(exist_ok=True)
    # Copy ref for comparison
    import shutil
    shutil.copy(ref_wav, ref_dir / "00_REFERENCE.wav")

    gpt_latent, speaker_emb = model.get_conditioning_latents(
        audio_path=ref_wav, gpt_cond_len=6, max_ref_length=30
    )

    for tag, text in DEMO_TEXTS:
        out = model.inference(
            text, language="vi",
            gpt_cond_latent=gpt_latent,
            speaker_embedding=speaker_emb,
            temperature=0.7, length_penalty=1.0,
            repetition_penalty=10.0, top_k=30, top_p=0.85,
        )
        wav_np = np.asarray(out["wav"], dtype=np.float32)
        path = ref_dir / f"{tag}.wav"
        sf.write(str(path), wav_np, 24000)
        print(f"    [{tag}] {len(wav_np)/24000:.1f}s — {text[:50]}...")

print(f"[3/3] Done. Listen at: {OUT}")
print("Structure:")
print("  ref_<voice>/")
print("    00_REFERENCE.wav       ← original voice (nghe trước để biết speaker gốc)")
print("    greeting.wav           ← XTTS FT clone nói câu chào")
print("    comic_short.wav        ← XTTS FT clone nói câu comic ngắn")
print("    comic_emotion.wav      ← XTTS FT clone nói câu cảm xúc")
print("    tone_minimal1.wav      ← Test tone sắc/ngã (mã/mả)")
print("    tone_minimal2.wav      ← Test tone hỏi/nặng (hỏi/hộp)")
print("    long_complex.wav       ← Test câu dài phức tạp")
