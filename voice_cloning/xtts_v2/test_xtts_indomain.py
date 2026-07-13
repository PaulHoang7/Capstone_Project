"""Test hypothesis: XTTS-FT conditioned on an IN-DOMAIN VieNeu speaker reference
(a voice it trained on) reads cleanly, vs the heldout zero-shot clone (CER 0.52).

Synth only. Saves wavs + sentences.json. STT/CER done separately in stt env.
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write as wav_write

sys.path.insert(0, "/home/bes/Desktop/Tin/Capstone_project/voice_cloning/xtts_v2/coqui_tts")
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt-dir", default="/mnt/nfs-data/tin_dataset/checkpoints/xtts_vieneu_ft")
_ap.add_argument("--out", default="/tmp/xtts_indomain")
_ap.add_argument("--sents", default="", help="optional json list of sentences")
_args = _ap.parse_args()
FT_DIR = _args.ckpt_dir
REF = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs/capybara1812_1003_193.wav"  # in-domain speaker
OUT = Path(_args.out); OUT.mkdir(parents=True, exist_ok=True)

SENTS = [
    "Mỗi nhóc phải thơm lên má của cậu Út một cái mới được một khúc dồi.",
    "Từ đầu tới chân, tôi giống như một triệu phú đô la vừa bước ra từ sở đúc tiền.",
    "Khi tôi đang mê mẩn ngắm nhìn người ấy, bố tôi nghiêm giọng nói.",
    "Chúng ta nên ngồi xuống nói chuyện với nhau một cách bình tĩnh và rõ ràng hơn.",
    "Chạy xe ngoài đường nguy hiểm lắm, vào công viên mà chơi.",
    "Bình tĩnh, đạp chân thẳng, dừng xe lại mau.",
]

if _args.sents:
    SENTS = json.load(open(_args.sents))

print("[1/3] load XTTS-FT")
cfg = XttsConfig(); cfg.load_json(os.path.join(FT_DIR, "config.json"))
xtts = Xtts.init_from_config(cfg)
xtts.load_checkpoint(cfg, checkpoint_dir=FT_DIR, use_deepspeed=False, eval=True)
xtts.cuda()

print("[2/3] conditioning latents from in-domain ref")
gpt_latent, spk_emb = xtts.get_conditioning_latents(audio_path=REF, gpt_cond_len=6, max_ref_length=30)

print("[3/3] synth")
rows = []
for i, text in enumerate(SENTS):
    out = xtts.inference(text, language="vi", gpt_cond_latent=gpt_latent, speaker_embedding=spk_emb,
                         temperature=0.7, length_penalty=1.0, repetition_penalty=2.0, top_k=50, top_p=0.85)
    wav = np.asarray(out["wav"], dtype=np.float32)
    fn = OUT / f"{i:02d}.wav"
    wav_write(fn, 24000, wav)
    rows.append({"i": i, "text": text, "wav": str(fn), "dur": round(len(wav) / 24000, 2)})
    print(f"  [{i}] dur={rows[-1]['dur']}s  {text[:50]}")

json.dump({"ref": REF, "rows": rows}, open(OUT / "sentences.json", "w"), ensure_ascii=False, indent=2)
print(f"saved -> {OUT}")
