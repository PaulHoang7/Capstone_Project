"""
Inference script for VITS2 Variant B (+ Tone Embedding).

Usage:
    cd /home/bes/Desktop/TTS
    python Capstone_project/scripts/run_inference_variant_b.py
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))
_VITS2_DIR = os.path.join(_PROJECT_ROOT, 'vits2_pytorch')

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _VITS2_DIR not in sys.path:
    sys.path.insert(0, _VITS2_DIR)

import torch
import utils
import commons
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

from Capstone_project.models.models_tone import SynthesizerTrnTone
from Capstone_project.tone_encoder.tone_utils import text_to_tone_sequence

# --- Configuration ---
CONFIG = "Capstone_project/configs/vits2_vieneu_variant_b.json"
MODEL = "vits2_pytorch/logs/vieneu_variant_b/G_344000.pth"
OUTPUT_DIR = "Capstone_project/samples_variant_b_344000"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test sentences (same as baseline for comparison)
test_samples = [
    # Short sentence, speaker 0
    (0, "mˈo6t̪  kˈəːn  bˈaː5w  kˈaːɜt̪  ɗˌaː5  sˈa4j  ɹˈaː,  ɗˌaː5  tˈəɪɜ  mˈo6t̪  kˈəːn  zˈɔɜ  lˈe-6ɲ  zˈaːɜ,  tʃˈəː2j  zˈoaː6  mˈyə,  sˈɛ  kˈo6  bˌi6  hˈɔ4ŋ."),
    # Medium sentence, speaker 15
    (15, "tʃˈuɜŋ  t̪ˈaː  ŋˈiɛn  kˈiɜw  lˈi6c  sˈy4  ɲˈa2m  ɲˈi2n  ɲˈə6n  ɹˈɔ5  hˈəːn  kˌaːɜj  t̪ˈi2ɲ  tˈeɜ  mˌaː2  tʃˈɔŋ  ɗˈɔɜ  tʃˈuɜŋ  t̪ˈaː  ɗˌaːŋ  kˈə2n  fˌaː4j  hˈe-2ɲ  ɗˈo6ŋ."),
    # Question sentence, speaker 42
    (42, "t̪ˈaː6j  sˈaːw  ɲˌy5ŋ  tʃˈɛ4  mˈaːŋ  lwˈaː6j  dʒˈɛn  sˈəɜw  nˈa2j  lˈaː6j  kˈɔɜ  sˈu  hˈyəɜŋ  zˈuɜp  ɗˈəː5,  tˈə6m  tʃˈiɜ  xˌi  xˌoŋ  ɗˌyə6c  ˈiɛw  kˈə2w?"),
    # Another speaker 57
    (57, "t̪ˈaː6j  sˈaːw  tʃˈuɜŋ  t̪ˈaː  xˌoŋ  ɗˌyə6c  kˈuŋ  kˈəɜp  ɲˈiɛ2w  tˈoŋ  t̪ˈin  hˈəːn  vˈe2  kˌaːɜc  vˈəɜn  ɗˈe2  lˈiɛn  kwˈaːn"),
    # Short with speaker 19
    (19, "ɗˈiɛ2w  tˈoj  mˈuəɜn  tʃwˈiɛ2n  t̪ˈaː4j  lˌaː2  tʃˈuɜŋ  t̪ˈaː  kˈɔɜ  tˈe4  kˈɔɜ  ɗˈəː2j  sˈoɜŋ  tʃˈan  ɣˈoɜj  kˈuə2ŋ  ɲˈiɛ6t̪  hˈəːn"),
]


def get_text_and_tone(text, hps):
    """Get aligned text and tone sequences."""
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    tone_norm = text_to_tone_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
        tone_norm = commons.intersperse(tone_norm, 0)
    assert len(text_norm) == len(tone_norm), (
        f"text/tone length mismatch: {len(text_norm)} vs {len(tone_norm)}"
    )
    return torch.LongTensor(text_norm), torch.LongTensor(tone_norm)


def main():
    # Load config
    hps = utils.get_hparams_from_file(CONFIG)

    # Create model — force CPU when GPU is busy with training
    device = torch.device("cpu")
    print(f"Using device: {device}")

    net_g = SynthesizerTrnTone(
        len(symbols),
        80,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    net_g.eval()

    # Load checkpoint
    utils.load_checkpoint(MODEL, net_g, None)
    print(f"Loaded checkpoint: {MODEL}")

    # Generate samples
    for i, (spk_id, text) in enumerate(test_samples):
        stn_tst, tone_tst = get_text_and_tone(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            tone_tst = tone_tst.to(device).unsqueeze(0)
            sid = torch.LongTensor([spk_id]).to(device)
            audio = net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                tone=tone_tst,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
            )[0][0, 0].data.cpu().float().numpy()

        out_path = os.path.join(OUTPUT_DIR, f"sample_{i}_spk{spk_id}.wav")
        write(out_path, hps.data.sampling_rate, audio)
        print(
            f"[{i+1}/{len(test_samples)}] Saved: {out_path} "
            f"(spk={spk_id}, len={len(audio)/hps.data.sampling_rate:.2f}s)"
        )

    print(f"\nAll samples saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
