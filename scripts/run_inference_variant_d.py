"""
Inference script for VITS2 Variant D (+ Dual-Path + Cross-Attention).

Usage:
    cd /home/bes/Desktop/TTS
    python Capstone_project/scripts/run_inference_variant_d.py
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

from Capstone_project.models import build_synthesizer
from Capstone_project.tone_encoder.tone_utils import text_to_tone_sequence

# --- Configuration ---
CONFIG = "Capstone_project/configs/vits2_vieneu_variant_d.json"
MODEL = "/home/bes/Desktop/TTS/vits2_pytorch/logs/vieneu_variant_d/G_438000.pth"
OUTPUT_DIR = "Capstone_project/samples_variant_d_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 15 test sentences from val set — diverse speakers, varying lengths
test_samples = [
    # 1. Medium sentence, speaker 0
    (0, "mˈo6t̪  kˈəːn  bˈaː5w  kˈaːɜt̪  ɗˌaː5  sˈa4j  ɹˈaː,  ɗˌaː5  tˈəɪɜ  mˈo6t̪  kˈəːn  zˈɔɜ  lˈe-6ɲ  zˈaːɜ,  tʃˈəː2j  zˈoaː6  mˈyə,  sˈɛ  kˈo6  bˌi6  hˈɔ4ŋ."),
    # 2. Long sentence, speaker 19
    (19, "ɗˈiɛ2w  t̪ˈoj  mˈuəɜn  tʃwˈiɛ2n  t̪ˈaː4j  lˌaː2  tʃˈuɜŋ  t̪ˈaː  kˈɔɜ  tˈe4  kˈɔɜ  ɗˈəː2j  sˈoɜŋ  tʃˈan  ɣˈoɜj  kˈuə2ŋ  ɲˈiɛ6t̪  hˈəːn,  vˈuj  vˈɛ4  hˈəːn,  tˈə6m  tʃˈiɜ  t̪ˈi2ɲ  zˈu6c  fˈu2  fˈiɜm  nˌeɜw  tʃˈuɜŋ  t̪ˈaː  ˈiɜt̪  bˌi6  kˈiɛ2m  tʃˈeɜ  bˌəː4j  tˈiɛn  hˈyəɜŋ  zˈən  tʃˈu4  kˌuə4  nˈe2n  vˈan  hwˈaːɜ  tʃˈuɜŋ  t̪ˈaː  tʃˈɔŋ  fˈɔ2ŋ  ŋˈu4."),
    # 3. Medium sentence, speaker 15
    (15, "tʃˈuɜŋ  t̪ˈaː  ŋˈiɛn  kˈiɜw  lˈi6c  sˈy4  ɲˈa2m  ɲˈi2n  ɲˈə6n  ɹˈɔ5  hˈəːn  kˌaːɜj  t̪ˈi2ɲ  tˈeɜ  mˌaː2  tʃˈɔŋ  ɗˈɔɜ  tʃˈuɜŋ  t̪ˈaː  ɗˌaːŋ  kˈə2n  fˌaː4j  hˈe-2ɲ  ɗˈo6ŋ."),
    # 4. Question sentence, speaker 42
    (42, "t̪ˈaː6j  sˈaːw  ɲˌy5ŋ  tʃˈɛ4  mˈaːŋ  lwˈaː6j  dʒˈɛn  sˈəɜw  nˈa2j  lˈaː6j  kˈɔɜ  sˈu  hˈyəɜŋ  zˈuɜp  ɗˈəː5,  tˈə6m  tʃˈiɜ  xˌi  xˌoŋ  ɗˌyə6c  ˈiɛw  kˈə2w?"),
    # 5. Long sentence with many words, speaker 57
    (57, "t̪ˈaː6j  sˈaːw  tʃˈuɜŋ  t̪ˈaː  xˌoŋ  ɗˌyə6c  kˈuŋ  kˈəɜp  ɲˈiɛ2w  tˈoŋ  t̪ˈin  hˈəːn  vˈe2  kˌaːɜc  vˈəɜn  ɗˈe2  lˈiɛn  kwˈaːn  ɗˌeɜn  kwˈiɛ2n  ɹˈiɛŋ  t̪ˈy  mˌaː2  snˈɔwzɛn  vˌaː2  ɲˌy5ŋ  ŋˈyə2j  xˈaːɜc  ɗˌaː5  vˈe-6c  ɹˈaː?"),
    # 6. Complex sentence, speaker 75
    (75, "kˌɔn  ɗˌaː5  t̪wˈiɛ6t̪  vˈɔ6ŋ  bˈiɛɜt̪  bˈaːw,  vˌaː2  tʃˈɔŋ  xwˈe-4ɲ  xˈaɜc  ˈəɪɜ,  t̪ˈəɜt̪  kˈaː4  kˈiɲ  ŋˈiɛ6m  t̪ˈo2j  t̪ˈe6  ɲˈəɜt̪  kˌuə4  kˌɔn  tʃˈen  mˌɔ6j  lˈi5ɲ  vˈy6c  bˈo5ŋ  zˈyŋ  kˈeɜt̪  hˈəː6p  lˈaː6j  mˈo6t̪  kˈe-ɜc  hwˈi  hwˈaː2ŋ."),
    # 7. Short sentence, speaker 84
    (84, "hˈaːj  mˈyəj  sˈaɜw  fˈə2n  tʃˈam  tʃˈɔŋ  sˈoɜ  ɲˌy5ŋ  ŋˈyə2j  ɗˌyə6c  tʃˈuŋ  t̪ˈəm  kˈɛnədi  vˈiɲ  zˈe-ɲ."),
    # 8. Very short, speaker 92
    (92, "ɲˌyŋ  tˈy6c  sˈy6  lˌaː2  bˈaːɜc  kˌɔ2n  hˈəːj  sˈuɜc  fˈaː6m  tʃˈaɜw  ɹˈo2j."),
    # 9. Short, speaker 75 (same speaker different text)
    (75, "ɗˈyəɜ  hˈəː6p  bˈoɜ  ɲˈəɜt̪  lˌaː2  vˈali,  lˌaː2  ŋˈyə2j  tˈyɜ  hˈaːj  bˈen  kˈe-6ɲ  mˈɛ6."),
    # 10. Medium, speaker 111
    (111, "sˈɔŋ  ɗˈə1w  ɗˈəɪɜ,  hˈaɜn  vˈo5  hˈaːj  kˌaːɜj  vˈaː2w  bˈu6ŋ  kˌɔn  vˈə6t̪,  ɹˈo2j  tˈaː4  nˈɔɜ  ɹˈaː  vˌaː2  nˈɔɜj  vˌəːɜj  ɗˈaːɜm  ɗˈoŋ  ɗˈyɜŋ  sˈuŋ  kwˈe-ɲ."),
    # 11. Short, speaker 120
    (120, "kˈaːɜw  nˈɔɜj,  lwˈaː2j  ŋˈyə2j  xˌoŋ  kˈɔɜ  ɗˈu4  tˈəː2j  zˈaːn  ɗˌe4  hˈɔ6k  bˈəɜt̪  kˈi2  ɗˈiɛ2w  zˈi2."),
    # 12. Medium, speaker 79
    (79, "sˈiɲ  ɹˈaː  t̪ˈaː6j  ɗˈyɜc,  fˈə2n  nˈaː2w  ɗˌyə6c  ŋˈyə2j  ˈoŋ  lˌaː2  zˈaːɜw  sˈi5  zˈɔ  tˈaːɜj  tʃwˈiɛ2n  tˈoɜŋ  nˈuəj  zˈyə5ŋ  vˌaː2  zˈoɜŋ  ɲˌy  bˈaɪəlˌɪk."),
    # 13. Long complex, speaker 133
    (133, "ɲˌyŋ  tʃˈɔŋ  tʃˈyə2ŋ  hˈəː6p  nˈa2j,  zˈɔ  kˈaː4  hˈaːj  ɗˈe2w  hwˈaː6t̪  ɗˈo6ŋ  vˌəːɜj  mˈyɜc  ɗˈo6  t̪ˈin  kˈəɪ6  kˈaːw,  tˈyəŋ  vˈu6  ɗˌyə6c  zˈaː4j  kwˈiɛɜt̪  tʃˈɔŋ  kˈuə6c  hˈɔ6p  kˈɛɜw  zˈaː2j  hˈaːj  zˈəː2  vˌaː2  kˈeɜt̪  tˈuɜc  bˈa2ŋ  kˌaːɜj  bˈaɜt̪  t̪ˈaj."),
    # 14. Medium emotional, speaker 82
    (82, "kˌɔn  kˌuə4  ɲˌy5ŋ  ŋˈyə2j  mˈɛ6  tʃˈə2m  kˈaː4m  kˈɔɜ  ŋwˈi  kˈəː  kˈaː4m  tˈəɪɜ  bˈəɜt̪  ˈaːn  vˌaː2  sˈəː6  hˈaː5j  ɲˈiɛ2w  ɗˈiɛ2w  xˌi  lˈəːɜn  lˈen."),
    # 15. Short, speaker 132
    (132, "ɗwˈaː2n  t̪ˈi2m  kˈiɛɜm  tʃˌyə  ɗˈi  ɗˌyə6c  bˈaːw  sˈaː  tˌi2  ɗˌaː5  ɣˈa6p  tʃˈaː."),
    # 16. Medium, speaker 151
    (151, "zˌu2  kˈɔɜ  tˈoŋ  mˈiɲ  sˈaːɜŋ  zˈaː6  tˌi2  mˈo6t̪  bˈɛɜ  ɣˈaːɜj  ŋˈɛ2w  xˈo4  ˈəː4  tʃˈə1w  fˈi  kˈu5ŋ  tʃˈi4  ɗˌyə6c  kˈaɜp  sˈe-ɜc  ɗˌeɜn  tʃˈyə2ŋ  vˈaː2j  nˈam."),
    # 17. Medium with stress/emotion words, speaker 81
    (81, "tˈyɜ  t̪ˈaːɜm,  kˈaŋ  tˈa4ŋ  kˈɛɜw  zˈaː2j,  hˈaj  kˌɔ2n  ɣˈɔ6j  lˌaː2  stɹˈɛs  mˈaː5n  t̪ˈiɜɲ,  lˌaː2  ˈiɛɜw  t̪ˈoɜ  ŋwˈi  kˈəː  kwˈaːn  tʃˈɔ6ŋ  ɗˈoɜj  vˌəːɜj  kˌaːɜc  bˈe6ɲ  t̪ˈim  mˈe-6c."),
    # 18. Short, speaker 62
    (62, "ɗˈiə5  tʃˈu6c  lˌaː2  mˈəɪɜ  kˌaːɜj  ɗˈiə5  tʃˈɔ2n  zˈyə5  hˈaːj  bˈe-ɜɲ  tʃˈyəɜc,  nˈa2m  ˈəː4  hˈaːj  ɗˈə2w  kˌuə4  tʃˈu6c  sˈɛ."),
    # 19. Very long with English loanword, speaker 161
    (161, "ɹˈɔ5  ɹˈaː2ŋ  lˌaː2  mˈaɪkɹəsˌɒft  ɗˌaː5  lˈaː2m  ɹˈəɜt̪  t̪ˈoɜt̪  vˌaː2  ɹˈəɜt̪  tˈe-2ɲ  kˈoŋ  tʃˈɔŋ  vˈiɛ6c  xˈaːj  sˈaːɜŋ  ɹˈaː  nˈe2n  t̪ˈaː4ŋ  hˈe6  ɗˈiɛ2w  hˈe-2ɲ  wˈɪndəʊz  kwˈaː  vˈiɛ6c  tˈaːm  xˈaː4w  fˈə2n  kˈyɜŋ  t̪ˈiɛw  tʃwˈə4n  ɗˌyə6c  sˈaːɜŋ  tʃˈeɜ  ɗˈə2w  t̪ˈiɛn  bˌəː4j  ˈibm  vˌaː2  ɗˈaː4m  bˈaː4w  ɹˈa2ŋ  kˌaːɜc  fˈə2n  mˈe2m  kˌuə4  hˈɔ6  hwˈaː6t̪  ɗˈo6ŋ  ˈo4n  ɗˈi6ɲ  tʃˈen  nˈe2n  t̪ˈaː4ŋ  nˈa2j."),
    # 20. Medium, speaker 132 (same speaker as #15, different text)
    (132, "zˌu2  fˌaː4j  nˈɔɜj  ɹˈa2ŋ  mˌo5j  fˈaːɜt̪  mˈiɲ  kˌuə4  t̪ˈoj  fˈə2n  ɲˈiɛ2w  ɗˈe2w  ɲˈəː2  tˈyə2  hˈyə4ŋ  t̪ˌy2  mˈɛ6,  ɲˌyŋ  sˈy6  hwˈəɜn  lwˈiɛ6n  kˌuə4  tʃˈaː  kˈu5ŋ  ɹˈəɜt̪  hˈi5w  ˈiɜc."),
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

    # Force CPU when GPU is busy with training
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Build model using variant-aware factory
    net_g = build_synthesizer(hps, len(symbols)).to(device)
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
