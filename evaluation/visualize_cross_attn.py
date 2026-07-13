"""
Visualize cross-attention maps from Variant D.

Usage:
    python Capstone_project/evaluation/visualize_cross_attn.py \
        --config Capstone_project/configs/vits2_vieneu_variant_d_v2.json \
        --checkpoint path/to/G_xxx.pth \
        --text "xin chào các bạn" \
        --speaker-id 0 \
        --output-dir /tmp/cross_attn_viz/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))
_VITS2_DIR = os.path.join(_PROJECT_ROOT, 'vits2_pytorch')
for p in [_PROJECT_ROOT, _VITS2_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import commons
import utils
from text import text_to_sequence
from text.symbols import symbols
from Capstone_project.tone_encoder.tone_utils import extract_tone_label_from_vietnamese


def visualize(args):
    hps = utils.get_hparams_from_file(args.config)

    from Capstone_project.models import build_synthesizer
    net_g = build_synthesizer(hps, n_vocab=len(symbols)).to(args.device)
    net_g.eval()
    utils.load_checkpoint(args.checkpoint, net_g, None)

    # Prepare input
    text_norm = text_to_sequence(args.text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    x = torch.LongTensor(text_norm).to(args.device).unsqueeze(0)
    x_lengths = torch.LongTensor([len(text_norm)]).to(args.device)
    sid = torch.LongTensor([args.speaker_id]).to(args.device)

    # Extract tones
    tones = extract_tone_label_from_vietnamese(args.text)
    if hps.data.add_blank:
        tones = commons.intersperse(tones, 0)
    tone = torch.LongTensor(tones).to(args.device).unsqueeze(0)

    # Inference
    with torch.no_grad():
        audio = net_g.infer(
            x, x_lengths, sid=sid, tone=tone,
            noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0,
        )[0][0, 0].cpu().float().numpy()

    # Get attention weights from cross-attention module
    cross_attn = net_g.enc_p.cross_attn
    attn_l2t = cross_attn._attn_weights_l2t  # [b, t, t]
    attn_t2l = cross_attn._attn_weights_t2l  # [b, t, t]

    if attn_l2t is None:
        print("No attention weights stored. Make sure model ran inference.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot L→T attention
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    a_l2t = attn_l2t[0].cpu().numpy()
    a_t2l = attn_t2l[0].cpu().numpy()

    # Trim to actual length
    seq_len = x_lengths[0].item()
    a_l2t = a_l2t[:seq_len, :seq_len]
    a_t2l = a_t2l[:seq_len, :seq_len]

    im0 = axes[0].imshow(a_l2t, aspect='auto', origin='lower', cmap='hot')
    axes[0].set_title(f"L→T: Linguistic attends to Tonal\n'{args.text}'")
    axes[0].set_xlabel("Tonal position")
    axes[0].set_ylabel("Linguistic position")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(a_t2l, aspect='auto', origin='lower', cmap='hot')
    axes[1].set_title(f"T→L: Tonal attends to Linguistic\n'{args.text}'")
    axes[1].set_xlabel("Linguistic position")
    axes[1].set_ylabel("Tonal position")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "cross_attention_maps.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # Save audio
    from scipy.io.wavfile import write as write_wav
    audio_path = os.path.join(args.output_dir, "sample.wav")
    write_wav(audio_path, hps.data.sampling_rate, (audio * 32767).astype(np.int16))
    print(f"Saved: {audio_path}")

    # Print gate values (from cross-attention)
    gate_l = torch.sigmoid(cross_attn.gate_l).mean().item()
    gate_t = torch.sigmoid(cross_attn.gate_t).mean().item()
    print(f"Cross-attn gate values: L={gate_l:.3f}, T={gate_t:.3f}")
    print(f"  (0=ignore cross-attn, 1=fully use cross-attn)")

    # Print fusion gate stats
    if hasattr(net_g.enc_p, 'gate'):
        print("Fusion gate statistics available after training analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default="xin chào các bạn")
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="/tmp/cross_attn_viz/")
    parser.add_argument("--device", type=str, default="cpu")
    visualize(parser.parse_args())