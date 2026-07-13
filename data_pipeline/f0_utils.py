"""
F0 extraction utilities for Variant E auxiliary loss.

Extracts F0 from audio using PyWorld DIO, returns log-F0 aligned to
the spectrogram frame rate (hop_length).
"""

import numpy as np
import torch


def extract_f0_from_audio(audio_np, sr=24000, hop_length=256):
    """Extract F0 from audio waveform using PyWorld.

    Args:
        audio_np: numpy array [T_samples], float64.
        sr: sample rate.
        hop_length: hop length (to match spectrogram frames).
    Returns:
        log_f0: numpy array [T_frames], log-F0 (0.0 for unvoiced).
    """
    import pyworld as pw

    audio_f64 = audio_np.astype(np.float64)
    f0, _ = pw.dio(audio_f64, sr, frame_period=hop_length / sr * 1000)
    f0 = pw.stonemask(audio_f64, f0, _, sr)

    # Convert to log scale (unvoiced → 0.0)
    log_f0 = np.where(f0 > 0, np.log(f0), 0.0).astype(np.float32)
    return log_f0


def align_f0_to_text(f0_frames, attn):
    """Align frame-level F0 to text-level using attention matrix.

    Computes weighted average of frame F0 per text position.

    Args:
        f0_frames: [b, 1, T_mel] frame-level log-F0.
        attn: [b, 1, T_mel, T_text] hard attention from MAS.
    Returns:
        f0_text: [b, 1, T_text] text-level log-F0.
    """
    # attn: [b, 1, T_mel, T_text] from VITS2 MAS output
    # f0_frames: [b, 1, T_f0] — may differ from T_mel
    # We want: for each text position, average F0 over aligned mel frames

    # Transpose attn to [b, T_text, T_mel] for weighted average
    attn_sq = attn.squeeze(1).transpose(1, 2)  # [b, 1, T_mel, T_text] → [b, T_text, T_mel]
    f0_sq = f0_frames.squeeze(1)  # [b, T_f0]

    T_mel = attn_sq.size(-1)  # mel dimension from attention
    T_f0 = f0_sq.size(-1)

    # Align F0 length to attention mel dimension
    if T_f0 < T_mel:
        f0_sq = torch.nn.functional.pad(f0_sq, (0, T_mel - T_f0))
    elif T_f0 > T_mel:
        f0_sq = f0_sq[:, :T_mel]

    # Weighted sum: sum_mel(attn[text, mel] * f0[mel]) / sum_mel(attn[text, mel])
    numerator = torch.matmul(attn_sq, f0_sq.unsqueeze(-1))  # [b, T_text, 1]
    denominator = attn_sq.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [b, T_text, 1]

    f0_text = (numerator / denominator).transpose(1, 2)  # [b, 1, T_text]
    return f0_text
