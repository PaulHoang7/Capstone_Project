"""
Full evaluation pipeline for VITS2 Vietnamese TTS variants.

Computes:
  1. Tone Confusion Matrix (6x6) via Whisper large-v3 STT
  2. F0 RMSE (PyWorld) against ground truth audio
  3. MCD (Mel Cepstral Distortion) against ground truth
  4. Speaker Similarity (ECAPA-TDNN cosine similarity)
  5. Generates 20+ audio samples from different speakers

Usage:
    cd /home/bes/Desktop/TTS
    python Capstone_project/evaluation/evaluate_tts.py \
        --config vits2_pytorch/configs/vits2_vieneu_base.json \
        --checkpoint vits2_pytorch/logs/vieneu_base/G_438000.pth \
        --test-set Capstone_project/evaluation/tone_test_set.json \
        --output-dir /mnt/nfs-data/tin_dataset/experiments/variant_A_baseline/ \
        --variant-name "A_baseline" \
        --whisper-model large-v3
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))
_VITS2_DIR = os.path.join(_PROJECT_ROOT, 'vits2_pytorch')

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _VITS2_DIR not in sys.path:
    sys.path.insert(0, _VITS2_DIR)

import torch
from scipy.io.wavfile import write as write_wav

import commons
import utils
from text import text_to_sequence
from text.symbols import symbols

from Capstone_project.tone_encoder.tone_utils import (
    extract_tone_label_from_vietnamese,
    text_to_tone_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TTS variant")
    parser.add_argument("--config", type=str, required=True, help="Config JSON path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--test-set", type=str, required=True, help="Test set JSON path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--variant-name", type=str, required=True, help="Variant name")
    parser.add_argument("--whisper-model", type=str, default="large-v3")
    parser.add_argument("--use-tone", action="store_true", help="Use tone-aware model")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip Whisper STT")
    parser.add_argument("--skip-f0", action="store_true", help="Skip F0 RMSE")
    parser.add_argument("--skip-mcd", action="store_true", help="Skip MCD")
    parser.add_argument("--skip-spk-sim", action="store_true", help="Skip speaker sim")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    return parser.parse_args()


def load_model(config_path, checkpoint_path, use_tone=False, device="cuda"):
    """Load TTS model from checkpoint.

    Auto-detects variant from config: if hps.model.variant exists and is C+,
    uses the build_synthesizer factory. Otherwise falls back to legacy loading.
    """
    hps = utils.get_hparams_from_file(config_path)
    variant = getattr(hps.model, "variant", None)

    if variant and variant not in ("A", "B"):
        from Capstone_project.models import build_synthesizer
        net_g = build_synthesizer(hps, n_vocab=len(symbols)).to(device)
    elif use_tone:
        from Capstone_project.models.models_tone import SynthesizerTrnTone
        net_g = SynthesizerTrnTone(
            len(symbols), 80,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    else:
        from models import SynthesizerTrn
        net_g = SynthesizerTrn(
            len(symbols), 80,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

    net_g.eval()
    utils.load_checkpoint(checkpoint_path, net_g, None)
    print(f"Loaded: {checkpoint_path} (variant={variant or 'A'}, device={device})")
    return net_g, hps


def synthesize_text(net_g, hps, ipa_text, speaker_id, use_tone=False):
    """Synthesize audio from IPA text."""
    text_norm = text_to_sequence(ipa_text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    device = next(net_g.parameters()).device
    x = torch.LongTensor(text_norm).to(device).unsqueeze(0)
    x_lengths = torch.LongTensor([len(text_norm)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    tone = None
    if use_tone:
        tone_norm = text_to_tone_sequence(ipa_text, hps.data.text_cleaners)
        if hps.data.add_blank:
            tone_norm = commons.intersperse(tone_norm, 0)
        tone = torch.LongTensor(tone_norm).to(device).unsqueeze(0)

    with torch.no_grad():
        kwargs = dict(noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)
        if tone is not None:
            kwargs['tone'] = tone
        audio = net_g.infer(
            x, x_lengths, sid=sid, **kwargs,
        )[0][0, 0].data.cpu().float().numpy()

    return audio


# ========== Metric Computation ==========

def compute_tone_confusion_matrix(synth_dir, test_set, whisper_model_name):
    """Compute 6x6 tone confusion matrix using Whisper STT."""
    import whisper

    print(f"Loading Whisper {whisper_model_name}...")
    model = whisper.load_model(whisper_model_name)

    # Tone labels for the matrix (1-6, excluding 0=pad and 7=rare)
    tone_names = {1: 'ngang', 2: 'sắc', 3: 'huyền', 4: 'hỏi', 5: 'ngã', 6: 'nặng'}
    matrix = np.zeros((6, 6), dtype=int)  # rows=actual, cols=predicted

    transcriptions = []
    for entry in test_set:
        wav_path = os.path.join(synth_dir, f"{entry['id']}.wav")
        if not os.path.exists(wav_path):
            continue

        # Transcribe with Whisper
        result = model.transcribe(wav_path, language="vi", task="transcribe")
        transcribed = result["text"].strip()

        # Extract tones from transcription
        predicted_tones = extract_tone_label_from_vietnamese(transcribed)
        predicted_tone_list = [t for _, t in predicted_tones]

        # Expected tones from the original (filter out pad=0 and rare=7)
        expected_tones = [t for t in entry['expected_tones'] if 1 <= t <= 6]

        # Align and compare (use min length for safety)
        min_len = min(len(expected_tones), len(predicted_tone_list))
        for i in range(min_len):
            actual = expected_tones[i]
            predicted = predicted_tone_list[i]
            if 1 <= actual <= 6 and 1 <= predicted <= 6:
                matrix[actual - 1][predicted - 1] += 1

        transcriptions.append({
            'id': entry['id'],
            'expected_text_preview': entry['ipa_text'][:50],
            'transcribed': transcribed,
            'expected_tone_count': len(expected_tones),
            'predicted_tone_count': len(predicted_tone_list),
        })

    # Compute accuracy
    total = matrix.sum()
    correct = np.trace(matrix)
    accuracy = correct / total if total > 0 else 0

    # Per-tone accuracy
    per_tone_acc = {}
    for i in range(6):
        row_sum = matrix[i].sum()
        per_tone_acc[tone_names[i + 1]] = (
            matrix[i][i] / row_sum if row_sum > 0 else 0
        )

    return {
        'matrix': matrix.tolist(),
        'tone_names': list(tone_names.values()),
        'overall_accuracy': float(accuracy),
        'per_tone_accuracy': per_tone_acc,
        'total_comparisons': int(total),
        'transcriptions_sample': transcriptions[:10],
    }


def compute_f0_rmse(synth_dir, test_set, sampling_rate=24000):
    """Compute F0 RMSE between synthesized and ground truth audio."""
    import pyworld as pw
    import librosa

    f0_errors = []
    per_tone_f0 = defaultdict(list)

    for entry in test_set:
        gt_path = entry.get('ground_truth_audio')
        synth_path = os.path.join(synth_dir, f"{entry['id']}.wav")

        if not gt_path or not os.path.exists(gt_path) or not os.path.exists(synth_path):
            continue

        try:
            # Load audio
            gt_audio, _ = librosa.load(gt_path, sr=sampling_rate)
            synth_audio, _ = librosa.load(synth_path, sr=sampling_rate)

            # Extract F0
            gt_audio_f64 = gt_audio.astype(np.float64)
            synth_audio_f64 = synth_audio.astype(np.float64)

            gt_f0, gt_t = pw.dio(gt_audio_f64, sampling_rate)
            gt_f0 = pw.stonemask(gt_audio_f64, gt_f0, gt_t, sampling_rate)

            synth_f0, synth_t = pw.dio(synth_audio_f64, sampling_rate)
            synth_f0 = pw.stonemask(synth_audio_f64, synth_f0, synth_t, sampling_rate)

            # Only compare voiced frames
            gt_voiced = gt_f0 > 0
            synth_voiced = synth_f0 > 0

            # Use DTW for alignment
            from dtw import dtw as dtw_func
            gt_feat = gt_f0[gt_voiced].reshape(-1, 1)
            synth_feat = synth_f0[synth_voiced].reshape(-1, 1)

            if len(gt_feat) < 5 or len(synth_feat) < 5:
                continue

            alignment = dtw_func(synth_feat, gt_feat)
            aligned_synth = synth_feat[alignment.index1].flatten()
            aligned_gt = gt_feat[alignment.index2].flatten()

            rmse = np.sqrt(np.mean((aligned_synth - aligned_gt) ** 2))
            f0_errors.append(rmse)

        except Exception as e:
            print(f"  F0 error for {entry['id']}: {e}")
            continue

    if not f0_errors:
        return {'f0_rmse': None, 'f0_rmse_std': None, 'n_samples': 0}

    return {
        'f0_rmse': float(np.mean(f0_errors)),
        'f0_rmse_std': float(np.std(f0_errors)),
        'f0_rmse_median': float(np.median(f0_errors)),
        'n_samples': len(f0_errors),
    }


def compute_mcd(synth_dir, test_set, sampling_rate=24000):
    """Compute Mel Cepstral Distortion between synthesized and ground truth."""
    import librosa

    mcd_values = []

    for entry in test_set:
        gt_path = entry.get('ground_truth_audio')
        synth_path = os.path.join(synth_dir, f"{entry['id']}.wav")

        if not gt_path or not os.path.exists(gt_path) or not os.path.exists(synth_path):
            continue

        try:
            gt_audio, _ = librosa.load(gt_path, sr=sampling_rate)
            synth_audio, _ = librosa.load(synth_path, sr=sampling_rate)

            # Extract MFCCs
            gt_mfcc = librosa.feature.mfcc(y=gt_audio, sr=sampling_rate, n_mfcc=13).T
            synth_mfcc = librosa.feature.mfcc(
                y=synth_audio, sr=sampling_rate, n_mfcc=13
            ).T

            if len(gt_mfcc) < 5 or len(synth_mfcc) < 5:
                continue

            # DTW alignment
            from dtw import dtw as dtw_func
            alignment = dtw_func(synth_mfcc, gt_mfcc)
            aligned_synth = synth_mfcc[alignment.index1]
            aligned_gt = gt_mfcc[alignment.index2]

            # MCD formula: (10/ln(10)) * sqrt(2 * sum((c_s - c_g)^2))
            diff = aligned_synth[:, 1:] - aligned_gt[:, 1:]  # skip c0
            frame_mcd = (10.0 / np.log(10)) * np.sqrt(
                2.0 * np.sum(diff ** 2, axis=1)
            )
            mcd = np.mean(frame_mcd)
            mcd_values.append(mcd)

        except Exception as e:
            print(f"  MCD error for {entry['id']}: {e}")
            continue

    if not mcd_values:
        return {'mcd': None, 'mcd_std': None, 'n_samples': 0}

    return {
        'mcd': float(np.mean(mcd_values)),
        'mcd_std': float(np.std(mcd_values)),
        'mcd_median': float(np.median(mcd_values)),
        'n_samples': len(mcd_values),
    }


def compute_speaker_similarity(synth_dir, test_set, sampling_rate=24000):
    """Compute speaker similarity using ECAPA-TDNN embeddings."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("SpeechBrain not installed. Skipping speaker similarity.")
        return {'speaker_similarity': None, 'n_samples': 0}

    import torchaudio

    print("Loading ECAPA-TDNN speaker encoder...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    similarities = []
    per_speaker = defaultdict(list)

    for entry in test_set:
        gt_path = entry.get('ground_truth_audio')
        synth_path = os.path.join(synth_dir, f"{entry['id']}.wav")

        if not gt_path or not os.path.exists(gt_path) or not os.path.exists(synth_path):
            continue

        try:
            gt_signal, gt_sr = torchaudio.load(gt_path)
            synth_signal, synth_sr = torchaudio.load(synth_path)

            # Resample to 16kHz if needed (ECAPA-TDNN expects 16kHz)
            if gt_sr != 16000:
                gt_signal = torchaudio.functional.resample(gt_signal, gt_sr, 16000)
            if synth_sr != 16000:
                synth_signal = torchaudio.functional.resample(
                    synth_signal, synth_sr, 16000
                )

            gt_emb = classifier.encode_batch(gt_signal.cuda())
            synth_emb = classifier.encode_batch(synth_signal.cuda())

            cos_sim = torch.nn.functional.cosine_similarity(
                gt_emb.squeeze(), synth_emb.squeeze(), dim=0
            ).item()

            similarities.append(cos_sim)
            per_speaker[entry['speaker_id']].append(cos_sim)

        except Exception as e:
            print(f"  Speaker sim error for {entry['id']}: {e}")
            continue

    if not similarities:
        return {'speaker_similarity': None, 'n_samples': 0}

    per_spk_avg = {
        str(spk): float(np.mean(sims)) for spk, sims in per_speaker.items()
    }

    return {
        'speaker_similarity': float(np.mean(similarities)),
        'speaker_similarity_std': float(np.std(similarities)),
        'per_speaker_similarity': per_spk_avg,
        'n_samples': len(similarities),
    }


def save_confusion_matrix_plot(matrix, tone_names, output_path):
    """Save confusion matrix as a heatmap PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(range(len(tone_names)))
    ax.set_yticks(range(len(tone_names)))
    ax.set_xticklabels(tone_names, rotation=45, ha='right')
    ax.set_yticklabels(tone_names)
    ax.set_xlabel('Predicted Tone')
    ax.set_ylabel('Actual Tone')
    ax.set_title('Tone Confusion Matrix')

    # Add text annotations
    for i in range(len(tone_names)):
        for j in range(len(tone_names)):
            val = matrix[i][j]
            color = 'white' if val > np.max(matrix) / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def generate_results_markdown(results, variant_name):
    """Generate human-readable markdown summary."""
    lines = [
        f"# Evaluation Results: Variant {variant_name}",
        f"",
        f"**Checkpoint**: {results.get('checkpoint', 'N/A')}",
        f"**Test set size**: {results.get('test_set_size', 'N/A')}",
        f"**Timestamp**: {results.get('timestamp', 'N/A')}",
        f"",
        f"## Metrics Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    if results.get('tone_confusion'):
        tc = results['tone_confusion']
        lines.append(
            f"| Tone Accuracy | {tc['overall_accuracy']:.4f} "
            f"({tc['total_comparisons']} comparisons) |"
        )
        for tone, acc in tc.get('per_tone_accuracy', {}).items():
            lines.append(f"| - {tone} | {acc:.4f} |")

    if results.get('f0_rmse'):
        f0 = results['f0_rmse']
        if f0.get('f0_rmse') is not None:
            lines.append(
                f"| F0 RMSE | {f0['f0_rmse']:.2f} Hz "
                f"(+/- {f0['f0_rmse_std']:.2f}, n={f0['n_samples']}) |"
            )

    if results.get('mcd'):
        mcd = results['mcd']
        if mcd.get('mcd') is not None:
            lines.append(
                f"| MCD | {mcd['mcd']:.2f} dB "
                f"(+/- {mcd['mcd_std']:.2f}, n={mcd['n_samples']}) |"
            )

    if results.get('speaker_similarity'):
        ss = results['speaker_similarity']
        if ss.get('speaker_similarity') is not None:
            lines.append(
                f"| Speaker Similarity | {ss['speaker_similarity']:.4f} "
                f"(+/- {ss['speaker_similarity_std']:.4f}, n={ss['n_samples']}) |"
            )

    lines.extend([
        f"",
        f"## Audio Samples",
        f"",
        f"Generated {results.get('n_audio_samples', 0)} audio samples in `audio/`",
    ])

    return "\n".join(lines)


def main():
    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Load test set
    with open(args.test_set, 'r') as f:
        test_set = json.load(f)
    print(f"Loaded {len(test_set)} test entries")

    # Load model
    net_g, hps = load_model(args.config, args.checkpoint, use_tone=args.use_tone, device=args.device)

    # Copy config to output
    shutil.copy2(args.config, os.path.join(args.output_dir, "config.json"))
    with open(os.path.join(args.output_dir, "checkpoint_info.json"), "w") as f:
        json.dump({"path": args.checkpoint, "variant": args.variant_name}, f, indent=2)

    # ===== Step 1: Synthesize all test sentences =====
    print("\n--- Synthesizing test sentences ---")
    for i, entry in enumerate(test_set):
        audio = synthesize_text(
            net_g, hps, entry['ipa_text'], entry['speaker_id'],
            use_tone=args.use_tone,
        )
        wav_path = os.path.join(audio_dir, f"{entry['id']}.wav")
        write_wav(wav_path, hps.data.sampling_rate, audio)
        if (i + 1) % 50 == 0:
            print(f"  Synthesized {i+1}/{len(test_set)}")
    print(f"  Synthesized {len(test_set)} sentences total")

    results = {
        'variant_name': args.variant_name,
        'checkpoint': args.checkpoint,
        'test_set_size': len(test_set),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_audio_samples': len(test_set),
    }

    # ===== Step 2: Tone Confusion Matrix =====
    if not args.skip_whisper:
        print("\n--- Computing tone confusion matrix ---")
        tone_results = compute_tone_confusion_matrix(
            audio_dir, test_set, args.whisper_model
        )
        results['tone_confusion'] = tone_results
        save_confusion_matrix_plot(
            np.array(tone_results['matrix']),
            tone_results['tone_names'],
            os.path.join(args.output_dir, "confusion_matrix.png"),
        )
    else:
        print("Skipping Whisper STT (--skip-whisper)")

    # ===== Step 3: F0 RMSE =====
    if not args.skip_f0:
        print("\n--- Computing F0 RMSE ---")
        f0_results = compute_f0_rmse(audio_dir, test_set, hps.data.sampling_rate)
        results['f0_rmse'] = f0_results
        print(f"  F0 RMSE: {f0_results.get('f0_rmse', 'N/A')}")
    else:
        print("Skipping F0 RMSE (--skip-f0)")

    # ===== Step 4: MCD =====
    if not args.skip_mcd:
        print("\n--- Computing MCD ---")
        mcd_results = compute_mcd(audio_dir, test_set, hps.data.sampling_rate)
        results['mcd'] = mcd_results
        print(f"  MCD: {mcd_results.get('mcd', 'N/A')} dB")
    else:
        print("Skipping MCD (--skip-mcd)")

    # ===== Step 5: Speaker Similarity =====
    if not args.skip_spk_sim:
        print("\n--- Computing speaker similarity ---")
        spk_results = compute_speaker_similarity(
            audio_dir, test_set, hps.data.sampling_rate
        )
        results['speaker_similarity'] = spk_results
        print(f"  Speaker similarity: {spk_results.get('speaker_similarity', 'N/A')}")
    else:
        print("Skipping speaker similarity (--skip-spk-sim)")

    # ===== Save results =====
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results: {results_path}")

    # Save markdown summary
    md_content = generate_results_markdown(results, args.variant_name)
    md_path = os.path.join(args.output_dir, "results.md")
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Saved summary: {md_path}")

    print(f"\n{'='*60}")
    print(f"Evaluation complete for variant: {args.variant_name}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
