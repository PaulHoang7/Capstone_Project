"""MCD + F0 RMSE evaluation across pre-generated audio dirs.

Replicates the same (sid, gen_idx) selection used by eval_xtts_heldout.py
(seed=42, n_per_speaker=3, HELDOUT_SIDS), looks up the target ground-truth
audio from heldout.csv, then computes:

  - F0 RMSE (PyWorld DIO + STONEMASK)
  - MCD-DTW (DTW-aligned Mel-Cepstral Distortion, 13 coefficients)

per (synth_wav, target_wav) pair.

Usage:
    python eval_mcd_f0.py \
        --systems xtts_ft:eval_ft/generated \
                  xtts_ctc:eval_ctc/generated \
                  xtts_vanilla:eval_vanilla/generated \
                  gwen:eval_gwen_aligned/generated \
        --out eval_mcd_f0_4way
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
HELDOUT_SIDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
DEFAULT_HELDOUT_CSV = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits/heldout.csv"
SPEAKER_MAP        = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/speaker_map.json"


def load_heldout_samples(csv_path):
    with open(SPEAKER_MAP) as f:
        spk2sid = json.load(f)
    by_speaker = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for r in reader:
            spk = r["speaker_name"]
            sid = spk2sid.get(spk, -1)
            if sid in HELDOUT_SIDS:
                by_speaker.setdefault(spk, []).append(
                    (r["audio_file"], r["text"], sid)
                )
    return by_speaker


def get_selection(by_speaker, n_per_speaker=3, seed=42):
    """Reproduce the exact (sid, gen_idx) → target_wav mapping."""
    random.seed(seed)
    selection = {}   # (sid, gen_idx) → (target_wav, target_text)
    for spk_name, items in sorted(by_speaker.items()):
        if len(items) < n_per_speaker + 1:
            continue
        chosen = random.sample(items, n_per_speaker + 1)
        ref_wav, ref_text, sid = chosen[0]
        targets = chosen[1:]
        for idx, (tgt_wav, tgt_text, _) in enumerate(targets):
            selection[(sid, idx)] = (tgt_wav, tgt_text)
    return selection


def load_audio(path, target_sr=SAMPLE_RATE):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def f0_rmse(synth_wav, gt_wav, sr=SAMPLE_RATE):
    """Computed only on voiced frames in BOTH signals. Hz scale."""
    import pyworld as pw
    f0_s, t = pw.dio(synth_wav, sr, frame_period=10.0)
    f0_s = pw.stonemask(synth_wav, f0_s, t, sr)
    f0_g, t2 = pw.dio(gt_wav, sr, frame_period=10.0)
    f0_g = pw.stonemask(gt_wav, f0_g, t2, sr)
    # Align by truncating to min length
    L = min(len(f0_s), len(f0_g))
    f0_s, f0_g = f0_s[:L], f0_g[:L]
    voiced = (f0_s > 0) & (f0_g > 0)
    if voiced.sum() < 10:
        return None
    rmse = float(np.sqrt(np.mean((f0_s[voiced] - f0_g[voiced]) ** 2)))
    return rmse


def mcd_dtw(synth_wav, gt_wav, sr=SAMPLE_RATE, n_mfcc=13):
    """MCD with DTW alignment. Lower = closer to GT spectrum."""
    import librosa
    # MFCC (log-mel cepstrum proxy) — 13 coeffs is standard
    def mfcc(y):
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256).T
    M_s = mfcc(synth_wav.astype(np.float32))
    M_g = mfcc(gt_wav.astype(np.float32))
    # Skip coeff[0] (energy) for MCD
    M_s, M_g = M_s[:, 1:], M_g[:, 1:]
    # DTW alignment via librosa
    D, wp = librosa.sequence.dtw(X=M_s.T, Y=M_g.T, metric="euclidean")
    # MCD constant = 10 / ln(10) × sqrt(2)
    K = 10.0 / np.log(10.0) * np.sqrt(2.0)
    diffs = []
    for i, j in wp:
        diffs.append(np.sum((M_s[i] - M_g[j]) ** 2))
    if not diffs:
        return None
    mcd = K * float(np.mean(np.sqrt(diffs)))
    return mcd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", required=True,
                   help="name:dir pairs, e.g. xtts_ft:eval_ft/generated")
    p.add_argument("--out", required=True)
    p.add_argument("--heldout-csv", default=DEFAULT_HELDOUT_CSV)
    p.add_argument("--n-per-speaker", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vieneu-wavs-base",
                   default="/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/wavs")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_speaker = load_heldout_samples(args.heldout_csv)
    selection = get_selection(by_speaker, args.n_per_speaker, args.seed)
    print(f"[selection] {len(selection)} (sid, gen_idx) pairs")

    all_results = {}
    for sys_spec in args.systems:
        name, gen_dir = sys_spec.split(":", 1)
        gen_dir = Path(gen_dir)
        print(f"\n=== {name} ===  ({gen_dir})")
        results = []
        for (sid, gen_idx), (tgt_rel, tgt_text) in sorted(selection.items()):
            synth_path = gen_dir / f"sid{sid:03d}_gen{gen_idx}.wav"
            if not synth_path.exists():
                continue
            gt_path = Path(args.vieneu_wavs_base) / Path(tgt_rel).name
            if not gt_path.exists():
                gt_path = Path(tgt_rel)
                if not gt_path.is_absolute():
                    gt_path = Path(args.vieneu_wavs_base) / tgt_rel
            if not gt_path.exists():
                print(f"  sid={sid:03d} gen{gen_idx}: GT missing — {gt_path}")
                continue
            try:
                synth_audio = load_audio(synth_path)
                gt_audio    = load_audio(gt_path)
            except Exception as exc:
                print(f"  sid={sid:03d} gen{gen_idx}: load fail: {exc}")
                continue
            try:
                f0 = f0_rmse(synth_audio, gt_audio)
                mcd = mcd_dtw(synth_audio, gt_audio)
            except Exception as exc:
                print(f"  sid={sid:03d} gen{gen_idx}: compute fail: {exc}")
                continue
            results.append({"sid": sid, "gen_idx": gen_idx,
                             "f0_rmse_hz": f0, "mcd_dtw": mcd})
            print(f"  sid={sid:03d} gen{gen_idx}: F0_RMSE={f0:.1f}Hz  MCD={mcd:.2f}")
        all_results[name] = results

    # Summary
    summary = {}
    for name, results in all_results.items():
        f0s = [r["f0_rmse_hz"] for r in results if r["f0_rmse_hz"] is not None]
        mcds = [r["mcd_dtw"]    for r in results if r["mcd_dtw"]    is not None]
        summary[name] = {
            "n":           len(results),
            "f0_mean_hz":  float(np.mean(f0s))  if f0s else None,
            "f0_median":   float(np.median(f0s)) if f0s else None,
            "mcd_mean":    float(np.mean(mcds)) if mcds else None,
            "mcd_median":  float(np.median(mcds)) if mcds else None,
        }
    with open(out_dir / "mcd_f0_results.json", "w") as f:
        json.dump({"per_sample": all_results, "summary": summary}, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON (lower = better for both)")
    print("=" * 60)
    print(f"{'system':14s} {'n':>4s} {'F0_RMSE_mean':>14s} {'MCD_mean':>10s}")
    for name, s in summary.items():
        f0 = f"{s['f0_mean_hz']:.1f}" if s['f0_mean_hz'] is not None else "—"
        mcd = f"{s['mcd_mean']:.2f}"    if s['mcd_mean']    is not None else "—"
        print(f"{name:14s} {s['n']:4d} {f0:>14s} {mcd:>10s}")
    print(f"\nSaved: {out_dir}/mcd_f0_results.json")


if __name__ == "__main__":
    main()
