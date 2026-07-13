"""
Evaluate Phase 3 zero-shot voice cloning on 20 held-out speakers.

For each held-out speaker:
  1. Pick 2 samples: one as reference_audio, one as target_text
  2. Encode reference → speaker embedding g_ref
  3. Synthesize target_text conditioned on g_ref → generated_audio
  4. Encode generated_audio → g_gen
  5. Compute cosine_similarity(g_ref, g_gen)

Also compares against ground-truth audio (upper bound) and random speaker (lower bound).

Usage:
    python Capstone_project/scripts/eval_phase3_zeroshot.py \\
        --ckpt /home/bes/Desktop/Tin/logs/vieneu_clone_phase3/G_519000.pth \\
        --out /tmp/phase3_eval
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write as wav_write

_HERE = Path(__file__).resolve()
_TIN_ROOT = _HERE.parents[2]
_VITS2_ROOT = _TIN_ROOT / "vits2_pytorch"
for _p in (str(_TIN_ROOT), str(_VITS2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)

HELD_OUT = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

SAMPLE_RATE_ECAPA = 16_000
SAMPLE_RATE_TTS = 24_000


def load_samples_per_speaker(filelist_path: str, speakers: list[int]) -> dict:
    """Return {sid: [(wav, text), ...]} for each speaker."""
    out: dict = {}
    with open(filelist_path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            wav, sid, text = parts[0], int(parts[1]), parts[2]
            if sid in speakers:
                out.setdefault(sid, []).append((wav, text))
    return out


def encode_wav(spk_enc, wav_path: str, device) -> torch.Tensor:
    """Load wav → 16kHz mono → speaker embedding [1, 192]."""
    import soundfile as sf
    wav_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav_np).T  # [channels, T]
    if sr != SAMPLE_RATE_ECAPA:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_ECAPA)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(device)
    wav_lens = torch.ones(1, device=device)
    with torch.no_grad():
        emb = spk_enc.encode(wav, wav_lens)  # [1, 192]
    return emb  # for similarity comparison use raw 192-d embedding


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Phase 3 TTS checkpoint")
    p.add_argument("--config", default="Capstone_project/configs/vits2_vieneu_clone_phase3.json")
    p.add_argument("--spk-ckpt",
                   default="/mnt/nfs-data/tin_dataset/checkpoints/speaker_encoder/speaker_encoder_best.pth")
    p.add_argument("--filelist", default="vits2_pytorch/filelists/vieneu_train_filelist.txt")
    p.add_argument("--out", default="/tmp/phase3_eval")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-per-speaker", type=int, default=3,
                   help="Number of text samples to synthesize per speaker")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "generated"
    audio_dir.mkdir(exist_ok=True)

    # ── Load TTS + speaker encoder ──────────────────────────────────
    log.info(f"Loading Phase 3 checkpoint: {args.ckpt}")
    from Capstone_project.scripts.pipeline_end_to_end import TTSEngine
    engine = TTSEngine(
        config_path=args.config,
        tts_ckpt=args.ckpt,
        spk_ckpt=args.spk_ckpt,
        device=args.device,
    )
    spk_enc = engine.spk_enc
    device = engine.device

    # ── Collect samples per held-out speaker ────────────────────────
    samples = load_samples_per_speaker(args.filelist, HELD_OUT)
    log.info(f"Loaded samples for {len(samples)} held-out speakers")

    # ── Evaluate ─────────────────────────────────────────────────────
    results = []

    for sid in sorted(samples.keys()):
        spk_samples = samples[sid]
        if len(spk_samples) < args.n_per_speaker + 1:
            log.warning(f"  sid={sid} only has {len(spk_samples)} samples, skipping")
            continue

        chosen = random.sample(spk_samples, args.n_per_speaker + 1)
        ref_wav, ref_text = chosen[0]
        targets = chosen[1:]  # n target (text, wav) pairs

        log.info(f"── Speaker sid={sid} ────────────────────────")
        log.info(f"  Reference: {Path(ref_wav).name}")

        # Encode reference
        try:
            g_ref_emb = encode_wav(spk_enc, ref_wav, device)  # [1, 192]
        except Exception as exc:
            log.error(f"  Failed to encode ref: {exc}")
            continue

        # Project to TTS gin_channels for synthesis
        with torch.no_grad():
            g_ref_projected = spk_enc.projection(g_ref_emb).unsqueeze(-1)  # [1, 256, 1]

        for idx, (target_wav, target_text) in enumerate(targets):
            # Synthesize target text with reference speaker
            try:
                audio = engine.synthesise(target_text, g_ref_projected)
            except Exception as exc:
                log.error(f"  TTS failed: {exc}")
                continue

            if len(audio) < 8000:  # < 0.33s at 24kHz = probably failure
                log.warning(f"  Generated audio too short ({len(audio)} samples)")
                continue

            # Save generated audio
            gen_path = audio_dir / f"sid{sid:03d}_gen{idx}.wav"
            audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wav_write(str(gen_path), SAMPLE_RATE_TTS, audio_int16)

            # Encode generated audio
            try:
                g_gen_emb = encode_wav(spk_enc, str(gen_path), device)  # [1, 192]
            except Exception as exc:
                log.error(f"  Failed to encode generated: {exc}")
                continue

            # Also encode ground-truth target audio (upper bound reference)
            try:
                g_tgt_emb = encode_wav(spk_enc, target_wav, device)
            except Exception:
                g_tgt_emb = g_ref_emb

            # Cosine similarities (on raw 192-d embeddings)
            cos = torch.nn.functional.cosine_similarity

            sim_gen_vs_ref = cos(g_gen_emb, g_ref_emb).item()
            sim_tgt_vs_ref = cos(g_tgt_emb, g_ref_emb).item()  # upper bound

            log.info(
                f"  [gen{idx}] cos(gen,ref)={sim_gen_vs_ref:.3f}  "
                f"cos(tgt,ref)={sim_tgt_vs_ref:.3f}  "
                f"({len(audio)/SAMPLE_RATE_TTS:.1f}s)  "
                f'"{target_text[:40]}"'
            )

            results.append({
                "sid": sid,
                "ref_wav": ref_wav,
                "target_wav": target_wav,
                "target_text": target_text[:80],
                "gen_audio": str(gen_path),
                "duration_s": round(len(audio) / SAMPLE_RATE_TTS, 2),
                "cos_gen_vs_ref": round(sim_gen_vs_ref, 4),
                "cos_tgt_vs_ref": round(sim_tgt_vs_ref, 4),
            })

    # ── Aggregate stats ─────────────────────────────────────────────
    if not results:
        log.error("No results")
        return

    gen_sims = [r["cos_gen_vs_ref"] for r in results]
    tgt_sims = [r["cos_tgt_vs_ref"] for r in results]

    # Per-speaker averages
    per_spk = {}
    for r in results:
        per_spk.setdefault(r["sid"], []).append(r["cos_gen_vs_ref"])
    per_spk_avg = {sid: np.mean(vs) for sid, vs in per_spk.items()}

    summary = {
        "n_samples": len(results),
        "n_speakers": len(per_spk),
        "checkpoint": args.ckpt,
        "overall": {
            "cos_gen_vs_ref_mean": round(float(np.mean(gen_sims)), 4),
            "cos_gen_vs_ref_median": round(float(np.median(gen_sims)), 4),
            "cos_gen_vs_ref_std": round(float(np.std(gen_sims)), 4),
            "cos_gen_vs_ref_min": round(float(np.min(gen_sims)), 4),
            "cos_gen_vs_ref_max": round(float(np.max(gen_sims)), 4),
            "upper_bound_tgt_vs_ref_mean": round(float(np.mean(tgt_sims)), 4),
        },
        "per_speaker": {str(k): round(v, 4) for k, v in per_spk_avg.items()},
        "target_met": float(np.mean(gen_sims)) > 0.75,
        "samples": results,
    }

    with open(out_dir / "zeroshot_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print summary
    log.info("=" * 55)
    log.info(f"ZERO-SHOT EVALUATION SUMMARY ({len(results)} samples)")
    log.info("=" * 55)
    log.info(f"  Generated vs Reference (zero-shot clone quality):")
    log.info(f"    Mean:    {summary['overall']['cos_gen_vs_ref_mean']:.4f}")
    log.info(f"    Median:  {summary['overall']['cos_gen_vs_ref_median']:.4f}")
    log.info(f"    Std:     {summary['overall']['cos_gen_vs_ref_std']:.4f}")
    log.info(f"    Range:   [{summary['overall']['cos_gen_vs_ref_min']:.3f}, "
             f"{summary['overall']['cos_gen_vs_ref_max']:.3f}]")
    log.info("")
    log.info(f"  Upper bound (target ground-truth vs ref, same speaker):")
    log.info(f"    Mean:    {summary['overall']['upper_bound_tgt_vs_ref_mean']:.4f}")
    log.info("")
    log.info(f"  Target: > 0.75 → {'✅ PASSED' if summary['target_met'] else '❌ NOT MET'}")
    log.info(f"  Output: {out_dir / 'zeroshot_eval.json'}")
    log.info("")
    log.info("  Per-speaker averages:")
    for sid in sorted(per_spk_avg.keys()):
        bar = "█" * int(per_spk_avg[sid] * 20)
        log.info(f"    sid={sid:3d}: {per_spk_avg[sid]:.3f}  {bar}")


if __name__ == "__main__":
    main()
