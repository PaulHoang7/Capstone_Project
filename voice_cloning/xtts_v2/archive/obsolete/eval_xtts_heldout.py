"""Evaluate XTTS (fine-tuned or vanilla viXTTS) heldout cos_sim.

Mirrors Phase 3 VITS2 protocol for fair comparison:
  - 20 heldout speakers (sids 0, 10, ..., 190)
  - Per speaker: 1 ref + N target texts
  - Metric: cos_sim(ECAPA_raw_192d(ref), ECAPA_raw_192d(gen))
  - Upper bound: cos_sim(ref, ground_truth_target) [no TTS involved]

Usage:
    python eval_xtts_heldout.py --ckpt-dir <path_to_xtts_checkpoint_dir> --out <out_dir>
      --ckpt-dir: directory with model.pth, config.json, vocab.json
      For fine-tuned: copy best_model_761026.pth into a dir with viXTTS config+vocab
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from scipy.io.wavfile import write as wav_write

SAMPLE_RATE_TTS   = 24_000
SAMPLE_RATE_ECAPA = 16_000

HELDOUT_SIDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                100, 110, 120, 130, 140, 150, 160, 170, 180, 190]


def load_ecapa(device):
    """Load speechbrain ECAPA-TDNN (same model class Phase 3 used)."""
    from speechbrain.pretrained import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/mnt/nfs-data/tin_dataset/hf_cache/ecapa_voxceleb",
        run_opts={"device": device},
    )
    return model


def ecapa_embed(ecapa, wav_path, device):
    """Load wav → 16kHz mono → ECAPA 192-d embedding."""
    import torchaudio
    wav_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav_np).T  # [C, T]
    if sr != SAMPLE_RATE_ECAPA:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_ECAPA)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(device)
    with torch.no_grad():
        emb = ecapa.encode_batch(wav).squeeze(1)  # [1, 192]
    return emb


def load_heldout_samples(csv_path, heldout_sids):
    """Read heldout.csv → {speaker_name: [(wav_path, text, sid), ...]}."""
    import csv
    # Need speaker_map for sid
    with open("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/speaker_map.json") as f:
        spk2sid = json.load(f)

    samples = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for r in reader:
            spk = r["speaker_name"]
            sid = spk2sid.get(spk, -1)
            if sid in heldout_sids:
                samples.setdefault(spk, []).append((r["audio_file"], r["text"], sid))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True, help="Dir with model.pth + config.json + vocab.json")
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-speaker", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heldout-csv",
                   default="/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits/heldout.csv")
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load XTTS ────────────────────────────────────────────────
    print(f"[1/4] Loading XTTS from {args.ckpt_dir}")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    config = XttsConfig()
    config.load_json(str(Path(args.ckpt_dir) / "config.json"))
    xtts = Xtts.init_from_config(config)
    xtts.load_checkpoint(config, checkpoint_dir=str(args.ckpt_dir), use_deepspeed=False, eval=True)
    xtts.to(device)

    # ── Load ECAPA-TDNN (raw 192-d, same as Phase 3) ─────────────
    print(f"[2/4] Loading ECAPA-TDNN VoxCeleb")
    ecapa = load_ecapa(device)

    # ── Load heldout samples ─────────────────────────────────────
    print(f"[3/4] Loading heldout samples from {args.heldout_csv}")
    by_speaker = load_heldout_samples(args.heldout_csv, HELDOUT_SIDS)
    print(f"  Found {len(by_speaker)} heldout speakers")

    # ── Evaluate ─────────────────────────────────────────────────
    print(f"[4/4] Generating + evaluating ({args.n_per_speaker} per speaker)")
    results = []
    for spk_name, items in sorted(by_speaker.items()):
        if len(items) < args.n_per_speaker + 1:
            continue
        chosen = random.sample(items, args.n_per_speaker + 1)
        ref_wav, ref_text, sid = chosen[0]
        targets = chosen[1:]

        # Extract conditioning latents once per speaker
        try:
            gpt_latent, speaker_emb = xtts.get_conditioning_latents(
                audio_path=ref_wav, gpt_cond_len=6, max_ref_length=30)
        except Exception as exc:
            print(f"  sid={sid} {spk_name}: skip — cond extract failed: {exc}")
            continue

        # Reference ECAPA embedding (ground truth speaker)
        ref_emb = ecapa_embed(ecapa, ref_wav, device)  # [1,192]

        for idx, (tgt_wav, tgt_text, _) in enumerate(targets):
            # Generate TTS
            try:
                out = xtts.inference(
                    tgt_text, language="vi",
                    gpt_cond_latent=gpt_latent,
                    speaker_embedding=speaker_emb,
                    temperature=0.7, length_penalty=1.0,
                    repetition_penalty=10.0, top_k=30, top_p=0.85,
                )
            except Exception as exc:
                print(f"  sid={sid} tgt{idx}: synthesize failed: {exc}")
                continue

            wav_np = np.asarray(out["wav"], dtype=np.float32)
            if len(wav_np) < 8000:
                continue

            gen_path = gen_dir / f"sid{sid:03d}_gen{idx}.wav"
            sf.write(str(gen_path), wav_np, SAMPLE_RATE_TTS)

            # ECAPA emb for gen + ground truth target
            gen_emb = ecapa_embed(ecapa, str(gen_path), device)
            tgt_emb = ecapa_embed(ecapa, tgt_wav, device)

            cos = torch.nn.functional.cosine_similarity
            cos_gen_ref = cos(gen_emb, ref_emb).item()
            cos_tgt_ref = cos(tgt_emb, ref_emb).item()

            print(f"  sid={sid:3d} {spk_name} gen{idx}: cos_gen_ref={cos_gen_ref:.3f}  "
                  f"cos_tgt_ref={cos_tgt_ref:.3f}  ({len(wav_np)/24000:.1f}s)")

            results.append({
                "sid": sid, "speaker": spk_name, "target_text": tgt_text[:80],
                "gen_path": str(gen_path), "duration_s": round(len(wav_np)/24000, 2),
                "cos_gen_ref": round(cos_gen_ref, 4),
                "cos_tgt_ref": round(cos_tgt_ref, 4),
            })

    # ── Summary ─────────────────────────────────────────────────
    if not results:
        print("No results!")
        return

    gen_sims = [r["cos_gen_ref"] for r in results]
    tgt_sims = [r["cos_tgt_ref"] for r in results]
    per_spk = {}
    for r in results:
        per_spk.setdefault(r["sid"], []).append(r["cos_gen_ref"])

    summary = {
        "checkpoint_dir": args.ckpt_dir,
        "n_samples": len(results),
        "n_speakers": len(per_spk),
        "overall": {
            "cos_gen_ref_mean": round(float(np.mean(gen_sims)), 4),
            "cos_gen_ref_median": round(float(np.median(gen_sims)), 4),
            "cos_gen_ref_std": round(float(np.std(gen_sims)), 4),
            "cos_gen_ref_min": round(float(np.min(gen_sims)), 4),
            "cos_gen_ref_max": round(float(np.max(gen_sims)), 4),
            "upper_bound_tgt_ref_mean": round(float(np.mean(tgt_sims)), 4),
        },
        "per_speaker_mean": {str(k): round(float(np.mean(v)), 4) for k, v in per_spk.items()},
        "samples": results,
    }

    out_json = out_dir / "heldout_eval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 55)
    print(f"XTTS HELDOUT EVAL ({len(results)} samples, {len(per_spk)} speakers)")
    print("=" * 55)
    print(f"  cos(gen, ref) mean:   {summary['overall']['cos_gen_ref_mean']:.4f}")
    print(f"  cos(gen, ref) median: {summary['overall']['cos_gen_ref_median']:.4f}")
    print(f"  cos(gen, ref) range:  [{summary['overall']['cos_gen_ref_min']:.3f}, "
          f"{summary['overall']['cos_gen_ref_max']:.3f}]")
    print(f"  upper bound (tgt-ref): {summary['overall']['upper_bound_tgt_ref_mean']:.4f}")
    print(f"  VITS2+Dual-Path Phase 3 baseline: 0.43 (for comparison)")
    print(f"  Output: {out_json}")


if __name__ == "__main__":
    main()
