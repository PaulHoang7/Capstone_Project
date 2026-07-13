"""Evaluate XTTS-FT (with/without CTC head) on text-audio alignment.

For each heldout speaker:
  1. Pick 2 samples: one as ref_audio, one as target_text
  2. Synthesize target_text conditioned on ref → gen_wav
  3. Whisper-transcribe gen_wav → predicted_text
  4. Compute CER(predicted_text, target_text)
  5. Aligned = (CER < threshold)
Report:
  - mean CER
  - alignment rate (% samples with CER < threshold)
  - mean (gen_dur / expected_dur) — tail-hallucination indicator

Usage:
    python eval_xtts_alignment.py --best-model <path.pth> --out <out_dir>
"""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "coqui_tts"))

VIXTTS_DIR = Path("/mnt/nfs-data/tin_dataset/checkpoints/vixtts")
HELDOUT_CSV = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits/heldout.csv")
SPEAKER_MAP = Path("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/speaker_map.json")
HELDOUT_SIDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                100, 110, 120, 130, 140, 150, 160, 170, 180, 190]


def normalize_text(t: str) -> str:
    """Lowercase + strip diacritic-noise for fair CER (keep Vietnamese accents
    since they're tone-critical, but unify case + collapse whitespace)."""
    import re, unicodedata
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\sàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def cer(pred: str, ref: str) -> float:
    """Character Error Rate using edit distance / max(len)."""
    p = normalize_text(pred).replace(" ", "")
    r = normalize_text(ref).replace(" ", "")
    if not r:
        return 0.0 if not p else 1.0
    # Levenshtein
    m, n = len(p), len(r)
    if m == 0: return 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if p[i-1] == r[j-1] else 1))
            prev = cur
    return dp[n] / max(n, 1)


def load_xtts_with_ckpt(best_model_path: str, device: str = "cuda"):
    """Load XTTS architecture from viXTTS + apply best_model.pth weights
    (filters CTC head keys — not used at inference)."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    cfg = XttsConfig()
    cfg.load_json(str(VIXTTS_DIR / "config.json"))
    xtts = Xtts.init_from_config(cfg)

    print(f"[load] reading {best_model_path}")
    raw = torch.load(best_model_path, map_location="cpu", weights_only=False)
    full_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    # Drop CTC head keys — only the trained GPT/HiFi-GAN/DVAE matter at inference
    inf_sd = {k: v for k, v in full_sd.items() if "ctc_head" not in k}
    # The trainer wraps everything under "xtts." prefix; Xtts.state_dict doesn't.
    own_keys = set(xtts.state_dict().keys())
    if not any(k.startswith("xtts.") for k in own_keys):
        inf_sd = {k.removeprefix("xtts."): v for k, v in inf_sd.items()}

    missing, unexpected = xtts.load_state_dict(inf_sd, strict=False)
    print(f"  loaded: {len(inf_sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing and len(missing) < 10:
        print(f"  missing samples: {missing[:5]}")

    # Tokenizer needs vocab.json
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    xtts.tokenizer = VoiceBpeTokenizer(str(VIXTTS_DIR / "vocab.json"))
    xtts.to(device).eval()
    return xtts


def synth(xtts, text: str, ref_path: str):
    """Run XTTS inference with shared XTTS_GEN_CONFIG."""
    from xtts_gen_config import XTTS_GEN_CONFIG, clean_for_xtts
    gpt_latent, spk_emb = xtts.get_conditioning_latents(
        audio_path=ref_path, gpt_cond_len=6, max_ref_length=30
    )
    cleaned = clean_for_xtts(text)
    out = xtts.inference(
        cleaned, language="vi",
        gpt_cond_latent=gpt_latent, speaker_embedding=spk_emb,
        **XTTS_GEN_CONFIG,
    )
    return np.asarray(out["wav"], dtype=np.float32), cleaned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="v3")
    p.add_argument("--n-per-speaker", type=int, default=2)
    p.add_argument("--cer-threshold", type=float, default=0.30,
                   help="CER below this counts as 'aligned'")
    p.add_argument("--max-speakers", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gen_dir = out / "gen"; gen_dir.mkdir(exist_ok=True)

    # Load heldout samples
    with open(SPEAKER_MAP) as f:
        spk2sid = json.load(f)
    samples_by_spk = {}
    with open(HELDOUT_CSV) as f:
        for r in csv.DictReader(f, delimiter="|"):
            sid = spk2sid.get(r["speaker_name"], -1)
            if sid in HELDOUT_SIDS:
                samples_by_spk.setdefault(r["speaker_name"], []).append(
                    (r["audio_file"], r["text"], sid))
    print(f"[data] {len(samples_by_spk)} heldout speakers loaded")

    # Load XTTS
    print(f"[1/3] loading XTTS from {args.best_model}")
    t0 = time.time()
    xtts = load_xtts_with_ckpt(args.best_model)
    print(f"  loaded in {time.time()-t0:.1f}s · VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load Whisper
    print(f"[2/3] loading faster-whisper-small")
    from faster_whisper import WhisperModel
    asr = WhisperModel("small", device="cuda", compute_type="float16")

    # Eval loop
    print(f"[3/3] generating + scoring (n={args.n_per_speaker}/spk, threshold={args.cer_threshold})")
    results = []
    spks = sorted(samples_by_spk.items())[:args.max_speakers]
    for spk, items in spks:
        if len(items) < args.n_per_speaker + 1:
            continue
        chosen = random.sample(items, args.n_per_speaker + 1)
        ref_wav, ref_text, sid = chosen[0]
        targets = chosen[1:]
        for idx, (tgt_wav, tgt_text, _) in enumerate(targets):
            try:
                wav, cleaned_text = synth(xtts, tgt_text, ref_wav)
            except Exception as e:
                print(f"  sid={sid} tgt{idx}: SYNTH FAIL {e}")
                continue
            gen_path = gen_dir / f"sid{sid:03d}_t{idx}.wav"
            sf.write(str(gen_path), wav, 24000)

            # Whisper transcribe
            try:
                segs, _ = asr.transcribe(str(gen_path), language="vi",
                                          beam_size=1, vad_filter=False)
                pred_text = " ".join(s.text.strip() for s in segs).strip()
            except Exception as e:
                pred_text = ""

            gen_dur = len(wav) / 24000
            expected_dur = max(1.0, len(cleaned_text) / 6.0 + 0.8)
            er = cer(pred_text, tgt_text)
            results.append({
                "sid": sid, "target_text": tgt_text, "cleaned": cleaned_text,
                "pred_text": pred_text, "cer": er,
                "gen_dur": round(gen_dur, 2),
                "expected_dur": round(expected_dur, 2),
                "dur_ratio": round(gen_dur / expected_dur, 2),
                "aligned": er < args.cer_threshold,
            })
            mark = "✓" if er < args.cer_threshold else "✗"
            print(f"  sid={sid:03d} t{idx}: CER={er:.3f} {mark} dur={gen_dur:.1f}s"
                  f"  tgt={tgt_text[:40]!r} pred={pred_text[:40]!r}")

    # Aggregate
    if results:
        cers = [r["cer"] for r in results]
        ratios = [r["dur_ratio"] for r in results]
        aligned = sum(1 for r in results if r["aligned"])
        summary = {
            "label": args.label,
            "n_samples": len(results),
            "mean_cer": round(np.mean(cers), 4),
            "median_cer": round(float(np.median(cers)), 4),
            "alignment_rate": round(aligned / len(results), 4),
            "mean_dur_ratio": round(np.mean(ratios), 3),
            "best_model": args.best_model,
        }
        print(f"\n=== {args.label} SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        (out / "results.json").write_text(
            json.dumps({"summary": summary, "samples": results}, ensure_ascii=False, indent=2)
        )
        print(f"  → {out / 'results.json'}")


if __name__ == "__main__":
    main()
