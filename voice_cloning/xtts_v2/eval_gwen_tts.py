"""Evaluate Gwen-TTS on 20 heldout speakers — same protocol as XTTS eval.

Metric: ECAPA raw 192-d cosine similarity (ref vs gen audio).
Compare against VITS2 baseline (0.43) + XTTS FT (0.71).

Usage:
    python eval_gwen_tts.py --out <out_dir>
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

SAMPLE_RATE_ECAPA = 16_000
HELDOUT_SIDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                100, 110, 120, 130, 140, 150, 160, 170, 180, 190]


def load_ecapa(device):
    from speechbrain.inference import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/mnt/nfs-data/tin_dataset/hf_cache/ecapa_voxceleb",
        run_opts={"device": device},
    )


def ecapa_embed(ecapa, wav_path, device):
    import torchaudio
    wav_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav_np).T
    if sr != SAMPLE_RATE_ECAPA:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_ECAPA)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(device)
    with torch.no_grad():
        emb = ecapa.encode_batch(wav).squeeze(1)
    return emb


def load_heldout_samples(csv_path, heldout_sids):
    import csv
    with open("/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/speaker_map.json") as f:
        spk2sid = json.load(f)
    samples = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter="|")
        for r in reader:
            spk = r["speaker_name"]
            sid = spk2sid.get(spk, -1)
            if sid in heldout_sids:
                samples.setdefault(spk, []).append(
                    (r["audio_file"], r["text"], sid))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default="g-group-ai-lab/gwen-tts-0.6B")
    p.add_argument("--n-per-speaker", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heldout-csv",
                   default="/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits/heldout.csv")
    # Sampling tuning for reducing "lơ lớ" (non-native VN inflection)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=2.0)
    p.add_argument("--subtalker-temperature", type=float, default=0.1)
    p.add_argument("--subtalker-top-k", type=int, default=20)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    device = "cuda"

    print(f"[1/4] Loading Gwen-TTS ({args.model_id})")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        args.model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    gen_config = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=4096,
        repetition_penalty=args.repetition_penalty,
        subtalker_do_sample=True,
        subtalker_temperature=args.subtalker_temperature,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=1.0,
    )

    print("[2/4] Loading ECAPA-TDNN")
    ecapa = load_ecapa(device)

    print(f"[3/4] Loading heldout samples")
    by_speaker = load_heldout_samples(args.heldout_csv, HELDOUT_SIDS)
    print(f"  Found {len(by_speaker)} heldout speakers")

    print(f"[4/4] Synthesize + eval ({args.n_per_speaker} per speaker)")
    results = []
    for spk_name, items in sorted(by_speaker.items()):
        if len(items) < args.n_per_speaker + 1:
            continue
        chosen = random.sample(items, args.n_per_speaker + 1)
        ref_wav, ref_text, sid = chosen[0]
        targets = chosen[1:]

        ref_emb = ecapa_embed(ecapa, ref_wav, device)

        for idx, (tgt_wav, tgt_text, _) in enumerate(targets):
            try:
                wavs, sr = model.generate_voice_clone(
                    text=tgt_text,
                    language="Vietnamese",
                    ref_audio=ref_wav,
                    ref_text=ref_text,
                    **gen_config,
                )
            except Exception as e:
                print(f"  sid={sid} tgt{idx}: synth failed: {e}")
                continue

            wav_np = np.asarray(wavs[0], dtype=np.float32).flatten()
            if len(wav_np) < 8000:
                continue

            gen_path = gen_dir / f"sid{sid:03d}_gen{idx}.wav"
            sf.write(str(gen_path), wav_np, sr)

            gen_emb = ecapa_embed(ecapa, str(gen_path), device)
            tgt_emb = ecapa_embed(ecapa, tgt_wav, device)

            cos = torch.nn.functional.cosine_similarity
            cos_gen_ref = cos(gen_emb, ref_emb).item()
            cos_tgt_ref = cos(tgt_emb, ref_emb).item()

            print(f"  sid={sid:3d} {spk_name} gen{idx}: "
                  f"cos_gen_ref={cos_gen_ref:.3f}  cos_tgt_ref={cos_tgt_ref:.3f}  "
                  f"({len(wav_np)/sr:.1f}s) [sr={sr}]")

            results.append({
                "sid": sid, "speaker": spk_name, "target_text": tgt_text[:80],
                "gen_path": str(gen_path),
                "duration_s": round(len(wav_np)/sr, 2),
                "sample_rate": sr,
                "cos_gen_ref": round(cos_gen_ref, 4),
                "cos_tgt_ref": round(cos_tgt_ref, 4),
            })

    if not results:
        print("No results!")
        return

    gen_sims = [r["cos_gen_ref"] for r in results]
    tgt_sims = [r["cos_tgt_ref"] for r in results]
    per_spk = {}
    for r in results:
        per_spk.setdefault(r["sid"], []).append(r["cos_gen_ref"])

    summary = {
        "model": args.model_id,
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

    print("=" * 60)
    print(f"GWEN-TTS HELDOUT EVAL ({len(results)} samples, {len(per_spk)} speakers)")
    print("=" * 60)
    print(f"  cos(gen, ref) mean:   {summary['overall']['cos_gen_ref_mean']:.4f}")
    print(f"  cos(gen, ref) median: {summary['overall']['cos_gen_ref_median']:.4f}")
    print(f"  cos(gen, ref) range:  [{summary['overall']['cos_gen_ref_min']:.3f}, "
          f"{summary['overall']['cos_gen_ref_max']:.3f}]")
    print(f"  upper bound (tgt-ref): {summary['overall']['upper_bound_tgt_ref_mean']:.4f}")
    print(f"  ─── COMPARE ───")
    print(f"  VITS2+Dual-Path baseline:     0.4300")
    print(f"  XTTS FT (our training):       0.7121")
    print(f"  viXTTS vanilla:               0.7057")
    print(f"  Gwen-TTS (previous benchmark): 0.7751")
    print(f"  Output: {out_json}")


if __name__ == "__main__":
    main()
