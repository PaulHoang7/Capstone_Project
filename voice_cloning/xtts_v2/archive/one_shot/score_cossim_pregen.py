"""Compute cos_sim(gen, ref_audio) over pre-generated heldout dirs.

Mirrors the (sid, gen_idx) selection from eval_mcd_f0.py / eval_xtts_heldout.py.
Use to score VITS2 (or any pre-synthesized system) against the same ref audio
that XTTS/Gwen were conditioned on.

Usage:
    python score_cossim_pregen.py \
      --systems vits2_d:eval_vits2_d/generated xtts_ft:eval_ft/generated \
      --out cossim_results
"""
import argparse, csv, json, random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HELDOUT_SIDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
HELDOUT_CSV = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/xtts_splits/heldout.csv"
SPEAKER_MAP = "/mnt/nfs-data/tin_dataset/VieNeu-TTS-140h/speaker_map.json"
ECAPA_SR = 16000


def load_heldout(csv_path):
    spk2sid = json.load(open(SPEAKER_MAP))
    by_speaker = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f, delimiter="|"):
            spk = r["speaker_name"]
            sid = spk2sid.get(spk, -1)
            if sid in HELDOUT_SIDS:
                by_speaker.setdefault(spk, []).append(
                    (r["audio_file"], r["text"], sid)
                )
    return by_speaker


def build_selection(by_speaker, n_per_speaker=3, seed=42):
    random.seed(seed)
    sel = {}
    refs = {}
    for spk, items in sorted(by_speaker.items()):
        if len(items) < n_per_speaker + 1:
            continue
        chosen = random.sample(items, n_per_speaker + 1)
        ref_wav, _, sid = chosen[0]
        refs[sid] = ref_wav
        for idx, (tgt_wav, _, _) in enumerate(chosen[1:]):
            sel[(sid, idx)] = tgt_wav
    return sel, refs


def load_ecapa(device):
    from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/mnt/nfs-data/tin_dataset/hf_cache/ecapa_voxceleb",
        run_opts={"device": device},
    )


def embed(ecapa, wav_path, device):
    wav_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav_np).T  # [C, T]
    if sr != ECAPA_SR:
        wav = torchaudio.functional.resample(wav, sr, ECAPA_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(device)
    with torch.no_grad():
        emb = ecapa.encode_batch(wav).squeeze(1)
    return emb / emb.norm(dim=-1, keepdim=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", required=True,
                   help="name:dir pairs (dir contains sid{sid:03d}_gen{idx}.wav)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ecapa = load_ecapa(device)

    by_spk = load_heldout(HELDOUT_CSV)
    sel, refs = build_selection(by_spk)

    # Pre-embed refs (one per sid)
    ref_emb = {sid: embed(ecapa, path, device) for sid, path in refs.items()}
    # Upper bound: cos(ref, ground-truth target audio)
    ub_scores = []
    for (sid, idx), tgt_wav in sel.items():
        e_t = embed(ecapa, tgt_wav, device)
        ub_scores.append(float((ref_emb[sid] * e_t).sum()))
    ub_mean = float(np.mean(ub_scores))

    results = {}
    for spec in args.systems:
        name, gen_dir = spec.split(":", 1)
        gen_dir = Path(gen_dir)
        scores = []
        per_speaker = {}
        for (sid, idx), _ in sorted(sel.items()):
            wav = gen_dir / f"sid{sid:03d}_gen{idx}.wav"
            if not wav.exists():
                print(f"  MISS {wav}")
                continue
            e_g = embed(ecapa, wav, device)
            s = float((ref_emb[sid] * e_g).sum())
            scores.append(s)
            per_speaker.setdefault(sid, []).append(s)
            print(f"  {name} sid={sid:03d} gen{idx}: cos={s:.4f}")
        results[name] = {
            "n": len(scores),
            "cos_gen_ref_mean":   float(np.mean(scores)),
            "cos_gen_ref_median": float(np.median(scores)),
            "cos_gen_ref_std":    float(np.std(scores)),
            "cos_gen_ref_min":    float(np.min(scores)),
            "cos_gen_ref_max":    float(np.max(scores)),
            "per_speaker_mean": {str(k): float(np.mean(v)) for k, v in per_speaker.items()},
        }
    results["upper_bound_tgt_ref_mean"] = ub_mean

    out_json = out / "cossim_results.json"
    json.dump(results, open(out_json, "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("COSINE SIMILARITY (higher = better)")
    print("=" * 60)
    print(f"  upper_bound (ref vs target_gt): {ub_mean:.4f}")
    for name, r in results.items():
        if name == "upper_bound_tgt_ref_mean":
            continue
        print(f"  {name:20s} n={r['n']:3d}  mean={r['cos_gen_ref_mean']:.4f}  median={r['cos_gen_ref_median']:.4f}")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
