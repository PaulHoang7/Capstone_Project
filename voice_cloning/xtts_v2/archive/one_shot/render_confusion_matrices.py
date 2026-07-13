"""Render 6x6 Tone Confusion Matrix → PNG for each system in tone_eval_4way.

Output: <out_dir>/{system}_confmat.png — one PNG per system, for slide use.
Also produces a combined 2x2 grid figure.

Usage:
    python render_confusion_matrices.py \
        --tone-eval-json Capstone_project/voice_cloning/xtts_v2/tone_eval_4way/tone_eval.json \
        --out-dir Capstone_project/voice_cloning/xtts_v2/tone_eval_4way/figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

TONE_NAMES = ["ngang", "sắc", "huyền", "hỏi", "ngã", "nặng"]


def plot_cm(matrix, tone_names, title, ax=None, normalize=True):
    """Plot a 6x6 confusion matrix with diagonal highlighted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    M = np.array(matrix, dtype=float)
    if normalize:
        row_sums = M.sum(axis=1, keepdims=True)
        Mn = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
    else:
        Mn = M
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1 if normalize else None,
                   aspect="auto")
    ax.set_xticks(range(len(tone_names)))
    ax.set_yticks(range(len(tone_names)))
    ax.set_xticklabels(tone_names, rotation=45, ha="right")
    ax.set_yticklabels(tone_names)
    ax.set_xlabel("Predicted tone")
    ax.set_ylabel("True tone")
    ax.set_title(title)
    # Annotate cells
    for i in range(len(tone_names)):
        for j in range(len(tone_names)):
            val = Mn[i, j]
            raw = int(M[i, j])
            txt = f"{val:.2f}\n({raw})" if normalize else f"{raw}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=8)
    # Highlight hard pairs (hỏi↔ngã, sắc↔nặng)
    hard = [(3, 4), (4, 3), (1, 5), (5, 1)]
    for (r, c) in hard:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False,
                                    edgecolor="red", linewidth=1.5))
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tone-eval-json", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(args.tone_eval_json))

    # Per-system PNG
    summary_rows = []
    for sys_name, r in data.items():
        cm = r["confusion_6x6"]
        align = r.get("alignable_rate", 0.0) * 100
        acc = r.get("overall_tone_accuracy", 0.0) * 100
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        title = f"{sys_name}  —  align {align:.1f}%, tone acc {acc:.1f}%"
        plot_cm(cm, TONE_NAMES, title, ax=ax)
        fig.tight_layout()
        out_path = out_dir / f"{sys_name}_confmat.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")
        summary_rows.append((sys_name, align, acc))

    # Combined 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    items = list(data.items())[:4]
    for ax, (sys_name, r) in zip(axes.flat, items):
        cm = r["confusion_6x6"]
        align = r.get("alignable_rate", 0.0) * 100
        acc = r.get("overall_tone_accuracy", 0.0) * 100
        title = f"{sys_name}  (align {align:.1f}%, tone {acc:.1f}%)"
        plot_cm(cm, TONE_NAMES, title, ax=ax)
    fig.suptitle("6×6 Tone Confusion Matrices  (red box = hard pairs hỏi↔ngã, sắc↔nặng)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "all_systems_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {out_path}")

    print("\nSummary:")
    for s, a, c in summary_rows:
        print(f"  {s:14s}  align {a:5.1f}%  tone {c:5.1f}%")


if __name__ == "__main__":
    main()
