"""Render 3-way Tone Confusion Matrix grid for defense slides.

Handles systems with 0 alignable samples by replacing the CM with an
explanation panel (instead of an empty heatmap).
"""
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TONE_NAMES = ["ngang", "sắc", "huyền", "hỏi", "ngã", "nặng"]
HARD_PAIRS = [(3, 4), (4, 3), (1, 5), (5, 1)]


def plot_cm(matrix, ax, title, n_syl):
    M = np.array(matrix, dtype=float)
    row_sums = M.sum(axis=1, keepdims=True)
    Mn = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(TONE_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(TONE_NAMES, fontsize=10)
    ax.set_xlabel("Predicted tone (STT)", fontsize=10)
    ax.set_ylabel("True tone", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(6):
        for j in range(6):
            val = Mn[i, j]; raw = int(M[i, j])
            ax.text(j, i, f"{val:.2f}\n({raw})", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)
    for (r, c) in HARD_PAIRS:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=2))


def plot_unalignable_panel(ax, title, n_total):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.5, 0.62,
            "0 / {} samples alignable".format(n_total),
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="darkred", transform=ax.transAxes)
    ax.text(0.5, 0.42,
            "STT transcript ↔ target text\nsyllable count mismatch\n→ no tone alignment possible",
            ha="center", va="center", fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.18,
            "(See audio samples — speech is\nproduced but content drifts.\nF0 RMSE 46Hz: best pitch fidelity.)",
            ha="center", va="center", fontsize=9, style="italic",
            color="dimgray", transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor("darkred"); spine.set_linewidth(1.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tone-eval-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--order", default="vits2_d,xtts_ft,gwen",
                   help="Comma-separated system order for left→right grid")
    p.add_argument("--labels", default="VITS2 + Dual-Path,XTTS FT (VieNeu),Gwen-TTS (0.6B)",
                   help="Display labels for each system, same order")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(args.tone_eval_json))
    order  = args.order.split(",")
    labels = args.labels.split(",")
    assert len(order) == len(labels)

    # Per-system standalone PNGs
    for sys_name, display in zip(order, labels):
        r = data[sys_name]
        align = r.get("alignable_rate", 0.0) * 100
        acc   = r.get("overall_tone_accuracy", 0.0) * 100
        n_tot = r.get("n_total_samples", 60)
        n_syl = r.get("n_syllables_compared", 0)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        title = f"{display}\nalign {align:.1f}%  ·  tone acc {acc:.1f}%  ·  N={n_syl} syl"
        if n_syl == 0:
            plot_unalignable_panel(ax, title, n_tot)
        else:
            plot_cm(r["confusion_6x6"], ax, title, n_syl)
        fig.tight_layout()
        out_path = out_dir / f"{sys_name}_cm.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")

    # 1x3 grid for slides
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, sys_name, display in zip(axes, order, labels):
        r = data[sys_name]
        align = r.get("alignable_rate", 0.0) * 100
        acc   = r.get("overall_tone_accuracy", 0.0) * 100
        n_tot = r.get("n_total_samples", 60)
        n_syl = r.get("n_syllables_compared", 0)
        title = f"{display}\nalign {align:.1f}%  ·  tone {acc:.1f}%  ·  N={n_syl}"
        if n_syl == 0:
            plot_unalignable_panel(ax, title, n_tot)
        else:
            plot_cm(r["confusion_6x6"], ax, title, n_syl)
    fig.suptitle("6×6 Tone Confusion Matrix — 3-way comparison (red box = hard pairs hỏi↔ngã, sắc↔nặng)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_grid = out_dir / "grid_3way.png"
    fig.savefig(out_grid, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_grid}")


if __name__ == "__main__":
    main()
