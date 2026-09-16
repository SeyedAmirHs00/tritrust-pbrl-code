"""Plot T sweep (lr scaled) from T_sweep_lr_scaled.csv."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import savefig_png_pdf
from synthetic_shared_core import SHARED_BRANCH_VARIANTS


def plot_T_figure(df: pd.DataFrame, out_root: str, Ts) -> None:
    colors = {"standard": "#4c72b0", "stabilized": "#dd8452"}
    markers = {"standard": "o", "stabilized": "s"}
    order = ["3R1N", "3R1A", "1R3A"]
    letters = ["a", "b", "c"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, cfg in zip(axes, order):
        for v in SHARED_BRANCH_VARIANTS:
            sub = df[(df.config == cfg) & (df.variant == v.name)].sort_values("T")
            ax.plot(
                sub["T"],
                100.0 * sub["correct_branch_rate"],
                color=colors[v.name],
                marker=markers[v.name],
                lw=2.0,
                ms=7,
                label=v.label,
            )
        ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.set_title(cfg, fontsize=11)
        ax.set_xlabel(r"Trajectory length $T$")
        ax.set_xscale("log")
        ax.set_xticks(list(Ts))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.grid(True, ls=":", alpha=0.45)
        ax.set_ylim(-2, 105)
    axes[0].set_ylabel("Correct-branch rate (%)")
    axes[-1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    savefig_png_pdf(
        fig, os.path.join(out_root, "T_sweep_lr_scaled.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    for cfg, letter in zip(order, letters):
        fig, ax = plt.subplots(figsize=(3.9, 3.2))
        for v in SHARED_BRANCH_VARIANTS:
            sub = df[(df.config == cfg) & (df.variant == v.name)].sort_values("T")
            ax.plot(
                sub["T"],
                100.0 * sub["correct_branch_rate"],
                color=colors[v.name],
                marker=markers[v.name],
                lw=2.0,
                ms=7,
                label=v.label,
            )
        ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel(r"Trajectory length $T$")
        ax.set_ylabel("Correct-branch rate (%)")
        ax.set_xscale("log")
        ax.set_xticks(list(Ts))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.grid(True, ls=":", alpha=0.45)
        ax.set_ylim(-2, 105)
        if letter == "c":
            ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        savefig_png_pdf(
            fig,
            os.path.join(out_root, f"T_sweep_lr_scaled_{letter}_{cfg}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot T sweep lr-scaled figures")
    p.add_argument("--out_root", default="results/synthetic_T_sweep_lr_scaled")
    args = p.parse_args()
    df = pd.read_csv(os.path.join(args.out_root, "T_sweep_lr_scaled.csv"))
    Ts = sorted(df["T"].unique().tolist())
    plot_T_figure(df, args.out_root, Ts)
    print(f"OUT T sweep plots: {args.out_root}")


if __name__ == "__main__":
    main()
