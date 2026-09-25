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


DEFAULT_DATA_DIR = "exp_synthesis/synthetic_T_sweep_lr_scaled"
DEFAULT_PLOT_DIR = "result/synthetic_T_sweep_lr_scaled"


def plot_T_figure(df: pd.DataFrame, plot_dir: str = DEFAULT_PLOT_DIR, Ts=None) -> None:
    if Ts is None:
        Ts = sorted(df["T"].unique().tolist())
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
        fig,
        os.path.join(plot_dir, "T_sweep_lr_scaled.png"),
        dpi=200,
        bbox_inches="tight",
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
            os.path.join(plot_dir, f"T_sweep_lr_scaled_{letter}_{cfg}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot T sweep lr-scaled figures")
    p.add_argument(
        "--data_dir",
        "--out_root",
        dest="data_dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing T_sweep_lr_scaled.csv",
    )
    p.add_argument(
        "--plot_dir",
        "--plot_root",
        dest="plot_dir",
        default=DEFAULT_PLOT_DIR,
        help="Directory to save plots",
    )
    args = p.parse_args()
    df = pd.read_csv(os.path.join(args.data_dir, "T_sweep_lr_scaled.csv"))
    Ts = sorted(df["T"].unique().tolist())
    plot_T_figure(df, plot_dir=args.plot_dir, Ts=Ts)
    print(f"OUT T sweep plots: {args.plot_dir}")


if __name__ == "__main__":
    main()
