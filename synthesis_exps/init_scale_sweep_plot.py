"""Plot init-scale sweep from init_scale_sweep.csv."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import savefig_png_pdf


DEFAULT_DATA_DIR = "exp_synthesis/synthetic_init_scale_sweep"
DEFAULT_PLOT_DIR = "results/synthetic_init_scale_sweep"


def plot_init_scale_figure(table: pd.DataFrame, plot_dir: str = DEFAULT_PLOT_DIR) -> None:
    order = ["3R1A", "2R1A1N", "3R1N", "1R3A"]
    letters = ["a", "b", "c", "d"]
    methods = [
        ("no_cons", 0.0),
    ]
    legend_names = {"no_cons": "no cons."}
    colors = {"no_cons": "#4c72b0"}

    fig, axes = plt.subplots(1, len(order), figsize=(4.0 * len(order), 3.2), sharey=True)
    for ax, cfg, letter in zip(axes, order, letters):
        for mname, _ in methods:
            sub = table[(table["config"] == cfg) & (table["method"] == mname)].sort_values(
                "init_rms_deltaR"
            )
            x = np.maximum(sub["init_rms_deltaR"].to_numpy(), 1e-4)
            ax.plot(
                x,
                sub["correct_branch_rate"],
                "o-",
                color=colors[mname],
                label=legend_names[mname],
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"Init $\mathrm{rms}|\Delta R|$")
        ax.set_title(cfg, fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.45)
        if ax is axes[0]:
            ax.set_ylabel("Correct-branch rate")
        if ax is axes[-1]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_png_pdf(
        fig,
        os.path.join(plot_dir, "init_scale_sweep.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    for cfg, letter in zip(order, letters):
        fig, ax = plt.subplots(figsize=(3.9, 3.2))
        for mname, _ in methods:
            sub = table[(table["config"] == cfg) & (table["method"] == mname)].sort_values(
                "init_rms_deltaR"
            )
            x = np.maximum(sub["init_rms_deltaR"].to_numpy(), 1e-4)
            ax.plot(
                x,
                sub["correct_branch_rate"],
                "o-",
                color=colors[mname],
                label=legend_names[mname],
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"Init $\mathrm{rms}|\Delta R|$")
        ax.set_ylabel("Correct-branch rate")
        ax.grid(True, which="both", ls=":", alpha=0.45)
        if letter == "d":
            ax.legend(fontsize=8)
        fig.tight_layout()
        savefig_png_pdf(
            fig,
            os.path.join(plot_dir, f"init_scale_sweep_{letter}_{cfg}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot init-scale sweep")
    p.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing init_scale_sweep.csv",
    )
    p.add_argument(
        "--plot_dir",
        "--plot-dir",
        dest="plot_dir",
        default=DEFAULT_PLOT_DIR,
        help="Directory to save plots",
    )
    p.add_argument("--out_dir", dest="data_dir", help="Alias for --data_dir")
    args = p.parse_args()
    table = pd.read_csv(os.path.join(args.data_dir, "init_scale_sweep.csv"))
    plot_init_scale_figure(table, plot_dir=args.plot_dir)
    print(f"OUT init-scale plots: {args.plot_dir}")


if __name__ == "__main__":
    main()
