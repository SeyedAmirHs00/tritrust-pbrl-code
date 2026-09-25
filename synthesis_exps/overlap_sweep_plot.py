"""Plot Overlap counterfactual from overlap_shared.csv."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import savefig_png_pdf


DEFAULT_DATA_DIR = "exp_synthesis/synthetic_overlap_sweep"
DEFAULT_PLOT_DIR = "result/synthetic_overlap_sweep"


def plot_overlap_figure(
    table: pd.DataFrame,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> None:
    colors = {"ttp": "#4c72b0", "no_alpha": "#dd8452", "ds_sym": "#55a868"}
    labels = {"ttp": "TTP", "no_alpha": r"No-$\alpha$", "ds_sym": "DS-Sym"}
    methods = ("ttp", "no_alpha", "ds_sym")

    qs = sorted(table.q.unique())
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    if len(qs) == 1:
        x = np.arange(len(methods))
        for mi, method in enumerate(methods):
            sub = table[table.method == method].iloc[0]
            ax.bar(
                mi,
                sub.signed_med,
                color=colors[method],
                yerr=[[sub.signed_med - sub.signed_q25], [sub.signed_q75 - sub.signed_med]],
                capsize=4,
                label=labels[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_xlabel(rf"method ($q={qs[0]:g}$, $n=500$)")
    else:
        for method in methods:
            sub = table[table.method == method].sort_values("q")
            ax.fill_between(
                sub.q, sub.signed_q25, sub.signed_q75, color=colors[method], alpha=0.15
            )
            ax.plot(sub.q, sub.signed_med, "o-", color=colors[method], label=labels[method])
        ax.set_xlabel("shared-pair fraction $q$")
        ax.legend(fontsize=8, loc="lower right")
    ax.axhline(0.0, color="gray", ls=":", lw=0.9)
    ax.set_ylabel(r"median signed $\mathrm{corr}(\hat R,R^*)$")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    savefig_png_pdf(
        fig,
        os.path.join(plot_dir, "3R1A_overlap_global_vs_local.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot overlap sweep figure")
    p.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing overlap_shared.csv",
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
    table = pd.read_csv(os.path.join(args.data_dir, "overlap_shared.csv"))
    plot_overlap_figure(table, plot_dir=args.plot_dir)
    print(f"OUT overlap plot: {args.plot_dir}")


if __name__ == "__main__":
    main()
