"""Plot Partial/stochastic adversary bars from partial_adversary_shared.csv."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import savefig_png_pdf


DEFAULT_DATA_DIR = "exp_synthesis/synthetic_partial_adversary"
DEFAULT_PLOT_DIR = "results/synthetic_partial_adversary"


def plot_partial_adversary_figures(
    table: pd.DataFrame,
    plot_dir: str = DEFAULT_PLOT_DIR,
    settings_order: list[str] | None = None,
) -> None:
    if settings_order is None:
        settings_order = list(dict.fromkeys(table["setting"].tolist()))
    methods = [("stabilized", {}), ("standard", {})]
    x = np.arange(len(settings_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for mi, (mname, _) in enumerate(methods):
        vals = [
            table[(table.setting == s) & (table.method == mname)].iloc[0].correct
            for s in settings_order
        ]
        ax.bar(x + (mi - 0.5) * width, vals, width, label=mname)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_order, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correct-branch rate")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    savefig_png_pdf(
        fig,
        os.path.join(plot_dir, "partial_adversary_correct_branch.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for mi, (mname, _) in enumerate(methods):
        aA = [
            table[(table.setting == s) & (table.method == mname)].iloc[0].abar_A
            for s in settings_order
        ]
        aR = [
            table[(table.setting == s) & (table.method == mname)].iloc[0].abar_R
            for s in settings_order
        ]
        ax.bar(x + (mi - 0.5) * width, aA, width, label=f"{mname} A")
        ax.scatter(x + (mi - 0.5) * width, aR, marker="D", color="black", zorder=3, s=20)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(settings_order, rotation=20, ha="right")
    ax.set_ylabel(r"mean $\bar\alpha$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_png_pdf(
        fig,
        os.path.join(plot_dir, "partial_adversary_recovered_trust.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot partial adversary figures")
    p.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing partial_adversary_shared.csv",
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
    table = pd.read_csv(os.path.join(args.data_dir, "partial_adversary_shared.csv"))
    settings_order = list(dict.fromkeys(table["setting"].tolist()))
    plot_partial_adversary_figures(table, plot_dir=args.plot_dir, settings_order=settings_order)
    print(f"OUT partial adversary plots: {args.plot_dir}")


if __name__ == "__main__":
    main()
