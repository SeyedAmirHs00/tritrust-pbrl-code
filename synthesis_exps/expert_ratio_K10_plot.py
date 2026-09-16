"""Plot K=10 expert-ratio heatmaps from expert_ratio_shared.csv."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import savefig_png_pdf
from synthetic_shared_core import SHARED_BRANCH_VARIANTS


def _annotate(ax, g: np.ndarray, k: int, signed: bool = False) -> None:
    for ni in range(k + 1):
        for na in range(k + 1):
            val = g[ni, na]
            if not np.isfinite(val):
                continue
            if signed:
                txt = f"{val:+.2f}"
            else:
                txt = f"{val:.2f}"
            ax.text(
                na,
                ni,
                txt,
                ha="center",
                va="center",
                fontsize=5.2,
                color="black",
                zorder=3,
            )


def _panel(
    ax,
    g: np.ndarray,
    k: int,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    xlabel: bool,
    ylabel: bool,
    signed: bool,
):
    im = ax.imshow(
        np.ma.masked_invalid(g),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[-0.5, k + 0.5, -0.5, k + 0.5],
        aspect="equal",
    )
    ax.plot([0, k / 2], [k, 0], color="black", lw=1.15, ls="--", zorder=2)
    _annotate(ax, g, k, signed=signed)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlim(-0.5, k + 0.5)
    ax.set_ylim(-0.5, k + 0.5)
    ax.set_xticks(range(0, k + 1, 2))
    ax.set_yticks(range(0, k + 1, 2))
    if xlabel:
        ax.set_xlabel("# adversarial experts", fontsize=9)
    if ylabel:
        ax.set_ylabel("# noisy experts", fontsize=9)
    return im


def _row_spec() -> List[Tuple[str, str, str, float, float, bool, str]]:
    return [
        ("correct", r"(a) Correct-branch rate", "correct-branch rate", 0.0, 1.0, False, "RdYlGn"),
        ("adv", r"(b) Mean $\bar\alpha_A$", r"mean $\bar\alpha_A$", -1.0, 1.0, True, "RdBu"),
        ("noisy", r"(c) Mean $\bar\alpha_N$", r"mean $\bar\alpha_N$", -1.0, 1.0, True, "RdBu"),
        ("rel", r"(d) Mean $\bar\alpha_R$", r"mean $\bar\alpha_R$", -1.0, 1.0, True, "RdBu"),
    ]


def plot_heatmaps(
    out_dir: str,
    k: int,
    grids: Dict[str, Dict[str, np.ndarray]],
) -> None:
    from matplotlib.gridspec import GridSpec

    col_labels = {
        "standard": "(i) Standard",
        "stabilized": "(ii) Stabilized",
    }
    names = [v.name for v in SHARED_BRANCH_VARIANTS]
    n_cols = len(names)
    rows = _row_spec()
    n_rows = len(rows)
    width_ratios = [1.0] * n_cols + [0.045]

    fig = plt.figure(figsize=(4.2 * n_cols + 0.8, 3.7 * n_rows + 0.4))
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.22,
        hspace=0.28,
        left=0.06,
        right=0.94,
        top=0.97,
        bottom=0.04,
    )
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(n_rows)])

    for i, (key, _row_caption, cbar_label, vmin, vmax, signed, cmap) in enumerate(rows):
        im = None
        for j, name in enumerate(names):
            im = _panel(
                axes[i, j],
                grids[key][name],
                k,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=col_labels.get(name, name) if i == 0 else "",
                xlabel=(i == n_rows - 1),
                ylabel=(j == 0),
                signed=signed,
            )
        cax = fig.add_subplot(gs[i, n_cols])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=9)

    combined = os.path.join(out_dir, "expert_ratio_sweep.png")
    savefig_png_pdf(fig, combined, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    def row_fig(
        grid_key: str,
        fname: str,
        cmap: str,
        vmin: float,
        vmax: float,
        cbar_label: str,
        signed: bool,
    ):
        f = plt.figure(figsize=(4.2 * n_cols + 0.8, 3.9))
        gs1 = GridSpec(
            1,
            n_cols + 1,
            figure=f,
            width_ratios=width_ratios,
            wspace=0.22,
            left=0.06,
            right=0.94,
            top=0.88,
            bottom=0.18,
        )
        axs = [f.add_subplot(gs1[0, j]) for j in range(n_cols)]
        cax = f.add_subplot(gs1[0, n_cols])
        im = None
        for j, name in enumerate(names):
            im = _panel(
                axs[j],
                grids[grid_key][name],
                k,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=col_labels.get(name, name),
                xlabel=True,
                ylabel=(j == 0),
                signed=signed,
            )
        cb = f.colorbar(im, cax=cax)
        cb.set_label(cbar_label, fontsize=9)
        savefig_png_pdf(
            f, os.path.join(out_dir, fname), dpi=220, bbox_inches="tight", facecolor="white"
        )
        plt.close(f)

    for key, _row_caption, cbar_label, vmin, vmax, signed, cmap in rows:
        fname = {
            "correct": "expert_ratio_correct_branch.png",
            "adv": "expert_ratio_adv_trust.png",
            "noisy": "expert_ratio_noisy_trust.png",
            "rel": "expert_ratio_rel_trust.png",
        }[key]
        row_fig(key, fname, cmap, vmin, vmax, cbar_label, signed)


def _nan_grid(k: int) -> Dict[str, np.ndarray]:
    return {v.name: np.full((k + 1, k + 1), np.nan) for v in SHARED_BRANCH_VARIANTS}


def grids_from_table(table: pd.DataFrame, k: int) -> Dict[str, Dict[str, np.ndarray]]:
    grids = {
        "correct": _nan_grid(k),
        "adv": _nan_grid(k),
        "noisy": _nan_grid(k),
        "rel": _nan_grid(k),
    }
    for _, r in table.iterrows():
        ni, na = int(r["n_N"]), int(r["n_A"])
        name = r["variant"]
        grids["correct"][name][ni, na] = float(r["correct_branch_rate"])
        grids["adv"][name][ni, na] = float(r["mean_adv_trust"])
        grids["noisy"][name][ni, na] = float(r["mean_noisy_trust"])
        grids["rel"][name][ni, na] = float(r["mean_rel_trust"])
    return grids


def replot_from_csv(out_dir: str) -> None:
    table = pd.read_csv(os.path.join(out_dir, "expert_ratio_shared.csv"))
    required = {"mean_noisy_trust", "mean_rel_trust"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"CSV missing {sorted(missing)}; re-run data script to recompute trusts."
        )
    k = int((table["n_R"] + table["n_N"] + table["n_A"]).iloc[0])
    grids = grids_from_table(table, k)
    plot_heatmaps(out_dir, k, grids)
    print(f"replot OK: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot expert-ratio K=10 heatmaps")
    p.add_argument("--out_dir", default="results/synthetic_expert_ratio_sweep_K10")
    args = p.parse_args()
    replot_from_csv(args.out_dir)


if __name__ == "__main__":
    main()
