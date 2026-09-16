"""Plot partial-adversary α learning curves from train CSVs.

Aggregates over seeds: mean curve with a shaded variability band
(``std`` / ``sem`` / ``var``). Optionally also writes per-seed figures.

Examples
--------
  python partial_adversary_alpha_curve_data.py --out_dir results/foo --overwrite
  python partial_adversary_alpha_curve_plot.py --run_dir results/foo

  python partial_adversary_alpha_curve_plot.py --run_dir results/foo --ci std
  python partial_adversary_alpha_curve_plot.py --run_dir results/foo --ci sem --no-per-seed

  python partial_adversary_alpha_curve_plot.py \
      --csv path/to/alpha_learning_curve_per_step.csv \
      --out_dir path/to/figures
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import savefig_png_pdf

HIST_CSV_NAME = "alpha_learning_curve_per_step.csv"
EXPERT_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
EXPERT_LABELS = {
    0: r"$\alpha_0$ (R)",
    1: r"$\alpha_1$ (R)",
    2: r"$\alpha_2$ (R)",
    3: r"$\alpha_3$ (A)",
}


def aggregate_over_seeds(
    esub: pd.DataFrame,
    value_col: str,
    ci: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(steps, mean, band)`` aggregated over ``seed_idx``."""
    g = esub.groupby("step")[value_col]
    mean = g.mean()
    n = g.count().clip(lower=1)
    # Population/sample std across seeds; single seed → zero band.
    std = g.std(ddof=1).fillna(0.0)
    var = g.var(ddof=1).fillna(0.0)

    if ci == "none":
        band = pd.Series(0.0, index=mean.index)
    elif ci == "std":
        band = std
    elif ci == "sem":
        band = std / np.sqrt(n.to_numpy(dtype=float))
    elif ci == "var":
        band = var
    else:
        raise ValueError(f"Unknown ci={ci!r}; use std|sem|var|none")

    steps = mean.index.to_numpy(dtype=float)
    return steps, mean.to_numpy(dtype=float), band.to_numpy(dtype=float)


def _ci_legend_suffix(ci: str) -> str:
    return {
        "std": "mean ± std",
        "sem": "mean ± SEM",
        "var": "mean ± var",
        "none": "mean",
    }.get(ci, f"mean ± {ci}")


def plot_curves(
    hist: pd.DataFrame,
    out_path: str,
    value_col: str,
    ylabel: str,
    *,
    ci: str = "std",
) -> None:
    """One panel per (setting, method); expert mean ± seed band vs step."""
    settings = list(hist["setting"].unique())
    methods = list(hist["method"].unique())
    n_seeds = int(hist["seed_idx"].nunique())
    n_rows, n_cols = len(methods), len(settings)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 2.9 * n_rows), sharex=True, sharey=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    for mi, mname in enumerate(methods):
        for si, sname in enumerate(settings):
            ax = axes[mi, si]
            sub = hist[(hist.setting == sname) & (hist.method == mname)]
            for e in range(4):
                esub = sub[sub.expert == e]
                if esub.empty:
                    continue
                steps, mean, band = aggregate_over_seeds(esub, value_col, ci)
                color = EXPERT_COLORS[e]
                ax.plot(steps, mean, color=color, lw=1.6, label=EXPERT_LABELS[e])
                if n_seeds > 1 and ci != "none" and np.any(band > 0):
                    ax.fill_between(
                        steps,
                        mean - band,
                        mean + band,
                        color=color,
                        alpha=0.22,
                        linewidth=0,
                    )
            if mi == 0:
                ax.set_title(sname, fontsize=9)
            if si == 0:
                ax.set_ylabel(f"{mname}\n{ylabel}", fontsize=9)
            if mi == n_rows - 1:
                ax.set_xlabel("step")
            ax.axhline(0, color="gray", ls=":", lw=0.8)
            ax.grid(True, ls=":", alpha=0.35)
            if mi == 0 and si == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Seed-aggregated α curves ({_ci_legend_suffix(ci)}, n={n_seeds})",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    savefig_png_pdf(fig, out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_per_seed(
    hist: pd.DataFrame,
    out_dir: str,
    value_col: str,
    tag: str,
) -> None:
    """Separate figure per (setting, method, seed_idx)."""
    os.makedirs(out_dir, exist_ok=True)
    for (sname, mname, sid), sub in hist.groupby(["setting", "method", "seed_idx"]):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        for e in range(4):
            esub = sub[sub.expert == e].sort_values("step")
            role = "A" if e == 3 else "R"
            ax.plot(
                esub["step"],
                esub[value_col],
                color=EXPERT_COLORS[e],
                lw=1.6,
                label=f"expert {e} ({role})",
            )
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel(tag)
        ax.set_title(f"{sname} / {mname} / seed_idx={sid}")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)
        fig.tight_layout()
        fname = f"alpha_curve_{tag}_{sname}_{mname}_seed{sid}.png".replace("/", "_")
        out_path = os.path.join(out_dir, fname)
        savefig_png_pdf(fig, out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_per_experiment(
    hist: pd.DataFrame,
    out_dir: str,
    *,
    ci: str = "std",
) -> None:
    """Generate separate figures per (setting, method) experiment."""
    os.makedirs(out_dir, exist_ok=True)
    for (sname, mname), sub in hist.groupby(["setting", "method"], sort=False):
        n_seeds = int(sub["seed_idx"].nunique())

        # 1. Combined 2-panel figure: [tilde_alpha, abar_alpha]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=True)
        for e in range(4):
            esub = sub[sub.expert == e]
            if esub.empty:
                continue
            color = EXPERT_COLORS[e]
            label = EXPERT_LABELS[e]

            # Left panel: tilde_alpha
            steps1, mean1, band1 = aggregate_over_seeds(esub, "tilde_alpha", ci)
            ax1.plot(steps1, mean1, color=color, lw=1.8, label=label)
            if n_seeds > 1 and ci != "none" and np.any(band1 > 0):
                ax1.fill_between(
                    steps1,
                    mean1 - band1,
                    mean1 + band1,
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                )

            # Right panel: abar_alpha
            steps2, mean2, band2 = aggregate_over_seeds(esub, "abar_alpha", ci)
            ax2.plot(steps2, mean2, color=color, lw=1.8, label=label)
            if n_seeds > 1 and ci != "none" and np.any(band2 > 0):
                ax2.fill_between(
                    steps2,
                    mean2 - band2,
                    mean2 + band2,
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                )

        ax1.axhline(0, color="gray", ls=":", lw=0.8)
        ax1.grid(True, ls=":", alpha=0.35)
        ax1.set_xlabel("step")
        ax1.set_ylabel(r"$\tilde\alpha=\tanh(\alpha)$")
        ax1.set_title(r"Bounded Trust $\tilde\alpha$", fontsize=10)
        ax1.legend(fontsize=8, loc="best")

        ax2.axhline(0, color="gray", ls=":", lw=0.8)
        ax2.grid(True, ls=":", alpha=0.35)
        ax2.set_xlabel("step")
        ax2.set_ylabel(r"$\bar\alpha$")
        ax2.set_title(r"Max-Normalized Trust $\bar\alpha$", fontsize=10)
        ax2.legend(fontsize=8, loc="best")

        fig.suptitle(
            f"{sname} ({mname}) — α curves ({_ci_legend_suffix(ci)}, n={n_seeds})",
            fontsize=11,
        )
        fig.tight_layout()
        fname_combo = f"alpha_curve_{sname}_{mname}.png".replace("/", "_")
        out_path_combo = os.path.join(out_dir, fname_combo)
        savefig_png_pdf(fig, out_path_combo, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path_combo}")

        # 2. Individual single-panel figures for focused views
        for col, ylabel, tag in [
            ("tilde_alpha", r"$\tilde\alpha=\tanh(\alpha)$", "tilde"),
            ("abar_alpha", r"$\bar\alpha$", "abar"),
        ]:
            fig_s, ax_s = plt.subplots(figsize=(5.6, 3.6))
            for e in range(4):
                esub = sub[sub.expert == e]
                if esub.empty:
                    continue
                color = EXPERT_COLORS[e]
                label = EXPERT_LABELS[e]
                steps, mean, band = aggregate_over_seeds(esub, col, ci)
                ax_s.plot(steps, mean, color=color, lw=1.8, label=label)
                if n_seeds > 1 and ci != "none" and np.any(band > 0):
                    ax_s.fill_between(
                        steps,
                        mean - band,
                        mean + band,
                        color=color,
                        alpha=0.22,
                        linewidth=0,
                    )

            ax_s.axhline(0, color="gray", ls=":", lw=0.8)
            ax_s.grid(True, ls=":", alpha=0.35)
            ax_s.set_xlabel("step")
            ax_s.set_ylabel(ylabel)
            ax_s.set_title(
                f"{sname} ({mname}) — {ylabel} ({_ci_legend_suffix(ci)}, n={n_seeds})",
                fontsize=10,
            )
            ax_s.legend(fontsize=8, loc="best")
            fig_s.tight_layout()
            fname_s = f"alpha_curve_{tag}_{sname}_{mname}.png".replace("/", "_")
            out_path_s = os.path.join(out_dir, fname_s)
            savefig_png_pdf(fig_s, out_path_s, dpi=200, bbox_inches="tight")
            plt.close(fig_s)
            print(f"Saved {out_path_s}")


def write_alpha_curve_figures(
    hist: pd.DataFrame,
    out_dir: str,
    *,
    ci: str = "std",
    per_experiment: bool = True,
    per_seed: bool = False,
) -> None:
    """Write aggregated mean±band grids (+ optional per-experiment, per-seed) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    plot_curves(
        hist,
        os.path.join(out_dir, "alpha_curve_tilde_grid.png"),
        "tilde_alpha",
        r"$\tilde\alpha=\tanh(\alpha)$",
        ci=ci,
    )
    plot_curves(
        hist,
        os.path.join(out_dir, "alpha_curve_abar_grid.png"),
        "abar_alpha",
        r"$\bar\alpha$",
        ci=ci,
    )
    if per_experiment:
        plot_per_experiment(hist, out_dir, ci=ci)
    if per_seed:
        plot_per_seed(hist, out_dir, "tilde_alpha", "tilde")
        plot_per_seed(hist, out_dir, "abar_alpha", "abar")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=f"Directory containing {HIST_CSV_NAME} (also used as --out_dir if unset).",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to alpha_learning_curve_per_step.csv (overrides --run_dir).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to write figures (default: --run_dir or CSV parent).",
    )
    p.add_argument(
        "--ci",
        choices=["std", "sem", "var", "none"],
        default="std",
        help="Seed-aggregation band: mean±std (default), ±SEM, ±variance, or mean only.",
    )
    p.add_argument(
        "--per-experiment",
        action="store_true",
        default=True,
        help="Write separate figures per (setting, method) experiment (default: True).",
    )
    p.add_argument(
        "--no-per-experiment",
        dest="per_experiment",
        action="store_false",
        help="Do not write figures per experiment.",
    )
    p.add_argument(
        "--per-seed",
        action="store_true",
        help="Also write one figure per (setting, method, seed). Off by default.",
    )
    p.add_argument(
        "--no-per-seed",
        action="store_true",
        help=argparse.SUPPRESS,  # kept for back-compat; default already skips per-seed
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.csv:
        csv_path = args.csv
    elif args.run_dir:
        csv_path = os.path.join(args.run_dir, HIST_CSV_NAME)
    else:
        raise SystemExit("Provide --run_dir or --csv")

    if not os.path.isfile(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.out_dir:
        out_dir = args.out_dir
    elif args.run_dir:
        out_dir = args.run_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(csv_path)) or "."

    hist = pd.read_csv(csv_path)
    required = {
        "setting",
        "method",
        "seed_idx",
        "expert",
        "step",
        "tilde_alpha",
        "abar_alpha",
    }
    missing = required - set(hist.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    n_seeds = int(hist["seed_idx"].nunique())
    print(
        f"Plotting from {csv_path} → {out_dir}  "
        f"(n_seeds={n_seeds}, ci={args.ci}, per_experiment={args.per_experiment}, per_seed={args.per_seed})"
    )
    write_alpha_curve_figures(
        hist,
        out_dir,
        ci=args.ci,
        per_experiment=args.per_experiment,
        per_seed=args.per_seed and not args.no_per_seed,
    )
    print(f"OUT: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
