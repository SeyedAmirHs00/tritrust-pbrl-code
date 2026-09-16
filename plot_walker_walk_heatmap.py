#!/usr/bin/env python3
"""Heatmap of final performance: teacher β (rows) × max_feedback (cols).

Uses the **last eval point** per seed by default, then averages across seeds
for each Hydra config folder under ``<root>/<env>/``.

Example (matches the walker_walk zero-last SGD sweep)::

  python plot_heatmap.py \\
      --root exp_pebble_mixture_zero_last_wk_sgd \\
      --env walker_walk

Figures land in ``results/<root>/<env>/heatmap_<metric>.png`` (and ``.pdf`` / ``.csv``).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_experiments import (
    abs_under_repo,
    default_metric_for_env,
    find_csv_files,
    list_run_configs,
    load_seed_series,
    repo_root,
    resolve_env,
    teacher_betas_dirname,
    trailing_seed_scores,
)


def collect_final_table(
    root: str,
    env: str,
    *,
    metric: str,
    last_n: int = 1,
    seeds: Optional[Sequence[int]] = None,
    max_feedbacks: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """One row per (teacher_betas, max_feedback) with mean/std of trailing scores."""
    env_dir = os.path.join(root, env)
    configs = list_run_configs(env_dir, env)
    if max_feedbacks is not None:
        keep = {int(x) for x in max_feedbacks}
        configs = [c for c in configs if c.max_feedback in keep and not c.beta_agnostic]
    else:
        configs = [c for c in configs if not c.beta_agnostic]

    rows: List[dict] = []
    for cfg in configs:
        eval_files = find_csv_files(cfg.path, "eval", seeds=seeds)
        if not eval_files:
            print(f"  skip {cfg.label}: no eval.csv")
            continue
        try:
            _, Y = load_seed_series(eval_files, metric)
        except ValueError as exc:
            print(f"  skip {cfg.label}: {exc}")
            continue
        scores = trailing_seed_scores(Y, last_n)
        n_ok = int(np.sum(np.isfinite(scores)))
        if n_ok == 0:
            print(f"  skip {cfg.label}: no finite trailing scores")
            continue
        final_mu = float(np.nanmean(scores))
        final_sd = float(np.nanstd(scores, ddof=1)) if n_ok > 1 else 0.0
        beta_label = teacher_betas_dirname(cfg.teacher_betas)
        rows.append(
            {
                "teacher_betas": beta_label,
                "teacher_betas_tuple": cfg.teacher_betas,
                "max_feedback": cfg.max_feedback,
                "n_seeds": n_ok,
                "final_mean": final_mu,
                "final_std": final_sd,
                "last_n": last_n,
                "metric": metric,
            }
        )
        print(
            f"  [{beta_label:24s} fb={cfg.max_feedback:<5d}] "
            f"seeds={n_ok}  last-{last_n} mean={final_mu:.2f} ± {final_sd:.2f}"
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pivot_means(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = teacher β (lexicographic), columns = max_feedback (numeric ascending)."""
    ordered = df.sort_values(["teacher_betas_tuple", "max_feedback"])
    beta_order = list(dict.fromkeys(ordered["teacher_betas"].tolist()))
    fb_order = sorted(ordered["max_feedback"].unique().tolist())
    pivot = ordered.pivot_table(
        index="teacher_betas",
        columns="max_feedback",
        values="final_mean",
        aggfunc="mean",
    )
    return pivot.reindex(index=beta_order, columns=fb_order)


def plot_heatmap(
    pivot: pd.DataFrame,
    *,
    out_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> None:
    n_rows, n_cols = pivot.shape
    fig_w = max(6.0, 1.6 * n_cols + 2.5)
    fig_h = max(4.0, 0.55 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.6,
        linecolor="#cccccc",
        cbar_kws={"label": "value"},
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Max feedback")
    ax.set_ylabel("Teacher betas")
    if title:
        ax.set_title(title)
    # Keep categorical tick labels as plain ints / beta strings.
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=0)
    ax.set_yticklabels(list(pivot.index), rotation=0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teacher-β × max-feedback heatmap of final eval performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--root",
        type=str,
        default="exp_pebble_mixture_zero_last_wk_sgd",
        help="Experiment root containing <env>/max_feedbackN_... folders",
    )
    p.add_argument(
        "--env",
        type=str,
        default="walker_walk",
        help="Environment folder or alias (default: walker_walk)",
    )
    p.add_argument(
        "--metric",
        type=str,
        default=None,
        help="eval.csv column (default: env primary metric)",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=1,
        help="Trailing eval points averaged per seed (default: 1 = final eval only)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only include these seeds",
    )
    p.add_argument(
        "--max-feedback",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Only include these max_feedback values (default: all found)",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib / seaborn colormap (default: viridis)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: results/<root>/<env>/)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title (default: none)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = abs_under_repo(args.root)
    env = resolve_env(args.env)
    metric = args.metric or default_metric_for_env(env)

    if not os.path.isdir(os.path.join(root, env)):
        print(f"ERROR: env dir not found: {os.path.join(root, env)}")
        return 1

    out_dir = (
        abs_under_repo(args.out)
        if args.out
        else os.path.join(repo_root(), "results", os.path.basename(os.path.normpath(root)), env)
    )

    print("Heatmap")
    print(f"  root   : {root}")
    print(f"  env    : {env}")
    print(f"  metric : {metric}")
    print(f"  last_n : {args.last_n}")
    print(f"  out    : {out_dir}")

    table = collect_final_table(
        root,
        env,
        metric=metric,
        last_n=args.last_n,
        seeds=args.seeds,
        max_feedbacks=args.max_feedback,
    )
    if table.empty:
        print("No runs found to plot.")
        return 1

    pivot = pivot_means(table)
    print("\nPivot (final mean):")
    print(pivot.to_string(float_format=lambda v: f"{v:.2f}"))

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"heatmap_{metric}.csv")
    pivot.to_csv(csv_path, float_format="%.4f")
    print(f"Saved {csv_path}")

    long_csv = os.path.join(out_dir, f"heatmap_{metric}_long.csv")
    table.drop(columns=["teacher_betas_tuple"]).to_csv(
        long_csv, index=False, float_format="%.4f"
    )
    print(f"Saved {long_csv}")

    plot_heatmap(
        pivot,
        out_path=os.path.join(out_dir, f"heatmap_{metric}.png"),
        title=args.title,
        cmap=args.cmap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
