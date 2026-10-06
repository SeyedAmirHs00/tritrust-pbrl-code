"""Overlap counterfactual with DS-Sym (init q=1, s=1, theta=0) vs TTP / No-alpha.

Defaults to setting: betas = [1.0, 1.0, 0.0, 0.0, -1.0] (2R2N1A).

Example:
  python overlap_ds_init_sweep.py --seeds 120 --overwrite
  python overlap_ds_init_sweep.py --betas 1 1 0 0 -1 --seeds 120 --overwrite
  python overlap_ds_init_sweep.py --replot
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from overlap_ds_init_sweep_data import DEFAULT_BETAS, run_overlap_data
from overlap_ds_init_sweep_plot import plot_overlap_figure
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser(
        description="Overlap counterfactual (TTP / No-alpha / DS-Sym init q=1, s=1, theta=0)"
    )
    p.add_argument("--out_dir", default="results/synthetic_overlap_ds_init_sweep")
    p.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=list(DEFAULT_BETAS),
        help="Teacher rationality mixture (default: 1.0 1.0 0.0 0.0 -1.0 for 2R2N1A).",
    )
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replot", action="store_true", help="Only regenerate figures from CSV")
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    p.add_argument("--q-init", type=float, default=1.0, help="Initial DS-Sym expert reliability q (default: 1.0).")
    p.add_argument("--s-init", type=float, default=1.0, help="Initial DS-Sym scale parameter s (default: 1.0).")
    p.add_argument("--lr-ds", type=float, default=0.05, help="Learning rate for DS-Sym (default: 0.05).")
    args = p.parse_args()

    csv_path = os.path.join(args.out_dir, "overlap_shared.csv")
    if not os.path.exists(csv_path):
        alt_path = os.path.join(args.out_dir, "overlap_ds_init_shared.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path

    if args.replot:
        plot_overlap_figure(
            pd.read_csv(csv_path),
            args.out_dir,
        )
        print(f"replot OK: {args.out_dir}")
        return

    run_overlap_data(
        args.out_dir,
        args.seeds,
        args.steps,
        args.overwrite,
        betas=tuple(args.betas),
        coef_max_delta=args.coef_max_delta,
        q_init=args.q_init,
        s_init=args.s_init,
        lr_ds=args.lr_ds,
    )
    plot_overlap_figure(
        pd.read_csv(csv_path),
        args.out_dir,
    )
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
