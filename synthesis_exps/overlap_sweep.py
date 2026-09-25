"""
Overlap counterfactual (TTP / No-alpha / DS-Sym).
Label: fig:synthetic-overlap-sweep

Example:
  python overlap_sweep.py --seeds 120 --overwrite
  python overlap_sweep.py --replot
"""


from __future__ import annotations

import argparse
import os

import pandas as pd

from overlap_sweep_data import run_overlap_data
from overlap_sweep_plot import plot_overlap_figure
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="exp_synthesis/synthetic_overlap_sweep", help="Directory for CSV data")
    p.add_argument("--plot_dir", default="result/synthetic_overlap_sweep", help="Directory for plots")
    p.add_argument("--out_dir", dest="data_dir", help="Alias for --data_dir")
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
    args = p.parse_args()

    if args.replot:
        plot_overlap_figure(
            pd.read_csv(os.path.join(args.data_dir, "overlap_shared.csv")),
            plot_dir=args.plot_dir,
        )
        print(f"replot OK: {args.plot_dir}")
        return

    run_overlap_data(
        args.data_dir,
        args.seeds,
        args.steps,
        args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )
    plot_overlap_figure(
        pd.read_csv(os.path.join(args.data_dir, "overlap_shared.csv")),
        plot_dir=args.plot_dir,
    )
    print(f"OUT: data={args.data_dir}, plot={args.plot_dir}")


if __name__ == "__main__":
    main()
