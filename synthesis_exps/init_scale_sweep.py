"""
Init-scale (rms|ΔR|) sweep with no consensus.
Label: fig:synthetic-sigma-consensus

Produces: results/synthetic_init_scale_sweep/

Example:
  python init_scale_sweep.py --seeds 200 --overwrite
  python init_scale_sweep.py --replot
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from init_scale_sweep_data import run_init_scale_data
from init_scale_sweep_plot import plot_init_scale_figure
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser(
        description="Init-scale sweep, no consensus (fig:synthetic-sigma-consensus)"
    )
    p.add_argument("--data_dir", default="exp_synthesis/synthetic_init_scale_sweep", help="Directory for CSV data")
    p.add_argument("--plot_dir", default="result/synthetic_init_scale_sweep", help="Directory for plots")
    p.add_argument("--out_dir", dest="data_dir", help="Alias for --data_dir")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replot", action="store_true")
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    args = p.parse_args()

    if args.replot:
        table = pd.read_csv(os.path.join(args.data_dir, "init_scale_sweep.csv"))
        plot_init_scale_figure(table, plot_dir=args.plot_dir)
        print(f"replot OK: {args.plot_dir}")
        return

    out = run_init_scale_data(
        args.data_dir, args.seeds, args.steps, args.overwrite, coef_max_delta=args.coef_max_delta
    )
    table = pd.read_csv(os.path.join(out, "init_scale_sweep.csv"))
    plot_init_scale_figure(table, plot_dir=args.plot_dir)
    print(f"OUT: data={args.data_dir}, plot={args.plot_dir}")


if __name__ == "__main__":
    main()
