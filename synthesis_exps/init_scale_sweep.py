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
    p.add_argument("--out_dir", default="results/synthetic_init_scale_sweep")
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
        table = pd.read_csv(os.path.join(args.out_dir, "init_scale_sweep.csv"))
        plot_init_scale_figure(table, args.out_dir)
        print(f"replot OK: {args.out_dir}")
        return

    out = run_init_scale_data(
        args.out_dir, args.seeds, args.steps, args.overwrite, coef_max_delta=args.coef_max_delta
    )
    table = pd.read_csv(os.path.join(out, "init_scale_sweep.csv"))
    plot_init_scale_figure(table, out)


if __name__ == "__main__":
    main()
