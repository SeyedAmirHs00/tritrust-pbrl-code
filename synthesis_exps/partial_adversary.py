"""
Partial / stochastic adversaries.
Label: fig:partial-adversary

Example:
  python partial_adversary.py --seeds 200 --overwrite
  python partial_adversary.py --replot
"""


from __future__ import annotations

import argparse
import os

import pandas as pd

from partial_adversary_data import SETTINGS, run_partial_adversary_data
from partial_adversary_plot import plot_partial_adversary_figures
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/synthetic_partial_adversary")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()

    settings_order = [s for s, _ in SETTINGS]

    if args.replot:
        table = pd.read_csv(os.path.join(args.out_dir, "partial_adversary_shared.csv"))
        plot_partial_adversary_figures(table, args.out_dir, settings_order)
        print(f"replot OK: {args.out_dir}")
        return

    run_partial_adversary_data(
        args.out_dir,
        args.seeds,
        args.steps,
        args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )
    table = pd.read_csv(os.path.join(args.out_dir, "partial_adversary_shared.csv"))
    plot_partial_adversary_figures(table, args.out_dir, settings_order)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
