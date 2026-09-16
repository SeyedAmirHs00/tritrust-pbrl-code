"""
Branch-symmetry trust bars + correct-branch table.
Label: fig:synthetic-branch

Produces: results/synthetic_branch_symmetry/

Example:
  python branch_symmetry.py --seeds 200 --overwrite
  python branch_symmetry.py --replot
"""

from __future__ import annotations

import argparse
import os

from branch_symmetry_data import run_branch_data
from branch_symmetry_plot import plot_branch_bars
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser(description="Branch-symmetry (fig:synthetic-branch)")
    p.add_argument("--out_dir", default="results/synthetic_branch_symmetry")
    p.add_argument("--seeds", type=int, default=200)
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
        plot_branch_bars(args.out_dir)
        print(f"replot OK: {args.out_dir}")
        return

    out = run_branch_data(
        args.out_dir, args.seeds, args.steps, args.overwrite, coef_max_delta=args.coef_max_delta
    )
    plot_branch_bars(out)


if __name__ == "__main__":
    main()
