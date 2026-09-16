"""
K=10 expert-ratio heatmaps.
Label: fig:synthetic-ratio-sweep

Example:
  python expert_ratio_K10.py --seeds 100 --overwrite
"""

from __future__ import annotations

import argparse

from expert_ratio_K10_data import run_expert_ratio_data
from expert_ratio_K10_plot import replot_from_csv
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/synthetic_expert_ratio_sweep_K10")
    p.add_argument("--n_experts", type=int, default=10)
    p.add_argument("--seeds", type=int, default=100)
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
        replot_from_csv(args.out_dir)
        return

    run_expert_ratio_data(
        args.out_dir,
        n_experts=args.n_experts,
        seeds=args.seeds,
        steps=args.steps,
        overwrite=args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )
    replot_from_csv(args.out_dir)
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
