"""
Enhancement ablation on shared head R = sum_t tanh(θ^T s_t).
Label: tab:enhancement-ablation

Example:
  python enhancement_ablation.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse

from enhancement_ablation_data import run_ablation_data
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/synthetic_enhancement_ablation")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    args = p.parse_args()

    run_ablation_data(
        args.out_dir,
        seeds=args.seeds,
        steps=args.steps,
        n=args.n,
        pairs=args.pairs,
        q=args.q,
        overwrite=args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )


if __name__ == "__main__":
    main()
