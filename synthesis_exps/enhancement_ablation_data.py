"""
Enhancement ablation table data (no plotting).
Label: tab:enhancement-ablation

Example:
  python enhancement_ablation_data.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Tuple

import numpy as np
import pandas as pd

from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA, SharedVariant, run_shared_variant

ABLATION_VARIANTS: Tuple[SharedVariant, ...] = (
    SharedVariant(
        "raw",
        "Raw",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=False,
        use_maxnorm=False,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "tanh",
        "+Tanh",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=False,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "maxnorm",
        "+Max-norm",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "full",
        "Full (detached $w_k$)",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=True,
        detach_weights=True,
    ),
    SharedVariant(
        "no_w",
        "w/o $w_k$",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=False,
    ),
    SharedVariant(
        "attach_w",
        "Attached $w_k$",
        target_rms=0.0,
        use_tanh=True,
        use_alpha_tanh=True,
        use_maxnorm=True,
        use_confidence_weights=True,
        detach_weights=False,
    ),
)

BETAS = (1.0, 1.0, 0.0, -1.0)
CFG = "2R1N1A"


def summarize(variant: SharedVariant, rho: np.ndarray, abar: np.ndarray) -> dict:
    return {
        "config": CFG,
        "variant": variant.name,
        "label": variant.label,
        "use_alpha_tanh": variant.use_alpha_tanh,
        "use_maxnorm": variant.use_maxnorm,
        "use_w": variant.use_confidence_weights,
        "detach_w": variant.detach_weights,
        "correct": float((rho > 0.5).mean()),
        "mean_abs_corr": float(np.abs(rho).mean()),
        "mean_signed_corr": float(rho.mean()),
        "abar_R": float(abar[:, :2].mean()),
        "abar_N": float(abar[:, 2].mean()),
        "abar_A": float(abar[:, 3].mean()),
    }


def run_ablation_data(
    out_dir: str,
    *,
    seeds: int,
    steps: int,
    n: int,
    pairs: int,
    q: float,
    overwrite: bool,
    coef_max_delta: float = DEFAULT_COEF_MAX_DELTA,
) -> str:
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    rows: List[dict] = []
    for idx, v in enumerate(ABLATION_VARIANTS):
        rho, abar, _ = run_shared_variant(
            BETAS,
            v,
            seeds=seeds,
            steps=steps,
            n_seg=n,
            pairs=pairs,
            q=q,
            seed=9100 + 17 * idx,
            coef_max_delta=coef_max_delta,
        )
        row = summarize(v, rho, abar)
        rows.append(row)
        print(
            f"[{CFG}] {v.name:10s} correct={row['correct']:.3f} "
            f"|corr|={row['mean_abs_corr']:.3f} "
            f"aR={row['abar_R']:+.3f} aN={row['abar_N']:+.3f} aA={row['abar_A']:+.3f}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "enhancement_ablation.csv"), index=False)
    table.to_csv(os.path.join(out_dir, "enhancement_ablation_2R1N1A.csv"), index=False)
    print(f"OUT ablation data: {out_dir}")
    return out_dir


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
