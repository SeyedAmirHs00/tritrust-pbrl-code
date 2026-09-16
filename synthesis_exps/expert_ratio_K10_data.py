"""Generate K=10 expert-ratio sweep CSV (no plotting).

Example:
  python expert_ratio_K10_data.py --seeds 100 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA, SHARED_BRANCH_VARIANTS, run_shared_variant


def _mean_slice(abar: np.ndarray, start: int, stop: int) -> float:
    if stop <= start:
        return float(np.nan)
    return float(np.mean(abar[:, start:stop]))


def run_expert_ratio_data(
    out_dir: str,
    *,
    n_experts: int,
    seeds: int,
    steps: int,
    overwrite: bool,
    coef_max_delta: float = DEFAULT_COEF_MAX_DELTA,
) -> str:
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    k = n_experts
    rows = []
    idx = 0
    for n_a in range(k + 1):
        for n_n in range(k - n_a + 1):
            n_r = k - n_a - n_n
            betas = tuple([1.0] * n_r + [0.0] * n_n + [-1.0] * n_a)
            name = f"{n_r}R{n_n}N{n_a}A"
            for v in SHARED_BRANCH_VARIANTS:
                idx += 1
                rho, abar, _ = run_shared_variant(
                    betas,
                    v,
                    seeds=seeds,
                    steps=steps,
                    seed=8000 + idx,
                    coef_max_delta=coef_max_delta,
                )
                rel_trust = _mean_slice(abar, 0, n_r)
                noisy_trust = _mean_slice(abar, n_r, n_r + n_n)
                adv_trust = _mean_slice(abar, n_r + n_n, n_r + n_n + n_a)
                rows.append(
                    {
                        "n_R": n_r,
                        "n_N": n_n,
                        "n_A": n_a,
                        "variant": v.name,
                        "correct_branch_rate": float((rho > 0.5).mean()),
                        "mean_adv_trust": adv_trust,
                        "mean_noisy_trust": noisy_trust,
                        "mean_rel_trust": rel_trust,
                        "mean_rho": float(rho.mean()),
                    }
                )
            sub = [r for r in rows if r["n_R"] == n_r and r["n_N"] == n_n and r["n_A"] == n_a]
            msg = " | ".join(f"{r['variant'][:3]}={r['correct_branch_rate']:.2f}" for r in sub)
            print(f"  {name:12s} {msg}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "expert_ratio_shared.csv"), index=False)
    print(f"OUT data: {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/synthetic_expert_ratio_sweep_K10")
    p.add_argument("--n_experts", type=int, default=10)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    args = p.parse_args()

    run_expert_ratio_data(
        args.out_dir,
        n_experts=args.n_experts,
        seeds=args.seeds,
        steps=args.steps,
        overwrite=args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )


if __name__ == "__main__":
    main()
