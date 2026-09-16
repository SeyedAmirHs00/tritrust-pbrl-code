"""Generate branch-symmetry CSVs (no plotting).

Example:
  python branch_symmetry_data.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from synthetic_shared_core import (
    DEFAULT_COEF_MAX_DELTA,
    DEFAULT_STANDARD_TARGET_RMS,
    SharedVariant,
    build_k4_configs,
    run_shared_variant,
)

# Standard: rms|ΔR|_0 ≈ 1.4; Stabilized: θ=0.
BRANCH_VARIANTS = (
    SharedVariant(
        "standard",
        "Standard",
        target_rms=DEFAULT_STANDARD_TARGET_RMS,
        consensus_coef=0.0,
        use_tanh=True,
    ),
    SharedVariant("stabilized", "Stabilized", target_rms=0.0, consensus_coef=0.0, use_tanh=True),
)


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def run_branch_data(out_dir: str, seeds: int, steps: int, overwrite: bool, *, coef_max_delta: float = DEFAULT_COEF_MAX_DELTA) -> str:
    ensure_dir(out_dir, overwrite)
    configs = build_k4_configs()
    order = ["3R1N", "3R1A", "1R3A"]
    rows = []
    bar_rows = []
    cal_rng = np.random.default_rng(9017)
    idx = 0
    for cfg in order:
        betas = configs[cfg]
        k = len(betas)
        for v in BRANCH_VARIANTS:
            idx += 1
            rho, abar, rms0 = run_shared_variant(
                betas,
                v,
                seeds=seeds,
                steps=steps,
                seed=9017 + 31 * idx,
                cal_rng=cal_rng,
                coef_max_delta=coef_max_delta,
            )
            correct = rho > 0.5
            flipped = rho < -0.5
            if float(correct.mean()) >= float(flipped.mean()):
                mask = correct
                branch_tag = "correct"
            else:
                mask = flipped
                branch_tag = "flipped"
            abar_m = abar[mask] if mask.any() else abar
            rows.append(
                {
                    "config": cfg,
                    "variant": v.name,
                    "init_rms": rms0,
                    "correct_branch_rate": float(correct.mean()),
                    "flipped_branch_rate": float(flipped.mean()),
                    "mean_rho": float(rho.mean()),
                    "trust_conditioned_on": branch_tag,
                    "n_trust_seeds": int(mask.sum()),
                    "mean_abar_R": float(abar_m[:, :3].mean())
                    if cfg != "1R3A"
                    else float(abar_m[:, 0].mean()),
                    "mean_abar_N": float(abar_m[:, 3].mean()) if "N" in cfg else np.nan,
                    "mean_abar_A": float(abar_m[:, -1].mean()) if "A" in cfg else np.nan,
                }
            )
            mean = abar_m.mean(0)
            std = abar_m.std(0)
            for ei in range(k):
                bar_rows.append(
                    {
                        "config": cfg,
                        "variant": v.name,
                        "expert_idx": ei,
                        "mean_abar": float(mean[ei]),
                        "std_abar": float(std[ei]),
                    }
                )
            print(
                f"[branch] {cfg:6s} {v.name:10s} correct={rows[-1]['correct_branch_rate']:.3f} "
                f"rho={rows[-1]['mean_rho']:+.3f} rms0={rms0:.3f} "
                f"trust@{branch_tag} aR={rows[-1]['mean_abar_R']:+.2f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "paper_table_synthetic_symmetry_fix.csv"), index=False)

    wide = table.pivot(index="config", columns="variant", values="correct_branch_rate")
    wide = wide.reindex(columns=[v.name for v in BRANCH_VARIANTS])
    wide.to_csv(os.path.join(out_dir, "branch_correct_wide.csv"))

    pd.DataFrame(bar_rows).to_csv(os.path.join(out_dir, "alpha_bar_stats.csv"), index=False)
    print(f"OUT branch data: {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Branch-symmetry data")
    p.add_argument("--out_dir", default="results/synthetic_branch_symmetry")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    args = p.parse_args()

    run_branch_data(args.out_dir, args.seeds, args.steps, args.overwrite, coef_max_delta=args.coef_max_delta)


if __name__ == "__main__":
    main()
