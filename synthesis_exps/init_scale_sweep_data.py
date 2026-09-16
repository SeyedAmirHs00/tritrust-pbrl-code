"""Generate init-scale sweep CSVs (no plotting).

Example:
  python init_scale_sweep_data.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from synthetic_shared_core import (
    DEFAULT_COEF_MAX_DELTA,
    SharedVariant,
    build_k4_configs,
    calibrate_theta_scale,
    run_shared_variant,
)


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def run_init_scale_data(out_dir: str, seeds: int, steps: int, overwrite: bool, *, coef_max_delta: float = DEFAULT_COEF_MAX_DELTA) -> str:
    ensure_dir(out_dir, overwrite)
    configs = build_k4_configs()
    order = ["3R1A", "2R1A1N", "3R1N", "1R3A"]
    targets = [0.0, 0.05, 0.1, 0.25, 0.5, 1.4, 6.0]
    T = 50
    n_seg = 500
    methods = [
        ("no_cons", 0.0),
    ]
    cal_rng = np.random.default_rng(9201)
    # Large targets need a linear head: tanh saturates near rms≈√(2T)≈10 at T=50.
    theta_scales = {
        t: calibrate_theta_scale(
            t, seeds=40, n_seg=n_seg, T=T, d=16, rng=cal_rng, use_tanh=(t < 20.0)
        )
        for t in targets
    }

    rows = []
    idx = 0
    for cfg in order:
        betas = configs[cfg]
        for t in targets:
            for mname, ccoef in methods:
                idx += 1
                use_tanh = t < 20.0
                v = SharedVariant(
                    mname, mname, target_rms=t, consensus_coef=ccoef, use_tanh=use_tanh
                )
                rho, abar, rms0 = run_shared_variant(
                    betas,
                    v,
                    seeds=seeds,
                    steps=steps,
                    n_seg=n_seg,
                    T=T,
                    q=0.0,
                    seed=9201 + 37 * idx,
                    theta_scale=theta_scales[t],
                    coef_max_delta=coef_max_delta,
                )
                row = {
                    "config": cfg,
                    "method": mname,
                    "target_rms": t,
                    "step_sigma": t / np.sqrt(2 * T) if t > 0 else 0.0,
                    "init_rms_deltaR": rms0,
                    "correct_branch_rate": float((rho > 0.5).mean()),
                    "flipped_branch_rate": float((rho < -0.5).mean()),
                    "mean_rho": float(rho.mean()),
                }
                rows.append(row)
                print(
                    f"[init] {cfg:7s} {mname:10s} rms={t:<4g} correct={row['correct_branch_rate']:.3f} "
                    f"rho={row['mean_rho']:+.3f}"
                )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "init_scale_sweep.csv"), index=False)

    band_rows = []
    for t in targets:
        n = table[(table.config == "3R1N") & (table.method == "no_cons") & (table.target_rms == t)].iloc[
            0
        ]
        a = table[(table.config == "3R1A") & (table.method == "no_cons") & (table.target_rms == t)].iloc[
            0
        ]
        band_rows.append(
            {
                "target_rms": t,
                "3R1N_no": n.correct_branch_rate,
                "3R1A_no": a.correct_branch_rate,
                "both_no_ge_0.9": (n.correct_branch_rate >= 0.9) and (a.correct_branch_rate >= 0.9),
            }
        )
    pd.DataFrame(band_rows).to_csv(os.path.join(out_dir, "init_band_both_NA.csv"), index=False)
    print(f"OUT init-scale data: {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Init-scale sweep data")
    p.add_argument("--out_dir", default="results/synthetic_init_scale_sweep")
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

    run_init_scale_data(
        args.out_dir, args.seeds, args.steps, args.overwrite, coef_max_delta=args.coef_max_delta
    )


if __name__ == "__main__":
    main()
