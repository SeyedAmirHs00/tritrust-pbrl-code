"""Generate T sweep (lr ∝ 1/T) CSV (no plotting).

Example:
  python T_sweep_lr_scaled_data.py --seeds 120 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA, SHARED_BRANCH_VARIANTS, build_k4_configs, run_shared_variant

T0 = 50
LR0 = 0.05


def lr_theta_for_T(T: int) -> float:
    return LR0 * (T0 / float(T))


def ensure_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    os.makedirs(path)


def run_T_sweep_data(
    out_root: str,
    *,
    seeds: int,
    steps: int,
    Ts: list[int],
    overwrite: bool,
    coef_max_delta: float = DEFAULT_COEF_MAX_DELTA,
) -> str:
    ensure_dir(out_root, overwrite)
    configs = build_k4_configs()
    order = ["3R1N", "3R1A", "1R3A"]
    cal_rng = np.random.default_rng(9017)
    rows = []
    idx = 0

    for T in Ts:
        lr = lr_theta_for_T(T)
        for cfg in order:
            betas = configs[cfg]
            for v in SHARED_BRANCH_VARIANTS:
                idx += 1
                rho, abar, rms0 = run_shared_variant(
                    betas,
                    v,
                    seeds=seeds,
                    steps=steps,
                    T=T,
                    lr_theta=lr,
                    seed=9017 + 31 * idx,
                    cal_rng=cal_rng,
                    coef_max_delta=coef_max_delta,
                )
                correct = float((rho > 0.5).mean())
                flipped = float((rho < -0.5).mean())
                rows.append(
                    {
                        "T": T,
                        "lr_theta": lr,
                        "config": cfg,
                        "variant": v.name,
                        "label": v.label,
                        "correct_branch_rate": correct,
                        "flipped_branch_rate": flipped,
                        "mean_rho": float(rho.mean()),
                        "init_rms": rms0,
                        "mean_abar_R": float(abar[:, 0].mean())
                        if cfg == "1R3A"
                        else float(abar[:, :3].mean()),
                        "mean_abar_A": float(abar[:, -1].mean())
                        if "A" in cfg
                        else np.nan,
                    }
                )
                print(
                    f"T={T:3d} lr={lr:.4f} {cfg:6s} {v.name:10s} "
                    f"correct={correct:.3f} rho={rho.mean():+.3f} rms0={rms0:.3f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "T_sweep_lr_scaled.csv"), index=False)
    print(f"OUT T sweep data: {out_root}")
    return out_root


def main() -> None:
    p = argparse.ArgumentParser(description="T sweep with lr_theta ∝ 1/T (data only)")
    p.add_argument("--out_root", default="results/synthetic_T_sweep_lr_scaled")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument(
        "--Ts",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200],
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    args = p.parse_args()

    run_T_sweep_data(
        args.out_root,
        seeds=args.seeds,
        steps=args.steps,
        Ts=args.Ts,
        overwrite=args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )


if __name__ == "__main__":
    main()
