"""
Trajectory-length (T) sweep with lr_theta scaled as 1/T so sum-over-T
reward gradients keep a comparable effective step size.

Baseline: T0=50, lr0=0.05  =>  lr(T) = lr0 * (T0 / T)

Produces: results/synthetic_T_sweep_lr_scaled/

Example:
  python T_sweep_lr_scaled.py --seeds 120 --overwrite
  python T_sweep_lr_scaled.py --replot
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from T_sweep_lr_scaled_data import run_T_sweep_data
from T_sweep_lr_scaled_plot import plot_T_figure
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def main() -> None:
    p = argparse.ArgumentParser(description="T sweep with lr_theta ∝ 1/T")
    p.add_argument("--data_dir", default="exp_synthesis/synthetic_T_sweep_lr_scaled", help="Directory for CSV data")
    p.add_argument("--plot_dir", default="result/synthetic_T_sweep_lr_scaled", help="Directory for plots")
    p.add_argument("--out_root", dest="data_dir", help="Alias for --data_dir")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument(
        "--Ts",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200],
    )
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
        df = pd.read_csv(os.path.join(args.data_dir, "T_sweep_lr_scaled.csv"))
        Ts = sorted(df["T"].unique().tolist())
        plot_T_figure(df, plot_dir=args.plot_dir, Ts=Ts)
        print(f"replot OK: {args.plot_dir}")
        return

    run_T_sweep_data(
        args.data_dir,
        seeds=args.seeds,
        steps=args.steps,
        Ts=args.Ts,
        overwrite=args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )
    df = pd.read_csv(os.path.join(args.data_dir, "T_sweep_lr_scaled.csv"))
    Ts = sorted(df["T"].unique().tolist())
    plot_T_figure(df, plot_dir=args.plot_dir, Ts=Ts)
    print(f"OUT: data={args.data_dir}, plot={args.plot_dir}")


if __name__ == "__main__":
    main()
