"""
Partial-adversary α learning curves (Stabilized + Standard; detached w_k by default).

Example:
  python partial_adversary_alpha_curve.py --seeds 200 --overwrite
  python partial_adversary_alpha_curve.py --replot
"""

from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd

from partial_adversary_alpha_curve_data import (
    HIST_CSV_NAME,
    SUMMARY_CSV_NAME,
    resolve_lrs,
    resolve_methods,
    run_alpha_curve_experiment,
)
from partial_adversary_alpha_curve_plot import write_alpha_curve_figures
from synthetic_shared_core import DEFAULT_COEF_MAX_DELTA


def _default_out_dir(*, use_wk: bool, optimizer: str, reward_model: str) -> str:
    suffix_wk = "_wk" if use_wk else ""
    suffix_opt = "" if optimizer == "sgd" else f"_{optimizer}"
    suffix_model = "_mlp" if reward_model == "mlp" else ""
    return f"results/synthetic_partial_adversary_alpha_curve{suffix_wk}{suffix_opt}{suffix_model}"


def main() -> None:
    p = argparse.ArgumentParser(description="Partial-adversary α curves")
    p.add_argument("--out_dir", default=None, help="Default depends on --use-wk / optimizer / model.")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["stabilized", "standard", "subtract_init"],
        default=["stabilized", "standard"],
        help="Init variants (default: stabilized and standard; Standard uses rms|ΔR|≈1.4).",
    )
    p.add_argument(
        "--reward-model",
        choices=["linear", "mlp"],
        default="linear",
    )
    p.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw", "adam_sgd"],
        default="sgd",
    )
    p.add_argument("--lr-model", type=float, default=None)
    p.add_argument("--lr-alpha", type=float, default=None)
    p.add_argument(
        "--use-wk",
        dest="use_wk",
        action="store_true",
        default=True,
        help="Detached confidence weights w_k on reward loss (default: True).",
    )
    p.add_argument(
        "--no-wk",
        dest="use_wk",
        action="store_false",
        help="Disable w_k reweighting on reward loss.",
    )
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replot", action="store_true", help="Only regenerate figures from CSV")
    p.add_argument("--plot-ci", choices=["std", "sem", "var", "none"], default="std")
    p.add_argument("--plot-per-seed", action="store_true")
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = _default_out_dir(
            use_wk=args.use_wk, optimizer=args.optimizer, reward_model=args.reward_model
        )

    hist_path = os.path.join(args.out_dir, HIST_CSV_NAME)

    if args.replot:
        if not os.path.isfile(hist_path):
            raise FileNotFoundError(hist_path)
        hist_df = pd.read_csv(hist_path)
        write_alpha_curve_figures(
            hist_df,
            args.out_dir,
            ci=args.plot_ci,
            per_experiment=True,
            per_seed=args.plot_per_seed,
        )
        print(f"replot OK: {args.out_dir}")
        return

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise FileExistsError(f"{args.out_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    lr_model, lr_alpha = resolve_lrs(args.optimizer, args.lr_model, args.lr_alpha)
    hist_df, summary_df = run_alpha_curve_experiment(
        methods=resolve_methods(args.methods),
        reward_model=args.reward_model,
        seeds=args.seeds,
        steps=args.steps,
        log_every=args.log_every,
        hidden=args.hidden,
        n_layers=args.n_layers,
        optimizer=args.optimizer,
        lr_model=lr_model,
        lr_alpha=lr_alpha,
        use_wk=args.use_wk,
        coef_max_delta=args.coef_max_delta,
    )
    hist_df.to_csv(hist_path, index=False)
    summary_df.to_csv(os.path.join(args.out_dir, SUMMARY_CSV_NAME), index=False)
    write_alpha_curve_figures(
        hist_df,
        args.out_dir,
        ci=args.plot_ci,
        per_experiment=True,
        per_seed=args.plot_per_seed,
    )
    print(f"OUT: {args.out_dir}")


if __name__ == "__main__":
    main()
