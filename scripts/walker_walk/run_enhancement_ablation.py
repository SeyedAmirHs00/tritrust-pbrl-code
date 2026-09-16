#!/usr/bin/python3
 
"""Ablation of TriTrust-PBRL practical enhancements on Walker-Walk.

Mirrors Table ``enhancement-ablation`` in ``main_v2.tex`` (Section
``sec:enhancement-ablation``), but as end-to-end PEBBLE policy learning:

  Variants
  --------
  Raw                 : no tanh, no max-norm, no confidence weight w_k
  +Tanh               : tanh only
  +Tanh,+Max-norm     : tanh + max-norm  (same as w/o w_k)
  Full TTP            : tanh + max-norm + w_k
  w_k reward-only     : tanh + max-norm + w_k in reward CE (detached from alpha)
  w/o Max-norm        : tanh + w_k
  w/o Tanh            : max-norm + w_k

Expert mixture is the paper's 3R1N1A setting: teacher_betas=[1,1,1,0,-1].
Default: 5 seeds on walker_walk.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_tanh: bool
    use_max_norm: bool
    use_confidence_weight: bool
    use_confidence_weight_in_alpha: bool = True
    note: str = ""


# Matches Table enhancement-ablation in main_v2.tex (unique rows), plus
# wk_reward_only (w_k scales reward CE but is detached from alpha grads).
ABLATION_VARIANTS: Sequence[AblationVariant] = (
    AblationVariant("raw", False, False, False, note="no enhancements"),
    AblationVariant("tanh", True, False, False, note="+Tanh"),
    AblationVariant("tanh_maxn", True, True, False, note="+Tanh,+Max-norm / w/o w_k"),
    AblationVariant("full_ttp", True, True, True, note="Full TTP"),
    AblationVariant(
        "wk_reward_only",
        True,
        True,
        True,
        False,
        note="w_k in reward loss only (detached from alpha)",
    ),
    AblationVariant("wo_maxn", True, False, True, note="w/o Max-norm"),
    AblationVariant("wo_tanh", False, True, True, note="w/o Tanh"),
)

DEFAULT_SEEDS = [12345, 23451, 34512, 45123, 51234]
DEFAULT_TEACHER_BETAS = [1, 1, 1, 0, -1]  # 3R1N1A


def build_cmd(
    seed: int,
    variant: AblationVariant,
    teacher_betas: List[float],
    max_feedback: int,
    reward_batch: int,
    num_train_steps: int,
    device: str,
) -> List[str]:
    betas = "[" + ",".join(str(b) for b in teacher_betas) + "]"
    return [
        sys.executable,
        "train_PEBBLE_mixture_ablation.py",
        "env=walker_walk",
        f"seed={seed}",
        f"device={device}",
        "agent.params.actor_lr=0.0005",
        "agent.params.critic_lr=0.0005",
        f"num_train_steps={num_train_steps}",
        "agent.params.batch_size=1024",
        "double_q_critic.params.hidden_dim=1024",
        "double_q_critic.params.hidden_depth=2",
        "diag_gaussian_actor.params.hidden_dim=1024",
        "diag_gaussian_actor.params.hidden_depth=2",
        "num_unsup_steps=9000",
        f"reward_batch={reward_batch}",
        "num_interact=20000",
        f"max_feedback={max_feedback}",
        "feed_type=6",
        "reward_update=50",
        "reset_update=100",
        f"teacher_betas={betas}",
        f"use_tanh={str(variant.use_tanh).lower()}",
        f"use_max_norm={str(variant.use_max_norm).lower()}",
        f"use_confidence_weight={str(variant.use_confidence_weight).lower()}",
        f"use_confidence_weight_in_alpha={str(variant.use_confidence_weight_in_alpha).lower()}",
    ]


def run_one(cmd: List[str], dry_run: bool) -> bool:
    print("\n" + "=" * 88)
    print(" ".join(cmd))
    print("=" * 88)
    if dry_run:
        return True
    # Isolate process group so Ctrl-C can be handled by the caller if desired.
    result = subprocess.run(cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None)
    return result.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TriTrust-PBRL practical-enhancement ablations on Walker-Walk."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--teacher-betas",
        type=float,
        nargs="+",
        default=DEFAULT_TEACHER_BETAS,
        help=f"Expert rationalities (default 3R1N1A: {DEFAULT_TEACHER_BETAS})",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=[v.name for v in ABLATION_VARIANTS],
        choices=[v.name for v in ABLATION_VARIANTS],
        help="Subset of ablation variants to run",
    )
    parser.add_argument("--max-feedback", type=int, default=5000)
    parser.add_argument("--reward-batch", type=int, default=100)
    parser.add_argument("--num-train-steps", type=int, default=500000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    name_to_variant = {v.name: v for v in ABLATION_VARIANTS}
    selected = [name_to_variant[name] for name in args.variants]

    print("TriTrust-PBRL practical-enhancement ablation (Walker-Walk)")
    print(f"  seeds            : {args.seeds}")
    print(f"  teacher_betas    : {args.teacher_betas}")
    print(f"  max_feedback     : {args.max_feedback}")
    print(f"  reward_batch     : {args.reward_batch}")
    print(f"  num_train_steps  : {args.num_train_steps}")
    print("  variants:")
    for v in selected:
        flags = (
            f"tanh={int(v.use_tanh)} "
            f"maxn={int(v.use_max_norm)} "
            f"wk={int(v.use_confidence_weight)} "
            f"wa={int(v.use_confidence_weight_in_alpha)}"
        )
        print(f"    - {v.name:14s} [{flags}]  ({v.note})")

    failures = []
    for variant in selected:
        for seed in args.seeds:
            cmd = build_cmd(
                seed=seed,
                variant=variant,
                teacher_betas=list(args.teacher_betas),
                max_feedback=args.max_feedback,
                reward_batch=args.reward_batch,
                num_train_steps=args.num_train_steps,
                device=args.device,
            )
            ok = run_one(cmd, dry_run=args.dry_run)
            if not ok:
                failures.append((variant.name, seed))
                print(f"[FAILED] variant={variant.name} seed={seed}")

    if failures:
        print("\nFailed runs:")
        for name, seed in failures:
            print(f"  {name} / seed={seed}")
        return 1

    print("\nAll ablation runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
