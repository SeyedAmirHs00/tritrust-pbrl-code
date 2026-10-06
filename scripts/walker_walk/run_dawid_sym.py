#!/usr/bin/env python3
"""Runner for Dawid-Skene Symmetric (Dawid-Sym) experiments on Walker-Walk.

Defaults to teacher_betas = [1, 1, 0, 0, -1] (2R2N1A mixture: 2 reliable,
2 noisy/random, 1 adversarial expert) with initial q = 1 and s = 1.

Examples
--------
  # Standard run with default betas [1, 1, 0, 0, -1] and 1000 feedback
  python scripts/walker_walk/run_dawid_sym.py --max-feedback 1000

  # Single seed dry-run (prints command without running)
  python scripts/walker_walk/run_dawid_sym.py --max-feedback 1000 --seeds 12345 --dry-run

  # Feedback budget 3000 across all 5 seeds on GPU
  python scripts/walker_walk/run_dawid_sym.py --max-feedback 3000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

DEFAULT_SEEDS: Sequence[int] = (
    12345,
    23451,
    34512,
    45123,
    51234,
)

DEFAULT_TEACHER_BETAS: Sequence[int] = (1, 1, 0, 0, -1)

ENTRYPOINT = "train_PEBBLE_dawid_sym.py"
LOG_ROOT = "exp_pebble_dawid_sym"


@dataclass(frozen=True)
class FeedbackPreset:
    max_feedback: int
    reward_batch: int
    feed_type: int


FEEDBACK_PRESETS: dict[int, FeedbackPreset] = {
    500: FeedbackPreset(
        max_feedback=500,
        reward_batch=10,
        feed_type=6,
    ),
    1000: FeedbackPreset(
        max_feedback=1000,
        reward_batch=20,
        feed_type=6,
    ),
    2000: FeedbackPreset(
        max_feedback=2000,
        reward_batch=40,
        feed_type=6,
    ),
    3000: FeedbackPreset(
        max_feedback=3000,
        reward_batch=60,
        feed_type=6,
    ),
    5000: FeedbackPreset(
        max_feedback=5000,
        reward_batch=100,
        feed_type=6,
    ),
    7500: FeedbackPreset(
        max_feedback=7500,
        reward_batch=150,
        feed_type=6,
    ),
    10000: FeedbackPreset(
        max_feedback=10000,
        reward_batch=200,
        feed_type=6,
    ),
}

WALKER_BASE: Sequence[str] = (
    "env=walker_walk",
    "agent.params.actor_lr=0.0005",
    "agent.params.critic_lr=0.0005",
    "agent.params.batch_size=1024",
    "double_q_critic.params.hidden_dim=1024",
    "double_q_critic.params.hidden_depth=2",
    "diag_gaussian_actor.params.hidden_dim=1024",
    "diag_gaussian_actor.params.hidden_depth=2",
    "num_unsup_steps=9000",
    "num_interact=20000",
    "reward_update=50",
    "reset_update=100",
)


def format_betas(betas: Sequence[int | float]) -> str:
    return "[" + ",".join(str(b) for b in betas) + "]"


def build_cmd(
    *,
    seed: int,
    device: str,
    preset: FeedbackPreset,
    teacher_betas: Sequence[int | float],
    reward_batch: int | None = None,
    num_train_steps: int | None = None,
    reward_lr: float | None = None,
    expert_lr: float | None = None,
    q_init: float = 1.0,
    s_init: float = 1.0,
) -> List[str]:
    rb = preset.reward_batch if reward_batch is None else reward_batch
    steps = 500_000 if num_train_steps is None else num_train_steps
    extra: List[str] = [
        *WALKER_BASE,
        f"num_train_steps={steps}",
        f"reward_batch={rb}",
        f"max_feedback={preset.max_feedback}",
        f"feed_type={preset.feed_type}",
        f"teacher_betas={format_betas(teacher_betas)}",
        f"q_init={q_init}",
        f"s_init={s_init}",
    ]

    if reward_lr is not None:
        extra.append(f"reward_lr={reward_lr}")
    if expert_lr is not None:
        extra.append(f"expert_lr={expert_lr}")

    root_dir = Path(__file__).resolve().parents[2]
    entrypoint_path = str(root_dir / ENTRYPOINT)

    return [
        sys.executable,
        entrypoint_path,
        f"seed={seed}",
        f"device={device}",
        *extra,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runner for Dawid-Sym on Walker-Walk with custom teacher_betas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--teacher-betas",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEACHER_BETAS),
        metavar="B",
        help="Teacher rationality mixture (default: 1 1 0 0 -1 for 2R2N1A)",
    )
    parser.add_argument(
        "--max-feedback",
        type=int,
        choices=sorted(FEEDBACK_PRESETS),
        default=1000,
        help="Total preference-query budget",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds to evaluate",
    )
    parser.add_argument(
        "--reward-batch",
        type=int,
        default=None,
        help="Override preset reward_batch (default: auto from --max-feedback)",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Override num_train_steps (default: 500000)",
    )
    parser.add_argument(
        "--reward-lr",
        type=float,
        default=None,
        help="Reward ensemble learning rate (default: from config, 0.001)",
    )
    parser.add_argument(
        "--expert-lr",
        type=float,
        default=None,
        help="Expert reliability q and scale s learning rate (default: from config, 0.005)",
    )
    parser.add_argument(
        "--q-init",
        type=float,
        default=1.0,
        help="Initial expert reliability q (default: 1.0)",
    )
    parser.add_argument(
        "--s-init",
        type=float,
        default=1.0,
        help="Initial temperature scale s (default: 1.0)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset = FEEDBACK_PRESETS[args.max_feedback]
    rb = preset.reward_batch if args.reward_batch is None else args.reward_batch
    beta_label = format_betas(args.teacher_betas)

    print("=" * 80)
    print("Walker-Walk — Dawid-Skene Symmetric (Dawid-Sym)")
    print("=" * 80)
    print(f"  entrypoint     : {ENTRYPOINT}")
    print(f"  environment    : walker_walk")
    print(f"  teacher_betas  : {beta_label}")
    print(f"  q_init         : {args.q_init}")
    print(f"  s_init         : {args.s_init}")
    print(f"  max_feedback   : {preset.max_feedback}")
    print(f"  reward_batch   : {rb}")
    print(f"  feed_type      : {preset.feed_type}")
    print(f"  seeds          : {args.seeds}")
    print(f"  device         : {args.device}")
    print(f"  logs           : {LOG_ROOT}/walker_walk/...")
    print(f"  total runs     : {len(args.seeds)}")
    print("=" * 80)

    failures: List[tuple[str, int, int]] = []
    for seed in args.seeds:
        cmd = build_cmd(
            seed=seed,
            device=args.device,
            preset=preset,
            teacher_betas=args.teacher_betas,
            reward_batch=args.reward_batch,
            num_train_steps=args.num_train_steps,
            reward_lr=args.reward_lr,
            expert_lr=args.expert_lr,
            q_init=args.q_init,
            s_init=args.s_init,
        )
        print("\n" + "-" * 80)
        print(f"[walker_walk | dawid_sym] β={beta_label}  seed={seed}")
        print("-" * 80)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
        )
        if result.returncode != 0:
            failures.append((beta_label, seed, result.returncode))
            print(f"[FAILED] β={beta_label} seed={seed} (exit {result.returncode})")

    if failures:
        print("\nFailed runs:")
        for blabel, s, code in failures:
            print(f"  β={blabel}  seed={s}  exit={code}")
        return 1

    n = len(args.seeds)
    print(f"\nAll {n} walker_walk Dawid-Sym run(s) completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
