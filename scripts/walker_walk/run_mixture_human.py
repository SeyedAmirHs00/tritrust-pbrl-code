#!/usr/bin/env python3
"""Run mixture-human PEBBLE (zero-last + wk_sgd) on walker_walk.

Entrypoint: ``train_PEBBLE_mixture_human.py``
  - Stabilized init (zero last Linear of reward MLP)
  - SGD reward optimizer: network lr=0.001, alpha lr=0.005
  - Two-path w_k loss; human preference labels

Walker agent / feedback knobs match ``scripts/walker_walk/run_pebble.sh``
and ``config/train_PEBBLE_mixture_human.yaml`` (human budget).

Interactive: each feedback round prompts for human labels (1/2/3).

Examples
--------
  python scripts/walker_walk/run_mixture_human.py --dry-run
  python scripts/walker_walk/run_mixture_human.py --device cuda
  python scripts/walker_walk/run_mixture_human.py --seed 12345
  docker exec -it -w /workspace mixture_pbrl_container \\
    python scripts/walker_walk/run_mixture_human.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Sequence


DEFAULT_SEED: int = 12345

WALKER_EXTRA: Sequence[str] = (
    "env=walker_walk",
    "agent.params.actor_lr=0.0005",
    "agent.params.critic_lr=0.0005",
    "num_train_steps=1000000",
    "agent.params.batch_size=1024",
    "double_q_critic.params.hidden_dim=1024",
    "double_q_critic.params.hidden_depth=2",
    "diag_gaussian_actor.params.hidden_dim=1024",
    "diag_gaussian_actor.params.hidden_depth=2",
    "num_unsup_steps=9000",
    "reward_batch=10",
    "num_interact=20000",
    "max_feedback=100",
    "feed_type=1",
    "reward_update=50",
    "reset_update=100",
    "teacher_betas=[1,1,1,-1]",
    "reward_lr=0.001",
    "alpha_lr=0.005",
    "zero_last_layer=true",
)


def build_cmd(seed: int, device: str) -> List[str]:
    return [
        sys.executable,
        "train_PEBBLE_mixture_human.py",
        f"seed={seed}",
        f"device={device}",
        *WALKER_EXTRA,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of seeds (overrides --seed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]

    print("Mixture-human / zero-last + wk_sgd — walker_walk  β=[1,1,1,-1]")
    print(f"  device : {args.device}")
    print(f"  seeds  : {seeds}")
    print("  logs   : exp_pebble_mixture_zero_last_wk_sgd_human/walker_walk/...")

    failures: List[tuple[int, int]] = []
    for seed in seeds:
        cmd = build_cmd(seed, args.device)
        print("\n" + "=" * 88)
        print(f"[walker_walk] seed={seed}  {' '.join(cmd)}")
        print("=" * 88)
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
        )
        if result.returncode != 0:
            failures.append((seed, result.returncode))
            print(f"[FAILED] walker_walk seed={seed} (exit {result.returncode})")

    if failures:
        print("\nFailed runs:")
        for seed, code in failures:
            print(f"  seed={seed}  exit={code}")
        return 1

    print(f"\nAll {len(seeds)} walker_walk mixture-human runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
