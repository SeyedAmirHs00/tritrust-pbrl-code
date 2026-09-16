#!/usr/bin/env python3
"""Run buffer / RMS-ΔR diagnostics on each paper environment (one seed).

Uses ``train_PEBBLE_mixture_diagnostics.py``, which appends to
``buffer_diagnostics.csv`` under ``exp_pebble_mixture_diagnostics/...``
after every preference round at two checkpoints:

  pre_train  — after query sampling, before reward-model training
  post_train — after reward-model training

  rms_delta_r, median_sq_delta_r, std_delta_r, var_delta_r, mean_abs_delta_r, n_pairs,
  corr_r_rstar, corr_segment_r_rstar, n_corr_transitions, n_corr_segments,
  mean_state_{var,std,second_moment},
  mean_action_{var,std,second_moment},
  mean_sa_{var,std,second_moment}, n_transitions

Example
-------
  python scripts/run_buffer_diagnostics.py --dry-run
  python scripts/run_buffer_diagnostics.py --seed 12345
  python scripts/run_buffer_diagnostics.py --envs walker_walk cheetah_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class EnvRun:
    name: str
    env: str
    extra: Sequence[str]


# Hyperparameters match the existing per-env mixture scripts.
ENV_RUNS: Sequence[EnvRun] = (
    EnvRun(
        "walker_walk",
        "walker_walk",
        (
            "agent.params.actor_lr=0.0005",
            "agent.params.critic_lr=0.0005",
            "num_train_steps=500000",
            "agent.params.batch_size=1024",
            "double_q_critic.params.hidden_dim=1024",
            "double_q_critic.params.hidden_depth=2",
            "diag_gaussian_actor.params.hidden_dim=1024",
            "diag_gaussian_actor.params.hidden_depth=2",
            "num_unsup_steps=9000",
            "reward_batch=100",
            "num_interact=20000",
            "max_feedback=5000",
            "feed_type=6",
            "reward_update=50",
            "reset_update=100",
            "teacher_betas=[1,1,1,0,-1]",
        ),
    ),
    EnvRun(
        "cheetah_run",
        "cheetah_run",
        (
            "agent.params.actor_lr=0.0005",
            "agent.params.critic_lr=0.0005",
            "num_train_steps=1000000",
            "agent.params.batch_size=1024",
            "double_q_critic.params.hidden_dim=1024",
            "double_q_critic.params.hidden_depth=2",
            "diag_gaussian_actor.params.hidden_dim=1024",
            "diag_gaussian_actor.params.hidden_depth=2",
            "num_unsup_steps=9000",
            "reward_batch=100",
            "num_interact=20000",
            "max_feedback=4000",
            "feed_type=6",
            "reward_update=50",
            "reset_update=100",
            "teacher_betas=[1,1,1,-1]",
        ),
    ),
    EnvRun(
        "door_open",
        "metaworld_door-open-v2",
        (
            "agent.params.actor_lr=0.0003",
            "agent.params.critic_lr=0.0003",
            "activation=tanh",
            "num_unsup_steps=9000",
            "num_train_steps=1000000",
            "agent.params.batch_size=512",
            "double_q_critic.params.hidden_dim=256",
            "double_q_critic.params.hidden_depth=3",
            "diag_gaussian_actor.params.hidden_dim=256",
            "diag_gaussian_actor.params.hidden_depth=3",
            "reward_update=10",
            "num_interact=5000",
            "max_feedback=40000",
            "reward_batch=50",
            "feed_type=6",
            "teacher_betas=[1,1,1,-1]",
        ),
    ),
    EnvRun(
        "sweep_into",
        "metaworld_sweep-into-v2",
        (
            "agent.params.actor_lr=0.0003",
            "agent.params.critic_lr=0.0003",
            "activation=tanh",
            "num_unsup_steps=9000",
            "num_train_steps=1000000",
            "agent.params.batch_size=512",
            "double_q_critic.params.hidden_dim=256",
            "double_q_critic.params.hidden_depth=3",
            "diag_gaussian_actor.params.hidden_dim=256",
            "diag_gaussian_actor.params.hidden_depth=3",
            "reward_update=10",
            "num_interact=5000",
            "max_feedback=40000",
            "reward_batch=50",
            "feed_type=6",
            "teacher_betas=[1,1,1,-1]",
        ),
    ),
)


def build_cmd(run: EnvRun, seed: int, device: str) -> List[str]:
    return [
        sys.executable,
        "train_PEBBLE_mixture_diagnostics.py",
        f"env={run.env}",
        f"seed={seed}",
        f"device={device}",
        *run.extra,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=[r.name for r in ENV_RUNS],
        choices=[r.name for r in ENV_RUNS],
        help="Subset of environments to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [r for r in ENV_RUNS if r.name in args.envs]
    print("Reward-buffer diagnostics (one seed per env)")
    print(f"  seed   : {args.seed}")
    print(f"  device : {args.device}")
    print(f"  envs   : {[r.name for r in selected]}")
    print("  logs   : exp_pebble_mixture_diagnostics/<env>/.../buffer_diagnostics.csv")

    failures = []
    for run in selected:
        cmd = build_cmd(run, args.seed, args.device)
        print("\n" + "=" * 88)
        print(f"[{run.name}] {' '.join(cmd)}")
        print("=" * 88)
        if args.dry_run:
            continue
        result = subprocess.run(
            cmd, preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None
        )
        if result.returncode != 0:
            failures.append(run.name)
            print(f"[FAILED] {run.name}")

    if failures:
        print("\nFailed environments:")
        for name in failures:
            print(f"  {name}")
        return 1

    print("\nAll diagnostic runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
