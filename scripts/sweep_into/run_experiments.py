#!/usr/bin/env python3
"""Unified runner for Meta-World Sweep-Into experiments.

Consolidates all experiments for Sweep-Into across methods:
  - ``pebble_mixture`` : TriTrust-PBRL mixture (``train_PEBBLE_mixture.py``)
  - ``rime_mixture``   : RIME mixture baseline (``train_RIME_mixture.py``)
  - ``mixup_mixture``  : Mixup mixture baseline (``train_PEBBLE_mixup_mixture.py``)
  - ``mixture_raw``    : Raw mixture baseline (``train_PEBBLE_mixture_raw.py``)
  - ``pebble``         : Standard single-teacher PEBBLE (``train_PEBBLE.py``)
  - ``sac``            : Oracle SAC baseline (``train_SAC.py``)

Key Settings for Sweep-Into:
  - Environment: ``metaworld_sweep-into-v2``
  - Actor / Critic hidden dim: 256, depth: 3, batch size: 512, activation: tanh
  - Actor / Critic lr: 0.0003
  - Mixture feedback budget: 40000, reward batch: 50, num_interact: 5000
  - Reward update: 10
  - Experiment reward_lr defaults (per method configs):
    * pebble_mixture : 0.05 (SGD network lr), alpha_lr: 0.005
    * rime_mixture   : 0.0003 (Adam lr)
    * mixup_mixture  : 0.0003 (Adam lr)
    * mixture_raw    : 0.0003 (Adam lr)
    * pebble         : 0.0003 (Adam lr)

Features:
  - Method selection (single method or multiple / ``all``)
  - Automatic experiment-aware reward_lr defaults matching each method's config
  - Preset beta configs (``1,1,1,0``, ``1,1,1,-1``, or ``all``) or custom betas
  - Full override options for seeds, device, hyperparameters, and Hydra overrides
  - Graceful interrupt handling (Ctrl+C shows run details, asks to stop, skip, or continue)
  - Dry-run mode for previewing full commands before launch

Examples
--------
  # Preview all pebble_mixture experiments (both beta configurations, 10 seeds):
  python scripts/sweep_into/run_experiments.py --dry-run

  # Run pebble_mixture with [1,1,1,0] preset on specific seeds:
  python scripts/sweep_into/run_experiments.py --method pebble_mixture --preset 1,1,1,0 --seeds 12345 23451

  # Run rime_mixture with [1,1,1,-1] preset:
  python scripts/sweep_into/run_experiments.py --method rime_mixture --preset 1,1,1,-1

  # Run mixup_mixture with custom teacher betas:
  python scripts/sweep_into/run_experiments.py --method mixup_mixture --teacher-betas 1 1 1 0

  # Run all mixture methods on [1,1,1,-1]:
  python scripts/sweep_into/run_experiments.py --methods pebble_mixture rime_mixture mixup_mixture mixture_raw --preset 1,1,1,-1
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_SEEDS: Sequence[int] = (
    12345,
    23451,
    34512,
    45123,
    51234,
    67890,
    78906,
    89067,
    90678,
    6789,
)

BETA_PRESETS: Dict[str, Tuple[float, ...]] = {
    "1,1,1,0": (1.0, 1.0, 1.0, 0.0),
    "1,1,1,-1": (1.0, 1.0, 1.0, -1.0),
}

ENTRYPOINTS: Dict[str, str] = {
    "pebble_mixture": "train_PEBBLE_mixture.py",
    "rime_mixture": "train_RIME_mixture.py",
    "mixup_mixture": "train_PEBBLE_mixup_mixture.py",
    "mixture_raw": "train_PEBBLE_mixture_raw.py",
    "pebble": "train_PEBBLE.py",
    "sac": "train_SAC.py",
}

# Experiment-specific reward_lr defaults matching each method's config / optimizer:
# - pebble_mixture uses SGD (lr=0.05 on Sweep-Into)
# - rime_mixture, mixup_mixture, mixture_raw, pebble use Adam (lr=0.0003)
DEFAULT_REWARD_LRS: Dict[str, float] = {
    "pebble_mixture": 0.05,
    "rime_mixture": 0.0003,
    "mixup_mixture": 0.0003,
    "mixture_raw": 0.0003,
    "pebble": 0.0003,
}

MIXTURE_METHODS = {"pebble_mixture", "rime_mixture", "mixup_mixture", "mixture_raw"}


def format_betas(betas: Sequence[float | int]) -> str:
    """Format betas as a Hydra list string, e.g. [1,1,1,0] or [1,1,1,-1]."""
    items = []
    for b in betas:
        if isinstance(b, float) and b.is_integer():
            items.append(str(int(b)))
        else:
            items.append(str(b))
    return "[" + ",".join(items) + "]"


def resolve_reward_lr(method: str, user_override: Optional[float] = None) -> Optional[float]:
    """Resolve the effective reward_lr for a method, respecting user overrides."""
    if user_override is not None:
        return user_override
    return DEFAULT_REWARD_LRS.get(method)


def build_cmd(
    method: str,
    seed: int,
    device: str,
    teacher_betas: Sequence[float | int],
    reward_lr: Optional[float],
    alpha_lr: float,
    max_feedback: int,
    reward_batch: int,
    num_train_steps: int,
    num_unsup_steps: int,
    num_interact: int,
    reward_update: int,
    feed_type: int,
    mixup_alpha: float,
    env: str,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Optional[float]]:
    """Build command list for running the selected sweep_into experiment."""
    entrypoint = ENTRYPOINTS[method]
    effective_reward_lr = resolve_reward_lr(method, reward_lr)

    cmd = [
        sys.executable,
        entrypoint,
        f"env={env}",
        f"seed={seed}",
        f"device={device}",
        "agent.params.actor_lr=0.0003",
        "agent.params.critic_lr=0.0003",
        "activation=tanh",
        f"num_train_steps={num_train_steps}",
        "agent.params.batch_size=512",
        "double_q_critic.params.hidden_dim=256",
        "double_q_critic.params.hidden_depth=3",
        "diag_gaussian_actor.params.hidden_dim=256",
        "diag_gaussian_actor.params.hidden_depth=3",
    ]

    if method in MIXTURE_METHODS:
        cmd.extend([
            f"num_unsup_steps={num_unsup_steps}",
            f"reward_batch={reward_batch}",
            f"num_interact={num_interact}",
            f"max_feedback={max_feedback}",
            f"feed_type={feed_type}",
            f"reward_update={reward_update}",
            f"teacher_betas={format_betas(teacher_betas)}",
        ])
        if effective_reward_lr is not None:
            cmd.append(f"reward_lr={effective_reward_lr}")
        if method == "pebble_mixture":
            cmd.append(f"alpha_lr={alpha_lr}")
        elif method == "mixup_mixture":
            cmd.append(f"mixup_alpha={mixup_alpha}")

    elif method == "pebble":
        cmd.extend([
            f"num_unsup_steps={num_unsup_steps}",
            "reward_batch=50",
            f"num_interact={num_interact}",
            "max_feedback=10000",
            "feed_type=1",
            f"reward_update={reward_update}",
            "reset_update=100",
            "segment=25",
            "teacher_beta=-1",
            "teacher_gamma=1",
            "teacher_eps_skip=0",
            "teacher_eps_mistake=0",
            "teacher_eps_equal=0",
        ])
        if effective_reward_lr is not None:
            cmd.append(f"reward_lr={effective_reward_lr}")

    if extra_args:
        cmd.extend(extra_args)
    return cmd, effective_reward_lr


def _format_experiment_details(
    method: str,
    seed: int,
    device: str,
    betas: Sequence[float | int],
    reward_lr: Optional[float],
    alpha_lr: float,
    max_feedback: int,
    remaining_seeds: Sequence[int],
    remaining_tasks: int,
) -> str:
    lines = [
        "Experiment details:",
        f"  method            : {method} ({ENTRYPOINTS[method]})",
        "  env               : metaworld_sweep-into-v2",
        f"  seed              : {seed}",
        f"  device            : {device}",
    ]
    if reward_lr is not None:
        lines.append(f"  reward_lr         : {reward_lr}")
    if method == "pebble_mixture":
        lines.append(f"  alpha_lr          : {alpha_lr}")
    if method in MIXTURE_METHODS:
        lines.extend([
            f"  teacher_betas     : {format_betas(betas)}",
            f"  max_feedback      : {max_feedback}",
        ])
    lines.extend([
        f"  remaining_seeds   : {list(remaining_seeds)}",
        f"  remaining_tasks   : {remaining_tasks}",
    ])
    return "\n".join(lines)


class InterruptGuard:
    """Confirm Ctrl+C at the runner level; child runs in its own process group."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.method: str = "pebble_mixture"
        self.seed: Optional[int] = None
        self.device: str = "cuda"
        self.betas: Sequence[float | int] = ()
        self.reward_lr: Optional[float] = None
        self.alpha_lr: float = 0.005
        self.max_feedback: int = 40000
        self.remaining_seeds: Sequence[int] = ()
        self.remaining_tasks: int = 0
        self.confirming = False
        self.cancel_requested = False
        self.skip_requested = False

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handler)

    def set_current(
        self,
        proc: subprocess.Popen,
        method: str,
        seed: int,
        device: str,
        betas: Sequence[float | int],
        reward_lr: Optional[float],
        alpha_lr: float,
        max_feedback: int,
        remaining_seeds: Sequence[int],
        remaining_tasks: int,
    ) -> None:
        self.proc = proc
        self.method = method
        self.seed = seed
        self.device = device
        self.betas = betas
        self.reward_lr = reward_lr
        self.alpha_lr = alpha_lr
        self.max_feedback = max_feedback
        self.remaining_seeds = remaining_seeds
        self.remaining_tasks = remaining_tasks
        self.skip_requested = False

    def clear_current(self) -> None:
        self.proc = None
        self.seed = None

    def kill_child(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def _handler(self, signum: int, frame: Any) -> None:
        if self.confirming:
            print("\nSecond Ctrl+C received — forcing immediate exit.", flush=True)
            self.kill_child()
            os._exit(130)

        self.confirming = True
        try:
            print("\n" + "=" * 76, flush=True)
            print("Ctrl+C received — interrupt requested.", flush=True)
            if self.seed is not None:
                print(
                    _format_experiment_details(
                        method=self.method,
                        seed=self.seed,
                        device=self.device,
                        betas=self.betas,
                        reward_lr=self.reward_lr,
                        alpha_lr=self.alpha_lr,
                        max_feedback=self.max_feedback,
                        remaining_seeds=self.remaining_seeds,
                        remaining_tasks=self.remaining_tasks,
                    ),
                    flush=True,
                )
            else:
                print("  (no active training child)", flush=True)
            print("=" * 76, flush=True)
            print("Options: [y] Stop runner  |  [s] Skip this seed  |  [N/Enter] Continue")
            try:
                answer = input("Choose action [y/s/N]: ").strip().lower()
            except EOFError:
                answer = "y"

            if answer in ("y", "yes"):
                print("Cancelling current experiment and stopping runner.", flush=True)
                self.cancel_requested = True
                self.kill_child()
            elif answer in ("s", "skip"):
                print("Terminating current seed and advancing to next...", flush=True)
                self.skip_requested = True
                self.kill_child()
            else:
                print("Continuing current experiment...", flush=True)
        finally:
            self.confirming = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    method_group = parser.add_argument_group("Method Selection")
    method_group.add_argument(
        "--method",
        type=str,
        default="pebble_mixture",
        choices=list(ENTRYPOINTS.keys()) + ["all"],
        help="Experiment method to run (default: pebble_mixture)",
    )
    method_group.add_argument(
        "--methods",
        nargs="+",
        choices=list(ENTRYPOINTS.keys()),
        default=None,
        help="Multiple methods to execute sequentially (overrides --method)",
    )

    exp_group = parser.add_argument_group("Mixture Beta Selection")
    exp_group.add_argument(
        "--preset",
        choices=["all", "1,1,1,0", "1,1,1,-1"],
        default="all",
        help="Beta preset to run: '1,1,1,0', '1,1,1,-1', or 'all' (default: all)",
    )
    exp_group.add_argument(
        "--all-betas",
        action="store_true",
        help="Explicitly run all preset beta configurations ([1,1,1,0] and [1,1,1,-1])",
    )
    exp_group.add_argument(
        "--teacher-betas",
        nargs="+",
        type=float,
        default=None,
        metavar="B",
        help="Custom teacher rationality values (e.g. --teacher-betas 1 1 1 0). Overrides --preset.",
    )

    run_group = parser.add_argument_group("Run Configuration")
    run_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Random seeds to execute (default: {list(DEFAULT_SEEDS)})",
    )
    run_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (default: cuda)",
    )
    run_group.add_argument(
        "--env",
        type=str,
        default="metaworld_sweep-into-v2",
        help="Environment name (default: metaworld_sweep-into-v2)",
    )
    run_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training subprocesses",
    )
    run_group.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining experiments even if one fails",
    )

    param_group = parser.add_argument_group("Hyperparameter Overrides")
    param_group.add_argument(
        "--reward-lr",
        type=float,
        default=None,
        help=(
            "Learning rate for reward model parameters. If omitted, uses method-specific config default "
            "(pebble_mixture: 0.05; rime_mixture, mixup_mixture, mixture_raw, pebble: 0.0003)"
        ),
    )
    param_group.add_argument(
        "--alpha-lr",
        type=float,
        default=0.005,
        help="Learning rate for alpha trust weights in pebble_mixture (default: 0.005)",
    )
    param_group.add_argument(
        "--max-feedback",
        type=int,
        default=40000,
        help="Maximum feedback budget (default: 40000)",
    )
    param_group.add_argument(
        "--reward-batch",
        type=int,
        default=50,
        help="Reward query batch size (default: 50)",
    )
    param_group.add_argument(
        "--num-train-steps",
        type=int,
        default=1000000,
        help="Total training environment steps (default: 1000000)",
    )
    param_group.add_argument(
        "--num-unsup-steps",
        type=int,
        default=9000,
        help="Number of unsupervised exploratory steps (default: 9000)",
    )
    param_group.add_argument(
        "--num-interact",
        type=int,
        default=5000,
        help="Steps between teacher interactions (default: 5000)",
    )
    param_group.add_argument(
        "--reward-update",
        type=int,
        default=10,
        help="Reward model update epochs per query batch (default: 10)",
    )
    param_group.add_argument(
        "--feed-type",
        type=int,
        default=6,
        help="Feedback query selection type (default: 6)",
    )
    param_group.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.5,
        help="Mixup alpha parameter for mixup_mixture (default: 0.5)",
    )
    param_group.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Additional Hydra key=value overrides",
    )

    return parser.parse_args()


def determine_methods(args: argparse.Namespace) -> List[str]:
    if args.methods is not None:
        return args.methods
    if args.method == "all":
        return list(ENTRYPOINTS.keys())
    return [args.method]


def determine_beta_configurations(args: argparse.Namespace) -> List[Tuple[float | int, ...]]:
    if args.teacher_betas is not None:
        return [tuple(args.teacher_betas)]
    if args.all_betas or args.preset == "all":
        return [BETA_PRESETS["1,1,1,0"], BETA_PRESETS["1,1,1,-1"]]
    return [BETA_PRESETS[args.preset]]


def main() -> int:
    args = parse_args()
    methods = determine_methods(args)
    beta_configs = determine_beta_configurations(args)

    total_runs = 0
    for m in methods:
        if m in MIXTURE_METHODS:
            total_runs += len(beta_configs) * len(args.seeds)
        else:
            total_runs += len(args.seeds)

    print("=" * 80)
    print("Meta-World Sweep-Into Experiment Runner")
    print("=" * 80)
    print(f"  env               : {args.env}")
    print(f"  methods ({len(methods):2d})     : {methods}")
    print(f"  device            : {args.device}")
    if args.reward_lr is not None:
        print(f"  reward_lr         : {args.reward_lr} (user override for all methods)")
    else:
        lr_info = ", ".join(f"{m}={DEFAULT_REWARD_LRS[m]}" for m in methods if m in DEFAULT_REWARD_LRS)
        print(f"  reward_lr         : auto (config defaults: {lr_info})")
    print(f"  alpha_lr          : {args.alpha_lr}")
    print(f"  max_feedback      : {args.max_feedback}")
    print(f"  reward_batch      : {args.reward_batch}")
    print(f"  num_train_steps   : {args.num_train_steps}")
    print(f"  seeds ({len(args.seeds):2d})     : {args.seeds}")
    if any(m in MIXTURE_METHODS for m in methods):
        print(f"  beta sets ({len(beta_configs):2d}) : {[format_betas(b) for b in beta_configs]}")
    print(f"  total runs        : {total_runs}")
    if args.dry_run:
        print("  mode              : DRY RUN (no training launched)")
    print("=" * 80)

    guard = InterruptGuard()
    guard.install()

    failures: List[Tuple[str, str, int, int]] = []
    completed = 0
    current_run = 0

    for method in methods:
        configs_to_run = beta_configs if method in MIXTURE_METHODS else [()]
        for betas in configs_to_run:
            beta_str = format_betas(betas) if method in MIXTURE_METHODS else "N/A"

            for seed_idx, seed in enumerate(args.seeds):
                current_run += 1
                remaining_seeds = args.seeds[seed_idx:]
                remaining_tasks = total_runs - current_run

                cmd, effective_reward_lr = build_cmd(
                    method=method,
                    seed=seed,
                    device=args.device,
                    teacher_betas=betas,
                    reward_lr=args.reward_lr,
                    alpha_lr=args.alpha_lr,
                    max_feedback=args.max_feedback,
                    reward_batch=args.reward_batch,
                    num_train_steps=args.num_train_steps,
                    num_unsup_steps=args.num_unsup_steps,
                    num_interact=args.num_interact,
                    reward_update=args.reward_update,
                    feed_type=args.feed_type,
                    mixup_alpha=args.mixup_alpha,
                    env=args.env,
                    extra_args=args.extra_args,
                )

                lr_str = f"reward_lr={effective_reward_lr}" if effective_reward_lr is not None else "no reward_lr"
                print("\n" + "#" * 88)
                print(
                    f"[{current_run}/{total_runs}] [sweep_into] method={method} | β={beta_str} | seed={seed} | {lr_str}"
                )
                print("#" * 88)
                print(f"Command: {' '.join(cmd)}\n")

                if args.dry_run:
                    completed += 1
                    continue

                proc = subprocess.Popen(
                    cmd,
                    preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None,
                )
                guard.set_current(
                    proc=proc,
                    method=method,
                    seed=seed,
                    device=args.device,
                    betas=betas,
                    reward_lr=effective_reward_lr,
                    alpha_lr=args.alpha_lr,
                    max_feedback=args.max_feedback,
                    remaining_seeds=remaining_seeds,
                    remaining_tasks=remaining_tasks,
                )
                returncode = proc.wait()
                guard.clear_current()

                if guard.cancel_requested:
                    print(f"\nRunner halted by user interrupt during seed={seed}.")
                    return 130

                if guard.skip_requested:
                    print(f"[SKIPPED] Seed {seed} skipped by user.")
                    failures.append((method, beta_str, seed, -1))
                    continue

                if returncode != 0:
                    failures.append((method, beta_str, seed, returncode))
                    print(f"[FAILED] method={method} β={beta_str} seed={seed} (exit code {returncode})")
                    if not args.continue_on_error:
                        print("\nAborting runner (use --continue-on-error to keep running).")
                        return returncode
                else:
                    completed += 1
                    print(f"[SUCCESS] method={method} β={beta_str} seed={seed}")

    print("\n" + "=" * 80)
    print("Execution Summary")
    print("=" * 80)
    print(f"Total planned : {total_runs}")
    print(f"Completed     : {completed}")
    print(f"Failed/Skipped: {len(failures)}")

    if failures:
        print("\nFailed / Skipped Runs:")
        for m, b_str, seed, code in failures:
            status = "skipped" if code == -1 else f"exit code {code}"
            print(f"  {m:16s}  β={b_str:14s}  seed={seed:<7d}  ({status})")
        return 1

    print("\nAll experiments completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
