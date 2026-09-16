"""Generate Partial/stochastic adversary CSV (no plotting).

Example:
  python partial_adversary_data.py --seeds 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from synthetic_shared_core import (
    DEFAULT_COEF_MAX_DELTA,
    DEFAULT_STANDARD_TARGET_RMS,
    clamp_coef_after_step,
    sample_expert_pairs,
    sigmoid,
)

# Shared with partial_adversary_alpha_curve_data: same (setting, method) → same RNG seed.
SEED_BASE = 9400

SETTINGS: List[Tuple[str, Dict[str, Optional[float]]]] = [
    ("perfect_flip", dict(beta_adv=-1.0, flip_prob=None)),
    ("beta_-0.5", dict(beta_adv=-0.5, flip_prob=None)),
    ("beta_-0.25", dict(beta_adv=-0.25, flip_prob=None)),
    ("stoch_p0.5", dict(beta_adv=None, flip_prob=0.5)),
    ("stoch_p0.25", dict(beta_adv=None, flip_prob=0.25)),
]

METHODS: List[Tuple[str, Dict[str, float]]] = [
    ("stabilized", dict(target_rms=0.0, consensus_coef=0.0)),
    ("standard", dict(target_rms=DEFAULT_STANDARD_TARGET_RMS, consensus_coef=0.0)),
]

METHOD_SEED_ORDER: Tuple[str, ...] = tuple(m for m, _ in METHODS)


def partial_adversary_job_seed(
    setting_index: int,
    method_name: str,
    *,
    method_order: Sequence[str] = METHOD_SEED_ORDER,
) -> int:
    """Map (setting, method) to the historical ``9400 + idx`` grid.

    ``idx`` counts over ``SETTINGS × METHODS`` in that order (1-based), matching
    the original partial_adversary_data loop. Alpha-curve (stabilized-only) must
    use this helper so it shares RNGs with the bar experiment's stabilized rows.
    """
    try:
        method_index = list(method_order).index(method_name)
    except ValueError as exc:
        raise ValueError(
            f"method {method_name!r} not in seed order {list(method_order)}"
        ) from exc
    return SEED_BASE + setting_index * len(method_order) + method_index + 1


def make_partial_adv_labels(
    rng: np.random.Generator,
    *,
    r_star: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    beta_adv: float | None,
    flip_prob: float | None,
) -> np.ndarray:
    """Build y with shape (seeds, K=4, pairs) for 3R1A partial/stochastic adv."""
    seeds, k, pairs = i.shape
    assert k == 4
    y = np.zeros((seeds, k, pairs))
    for e in range(3):
        d_star = np.take_along_axis(r_star, i[:, e], 1) - np.take_along_axis(r_star, j[:, e], 1)
        y[:, e] = (rng.random((seeds, pairs)) < sigmoid(d_star)).astype(float)
    d_adv = np.take_along_axis(r_star, i[:, 3], 1) - np.take_along_axis(r_star, j[:, 3], 1)
    if flip_prob is not None:
        anti = (rng.random((seeds, pairs)) < sigmoid(-d_adv)).astype(float)
        rel = (rng.random((seeds, pairs)) < sigmoid(d_adv)).astype(float)
        use_anti = rng.random((seeds, pairs)) < flip_prob
        y[:, 3] = np.where(use_anti, anti, rel)
    else:
        ba = float(beta_adv)
        y[:, 3] = (rng.random((seeds, pairs)) < sigmoid(ba * d_adv)).astype(float)
    return y


def run_with_stochastic_adv(
    *,
    seeds: int,
    steps: int,
    target_rms: float,
    consensus_coef: float,
    beta_adv: float | None,
    flip_prob: float | None,
    seed: int,
    n_seg: int = 500,
    q: float = 0.0,
    pairs: int = 256,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    coef_max_delta: Optional[float] = DEFAULT_COEF_MAX_DELTA,
) -> tuple[np.ndarray, np.ndarray]:
    from synthetic_shared_core import init_theta0

    rng = np.random.default_rng(seed)
    k, T, d = 4, 50, 16

    theta_star = rng.normal(size=(seeds, d))
    theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
    states = rng.normal(size=(seeds, n_seg, T, d))
    r_star = np.tanh(np.einsum("sntd,sd->snt", states, theta_star)).sum(2)
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i, j = sample_expert_pairs(rng, seeds=seeds, k=k, n_seg=n_seg, pairs=pairs, q=q)
    y = make_partial_adv_labels(
        rng, r_star=r_star, i=i, j=j, beta_adv=beta_adv, flip_prob=flip_prob
    )

    consensus_target = y.mean(1)
    theta0 = init_theta0(
        target_rms, seeds=seeds, d=d, rng=rng, n_seg=n_seg, T=T, use_tanh=False
    )

    import torch
    import torch.nn.functional as F
    from synthetic_shared_core import _segment_returns, get_device, rowwise_corr

    device = get_device()
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    i_t = torch.as_tensor(i, dtype=torch.long, device=device)
    j_t = torch.as_tensor(j, dtype=torch.long, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    y_bar = torch.as_tensor(consensus_target, dtype=torch.float32, device=device)
    theta = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float32, device=device))
    alpha = torch.nn.Parameter(torch.full((seeds, k), 0.01, device=device))

    for _ in range(steps):
        if theta.grad is not None:
            theta.grad = None
        if alpha.grad is not None:
            alpha.grad = None
        R = _segment_returns(states_t, theta, False)
        delta = R.gather(1, i_t.reshape(seeds, -1)).reshape(seeds, k, pairs) - R.gather(
            1, j_t.reshape(seeds, -1)
        ).reshape(seeds, k, pairs)
        trust = torch.tanh(alpha)
        denom = trust.abs().amax(1, keepdim=True).clamp_min(1e-12).detach()
        coef = trust / denom
        coef_before = coef.detach().clone()
        w = (k * trust.abs() / trust.abs().sum(1, keepdim=True).clamp_min(1e-12)).detach()
        logits_R = coef.detach().unsqueeze(2) * delta
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y_t, reduction="none")
        loss_R = (w.unsqueeze(2) * bce_R).mean(dim=(1, 2)).sum()
        if consensus_coef > 0:
            loss_R = loss_R + consensus_coef * F.binary_cross_entropy_with_logits(
                delta.mean(dim=1), y_bar, reduction="none"
            ).mean(dim=1).sum()
        logits_A = coef.unsqueeze(2) * delta.detach()
        loss_A = F.binary_cross_entropy_with_logits(logits_A, y_t, reduction="none").mean(
            dim=(1, 2)
        ).sum()
        (loss_R + loss_A).backward()
        with torch.no_grad():
            g = theta.grad
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g.mul_(torch.clamp(10.0 / gn, max=1.0))
            theta.data.sub_(lr_theta * g)
            alpha.data.sub_(lr_alpha * alpha.grad)
        clamp_coef_after_step(alpha, coef_before, coef_max_delta)

    with torch.no_grad():
        R = _segment_returns(states_t, theta, False).cpu().numpy()
        trust = torch.tanh(alpha)
        abar = (trust / trust.abs().amax(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    rho = rowwise_corr(R, r_star)
    return rho, abar


def run_partial_adversary_data(
    out_dir: str,
    seeds: int,
    steps: int,
    overwrite: bool,
    *,
    coef_max_delta: float = DEFAULT_COEF_MAX_DELTA,
) -> str:
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    rows = []
    for si, (sname, skw) in enumerate(SETTINGS):
        for mname, mkw in METHODS:
            rho, abar = run_with_stochastic_adv(
                seeds=seeds,
                steps=steps,
                seed=partial_adversary_job_seed(si, mname),
                n_seg=500,
                q=0.0,
                coef_max_delta=coef_max_delta,
                **skw,
                **mkw,
            )
            rows.append(
                {
                    "setting": sname,
                    "method": mname,
                    "correct": float((rho > 0.5).mean()),
                    "mean_rho": float(rho.mean()),
                    "abar_R": float(abar[:, :3].mean()),
                    "abar_A": float(abar[:, 3].mean()),
                }
            )
            print(
                f"[partial] {sname:12s} {mname:10s} correct={rows[-1]['correct']:.3f} "
                f"aR={rows[-1]['abar_R']:+.2f} aA={rows[-1]['abar_A']:+.2f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "partial_adversary_shared.csv"), index=False)
    print(f"OUT partial adversary data: {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/synthetic_partial_adversary")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    run_partial_adversary_data(
        args.out_dir,
        args.seeds,
        args.steps,
        args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )


if __name__ == "__main__":
    main()
