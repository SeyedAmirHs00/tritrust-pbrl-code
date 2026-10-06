"""Generate Overlap counterfactual CSV with DS-Sym initialized at q=1, s=1, theta=0.

Supports custom expert rationality mixtures, defaulting to 2R2N1A:
  betas = [1.0, 1.0, 0.0, 0.0, -1.0]
    - 2 Reliable experts (beta = 1.0)
    - 2 Noisy / random experts (beta = 0.0)
    - 1 Adversarial expert (beta = -1.0)

Example:
  python overlap_ds_init_sweep_data.py --seeds 120 --overwrite
  python overlap_ds_init_sweep_data.py --betas 1 1 0 0 -1 --seeds 120 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from synthetic_shared_core import (
    DEFAULT_COEF_MAX_DELTA,
    clamp_coef_after_step,
    get_device,
    rowwise_corr,
    sigmoid_np,
)

DEFAULT_BETAS = (1.0, 1.0, 0.0, 0.0, -1.0)


def _returns(states: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return torch.tanh(torch.einsum("sntd,sd->snt", states, theta)).sum(dim=2)


def train_ttp(
    states: torch.Tensor,
    i_all: torch.Tensor,
    j_all: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    fix_alpha: bool = False,
    coef_max_delta: Optional[float] = DEFAULT_COEF_MAX_DELTA,
) -> Tuple[np.ndarray, np.ndarray]:
    seeds, n_total, T, d = states.shape
    k = y.shape[1]
    theta = torch.nn.Parameter(torch.zeros(seeds, d, device=states.device))
    if fix_alpha:
        alpha_param = None
    else:
        alpha_param = torch.nn.Parameter(torch.full((seeds, k), alpha_init, device=states.device))

    for _ in range(steps):
        if theta.grad is not None:
            theta.grad = None
        if alpha_param is not None and alpha_param.grad is not None:
            alpha_param.grad = None

        R = _returns(states, theta)
        if fix_alpha:
            coef = torch.ones(seeds, k, device=states.device)
            w = torch.ones(seeds, k, device=states.device)
            coef_before = None
        else:
            trust = torch.tanh(alpha_param)
            denom = trust.abs().amax(1, keepdim=True).clamp_min(1e-12).detach()
            coef = trust / denom
            coef_before = coef.detach().clone()
            w = (k * trust.abs() / trust.abs().sum(1, keepdim=True).clamp_min(1e-12)).detach()

        loss_R = 0.0
        loss_A = 0.0
        for e in range(k):
            ie, je = i_all[:, e], j_all[:, e]
            delta = R.gather(1, ie) - R.gather(1, je)
            logits_R = coef[:, e : e + 1].detach() * delta
            bce = F.binary_cross_entropy_with_logits(logits_R, y[:, e], reduction="none")
            loss_R = loss_R + (w[:, e : e + 1] * bce).mean(dim=1).sum() / k
            if not fix_alpha:
                logits_A = coef[:, e : e + 1] * delta.detach()
                loss_A = loss_A + F.binary_cross_entropy_with_logits(
                    logits_A, y[:, e], reduction="none"
                ).mean(dim=1).sum() / k

        (loss_R + loss_A).backward()
        with torch.no_grad():
            g = theta.grad
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g.mul_(torch.clamp(10.0 / gn, max=1.0))
            theta.data.sub_(lr_theta * g)
            if alpha_param is not None:
                alpha_param.data.sub_(lr_alpha * alpha_param.grad)
        if alpha_param is not None and coef_before is not None:
            clamp_coef_after_step(alpha_param, coef_before, coef_max_delta)

    with torch.no_grad():
        R = _returns(states, theta).cpu().numpy()
        if fix_alpha:
            abar = np.ones((seeds, k))
        else:
            trust = torch.tanh(alpha_param)
            abar = (trust / trust.abs().amax(1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
    return R, abar


def train_ds_sym(
    states: torch.Tensor,
    i_all: torch.Tensor,
    j_all: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float = 0.05,
    q_init: float = 1.0,
    s_init: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train Dawid-Skene Symmetric model.

    Initialized with theta = 0, scale s = s_init, and reliability q = q_init.
    q is parameterized directly in [0, 1] and clamped after each SGD step.
    """
    seeds, _, _, d = states.shape
    k = y.shape[1]
    # theta initialized to 0
    theta = torch.nn.Parameter(torch.zeros(seeds, d, device=states.device))
    # scale parameter initialized to s_init (default 1.0)
    s = torch.nn.Parameter(torch.full((seeds,), s_init, device=states.device))
    # expert reliability parameterized directly in [0, 1], initialized to q_init (default 1.0)
    q = torch.nn.Parameter(torch.full((seeds, k), q_init, device=states.device))
    opt = torch.optim.SGD([theta, s, q], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        R = _returns(states, theta)
        # Use variance + eps to ensure finite, non-NaN gradients when theta == 0
        var = torch.var(R, dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-12).clamp_min(1e-6)
        R = (R - R.mean(1, keepdim=True)) / std

        loss = 0.0
        for e in range(k):
            ie, je = i_all[:, e], j_all[:, e]
            delta = R.gather(1, ie) - R.gather(1, je)
            p_z = torch.sigmoid(s.unsqueeze(1) * delta)
            p_y = q[:, e : e + 1] * p_z + (1.0 - q[:, e : e + 1]) * (1.0 - p_z)
            p_y = p_y.clamp(1e-6, 1.0 - 1e-6)
            loss = loss + F.binary_cross_entropy(p_y, y[:, e], reduction="none").mean(dim=1).sum() / k
        loss.backward()
        with torch.no_grad():
            if theta.grad is not None:
                gn = theta.grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
                theta.grad.mul_(torch.clamp(10.0 / gn, max=1.0))
        opt.step()
        # Direct projection of reliability q into [0, 1]
        with torch.no_grad():
            q.data.clamp_(0.0, 1.0)

    with torch.no_grad():
        R = _returns(states, theta)
        var = torch.var(R, dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-12).clamp_min(1e-6)
        R = (R - R.mean(1, keepdim=True)) / std
        return R.cpu().numpy(), q.cpu().numpy()


def run_overlap_data(
    out_dir: str,
    seeds: int,
    steps: int,
    overwrite: bool,
    *,
    betas: Sequence[float] = DEFAULT_BETAS,
    coef_max_delta: float = DEFAULT_COEF_MAX_DELTA,
    q_init: float = 1.0,
    s_init: float = 1.0,
    lr_ds: float = 0.05,
) -> str:
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    betas_np = np.asarray(betas, dtype=float)
    k = len(betas_np)
    n_total, T, d, m = 500, 50, 16, 256
    bs = n_total // k

    # Expert masks according to rationality
    rel_mask = betas_np == 1.0
    noisy_mask = betas_np == 0.0
    adv_mask = betas_np == -1.0
    n_r, n_n, n_a = int(rel_mask.sum()), int(noisy_mask.sum()), int(adv_mask.sum())
    setting_name = f"{n_r}R{n_n}N{n_a}A"

    qs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    methods = ("ttp", "no_alpha", "ds_sym")
    rows = []
    device = get_device()
    n_jobs = len(qs) * len(methods)
    t0 = time.perf_counter()
    print(
        f"[overlap_ds_init] device={device} seeds={seeds} steps={steps} "
        f"setting={setting_name} betas={betas_np.tolist()} k={k} "
        f"qs={qs} methods={list(methods)} jobs={n_jobs} "
        f"q_init={q_init} s_init={s_init} theta_init=0 "
        f"coef_max_delta={coef_max_delta} out_dir={out_dir}",
        flush=True,
    )

    job = 0
    for qi, q in enumerate(qs):
        print(
            f"[overlap_ds_init] q={q:<4g} ({qi + 1}/{len(qs)}) building data "
            f"(n_shared≈{int(round(q * m))}/{m}) ...",
            flush=True,
        )
        rng = np.random.default_rng(7000 + qi)
        theta_star = rng.normal(size=(seeds, d))
        theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
        states_np = rng.normal(size=(seeds, n_total, T, d))
        r_star = np.tanh(np.einsum("sntd,sd->snt", states_np, theta_star)).sum(2)
        r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

        blocks = [np.arange(b * bs, (b + 1) * bs) for b in range(k)]
        n_shared = int(round(q * m))
        n_priv = m - n_shared

        i_all_np = np.zeros((seeds, k, m), dtype=np.int64)
        j_all_np = np.zeros((seeds, k, m), dtype=np.int64)
        if n_shared > 0:
            i_s = rng.integers(0, n_total, (seeds, n_shared))
            j_s = rng.integers(0, n_total, (seeds, n_shared))
            same = i_s == j_s
            while same.any():
                j_s[same] = rng.integers(0, n_total, int(same.sum()))
                same = i_s == j_s
            i_all_np[:, :, :n_shared] = i_s[:, None, :]
            j_all_np[:, :, :n_shared] = j_s[:, None, :]
        for e, blk in enumerate(blocks):
            if n_priv <= 0:
                continue
            i_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
            j_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
            same = i_p == j_p
            while same.any():
                j_p[same] = rng.choice(blk, size=int(same.sum()), replace=True)
                same = i_p == j_p
            i_all_np[:, e, n_shared:] = i_p
            j_all_np[:, e, n_shared:] = j_p

        y_np = np.zeros((seeds, k, m))
        for e in range(k):
            dstar = np.take_along_axis(r_star, i_all_np[:, e], 1) - np.take_along_axis(
                r_star, j_all_np[:, e], 1
            )
            y_np[:, e] = (rng.random((seeds, m)) < sigmoid_np(betas_np[e] * dstar)).astype(float)

        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        i_all = torch.as_tensor(i_all_np, dtype=torch.long, device=device)
        j_all = torch.as_tensor(j_all_np, dtype=torch.long, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)

        for method in methods:
            job += 1
            print(
                f"[overlap_ds_init] [{job}/{n_jobs}] q={q:<4g} {method:8s} training ({steps} steps) ...",
                flush=True,
            )
            t_job = time.perf_counter()
            abar: Optional[np.ndarray] = None
            q_final: Optional[np.ndarray] = None

            if method == "ttp":
                R, abar = train_ttp(
                    states, i_all, j_all, y, steps=steps, fix_alpha=False, coef_max_delta=coef_max_delta
                )
            elif method == "no_alpha":
                R, abar = train_ttp(
                    states, i_all, j_all, y, steps=steps, fix_alpha=True, coef_max_delta=coef_max_delta
                )
            else:
                R, q_final = train_ds_sym(
                    states,
                    i_all,
                    j_all,
                    y,
                    steps=steps,
                    lr=lr_ds,
                    q_init=q_init,
                    s_init=s_init,
                )

            # Alignment metrics
            rho = rowwise_corr(R, r_star)
            glob = np.abs(rho)
            locals_ = [np.abs(rowwise_corr(R[:, blk], r_star[:, blk])) for blk in blocks]
            loc = np.mean(np.stack(locals_, 0), 0)

            # Group trust values (for TTP abar in [-1, 1])
            if abar is not None:
                mean_trust_rel = float(abar[:, rel_mask].mean()) if n_r > 0 else float("nan")
                mean_trust_noisy = float(abar[:, noisy_mask].mean()) if n_n > 0 else float("nan")
                mean_trust_adv = float(abar[:, adv_mask].mean()) if n_a > 0 else float("nan")
            else:
                mean_trust_rel = float("nan")
                mean_trust_noisy = float("nan")
                mean_trust_adv = float("nan")

            # Group reliability values (for DS-Sym q in [0, 1])
            if q_final is not None:
                mean_rel_rel = float(q_final[:, rel_mask].mean()) if n_r > 0 else float("nan")
                mean_rel_noisy = float(q_final[:, noisy_mask].mean()) if n_n > 0 else float("nan")
                mean_rel_adv = float(q_final[:, adv_mask].mean()) if n_a > 0 else float("nan")
            else:
                mean_rel_rel = float("nan")
                mean_rel_noisy = float("nan")
                mean_rel_adv = float("nan")

            row = {
                # Setup metadata
                "overlap_ratio_q": q,
                "method": method,
                "setting": setting_name,
                "betas": str(betas_np.tolist()),
                # Primary correlation metrics
                "signed_corr_median": float(np.mean(rho)),
                "signed_corr_q25": float(np.percentile(rho, 25)),
                "signed_corr_q75": float(np.percentile(rho, 75)),
                "abs_corr_median": float(np.mean(glob)),
                "abs_corr_q25": float(np.percentile(glob, 25)),
                "abs_corr_q75": float(np.percentile(glob, 75)),
                "local_corr_median": float(np.mean(loc)),
                "correct_branch_rate": float((rho > 0.5).mean()),
                # TTP Trust parameters (abar in [-1, 1])
                "mean_trust_reliable": mean_trust_rel,
                "mean_trust_noisy": mean_trust_noisy,
                "mean_trust_adversary": mean_trust_adv,
                # DS-Sym Reliability parameters (q in [0, 1])
                "mean_reliability_reliable": mean_rel_rel,
                "mean_reliability_noisy": mean_rel_noisy,
                "mean_reliability_adversary": mean_rel_adv,
            }

            # Individual per-expert trust and reliability
            for e_idx in range(k):
                b_val = betas_np[e_idx]
                tag = "R" if b_val > 0 else ("N" if b_val == 0 else "A")
                row[f"trust_e{e_idx}_{tag}"] = (
                    float(abar[:, e_idx].mean()) if abar is not None else float("nan")
                )
                row[f"rel_e{e_idx}_{tag}"] = (
                    float(q_final[:, e_idx].mean()) if q_final is not None else float("nan")
                )

            # Compatibility aliases for legacy tools/plots
            row["q"] = q
            row["signed_med"] = row["signed_corr_median"]
            row["signed_q25"] = row["signed_corr_q25"]
            row["signed_q75"] = row["signed_corr_q75"]
            row["global_med"] = row["abs_corr_median"]
            row["global_q25"] = row["abs_corr_q25"]
            row["global_q75"] = row["abs_corr_q75"]
            row["local_med"] = row["local_corr_median"]
            row["correct"] = row["correct_branch_rate"]
            row["mean_abar_A"] = mean_trust_adv
            row["mean_q_A"] = mean_rel_adv

            rows.append(row)
            dt = time.perf_counter() - t_job
            elapsed = time.perf_counter() - t0
            print(
                f"[overlap_ds_init] [{job}/{n_jobs}] q={q:<4g} {method:8s} "
                f"corr_med={row['signed_corr_median']:+.3f} "
                f"correct={row['correct_branch_rate']:.3f} "
                f"trust[R={mean_trust_rel:+.2f},N={mean_trust_noisy:+.2f},A={mean_trust_adv:+.2f}] "
                f"rel[R={mean_rel_rel:.2f},N={mean_rel_noisy:.2f},A={mean_rel_adv:.2f}] "
                f"({dt:.1f}s job, {elapsed:.1f}s total)",
                flush=True,
            )

    table = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "overlap_shared.csv")
    table.to_csv(csv_path, index=False)
    table.to_csv(os.path.join(out_dir, "overlap_ds_init_shared.csv"), index=False)
    elapsed = time.perf_counter() - t0
    print(f"[overlap_ds_init] done in {elapsed:.1f}s → {out_dir}", flush=True)
    print(f"OUT overlap data: {csv_path}", flush=True)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Overlap sweep with DS-Sym init q=1, s=1, theta=0 and custom betas"
    )
    p.add_argument("--out_dir", default="results/synthetic_overlap_ds_init_sweep")
    p.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=list(DEFAULT_BETAS),
        help="Teacher rationality mixture (default: 1.0 1.0 0.0 0.0 -1.0 for 2R2N1A).",
    )
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    p.add_argument("--q-init", type=float, default=1.0, help="Initial DS-Sym expert reliability q (default: 1.0).")
    p.add_argument("--s-init", type=float, default=1.0, help="Initial DS-Sym scale parameter s (default: 1.0).")
    p.add_argument("--lr-ds", type=float, default=0.05, help="Learning rate for DS-Sym (default: 0.05).")
    args = p.parse_args()

    run_overlap_data(
        args.out_dir,
        args.seeds,
        args.steps,
        args.overwrite,
        betas=tuple(args.betas),
        coef_max_delta=args.coef_max_delta,
        q_init=args.q_init,
        s_init=args.s_init,
        lr_ds=args.lr_ds,
    )


if __name__ == "__main__":
    main()
