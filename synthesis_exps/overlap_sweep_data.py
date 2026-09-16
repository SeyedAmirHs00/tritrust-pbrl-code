"""Generate Overlap counterfactual CSV (no plotting).

Example:
  python overlap_sweep_data.py --seeds 120 --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import Optional, Tuple

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
    k, m = y.shape[1], y.shape[2]
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
) -> np.ndarray:
    seeds, _, _, d = states.shape
    k, m = y.shape[1], y.shape[2]
    theta = torch.nn.Parameter(0.01 * torch.randn(seeds, d, device=states.device))
    s = torch.nn.Parameter(torch.ones(seeds, device=states.device))
    q_raw = torch.nn.Parameter(torch.zeros(seeds, k, device=states.device))
    opt = torch.optim.SGD([theta, s, q_raw], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        R = _returns(states, theta)
        std = R.std(1, keepdim=True).clamp_min(1e-6)
        R = (R - R.mean(1, keepdim=True)) / std
        q = torch.sigmoid(q_raw)
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

    with torch.no_grad():
        R = _returns(states, theta)
        std = R.std(1, keepdim=True).clamp_min(1e-6)
        R = (R - R.mean(1, keepdim=True)) / std
        return R.cpu().numpy()


def run_overlap_data(
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

    k, n_total, T, d, m = 4, 500, 50, 16, 256
    bs = n_total // k
    qs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    methods = ("ttp", "no_alpha", "ds_sym")
    rows = []
    device = get_device()
    n_jobs = len(qs) * len(methods)
    t0 = time.perf_counter()
    print(
        f"[overlap] device={device} seeds={seeds} steps={steps} "
        f"qs={qs} methods={list(methods)} jobs={n_jobs} "
        f"coef_max_delta={coef_max_delta} out_dir={out_dir}",
        flush=True,
    )

    job = 0
    for qi, q in enumerate(qs):
        print(
            f"[overlap] q={q:<4g} ({qi + 1}/{len(qs)}) building data "
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

        betas = np.array([1.0, 1.0, 1.0, -1.0])
        y_np = np.zeros((seeds, k, m))
        for e in range(k):
            dstar = np.take_along_axis(r_star, i_all_np[:, e], 1) - np.take_along_axis(
                r_star, j_all_np[:, e], 1
            )
            y_np[:, e] = (rng.random((seeds, m)) < sigmoid_np(betas[e] * dstar)).astype(float)

        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        i_all = torch.as_tensor(i_all_np, dtype=torch.long, device=device)
        j_all = torch.as_tensor(j_all_np, dtype=torch.long, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)

        for method in methods:
            job += 1
            print(
                f"[overlap] [{job}/{n_jobs}] q={q:<4g} {method:8s} training ({steps} steps) ...",
                flush=True,
            )
            t_job = time.perf_counter()
            if method == "ttp":
                R, abar = train_ttp(
                    states, i_all, j_all, y, steps=steps, fix_alpha=False, coef_max_delta=coef_max_delta
                )
                aA = float(abar[:, -1].mean())
            elif method == "no_alpha":
                R, abar = train_ttp(
                    states, i_all, j_all, y, steps=steps, fix_alpha=True, coef_max_delta=coef_max_delta
                )
                aA = float("nan")
            else:
                R = train_ds_sym(states, i_all, j_all, y, steps=steps)
                aA = float("nan")

            rho = rowwise_corr(R, r_star)
            glob = np.abs(rho)
            locals_ = [np.abs(rowwise_corr(R[:, blk], r_star[:, blk])) for blk in blocks]
            loc = np.mean(np.stack(locals_, 0), 0)
            row = {
                "q": q,
                "method": method,
                "global_med": float(np.mean(glob)),
                "global_q25": float(np.percentile(glob, 25)),
                "global_q75": float(np.percentile(glob, 75)),
                "signed_med": float(np.mean(rho)),
                "signed_q25": float(np.percentile(rho, 25)),
                "signed_q75": float(np.percentile(rho, 75)),
                "local_med": float(np.mean(loc)),
                "correct": float((rho > 0.5).mean()),
                "mean_abar_A": aA,
            }
            rows.append(row)
            dt = time.perf_counter() - t_job
            elapsed = time.perf_counter() - t0
            print(
                f"[overlap] [{job}/{n_jobs}] q={q:<4g} {method:8s} "
                f"|rho|_med={row['global_med']:.3f} rho_med={row['signed_med']:+.3f} "
                f"correct={row['correct']:.3f}  ({dt:.1f}s job, {elapsed:.1f}s total)",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "overlap_shared.csv"), index=False)
    elapsed = time.perf_counter() - t0
    print(f"[overlap] done in {elapsed:.1f}s → {out_dir}", flush=True)
    print(f"OUT overlap data: {out_dir}", flush=True)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/synthetic_overlap_sweep")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--coef-max-delta",
        type=float,
        default=DEFAULT_COEF_MAX_DELTA,
        help="Limit per-expert coef change after each step (default: 0 disables).",
    )
    args = p.parse_args()

    run_overlap_data(
        args.out_dir,
        args.seeds,
        args.steps,
        args.overwrite,
        coef_max_delta=args.coef_max_delta,
    )


if __name__ == "__main__":
    main()
