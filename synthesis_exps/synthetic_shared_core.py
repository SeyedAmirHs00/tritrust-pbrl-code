"""
Shared-head synthetic core (PyTorch autograd).

Mean TTP loss over experts and pairs (proper /K scaling via .mean()).
Detached max-norm denom + detached w_k on the reward path; trust path unweighted.
Optional consensus majority anchor on the reward path only (disabled by default).

Defaults match the disjoint robotics regime: n_seg=500, q=0 (no shared pairs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def rowwise_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    num = (x * y).sum(axis=1)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1)) + eps
    return num / den


# Back-compat aliases used by runners
sigmoid = sigmoid_np

DEFAULT_COEF_MAX_DELTA = 0.0  # <=0 disables per-step trust coef clamp


def get_device() -> torch.device:
    """Prefer CUDA when available; otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clamp_coef_after_step(
    alpha: torch.Tensor,
    coef_before: torch.Tensor,
    coef_max_delta: Optional[float],
    *,
    use_alpha_tanh: bool = True,
    use_maxnorm: bool = True,
) -> None:
    """Limit per-expert coef change after one optimizer step.

    Reconstructs the same coef map used in the TTP loss (optional ``tanh`` and
    max-norm), clamps the step in coef-space, then writes the matching ``alpha``.
    """
    if coef_max_delta is None or coef_max_delta <= 0:
        return
    with torch.no_grad():
        trust = torch.tanh(alpha) if use_alpha_tanh else alpha
        if use_maxnorm:
            if trust.dim() == 1:
                scale = trust.abs().amax().clamp_min(1e-12)
            else:
                scale = trust.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            coef_after = trust / scale
        else:
            scale = None
            coef_after = trust
        coef_target = coef_before + (coef_after - coef_before).clamp(
            -float(coef_max_delta), float(coef_max_delta)
        )
        trust_target = coef_target * scale if use_maxnorm else coef_target
        if use_alpha_tanh:
            trust_target = trust_target.clamp(-0.999999, 0.999999)
            alpha.copy_(torch.atanh(trust_target))
        else:
            alpha.copy_(trust_target)


# Paper default Standard init: calibrate θ so rms|ΔR|_0 ≈ 1.4.
DEFAULT_STANDARD_TARGET_RMS = 1.4


@dataclass(frozen=True)
class SharedVariant:
    name: str
    label: str
    # None => DEFAULT_STANDARD_TARGET_RMS; 0 => θ=0 (Stabilized); >0 => calibrated rms|ΔR|.
    target_rms: Optional[float] = None
    consensus_coef: float = 0.0
    use_tanh: bool = True  # reward head: R = sum tanh(θ^T s)
    use_alpha_tanh: bool = True  # trust path: α ↦ tanh(α)
    use_maxnorm: bool = True
    use_confidence_weights: bool = True
    detach_maxnorm: bool = True
    detach_weights: bool = True


SHARED_BRANCH_VARIANTS = (
    # Standard: rms|ΔR|_0 ≈ 1.4. Stabilized: θ=0.
    SharedVariant(
        "standard",
        "Standard",
        target_rms=DEFAULT_STANDARD_TARGET_RMS,
        consensus_coef=0.0,
        use_tanh=False,
    ),
    SharedVariant("stabilized", "Stabilized", target_rms=0.0, consensus_coef=0.0, use_tanh=False),
)


def init_theta0(
    target_rms: Optional[float],
    *,
    seeds: int,
    d: int,
    rng: np.random.Generator,
    n_seg: int = 500,
    T: int = 50,
    use_tanh: bool = True,
    theta_scale: Optional[float] = None,
    cal_rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Initialize reward weights θ.

    - ``theta_scale`` set: explicit Gaussian scale / √d (init-scale sweep).
    - ``target_rms is None``: ``DEFAULT_STANDARD_TARGET_RMS`` (1.4).
    - ``target_rms <= 0``: zeros (Stabilized).
    - ``target_rms > 0``: Gaussian calibrated so rms|ΔR| ≈ target.
    """
    if theta_scale is not None:
        if theta_scale == 0.0:
            return np.zeros((seeds, d), dtype=np.float64)
        return rng.normal(scale=theta_scale / np.sqrt(d), size=(seeds, d))

    rms = DEFAULT_STANDARD_TARGET_RMS if target_rms is None else float(target_rms)
    if rms <= 0:
        return np.zeros((seeds, d), dtype=np.float64)

    crng = cal_rng if cal_rng is not None else rng
    scale = calibrate_theta_scale(
        rms,
        seeds=min(40, seeds),
        n_seg=n_seg,
        T=T,
        d=d,
        rng=crng,
        use_tanh=use_tanh,
    )
    if scale == 0.0:
        return np.zeros((seeds, d), dtype=np.float64)
    return rng.normal(scale=scale / np.sqrt(d), size=(seeds, d))


def calibrate_theta_scale(
    target_rms: float,
    *,
    seeds: int,
    n_seg: int,
    T: int,
    d: int,
    rng: np.random.Generator,
    use_tanh: bool = True,
) -> float:
    if target_rms <= 0:
        return 0.0

    def measure(scale: float) -> float:
        th = rng.normal(scale=scale / np.sqrt(d), size=(seeds, d))
        states = rng.normal(size=(seeds, n_seg, T, d))
        pre = np.einsum("sntd,sd->snt", states, th)
        R = (np.tanh(pre) if use_tanh else pre).sum(axis=2)
        i = rng.integers(0, n_seg, size=(seeds, 512))
        j = rng.integers(0, n_seg, size=(seeds, 512))
        dR = np.take_along_axis(R, i, axis=1) - np.take_along_axis(R, j, axis=1)
        return float(np.sqrt(np.mean(dR**2)))

    # With tanh, rms saturates near ~√(2T); without, scale grows roughly linearly.
    lo, hi = 1e-4, 20.0 if use_tanh else max(20.0, 2.0 * target_rms)
    # Expand hi until we bracket the target (or hit a cap).
    for _ in range(16):
        if measure(hi) >= target_rms:
            break
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if measure(mid) < target_rms:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _segment_returns(states: torch.Tensor, theta: torch.Tensor, use_tanh: bool) -> torch.Tensor:
    # states: [S,N,T,D], theta: [S,D] -> R: [S,N]
    pre = torch.einsum("sntd,sd->snt", states, theta)
    r = torch.tanh(pre) if use_tanh else pre
    return r.sum(dim=2)


def sample_expert_pairs(
    rng: np.random.Generator,
    *,
    seeds: int,
    k: int,
    n_seg: int,
    pairs: int,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (i, j) indices per expert with shared-pair fraction q.

    Returns
    -------
    i_all, j_all : (seeds, k, pairs)
    """
    q = float(np.clip(q, 0.0, 1.0))
    n_shared = int(round(q * pairs))
    n_priv = pairs - n_shared
    bs = n_seg // k
    if bs < 2:
        raise ValueError(f"n_seg={n_seg} too small for k={k} disjoint blocks")
    blocks = [np.arange(b * bs, (b + 1) * bs) for b in range(k)]

    i_all = np.zeros((seeds, k, pairs), dtype=np.int64)
    j_all = np.zeros((seeds, k, pairs), dtype=np.int64)

    if n_shared > 0:
        i_s = rng.integers(0, n_seg, (seeds, n_shared))
        j_s = rng.integers(0, n_seg, (seeds, n_shared))
        same = i_s == j_s
        while same.any():
            j_s[same] = rng.integers(0, n_seg, int(same.sum()))
            same = i_s == j_s
        i_all[:, :, :n_shared] = i_s[:, None, :]
        j_all[:, :, :n_shared] = j_s[:, None, :]

    for e, blk in enumerate(blocks):
        if n_priv <= 0:
            continue
        i_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
        j_p = rng.choice(blk, size=(seeds, n_priv), replace=True)
        same = i_p == j_p
        while same.any():
            j_p[same] = rng.choice(blk, size=int(same.sum()), replace=True)
            same = i_p == j_p
        i_all[:, e, n_shared:] = i_p
        j_all[:, e, n_shared:] = j_p

    return i_all, j_all


def run_shared_variant(
    betas: Tuple[float, ...],
    variant: SharedVariant,
    *,
    seeds: int,
    n_seg: int = 500,
    T: int = 50,
    d: int = 16,
    pairs: int = 256,
    q: float = 0.0,
    steps: int = 400,
    lr_theta: float = 0.05,
    lr_alpha: float = 0.005,
    alpha_init: float = 0.01,
    seed: int = 0,
    theta_scale: Optional[float] = None,
    cal_rng: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
    coef_max_delta: Optional[float] = DEFAULT_COEF_MAX_DELTA,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns
    -------
    rho : (seeds,) corr(R_hat, R*)
    alpha_bar : (seeds, K)
    init_rms : float empirical init rms|Delta R|
    """
    device = device or get_device()
    rng = np.random.default_rng(seed)
    k = len(betas)
    b = np.asarray(betas, dtype=np.float64)

    theta_star = rng.normal(size=(seeds, d))
    theta_star /= np.linalg.norm(theta_star, axis=1, keepdims=True) + 1e-12
    states_np = rng.normal(size=(seeds, n_seg, T, d))

    r_star = np.tanh(np.einsum("sntd,sd->snt", states_np, theta_star)).sum(2)
    r_star = (r_star - r_star.mean(1, keepdims=True)) / (r_star.std(1, keepdims=True) + 1e-12)

    i_np, j_np = sample_expert_pairs(
        rng, seeds=seeds, k=k, n_seg=n_seg, pairs=pairs, q=q
    )

    y_np = np.zeros((seeds, k, pairs), dtype=np.float64)
    for e in range(k):
        d_star = np.take_along_axis(r_star, i_np[:, e], 1) - np.take_along_axis(
            r_star, j_np[:, e], 1
        )
        y_np[:, e] = (rng.random((seeds, pairs)) < sigmoid_np(b[e] * d_star)).astype(np.float64)
    consensus_np = y_np.mean(1)

    crng = cal_rng if cal_rng is not None else np.random.default_rng(seed + 7)
    theta0 = init_theta0(
        variant.target_rms,
        seeds=seeds,
        d=d,
        rng=rng,
        n_seg=n_seg,
        T=T,
        use_tanh=variant.use_tanh,
        theta_scale=theta_scale,
        cal_rng=crng,
    )

    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    i = torch.as_tensor(i_np, dtype=torch.long, device=device)  # [S,K,P]
    j = torch.as_tensor(j_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    y_bar = torch.as_tensor(consensus_np, dtype=torch.float32, device=device)

    theta = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float32, device=device))
    alpha = torch.nn.Parameter(torch.full((seeds, k), float(alpha_init), device=device))

    with torch.no_grad():
        R0 = _segment_returns(states, theta, variant.use_tanh)
        # init rms over all expert pairs
        d0 = R0.gather(1, i.reshape(seeds, -1)) - R0.gather(1, j.reshape(seeds, -1))
        init_rms = float(torch.sqrt((d0**2).mean()).item())

    for _ in range(steps):
        if theta.grad is not None:
            theta.grad = None
        if alpha.grad is not None:
            alpha.grad = None

        R = _segment_returns(states, theta, variant.use_tanh)

        if variant.use_alpha_tanh:
            trust = torch.tanh(alpha)
        else:
            trust = alpha

        abs_t = trust.abs()
        if variant.use_maxnorm:
            denom = abs_t.amax(dim=1, keepdim=True).clamp_min(1e-12)
            if variant.detach_maxnorm:
                denom = denom.detach()
            coef = trust / denom
        else:
            coef = trust

        coef_before = coef.detach().clone()

        if variant.use_confidence_weights:
            w = k * abs_t / abs_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
            if variant.detach_weights:
                w = w.detach()
        else:
            w = torch.ones(seeds, k, device=device)

        # Per-expert deltas: [S,K,P]
        delta = R.gather(1, i.reshape(seeds, -1)).reshape(seeds, k, pairs) - R.gather(
            1, j.reshape(seeds, -1)
        ).reshape(seeds, k, pairs)

        # Reward path: hold coef (and usually w) fixed.
        logits_R = coef.detach().unsqueeze(2) * delta  # [S,K,P]
        bce_R = F.binary_cross_entropy_with_logits(logits_R, y, reduction="none")
        loss_R = (w.unsqueeze(2) * bce_R).mean(dim=(1, 2)).sum()

        if variant.consensus_coef > 0.0:
            # Majority anchor on shared-index positions (meaningful mainly at q≈1).
            loss_R = loss_R + variant.consensus_coef * F.binary_cross_entropy_with_logits(
                delta.mean(dim=1), y_bar, reduction="none"
            ).mean(dim=1).sum()

        # Trust path: unweighted; hold delta fixed so only alpha gets this term
        logits_A = coef.unsqueeze(2) * delta.detach()
        if variant.detach_weights or not variant.use_confidence_weights:
            loss_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none").mean(
                dim=(1, 2)
            ).sum()
        else:
            bce_A = F.binary_cross_entropy_with_logits(logits_A, y, reduction="none")
            loss_A = (w.unsqueeze(2) * bce_A).mean(dim=(1, 2)).sum()

        (loss_R + loss_A).backward()

        with torch.no_grad():
            g = theta.grad
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            g.mul_(torch.clamp(10.0 / gn, max=1.0))
            theta.data.sub_(lr_theta * g)
            alpha.data.sub_(lr_alpha * alpha.grad)
        clamp_coef_after_step(
            alpha,
            coef_before,
            coef_max_delta,
            use_alpha_tanh=variant.use_alpha_tanh,
            use_maxnorm=variant.use_maxnorm,
        )

    with torch.no_grad():
        R = _segment_returns(states, theta, variant.use_tanh).cpu().numpy()
        if variant.use_alpha_tanh:
            trust = torch.tanh(alpha)
        else:
            trust = alpha
        if variant.use_maxnorm:
            abar = (trust / trust.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
        else:
            abar = trust.cpu().numpy()

    return rowwise_corr(R, r_star), abar, init_rms


def build_k4_configs() -> Dict[str, Tuple[float, ...]]:
    return {
        "3R1N": (1.0, 1.0, 1.0, 0.0),
        "3R1A": (1.0, 1.0, 1.0, -1.0),
        "2R1A1N": (1.0, 1.0, -1.0, 0.0),
        "1R3A": (1.0, -1.0, -1.0, -1.0),
    }