"""Diagnostics for reward-model preference pairs and trajectory buffers.

Logged quantities
-----------------
rms_delta_r
    rms|ΔR|_0 = sqrt( mean_{(i,j) in D} (R(τ_i) - R(τ_j))^2 )
    where D is the labeled preference-pair buffer and R(τ) is the
    ensemble-mean predicted segment return (sum over timesteps).

median_sq_delta_r
    median_{(i,j) in D} (R(τ_i) - R(τ_j))^2 over the same preference pairs.

corr_r_rstar
    Pearson correlation between per-step learned rewards r̂(s, a) and
    environment rewards r*(s, a) over the trajectory buffer.

corr_segment_r_rstar
    Pearson correlation between segment returns R(τ) = Σ_t r̂(s_t, a_t) and
    R*(τ) = Σ_t r*(s_t, a_t) over all length-``size_segment`` windows in the
    trajectory buffer.

Each row also includes ``phase``:

pre_train
    Logged after preference query sampling, before reward-model training
    (untrained / previous-weights snapshot on the newly sampled buffer).

post_train
    Logged after reward-model training on the sampled preferences.

mean_sa_var / mean_sa_std / mean_sa_second_moment
    Moments of concatenated [obs, action] rows in the reward-model
    trajectory buffer (``inputs``).

mean_state_* / mean_action_*
    Same moments restricted to the state or action slice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

DIAGNOSTICS_CSV_FIELDNAMES: List[str] = [
    "corr_r_rstar",
    "corr_segment_r_rstar",
    "mean_abs_delta_r",
    "mean_action_second_moment",
    "mean_action_std",
    "mean_action_var",
    "mean_sa_second_moment",
    "mean_sa_std",
    "mean_sa_var",
    "mean_state_second_moment",
    "mean_state_std",
    "mean_state_var",
    "median_sq_delta_r",
    "n_corr_segments",
    "n_corr_transitions",
    "n_pairs",
    "n_transitions",
    "phase",
    "rms_delta_r",
    "std_delta_r",
    "step",
    "var_delta_r",
]

DIAGNOSTICS_NUMERIC_FIELDNAMES: List[str] = [
    f for f in DIAGNOSTICS_CSV_FIELDNAMES if f not in {"phase"}
]


def _as_traj_array(traj) -> Optional[np.ndarray]:
    if traj is None or len(traj) == 0:
        return None
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return arr


def collect_sa_from_inputs(inputs: Sequence, ds: int, da: int) -> Optional[np.ndarray]:
    """Flatten reward-model ``inputs`` trajectories into [N, ds+da]."""
    chunks: List[np.ndarray] = []
    expected = ds + da
    for traj in inputs:
        arr = _as_traj_array(traj)
        if arr is None:
            continue
        if arr.shape[-1] != expected:
            # Skip malformed / empty placeholder rows.
            continue
        chunks.append(arr)
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def gather_trajectory_pairs(reward_models: Sequence) -> tuple:
    """Return aligned (sa, r*) trajectory arrays and segment length."""
    for rm in reward_models:
        trajectories = []
        for sa_traj, r_traj in zip(rm.inputs, rm.targets):
            sa = _as_traj_array(sa_traj)
            r = _as_traj_array(r_traj)
            if sa is None or r is None:
                continue
            r = r.reshape(-1)
            if sa.shape[0] != r.shape[0]:
                continue
            trajectories.append((sa, r))
        if trajectories:
            return trajectories, int(rm.size_segment)
    return [], 1


@torch.no_grad()
def _ensemble_predict(
    ensemble: Sequence[torch.nn.Module],
    sa: np.ndarray,
    device: Union[str, torch.device],
    batch_size: int,
) -> np.ndarray:
    """Predict ensemble-mean reward for transitions or segment sums."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    sa = np.asarray(sa, dtype=np.float32)
    if sa.ndim == 2:
        outputs: List[np.ndarray] = []
        for start in range(0, sa.shape[0], batch_size):
            batch = torch.as_tensor(sa[start : start + batch_size], device=device)
            member_preds = []
            for member in ensemble:
                out = member(batch)
                if out.dim() == 3:
                    out = out.squeeze(-1)
                elif out.dim() == 2 and out.shape[-1] == 1:
                    out = out.squeeze(-1)
                member_preds.append(out)
            outputs.append(torch.stack(member_preds, dim=0).mean(dim=0).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    if sa.ndim == 3:
        seg_sums: List[np.ndarray] = []
        for start in range(0, sa.shape[0], batch_size):
            batch = torch.as_tensor(sa[start : start + batch_size], device=device)
            member_preds = []
            for member in ensemble:
                out = member(batch)
                if out.dim() == 3:
                    out = out.squeeze(-1)
                elif out.dim() == 2 and out.shape[-1] == 1:
                    out = out.squeeze(-1)
                member_preds.append(out.sum(dim=1))
            seg_sums.append(torch.stack(member_preds, dim=0).mean(dim=0).cpu().numpy())
        return np.concatenate(seg_sums, axis=0)

    raise ValueError(f"Expected sa with ndim 2 or 3, got shape {sa.shape}")


@torch.no_grad()
def reward_env_correlation_stats(
    ensemble: Sequence[torch.nn.Module],
    trajectories: Sequence[tuple],
    size_segment: int,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 256,
) -> Dict[str, float]:
    """Correlation between learned rewards R and environment rewards R*."""
    if not trajectories:
        return {
            "corr_r_rstar": 0.0,
            "corr_segment_r_rstar": 0.0,
            "n_corr_transitions": 0.0,
            "n_corr_segments": 0.0,
        }

    sa_steps = [sa for sa, _ in trajectories]
    rstar_steps = [r for _, r in trajectories]
    sa_all = np.concatenate(sa_steps, axis=0)
    rstar_all = np.concatenate(rstar_steps, axis=0)
    r_pred_all = _ensemble_predict(ensemble, sa_all, device=device, batch_size=batch_size)

    seg_sa: List[np.ndarray] = []
    seg_rstar: List[float] = []
    for sa, r in trajectories:
        if sa.shape[0] < size_segment:
            continue
        for start in range(0, sa.shape[0] - size_segment + 1):
            seg_sa.append(sa[start : start + size_segment])
            seg_rstar.append(float(r[start : start + size_segment].sum()))

    if seg_sa:
        seg_sa_arr = np.stack(seg_sa, axis=0)
        r_pred_seg = _ensemble_predict(
            ensemble, seg_sa_arr, device=device, batch_size=batch_size
        )
        corr_segment = _pearson_corr(r_pred_seg, np.asarray(seg_rstar, dtype=np.float64))
        n_segments = float(len(seg_rstar))
    else:
        corr_segment = 0.0
        n_segments = 0.0

    return {
        "corr_r_rstar": _pearson_corr(r_pred_all, rstar_all),
        "corr_segment_r_rstar": corr_segment,
        "n_corr_transitions": float(rstar_all.size),
        "n_corr_segments": n_segments,
    }


def sa_moment_stats(x: np.ndarray, ds: int, da: int) -> Dict[str, float]:
    """Compute mean variance / std / second-moment of state, action, and SA."""
    assert x.ndim == 2 and x.shape[1] == ds + da
    obs = x[:, :ds]
    actions = x[:, ds : ds + da]

    def _moments(arr: np.ndarray) -> Dict[str, float]:
        var_per_dim = np.var(arr, axis=0)
        second_per_dim = np.mean(arr ** 2, axis=0)
        std_per_dim = np.sqrt(np.maximum(var_per_dim, 0.0))
        return {
            "mean_var": float(var_per_dim.mean()),
            "mean_std": float(std_per_dim.mean()),
            "mean_second_moment": float(second_per_dim.mean()),
        }

    state_m = _moments(obs)
    action_m = _moments(actions)
    sa_m = _moments(x)
    return {
        "n_transitions": float(x.shape[0]),
        "mean_state_var": state_m["mean_var"],
        "mean_state_std": state_m["mean_std"],
        "mean_state_second_moment": state_m["mean_second_moment"],
        "mean_action_var": action_m["mean_var"],
        "mean_action_std": action_m["mean_std"],
        "mean_action_second_moment": action_m["mean_second_moment"],
        "mean_sa_var": sa_m["mean_var"],
        "mean_sa_std": sa_m["mean_std"],
        "mean_sa_second_moment": sa_m["mean_second_moment"],
    }


@torch.no_grad()
def rms_delta_r_from_pairs(
    ensemble: Sequence[torch.nn.Module],
    seg1: np.ndarray,
    seg2: np.ndarray,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 256,
) -> Dict[str, float]:
    """rms|ΔR|_0 and companion stats over preference pairs (seg1, seg2).

    R(τ) = mean_ensemble sum_t r_θ(s_t, a_t). Alpha / trust scaling is NOT applied.
    """
    n = int(seg1.shape[0])
    if n == 0:
        return {
            "n_pairs": 0.0,
            "rms_delta_r": 0.0,
            "std_delta_r": 0.0,
            "var_delta_r": 0.0,
            "mean_abs_delta_r": 0.0,
            "median_sq_delta_r": 0.0,
        }

    device = torch.device(device) if not isinstance(device, torch.device) else device
    deltas: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s1 = torch.as_tensor(seg1[start:end], dtype=torch.float32, device=device)
        s2 = torch.as_tensor(seg2[start:end], dtype=torch.float32, device=device)

        r1_members = []
        r2_members = []
        for member in ensemble:
            # member output: [B, T, 1] or [B, T] depending on Sequential
            out1 = member(s1)
            out2 = member(s2)
            if out1.dim() == 3:
                out1 = out1.squeeze(-1)
                out2 = out2.squeeze(-1)
            r1_members.append(out1.sum(dim=1))
            r2_members.append(out2.sum(dim=1))

        r1 = torch.stack(r1_members, dim=0).mean(dim=0)
        r2 = torch.stack(r2_members, dim=0).mean(dim=0)
        deltas.append((r1 - r2).detach().cpu().numpy())

    delta = np.concatenate(deltas, axis=0)
    sq = delta ** 2
    return {
        "n_pairs": float(n),
        "rms_delta_r": float(np.sqrt(sq.mean())),
        "std_delta_r": float(delta.std()),
        "var_delta_r": float(delta.var()),
        "mean_abs_delta_r": float(np.abs(delta).mean()),
        "median_sq_delta_r": float(np.median(sq)),
    }


def gather_preference_pairs(reward_models: Sequence) -> tuple:
    """Stack labeled preference segments from one or more RewardModel buffers."""
    seg1_list, seg2_list = [], []
    for rm in reward_models:
        n = rm.capacity if rm.buffer_full else rm.buffer_index
        if n <= 0:
            continue
        seg1_list.append(rm.buffer_seg1[:n])
        seg2_list.append(rm.buffer_seg2[:n])
    if not seg1_list:
        return (
            np.zeros((0, 1, 1), dtype=np.float32),
            np.zeros((0, 1, 1), dtype=np.float32),
        )
    return np.concatenate(seg1_list, axis=0), np.concatenate(seg2_list, axis=0)


def compute_reward_buffer_diagnostics(
    ensemble: Sequence[torch.nn.Module],
    reward_models: Sequence,
    ds: int,
    da: int,
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 256,
) -> Dict[str, float]:
    """Full diagnostic dict for mixture / multi-buffer reward models."""
    seg1, seg2 = gather_preference_pairs(reward_models)
    stats = rms_delta_r_from_pairs(
        ensemble, seg1, seg2, device=device, batch_size=batch_size
    )

    trajectories, size_segment = gather_trajectory_pairs(reward_models)
    stats.update(
        reward_env_correlation_stats(
            ensemble,
            trajectories,
            size_segment=size_segment,
            device=device,
            batch_size=batch_size,
        )
    )

    # Trajectory buffers are duplicated across experts; use the first non-empty.
    x = None
    for rm in reward_models:
        x = collect_sa_from_inputs(rm.inputs, ds, da)
        if x is not None:
            break
    if x is None:
        stats.update(
            {
                "n_transitions": 0.0,
                "mean_state_var": 0.0,
                "mean_state_std": 0.0,
                "mean_state_second_moment": 0.0,
                "mean_action_var": 0.0,
                "mean_action_std": 0.0,
                "mean_action_second_moment": 0.0,
                "mean_sa_var": 0.0,
                "mean_sa_std": 0.0,
                "mean_sa_second_moment": 0.0,
            }
        )
    else:
        stats.update(sa_moment_stats(x, ds, da))
    return stats


def log_reward_buffer_diagnostics(logger, stats: Dict[str, float], step: int) -> None:
    if logger is None:
        return
    for key, value in stats.items():
        logger.log(f"reward/{key}", float(value), step)


def read_buffer_diagnostics_csv(path: str) -> List[Dict[str, str]]:
    """Read diagnostics rows, tolerating legacy 16-column files and wide rows."""
    import csv
    import os

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    rows: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return rows
        header = [h.strip() for h in header if h and h.strip()]

        for line in reader:
            if not line:
                continue
            if len(line) == len(DIAGNOSTICS_CSV_FIELDNAMES):
                row = dict(zip(DIAGNOSTICS_CSV_FIELDNAMES, line))
            elif len(line) == len(header):
                row = dict(zip(header, line))
            else:
                continue
            if "phase" not in row or not str(row.get("phase", "")).strip():
                row["phase"] = "post_train"
            rows.append(row)
    return rows


def write_reward_buffer_diagnostics_csv(
    out_dir: str,
    stats: Dict[str, float],
    step: int,
    phase: str = "post_train",
    filename: str = "buffer_diagnostics.csv",
) -> str:
    """Write diagnostics to a dedicated CSV (avoids reward.csv fieldname lock-in)."""
    import csv
    import os

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    new_row = {
        "step": int(step),
        "phase": phase,
        **{k: float(v) for k, v in stats.items()},
    }

    existing = read_buffer_diagnostics_csv(path)
    existing.append({k: str(v) for k, v in new_row.items()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=DIAGNOSTICS_CSV_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in existing:
            out = {k: row.get(k, "") for k in DIAGNOSTICS_CSV_FIELDNAMES}
            writer.writerow(out)
    return path
