#!/usr/bin/env python3
"""Publication-style plots for TriTrust-PBRL experiments.

Supports two experiment layouts (``--mode auto`` picks from ``--root``):

**Ablation** (``exp_pebble_mixture_ablation/``)
  Reads ``<env>/ablation_t*_m*_w*/**/eval.csv`` and optionally reward CSVs.
  Writes learning curves, final-return bars, w_k comparison, alpha plots.

**Diagnostics** (``exp_pebble_mixture_diagnostics/``)
  Reads ``<env>/max_feedback*/seed*/buffer_diagnostics.csv`` (``phase`` =
  ``pre_train`` after sampling / before reward training, ``post_train`` after).
  and optional ``eval.csv``.
  Writes time-series curves, final-snapshot bar charts, and summary tables.

Examples
--------
  # Enhancement ablation (single env)
  python plot_enhancement_ablation.py --env metaworld_sweep-into-v2 --teacher-betas 1 1 1 0
  python plot_enhancement_ablation.py --env metaworld_sweep-into-v2 --max-feedback 40000

  # Buffer / RMS-ΔR diagnostics (single env)
  python plot_enhancement_ablation.py --mode diagnostics --env walker_walk --seeds 12345

  # Diagnostics across paper environments
  python plot_enhancement_ablation.py --mode diagnostics --envs walker_walk cheetah_run door_open sweep_into

  # Shared options
  python plot_enhancement_ablation.py --metric episode_reward --ci std --smooth 3
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reward_model.diagnostics import (
    DIAGNOSTICS_CSV_FIELDNAMES,
    DIAGNOSTICS_NUMERIC_FIELDNAMES,
    read_buffer_diagnostics_csv,
)

# ---------------------------------------------------------------------------
# Variant metadata (matches scripts/walker_walk/run_enhancement_ablation.py)
# ---------------------------------------------------------------------------

FOLDER_RE = re.compile(
    r"ablation_t(?P<tanh>True|False)_m(?P<maxn>True|False)_w(?P<wk>True|False)"
    r"(?:_wa(?P<wa>True|False))?$"
)
FEEDBACK_RE = re.compile(r"^max_feedback(?P<fb>\d+)_")
TEACHER_BETAS_RE = re.compile(r"_b\[(?P<betas>[^\]]+)\]_")

DEFAULT_TEACHER_BETAS: Dict[str, List[float]] = {
    "walker_walk": [1, 1, 1, 0, -1],  # 3R1N1A
    "metaworld_sweep-into-v2": [1, 1, 1, 0],  # 3R1A (ablation default)
    "door_open": [1, 1, 1, -1],  # 3R1A adversarial
}

DIAGNOSTICS_TEACHER_BETAS: Dict[str, List[float]] = {
    "walker_walk": [1, 1, 1, 0, -1],
    "cheetah_run": [1, 1, 1, -1],
    "metaworld_door-open-v2": [1, 1, 1, -1],
    "door_open": [1, 1, 1, -1],
    "metaworld_sweep-into-v2": [1, 1, 1, -1],
    "sweep_into": [1, 1, 1, -1],
}

# Short names used by scripts/run_buffer_diagnostics.py → Hydra env folder names.
DIAGNOSTICS_ENV_ALIASES: Dict[str, str] = {
    "door_open": "metaworld_door-open-v2",
    "sweep_into": "metaworld_sweep-into-v2",
}

SEED_RE = re.compile(r"seed(?P<seed>\d+)")

DIAGNOSTICS_METRIC_GROUPS: Dict[str, List[str]] = {
    "Reward pair spread": ["rms_delta_r", "median_sq_delta_r", "std_delta_r", "mean_abs_delta_r"],
    "Reward alignment": ["corr_r_rstar", "corr_segment_r_rstar"],
    "SA buffer moments": ["mean_sa_var", "mean_sa_std", "mean_sa_second_moment"],
    "State moments": ["mean_state_var", "mean_state_std"],
    "Action moments": ["mean_action_var", "mean_action_std"],
    "Buffer size": ["n_pairs", "n_transitions"],
}

DIAGNOSTICS_LABELS: Dict[str, str] = {
    "rms_delta_r": r"rms$|\Delta R|_0$",
    "median_sq_delta_r": r"median($\Delta R^2$)",
    "std_delta_r": r"std($\Delta R$)",
    "var_delta_r": r"var($\Delta R$)",
    "mean_abs_delta_r": r"mean$|\Delta R|$",
    "corr_r_rstar": r"corr($R$, $R^*$) per step",
    "corr_segment_r_rstar": r"corr($R(\tau)$, $R^*(\tau)$)",
    "n_corr_transitions": "# corr transitions",
    "n_corr_segments": "# corr segments",
    "mean_sa_var": "mean SA var",
    "mean_sa_std": "mean SA std",
    "mean_sa_second_moment": "mean SA 2nd moment",
    "mean_state_var": "mean state var",
    "mean_state_std": "mean state std",
    "mean_state_second_moment": "mean state 2nd moment",
    "mean_action_var": "mean action var",
    "mean_action_std": "mean action std",
    "mean_action_second_moment": "mean action 2nd moment",
    "n_pairs": "# preference pairs",
    "n_transitions": "# transitions",
}


@dataclass(frozen=True)
class VariantMeta:
    key: str  # folder suffix flags, e.g. tTrue_mTrue_wTrue_waTrue
    label: str
    short: str
    color: str
    linestyle: str
    order: int
    use_tanh: bool
    use_max_norm: bool
    use_confidence_weight: bool
    use_confidence_weight_in_alpha: bool = True


# Colour-blind friendly palette; Full TTP highlighted.
# Key: (tanh, max_norm, w_k, w_k_in_alpha)
VARIANT_META: Dict[Tuple[bool, bool, bool, bool], VariantMeta] = {
    (False, False, False, True): VariantMeta(
        "tFalse_mFalse_wFalse_waTrue", "Raw", "Raw", "#7A7A7A", "--", 0, False, False, False, True
    ),
    (True, False, False, True): VariantMeta(
        "tTrue_mFalse_wFalse_waTrue", "+Tanh", "+Tanh", "#4C78A8", "-", 1, True, False, False, True
    ),
    (True, True, False, True): VariantMeta(
        "tTrue_mTrue_wFalse_waTrue", "+Tanh, +Max-norm", "+Tanh+Max", "#F58518", "-", 2, True, True, False, True
    ),
    (True, True, True, True): VariantMeta(
        "tTrue_mTrue_wTrue_waTrue", "Full TTP", "Full TTP", "#E45756", "-", 3, True, True, True, True
    ),
    (True, True, True, False): VariantMeta(
        "tTrue_mTrue_wTrue_waFalse",
        "w_k reward-only",
        "w_k rew",
        "#FF9DA6",
        ":",
        3,
        True,
        True,
        True,
        False,
    ),
    (True, False, True, True): VariantMeta(
        "tTrue_mFalse_wTrue_waTrue", "w/o Max-norm", "w/o Max", "#54A24B", "-.", 4, True, False, True, True
    ),
    (False, True, True, True): VariantMeta(
        "tFalse_mTrue_wTrue_waTrue", "w/o Tanh", "w/o Tanh", "#B279A2", "-.", 5, False, True, True, True
    ),
}


def parse_variant_folder(name: str) -> Optional[VariantMeta]:
    m = FOLDER_RE.match(name)
    if not m:
        return None
    wk = m.group("wk") == "True"
    # Old folders omit _wa; treat as coupled alpha grads (wa=True).
    wa_group = m.group("wa")
    wa = True if wa_group is None else (wa_group == "True")
    # If w_k is off, wa is irrelevant; normalize to True for lookup.
    if not wk:
        wa = True
    flags = (
        m.group("tanh") == "True",
        m.group("maxn") == "True",
        wk,
        wa,
    )
    return VARIANT_META.get(flags)


# ---------------------------------------------------------------------------
# Data loading / aggregation
# ---------------------------------------------------------------------------


def normalize_teacher_betas(betas: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(b) for b in betas)


def parse_teacher_betas_from_run_name(name: str) -> Optional[Tuple[float, ...]]:
    m = TEACHER_BETAS_RE.search(name)
    if not m:
        return None
    parts = [p.strip() for p in m.group("betas").split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def teacher_betas_match(name: str, teacher_betas: Sequence[float]) -> bool:
    parsed = parse_teacher_betas_from_run_name(name)
    if parsed is None:
        return False
    return parsed == normalize_teacher_betas(teacher_betas)


def _format_beta_value(b: float) -> str:
    ib = int(b)
    return str(ib) if b == ib else str(b)


def format_teacher_betas_tag(teacher_betas: Sequence[float]) -> str:
    return "[" + ", ".join(_format_beta_value(b) for b in teacher_betas) + "]"


def default_teacher_betas_for_env(
    env: str, *, mode: str = "ablation"
) -> Optional[List[float]]:
    table = DIAGNOSTICS_TEACHER_BETAS if mode == "diagnostics" else DEFAULT_TEACHER_BETAS
    folder = resolve_env_folder(env)
    if folder in table:
        return list(table[folder])
    if env in table:
        return list(table[env])
    env_lower = env.lower()
    for key, betas in table.items():
        if key in env_lower or env_lower in key:
            return list(betas)
    return None


def resolve_env_folder(env: str) -> str:
    return DIAGNOSTICS_ENV_ALIASES.get(env, env)


def resolve_plot_mode(root: str, mode: str) -> str:
    if mode != "auto":
        return mode
    return "diagnostics" if "diagnostics" in os.path.basename(root.rstrip("/")) else "ablation"


def filter_paths_by_seeds(
    paths: Sequence[str], seeds: Optional[Sequence[int]]
) -> List[str]:
    if not seeds:
        return list(paths)
    seed_set = {int(s) for s in seeds}
    return [p for p in paths if parse_seed_from_path(p) in seed_set]


def collect_env_runs(
    env_dir: str,
    teacher_betas: Optional[Sequence[float]] = None,
    max_feedback: Optional[int] = None,
) -> List[str]:
    """List Hydra run directories directly under an environment folder."""
    feedback_runs = list_feedback_runs(env_dir, teacher_betas=teacher_betas)
    if max_feedback is not None:
        return feedback_runs.get(max_feedback, [])
    return [path for runs in feedback_runs.values() for path in runs]


def discover_env_feedback_budgets(
    env_dir: str,
    teacher_betas: Optional[Sequence[float]] = None,
) -> List[int]:
    return sorted(list_feedback_runs(env_dir, teacher_betas=teacher_betas).keys())


def find_run_csv_files(run_dirs: Sequence[str], csv_name: str) -> List[str]:
    files: List[str] = []
    for root in run_dirs:
        pattern = os.path.join(glob.escape(root), "**", f"{csv_name}.csv")
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(f for f in files if os.path.getsize(f) > 0)


def parse_seed_from_path(path: str) -> Optional[int]:
    m = SEED_RE.search(path)
    return int(m.group("seed")) if m else None


def load_diagnostics_table(csv_files: Sequence[str], env: str) -> pd.DataFrame:
    """Load all diagnostic snapshots from each seed run."""
    frames = []
    for path in csv_files:
        rows = read_buffer_diagnostics_csv(path)
        if not rows:
            continue
        chunk = pd.DataFrame(rows)
        chunk["env"] = env
        chunk["seed"] = parse_seed_from_path(path)
        chunk["source"] = path
        frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    return sanitize_diagnostics_table(pd.concat(frames, ignore_index=True))


def sanitize_diagnostics_table(table: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and drop rows that fail to parse."""
    if table.empty:
        return table
    out = table.copy()
    out["step"] = pd.to_numeric(out.get("step"), errors="coerce")
    out = out.dropna(subset=["step"])
    out["step"] = out["step"].astype(int)
    out = normalize_diagnostics_phase(out)

    for col in DIAGNOSTICS_NUMERIC_FIELDNAMES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "rms_delta_r" in out.columns:
        out = out.dropna(subset=["rms_delta_r"])
    return out.reset_index(drop=True)


def diagnostics_metric_values(table: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in table.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(table[metric], errors="coerce").dropna().to_numpy(dtype=float)
    return vals


def available_diagnostics_metrics(
    table: pd.DataFrame, metrics: Sequence[str]
) -> List[str]:
    return [m for m in metrics if diagnostics_metric_values(table, m).size > 0]


def normalize_diagnostics_phase(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "phase" not in out.columns:
        out["phase"] = "post_train"
    else:
        phase = out["phase"].astype(str).str.strip()
        unknown = ~phase.isin(["pre_train", "post_train"])
        phase = phase.mask(unknown, "post_train")
        out["phase"] = phase.fillna("post_train")
    return out


def diagnostics_final_by_seed(
    table: pd.DataFrame, phase: Optional[str] = "post_train"
) -> pd.DataFrame:
    """Last diagnostic snapshot per seed (for bar charts / cross-env summaries)."""
    if table.empty:
        return table
    table = normalize_diagnostics_phase(table)
    if phase is not None:
        phased = table[table["phase"] == phase]
        if not phased.empty:
            table = phased
    if table.empty:
        return table
    if "seed" in table.columns and "step" in table.columns:
        group_cols = ["seed"]
        if "phase" in table.columns:
            group_cols.append("phase")
        return table.sort_values("step").groupby(group_cols, as_index=False).tail(1)
    return table.iloc[[-1]].copy()


def plot_diagnostics_curves_by_phase(
    table: pd.DataFrame,
    metrics: Sequence[str],
    title_base: str,
    out_dir: str,
    ci: str,
) -> None:
    """Write one diagnostics curve figure per ``phase`` value."""
    table = normalize_diagnostics_phase(table)
    phases = [p for p in ["pre_train", "post_train"] if (table["phase"] == p).any()]
    if not phases:
        phases = sorted(table["phase"].unique())

    for phase in phases:
        sub = table[table["phase"] == phase]
        phase_label = "before reward training" if phase == "pre_train" else "after reward training"
        plot_diagnostics_curves(
            sub,
            metrics,
            title=f"Diagnostics ({phase_label}) — {title_base}",
            out_path=os.path.join(out_dir, f"diagnostics_curves_{phase}.png"),
            ci=ci,
        )


def plot_diagnostics_curves(
    table: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    out_path: str,
    ci: str = "sem",
) -> None:
    """Learning-style curves for diagnostics metrics vs training step."""
    if table.empty or "step" not in table.columns:
        return

    metrics = available_diagnostics_metrics(table, metrics)
    if not metrics:
        return

    n = len(metrics)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False, sharex=True
    )
    axes_flat = axes.ravel()
    cmap = plt.get_cmap("tab10")
    seeds = sorted(table["seed"].dropna().unique()) if "seed" in table.columns else [None]

    for ax, metric in zip(axes_flat, metrics):
        for i, seed in enumerate(seeds):
            if seed is None:
                sub = table.sort_values("step")
            else:
                sub = table[table["seed"] == seed].sort_values("step")
            sub = sub.copy()
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.dropna(subset=["step", metric])
            if sub.empty:
                continue
            x = sub["step"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            label = f"seed {int(seed)}" if seed is not None and len(seeds) > 1 else metric
            ax.plot(x, y, color=cmap(i % 10), linewidth=1.8, label=label)

        if len(seeds) > 1:
            plot_table = table.copy()
            plot_table[metric] = pd.to_numeric(plot_table[metric], errors="coerce")
            plot_table = plot_table.dropna(subset=["step", metric])
            grouped = plot_table.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            x = grouped["step"].to_numpy(dtype=float)
            mean = grouped["mean"].to_numpy(dtype=float)
            std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            band = std if ci == "std" else std / np.sqrt(max(len(seeds), 1))
            ax.plot(x, mean, color="black", linewidth=2.2, linestyle="--", label="mean")
            if ci != "none":
                ax.fill_between(x, mean - band, mean + band, color="black", alpha=0.12)

        ax.set_title(_pretty_diagnostics_metric(metric))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_xlabel("Environment steps")

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _pretty_diagnostics_metric(metric: str) -> str:
    return DIAGNOSTICS_LABELS.get(metric, metric.replace("_", " "))


def plot_diagnostics_metric_bars(
    table: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    out_path: str,
) -> None:
    """Bar chart of diagnostics metrics (mean ± SEM across seeds when available)."""
    metrics = available_diagnostics_metrics(table, metrics)
    if not metrics:
        return

    means, sems, labels = [], [], []
    for metric in metrics:
        vals = diagnostics_metric_values(table, metric)
        means.append(float(np.mean(vals)))
        sems.append(
            float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        )
        labels.append(_pretty_diagnostics_metric(metric))

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(metrics)), 4.8))
    x = np.arange(len(metrics))
    ax.bar(
        x,
        means,
        yerr=sems,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        alpha=0.9,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Value")
    y_top = max(means[i] + sems[i] for i in range(len(means)))
    for i, (mu, se) in enumerate(zip(means, sems)):
        ax.text(
            i,
            mu + se + 0.02 * y_top,
            f"{mu:.2g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diagnostics_panels(
    table: pd.DataFrame,
    env_label: str,
    out_path: str,
) -> None:
    """Multi-panel bar chart grouped by diagnostic category."""
    groups = [
        (name, available_diagnostics_metrics(table, metrics))
        for name, metrics in DIAGNOSTICS_METRIC_GROUPS.items()
    ]
    groups = [(name, metrics) for name, metrics in groups if metrics]
    if not groups:
        return

    n = len(groups)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()

    for ax, (group_name, metrics) in zip(axes_flat, groups):
        means, sems = [], []
        for metric in metrics:
            vals = diagnostics_metric_values(table, metric)
            means.append(float(np.mean(vals)))
            sems.append(
                float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else 0.0
            )
        x = np.arange(len(metrics))
        ax.bar(
            x,
            means,
            yerr=sems,
            color="#F58518",
            edgecolor="black",
            linewidth=0.6,
            capsize=3,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_pretty_diagnostics_metric(m) for m in metrics],
            rotation=25,
            ha="right",
            fontsize=8,
        )
        ax.set_title(group_name, fontsize=10)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))

    for ax in axes_flat[len(groups) :]:
        ax.axis("off")

    fig.suptitle(f"Reward-buffer diagnostics — {env_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diagnostics_cross_env(
    tables: Dict[str, pd.DataFrame],
    metrics: Sequence[str],
    title: str,
    out_path: str,
) -> None:
    """Grouped bars comparing selected metrics across environments."""
    envs = sorted(tables.keys())
    metrics = [
        m
        for m in metrics
        if any(diagnostics_metric_values(df, m).size > 0 for df in tables.values())
    ]
    if not envs or not metrics:
        return

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(envs)), 5.0))
    x = np.arange(len(envs))
    width = 0.8 / len(metrics)
    cmap = plt.get_cmap("tab10")

    for j, metric in enumerate(metrics):
        means, sems = [], []
        for env in envs:
            df = tables[env]
            if metric not in df.columns:
                means.append(np.nan)
                sems.append(0.0)
                continue
            vals = diagnostics_metric_values(df, metric)
            if vals.size == 0:
                means.append(np.nan)
                sems.append(0.0)
                continue
            means.append(float(np.mean(vals)))
            sems.append(
                float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
        offset = (j - (len(metrics) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            label=_pretty_diagnostics_metric(metric),
            color=cmap(j),
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diagnostics_eval_curve(
    eval_files: Sequence[str],
    metric: str,
    title: str,
    out_path: str,
    ci: str,
    smooth: int,
) -> None:
    if not eval_files:
        return
    x, Y = load_seed_series(eval_files, metric)
    mean, band = aggregate(Y, ci=ci)
    if smooth > 1:
        mean = smooth_curve(mean, smooth)
        band = smooth_curve(band, smooth)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x, mean, color="#E45756", linestyle="-", label=f"n={Y.shape[0]} seeds")
    ax.fill_between(x, mean - band, mean + band, color="#E45756", alpha=0.18, linewidth=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def list_feedback_runs(
    variant_dir: str,
    teacher_betas: Optional[Sequence[float]] = None,
) -> Dict[int, List[str]]:
    """Map max_feedback → list of config run dirs under a variant folder."""
    found: Dict[int, List[str]] = {}
    if not os.path.isdir(variant_dir):
        return found
    for name in sorted(os.listdir(variant_dir)):
        path = os.path.join(variant_dir, name)
        if not os.path.isdir(path):
            continue
        m = FEEDBACK_RE.match(name)
        if not m:
            continue
        if teacher_betas is not None and not teacher_betas_match(name, teacher_betas):
            continue
        fb = int(m.group("fb"))
        found.setdefault(fb, []).append(path)
    return found


def discover_feedback_budgets(
    variants: Sequence[Tuple[VariantMeta, str]],
    teacher_betas: Optional[Sequence[float]] = None,
) -> List[int]:
    budgets = set()
    for _, vdir in variants:
        budgets.update(list_feedback_runs(vdir, teacher_betas=teacher_betas).keys())
    return sorted(budgets)


def find_csv_files(
    variant_dir: str,
    csv_name: str,
    max_feedback: Optional[int] = None,
    teacher_betas: Optional[Sequence[float]] = None,
) -> List[str]:
    """Find non-empty ``{csv_name}.csv`` under a variant.

    If ``max_feedback`` is set, only search matching ``max_feedbackN_*`` run
    folders so different feedback budgets are not mixed. When ``teacher_betas``
    is set, only runs whose folder encodes the same expert rationalities are
    included (Hydra ``_b[...]_`` suffix).
    """
    feedback_runs = list_feedback_runs(variant_dir, teacher_betas=teacher_betas)
    search_roots: List[str]
    if max_feedback is None:
        if teacher_betas is not None:
            search_roots = [p for runs in feedback_runs.values() for p in runs]
            if not search_roots:
                return []
        else:
            search_roots = [variant_dir]
    else:
        runs = feedback_runs.get(max_feedback, [])
        if not runs:
            return []
        search_roots = runs

    files: List[str] = []
    for root in search_roots:
        pattern = os.path.join(glob.escape(root), "**", f"{csv_name}.csv")
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(f for f in files if os.path.getsize(f) > 0)


def load_seed_series(
    csv_files: Sequence[str], metric: str, x_col: str = "step"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, Y) where Y has shape (n_seeds, n_steps), aligned on shared x."""
    series = []
    xs = []
    for path in csv_files:
        df = pd.read_csv(path)
        if x_col not in df.columns or metric not in df.columns:
            continue
        xs.append(df[x_col].to_numpy(dtype=float))
        series.append(df[metric].to_numpy(dtype=float))

    if not series:
        raise ValueError(f"No usable series for metric={metric!r}")

    # Align on intersection of x grids (eval logs are usually identical).
    common_x = xs[0]
    for x in xs[1:]:
        common_x = np.intersect1d(common_x, x)
    if common_x.size == 0:
        raise ValueError("Seed runs have no overlapping x values")

    Y = np.stack(
        [s[np.isin(x, common_x)] for s, x in zip(series, xs)],
        axis=0,
    )
    return common_x, Y


def aggregate(Y: np.ndarray, ci: str = "sem") -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
    if ci == "std":
        band = std
    elif ci == "sem":
        band = std / np.sqrt(max(Y.shape[0], 1))
    elif ci == "none":
        band = np.zeros_like(mean)
    else:
        raise ValueError(f"Unknown ci={ci!r}; use sem|std|none")
    return mean, band


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / window
    # Reflect-pad to keep length.
    pad = window // 2
    yp = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
        }
    )


def plot_learning_curves(
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: str,
    title: str,
    out_path: str,
    xlabel: str = "Environment steps",
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for meta in sorted(curves.keys(), key=lambda m: m.order):
        x, mean, band = curves[meta]
        ax.plot(x, mean, color=meta.color, linestyle=meta.linestyle, label=meta.label)
        ax.fill_between(
            x, mean - band, mean + band, color=meta.color, alpha=0.18, linewidth=0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    # Compact scientific x ticks for large step counts.
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_final_bars(
    finals: Dict[VariantMeta, np.ndarray],
    metric: str,
    title: str,
    out_path: str,
    last_n: int,
) -> None:
    metas = sorted(finals.keys(), key=lambda m: m.order)
    means = [float(np.mean(finals[m])) for m in metas]
    sems = [
        float(np.std(finals[m], ddof=1) / np.sqrt(len(finals[m])))
        if len(finals[m]) > 1
        else 0.0
        for m in metas
    ]
    labels = [m.label for m in metas]
    colors = [m.color for m in metas]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(metas))
    bars = ax.bar(
        x,
        means,
        yerr=sems,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        alpha=0.9,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    # Mark Full TTP.
    for i, m in enumerate(metas):
        if m.short == "Full TTP":
            bars[i].set_linewidth(1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_title(title)
    # Annotate mean values.
    y_max = max(means[i] + sems[i] for i in range(len(means)))
    for i, (mu, se) in enumerate(zip(means, sems)):
        ax.text(
            i,
            mu + se + 0.01 * y_max,
            f"{mu:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}  (last {last_n} eval points averaged per seed)")


def plot_expert_coefs(
    reward_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
    n_experts: int = 5,
) -> None:
    """One subplot per variant: mean expert_coef_k over steps."""
    _plot_multichannel_panels(
        reward_curves,
        title=title,
        out_path=out_path,
        ylabel="expert coef",
        channel_prefix="expert",
        n_channels=n_experts,
    )


def plot_alphas(
    alpha_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
    n_experts: Optional[int] = None,
) -> None:
    """One subplot per variant: mean alpha_k over steps."""
    if not alpha_curves:
        return
    sample = next(iter(alpha_curves.values()))[1]
    n = n_experts if n_experts is not None else sample.shape[-1]
    _plot_multichannel_panels(
        alpha_curves,
        title=title,
        out_path=out_path,
        ylabel=r"$\alpha_k$",
        channel_prefix=r"$\alpha$",
        n_channels=n,
    )


def plot_alpha_abs_sum(
    abs_sum_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
) -> None:
    """Overlay mean ± band of alpha_abs_sum across ablation variants."""
    if not abs_sum_curves:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for meta in sorted(abs_sum_curves.keys(), key=lambda m: m.order):
        x, mean, band = abs_sum_curves[meta]
        ax.plot(x, mean, color=meta.color, linestyle=meta.linestyle, label=meta.label)
        ax.fill_between(
            x, mean - band, mean + band, color=meta.color, alpha=0.18, linewidth=0
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(r"$|\alpha|_1$ (alpha_abs_sum)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_multichannel_panels(
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
    ylabel: str,
    channel_prefix: str,
    n_channels: int,
) -> None:
    metas = sorted(curves.keys(), key=lambda m: m.order)
    n = len(metas)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True
    )
    axes_flat = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")

    for ax, meta in zip(axes_flat, metas):
        x, Y = curves[meta]  # Y: (n_seeds, n_steps, n_channels)
        mean = np.nanmean(Y, axis=0)
        for k in range(min(n_channels, mean.shape[1])):
            ax.plot(
                x,
                mean[:, k],
                color=cmap(k),
                label=f"{channel_prefix}_{k}",
                linewidth=1.6,
            )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        ax.set_title(meta.label, color=meta.color)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_xlabel("steps")
        ax.set_ylabel(ylabel)

    for ax in axes_flat[len(metas) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=min(n_channels, 6), frameon=False
    )
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def load_reward_channels(
    reward_files: Sequence[str],
    col_regex: str,
    max_points: int = 400,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Load aligned reward CSV channels → (x, Y[n_seeds,n_steps,n_ch], col_names)."""
    seed_mats = []
    xs = []
    cols_ref: Optional[List[str]] = None
    for path in reward_files:
        df = pd.read_csv(path)
        if "step" not in df.columns:
            continue
        cols = [c for c in df.columns if re.fullmatch(col_regex, c)]
        if not cols:
            continue
        cols = sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))
        if cols_ref is None:
            cols_ref = cols
        elif cols != cols_ref:
            # Keep intersection in reference order.
            cols = [c for c in cols_ref if c in cols]
            if not cols:
                continue
        xs.append(df["step"].to_numpy(dtype=float))
        seed_mats.append(df[cols].to_numpy(dtype=float))

    if not seed_mats or not cols_ref:
        return None

    common_x = xs[0]
    for xr in xs[1:]:
        common_x = np.intersect1d(common_x, xr)
    if common_x.size == 0:
        return None
    if common_x.size > max_points:
        idx = np.linspace(0, common_x.size - 1, max_points).astype(int)
        common_x = common_x[idx]

    mats = []
    for mat, xr in zip(seed_mats, xs):
        mask = np.isin(xr, common_x)
        mats.append(mat[mask])
    return common_x, np.stack(mats, axis=0), cols_ref


def load_reward_scalar(
    reward_files: Sequence[str],
    column: str,
    max_points: int = 400,
    ci: str = "sem",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load one scalar reward column → (x, mean, band)."""
    try:
        x, Y = load_seed_series(reward_files, column)
    except ValueError:
        return None
    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        x = x[idx]
        Y = Y[:, idx]
    mean, band = aggregate(Y, ci=ci)
    return x, mean, band


def _pretty_metric(metric: str) -> str:
    return {
        "true_episode_reward": "True episode return",
        "episode_reward": "Episode return (learned reward)",
        "success_rate": "Success rate (%)",
        "actor_loss": "Actor loss",
        "critic_loss": "Critic loss",
    }.get(metric, metric.replace("_", " ").title())


def default_metric_for_env(env: str) -> str:
    if "metaworld" in env.lower():
        return "success_rate"
    return "true_episode_reward"


def _backbone_label(meta: VariantMeta) -> str:
    parts = []
    if meta.use_tanh:
        parts.append("Tanh")
    if meta.use_max_norm:
        parts.append("Max-norm")
    return "+".join(parts) if parts else "Raw"


def matched_w_pairs(
    metas: Sequence[VariantMeta],
) -> List[Tuple[str, VariantMeta, VariantMeta]]:
    """Pairs that share tanh/max-norm and differ only in confidence weight w_k."""
    by_backbone: Dict[Tuple[bool, bool], Dict[bool, VariantMeta]] = {}
    for m in metas:
        key = (m.use_tanh, m.use_max_norm)
        slot = by_backbone.setdefault(key, {})
        if m.use_confidence_weight:
            # Prefer full TTP (w_k also in alpha loss) over reward-only.
            existing = slot.get(True)
            if existing is None or (
                m.use_confidence_weight_in_alpha
                and not existing.use_confidence_weight_in_alpha
            ):
                slot[True] = m
        else:
            slot[False] = m

    pairs = []
    for (tanh, maxn), d in sorted(by_backbone.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if False in d and True in d:
            without_w, with_w = d[False], d[True]
            pairs.append((_backbone_label(without_w), without_w, with_w))
    return pairs


def plot_w_comparison(
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    finals: Dict[VariantMeta, np.ndarray],
    metric: str,
    env: str,
    out_dir: str,
    last_n: int,
) -> Optional[pd.DataFrame]:
    """Learning-curve pairs + grouped bars for with vs without confidence weight w_k."""
    pairs = matched_w_pairs(list(curves.keys()))
    if not pairs:
        print("No matched with/without-w_k pairs found; skipping w comparison.")
        return None

    color_wo, color_w = "#4C78A8", "#E45756"

    # --- Side-by-side learning curves (one panel per backbone) ---
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), sharey=True)
    axes_list = np.atleast_1d(axes).ravel()
    for ax, (backbone, wo, wi) in zip(axes_list, pairs):
        for meta, color, ls, tag in (
            (wo, color_wo, "--", "without w_k"),
            (wi, color_w, "-", "with w_k"),
        ):
            x, mean, band = curves[meta]
            ax.plot(x, mean, color=color, linestyle=ls, label=f"{tag} ({meta.label})")
            ax.fill_between(
                x, mean - band, mean + band, color=color, alpha=0.18, linewidth=0
            )
        ax.set_title(f"Backbone: {backbone}")
        ax.set_xlabel("Environment steps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes_list[0].set_ylabel(_pretty_metric(metric))
    fig.suptitle(f"Confidence weight w_k — with vs without ({env})", y=1.02)
    fig.tight_layout()
    curve_path = os.path.join(out_dir, f"w_comparison_curves_{metric}.png")
    fig.savefig(curve_path, bbox_inches="tight")
    fig.savefig(curve_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {curve_path}")

    # --- Grouped final-return bars ---
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(len(pairs))
    width = 0.36
    means_wo, sems_wo, means_w, sems_w = [], [], [], []
    rows = []
    for backbone, wo, wi in pairs:
        yw, yi = finals[wo], finals[wi]
        m_wo, m_w = float(np.mean(yw)), float(np.mean(yi))
        s_wo = float(np.std(yw, ddof=1) / np.sqrt(len(yw))) if len(yw) > 1 else 0.0
        s_w = float(np.std(yi, ddof=1) / np.sqrt(len(yi))) if len(yi) > 1 else 0.0
        means_wo.append(m_wo)
        sems_wo.append(s_wo)
        means_w.append(m_w)
        sems_w.append(s_w)
        delta = m_w - m_wo
        rows.append(
            {
                "backbone": backbone,
                "without_w_label": wo.label,
                "with_w_label": wi.label,
                "without_w_mean": m_wo,
                "without_w_sem": s_wo,
                "with_w_mean": m_w,
                "with_w_sem": s_w,
                "delta_with_minus_without": delta,
                "n_seeds_without": len(yw),
                "n_seeds_with": len(yi),
                "last_n_eval_points": last_n,
            }
        )

    bars_wo = ax.bar(
        x - width / 2,
        means_wo,
        width,
        yerr=sems_wo,
        color=color_wo,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="without w_k",
        error_kw={"elinewidth": 1.1},
    )
    bars_w = ax.bar(
        x + width / 2,
        means_w,
        width,
        yerr=sems_w,
        color=color_w,
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        label="with w_k",
        error_kw={"elinewidth": 1.1},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _, _ in pairs])
    ax.set_ylabel(_pretty_metric(metric))
    ax.set_title(f"Final return: with vs without w_k (last {last_n} evals)")
    ax.legend(frameon=False, loc="lower right")
    y_top = max(
        max(means_wo[i] + sems_wo[i], means_w[i] + sems_w[i]) for i in range(len(pairs))
    )
    for i, (m_wo, m_w, s_wo, s_w) in enumerate(
        zip(means_wo, means_w, sems_wo, sems_w)
    ):
        ax.text(
            i - width / 2,
            m_wo + s_wo + 0.008 * y_top,
            f"{m_wo:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            i + width / 2,
            m_w + s_w + 0.008 * y_top,
            f"{m_w:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.annotate(
            f"Δ={m_w - m_wo:+.0f}",
            xy=(i, max(m_wo + s_wo, m_w + s_w)),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#333333",
        )
    del bars_wo, bars_w
    fig.tight_layout()
    bar_path = os.path.join(out_dir, f"w_comparison_bars_{metric}.png")
    fig.savefig(bar_path, bbox_inches="tight")
    fig.savefig(bar_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {bar_path}")

    # --- Delta-only chart ---
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    deltas = [r["delta_with_minus_without"] for r in rows]
    colors = ["#54A24B" if d >= 0 else "#E45756" for d in deltas]
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _, _ in pairs])
    ax.set_ylabel(f"Δ {_pretty_metric(metric)} (with − without w_k)")
    ax.set_title("Effect of confidence weight w_k")
    offset = 0.03 * (max(abs(d) for d in deltas) or 1.0)
    for i, d in enumerate(deltas):
        ax.text(
            i,
            d + offset if d >= 0 else d - offset,
            f"{d:+.1f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=10,
        )
    fig.tight_layout()
    delta_path = os.path.join(out_dir, f"w_comparison_delta_{metric}.png")
    fig.savefig(delta_path, bbox_inches="tight")
    fig.savefig(delta_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {delta_path}")

    table = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"w_comparison_{metric}.csv")
    table.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved {csv_path}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return table


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def collect_variants(env_dir: str) -> List[Tuple[VariantMeta, str]]:
    found = []
    for name in sorted(os.listdir(env_dir)):
        path = os.path.join(env_dir, name)
        if not os.path.isdir(path):
            continue
        meta = parse_variant_folder(name)
        if meta is None:
            print(f"  skip unrecognized folder: {name}")
            continue
        found.append((meta, path))
    return found


def build_summary_table(
    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    finals: Dict[VariantMeta, np.ndarray],
    last_n: int,
) -> pd.DataFrame:
    rows = []
    for meta in sorted(curves.keys(), key=lambda m: m.order):
        x, mean, band = curves[meta]
        seed_finals = finals[meta]
        rows.append(
            {
                "variant": meta.label,
                "tanh": meta.use_tanh,
                "max_norm": meta.use_max_norm,
                "w_k": meta.use_confidence_weight,
                "w_k_in_alpha": meta.use_confidence_weight_in_alpha,
                "n_seeds": len(seed_finals),
                "final_mean": float(np.mean(seed_finals)),
                "final_std": float(np.std(seed_finals, ddof=1))
                if len(seed_finals) > 1
                else 0.0,
                "final_sem": float(np.std(seed_finals, ddof=1) / np.sqrt(len(seed_finals)))
                if len(seed_finals) > 1
                else 0.0,
                "curve_last_mean": float(mean[-1]),
                "auc_mean": float(
                    (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(mean, x)
                    / max(x[-1] - x[0], 1.0)
                ),
                "last_n_eval_points": last_n,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("auto", "ablation", "diagnostics"),
        default="auto",
        help="Experiment layout: ablation variants, buffer diagnostics, or auto from --root",
    )
    p.add_argument("--root", default="exp_pebble_mixture_ablation")
    p.add_argument("--env", default="walker_walk", help="Environment folder name (ablation or diagnostics)")
    p.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Diagnostics only: plot/compare multiple envs (aliases: door_open, sweep_into)",
    )
    p.add_argument(
        "--metric",
        default=None,
        help="Column in eval.csv (default: success_rate for MetaWorld, else true_episode_reward)",
    )
    p.add_argument(
        "--max-feedback",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Feedback budget(s) to plot, matching max_feedbackN_* run folders "
            "(e.g. --max-feedback 20000 40000). Default: every budget found."
        ),
    )
    p.add_argument(
        "--teacher-betas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Expert rationalities; filters run folders by _b[...]_ suffix "
            "(default: env-specific, e.g. walker_walk → 1 1 1 0 -1, "
            "metaworld_sweep-into-v2 → 1 1 1 0)"
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Diagnostics only: plot selected seed(s) only (e.g. --seeds 12345)",
    )
    p.add_argument(
        "--ci",
        choices=("sem", "std", "none"),
        default="sem",
        help="Shaded band: standard error (default), std, or none",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=5,
        help="Average last N eval points per seed for final bar chart",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window on mean curves (1 = off)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/exp_pebble_mixture_ablation/<env>/enhancement_ablation)",
    )
    p.add_argument(
        "--skip-reward",
        action="store_true",
        help="Skip expert-coefficient / alpha plots from reward.csv",
    )
    p.add_argument(
        "--alphas-only",
        action="store_true",
        help="Ablation only: plot alpha_* / alpha_abs_sum from reward.csv (skip eval curves)",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Diagnostics only: skip eval.csv learning-curve plot",
    )
    return p.parse_args()


def plot_one_feedback_budget(
    variants: Sequence[Tuple[VariantMeta, str]],
    *,
    max_feedback: Optional[int],
    teacher_betas: Optional[Sequence[float]],
    out_dir: str,
    env: str,
    metric: str,
    ci: str,
    last_n: int,
    smooth: int,
    skip_reward: bool,
    alphas_only: bool,
) -> bool:
    """Build all figures for one feedback budget. Returns True if anything was plotted."""
    fb_tag = f"max_feedback={max_feedback}" if max_feedback is not None else "all feedbacks"
    beta_tag = (
        f"teacher_betas={format_teacher_betas_tag(teacher_betas)}"
        if teacher_betas is not None
        else "all teacher_betas"
    )
    title_suffix = env
    if teacher_betas is not None:
        title_suffix += f" ({format_teacher_betas_tag(teacher_betas)})"
    if max_feedback is not None:
        title_suffix += f", max_feedback={max_feedback}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 72}\nPlotting {fb_tag}, {beta_tag} → {out_dir}\n{'=' * 72}")

    curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    finals: Dict[VariantMeta, np.ndarray] = {}
    reward_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]] = {}
    alpha_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]] = {}
    alpha_bound_curves: Dict[VariantMeta, Tuple[np.ndarray, np.ndarray]] = {}
    alpha_abs_sum_curves: Dict[
        VariantMeta, Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    for meta, vdir in variants:
        variant_runs = list_feedback_runs(vdir, teacher_betas=teacher_betas)
        if max_feedback is not None and max_feedback not in variant_runs:
            print(
                f"  [{meta.label:18s}] no max_feedback={max_feedback} run "
                f"for {beta_tag} — skip"
            )
            continue

        if not alphas_only:
            eval_files = find_csv_files(
                vdir, "eval", max_feedback=max_feedback, teacher_betas=teacher_betas
            )
            if not eval_files:
                print(f"  [{meta.label}] no eval.csv — skipping eval")
            else:
                x, Y = load_seed_series(eval_files, metric)
                mean, band = aggregate(Y, ci=ci)
                if smooth > 1:
                    mean = smooth_curve(mean, smooth)
                    band = smooth_curve(band, smooth)
                curves[meta] = (x, mean, band)

                n = min(last_n, Y.shape[1])
                seed_scores = np.nanmean(Y[:, -n:], axis=1)
                finals[meta] = seed_scores
                print(
                    f"  [{meta.label:18s}] seeds={Y.shape[0]}  "
                    f"last-{n} mean={seed_scores.mean():.1f} ± "
                    f"{seed_scores.std(ddof=1):.1f}"
                )

        if not skip_reward:
            reward_files = find_csv_files(
                vdir, "reward", max_feedback=max_feedback, teacher_betas=teacher_betas
            )
            if not reward_files:
                print(f"  [{meta.label}] no reward.csv — skipping reward metrics")
                continue

            coef = load_reward_channels(reward_files, r"expert_coef_\d+")
            if coef is not None:
                x_c, Y_c, _ = coef
                reward_curves[meta] = (x_c, Y_c)

            alphas = load_reward_channels(reward_files, r"alpha_\d+")
            if alphas is not None:
                x_a, Y_a, cols_a = alphas
                alpha_curves[meta] = (x_a, Y_a)
                print(
                    f"  [{meta.label:18s}] alphas seeds={Y_a.shape[0]}  "
                    f"channels={cols_a}  steps={Y_a.shape[1]}"
                )

            bounds = load_reward_channels(reward_files, r"alpha_bound_\d+")
            if bounds is not None:
                x_b, Y_b, _ = bounds
                alpha_bound_curves[meta] = (x_b, Y_b)

            abs_sum = load_reward_scalar(reward_files, "alpha_abs_sum", ci=ci)
            if abs_sum is not None:
                alpha_abs_sum_curves[meta] = abs_sum

    if alphas_only:
        if not alpha_curves and not alpha_abs_sum_curves:
            print(f"No alpha metrics found for {fb_tag}.")
            return False
    elif not curves:
        print(f"Nothing to plot for {fb_tag}.")
        return False

    if curves:
        plot_learning_curves(
            curves,
            metric=metric,
            title=f"Enhancement ablation — {title_suffix}",
            out_path=os.path.join(out_dir, f"learning_curve_{metric}.png"),
        )
        plot_final_bars(
            finals,
            metric=metric,
            title=f"Final return (last {last_n} evals) — {title_suffix}",
            out_path=os.path.join(out_dir, f"final_bar_{metric}.png"),
            last_n=last_n,
        )
        print("\n=== With vs without w_k ===")
        plot_w_comparison(
            curves,
            finals,
            metric=metric,
            env=title_suffix,
            out_dir=out_dir,
            last_n=last_n,
        )
        table = build_summary_table(curves, finals, last_n)
        if teacher_betas is not None:
            table.insert(1, "teacher_betas", format_teacher_betas_tag(teacher_betas))
        if max_feedback is not None:
            table.insert(
                2 if teacher_betas is not None else 1, "max_feedback", max_feedback
            )
        table_path = os.path.join(out_dir, f"summary_{metric}.csv")
        table.to_csv(table_path, index=False, float_format="%.4f")
        print(f"Saved {table_path}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    if reward_curves:
        plot_expert_coefs(
            reward_curves,
            title=f"Expert coefficients — {title_suffix}",
            out_path=os.path.join(out_dir, "expert_coefficients.png"),
        )

    if alpha_curves:
        print("\n=== Alpha metrics ===")
        plot_alphas(
            alpha_curves,
            title=rf"Trust parameters $\alpha_k$ — {title_suffix}",
            out_path=os.path.join(out_dir, "alphas.png"),
        )
    if alpha_bound_curves:
        plot_alphas(
            alpha_bound_curves,
            title=rf"Bounded alphas — {title_suffix}",
            out_path=os.path.join(out_dir, "alpha_bounds.png"),
        )
    if alpha_abs_sum_curves:
        plot_alpha_abs_sum(
            alpha_abs_sum_curves,
            title=rf"$|\alpha|_1$ across variants — {title_suffix}",
            out_path=os.path.join(out_dir, "alpha_abs_sum.png"),
        )

    print(f"Figures → {out_dir}")
    return True


def main_diagnostics(args: argparse.Namespace) -> int:
    mode = "diagnostics"
    if args.metric is None:
        args.metric = default_metric_for_env(args.env)
    apply_style()

    repo = os.path.dirname(os.path.abspath(__file__))
    env_names = args.envs if args.envs else [args.env]
    if len(env_names) == 1 and env_names[0] == "all":
        root_dir = os.path.join(repo, args.root)
        env_names = sorted(
            name
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        )

    per_env_tables: Dict[str, pd.DataFrame] = {}
    any_ok = False

    for env_name in env_names:
        folder = resolve_env_folder(env_name)
        teacher_betas = args.teacher_betas
        if teacher_betas is None:
            teacher_betas = default_teacher_betas_for_env(folder, mode=mode)
            if teacher_betas is None:
                teacher_betas = default_teacher_betas_for_env(env_name, mode=mode)

        env_dir = os.path.join(repo, args.root, folder)
        if not os.path.isdir(env_dir):
            print(f"Environment directory not found: {env_dir}")
            continue

        available = discover_env_feedback_budgets(env_dir, teacher_betas=teacher_betas)
        if args.max_feedback is not None and len(args.max_feedback) == 0:
            budgets: List[Optional[int]] = available if available else [None]
        elif args.max_feedback:
            budgets = list(args.max_feedback)
        elif len(available) == 1:
            budgets = available
        elif len(available) > 1:
            budgets = available
            print(f"[{env_name}] multiple feedback budgets: {budgets}")
        else:
            budgets = [None]

        base_out = args.out_dir or os.path.join(
            repo, "results", args.root, folder, "buffer_diagnostics"
        )
        if teacher_betas is not None:
            beta_slug = "b" + format_teacher_betas_tag(teacher_betas).replace(" ", "")
            base_out = os.path.join(base_out, beta_slug)

        for fb in budgets:
            if len(budgets) == 1 and fb is None:
                out_dir = base_out
            elif len(available) > 1:
                out_dir = os.path.join(base_out, f"max_feedback{fb}")
            else:
                out_dir = base_out

            if args.seeds:
                if len(args.seeds) == 1:
                    out_dir = os.path.join(out_dir, f"seed{args.seeds[0]}")
                else:
                    out_dir = os.path.join(
                        out_dir, "seeds_" + "_".join(str(s) for s in args.seeds)
                    )

            run_dirs = collect_env_runs(
                env_dir, teacher_betas=teacher_betas, max_feedback=fb
            )
            if not run_dirs:
                print(f"[{env_name}] no runs for max_feedback={fb} — skip")
                continue

            diag_files = filter_paths_by_seeds(
                find_run_csv_files(run_dirs, "buffer_diagnostics"), args.seeds
            )
            if args.seeds and not diag_files:
                print(
                    f"[{env_name}] no buffer_diagnostics.csv for seeds={args.seeds} — skip"
                )
                continue
            if not diag_files:
                print(f"[{env_name}] no buffer_diagnostics.csv — skip")
                continue

            table = load_diagnostics_table(diag_files, env=env_name)
            if args.seeds:
                table = table[table["seed"].isin(args.seeds)]
            if table.empty:
                continue

            final_table = diagnostics_final_by_seed(table)
            n_seeds = final_table["seed"].nunique() if "seed" in final_table.columns else 1
            n_snapshots = len(table)

            os.makedirs(out_dir, exist_ok=True)
            fb_tag = f", max_feedback={fb}" if fb is not None else ""
            beta_tag = (
                format_teacher_betas_tag(teacher_betas)
                if teacher_betas is not None
                else "all betas"
            )
            title_base = f"{env_name} ({beta_tag}{fb_tag})"
            print(f"\n{'=' * 72}\nDiagnostics: {title_base} → {out_dir}\n{'=' * 72}")
            print(
                f"  seeds={n_seeds}  snapshots={n_snapshots}  "
                f"final post_train rms_delta_r="
                f"{diagnostics_metric_values(final_table, 'rms_delta_r').mean():.3g}"
            )

            summary = table.copy()
            if fb is not None:
                summary.insert(1, "max_feedback", fb)
            if teacher_betas is not None:
                summary.insert(1, "teacher_betas", format_teacher_betas_tag(teacher_betas))

            csv_path = os.path.join(out_dir, "buffer_diagnostics_summary.csv")
            summary.to_csv(csv_path, index=False, float_format="%.6g")
            print(f"Saved {csv_path}")

            curve_metrics = [
                "rms_delta_r",
                "corr_r_rstar",
                "corr_segment_r_rstar",
                "mean_sa_var",
                "n_pairs",
            ]
            plot_diagnostics_curves_by_phase(
                table,
                curve_metrics,
                title_base=title_base,
                out_dir=out_dir,
                ci=args.ci,
            )
            post_train = table[table["phase"] == "post_train"]
            if not post_train.empty:
                plot_diagnostics_curves(
                    post_train,
                    curve_metrics,
                    title=f"Diagnostics over training — {title_base}",
                    out_path=os.path.join(out_dir, "diagnostics_curves.png"),
                    ci=args.ci,
                )

            first_pre = table[
                (table["phase"] == "pre_train") & (table["step"] == table["step"].min())
            ]
            if not first_pre.empty:
                plot_diagnostics_metric_bars(
                    first_pre,
                    ["corr_r_rstar", "corr_segment_r_rstar", "rms_delta_r"],
                    title=f"After unsup + sampling, before training — {title_base}",
                    out_path=os.path.join(out_dir, "reward_alignment_pre_train.png"),
                )
            plot_diagnostics_metric_bars(
                final_table,
                ["corr_r_rstar", "corr_segment_r_rstar", "rms_delta_r"],
                title=f"Reward alignment — {title_base}",
                out_path=os.path.join(out_dir, "reward_alignment_final.png"),
            )
            plot_diagnostics_metric_bars(
                final_table,
                ["rms_delta_r", "std_delta_r", "mean_abs_delta_r"],
                title=f"Final reward pair spread — {title_base}",
                out_path=os.path.join(out_dir, "rms_delta_r_final.png"),
            )
            plot_diagnostics_panels(
                final_table,
                env_label=f"{title_base} (final snapshot)",
                out_path=os.path.join(out_dir, "diagnostics_panels_final.png"),
            )

            if not args.skip_eval:
                eval_files = filter_paths_by_seeds(
                    find_run_csv_files(run_dirs, "eval"), args.seeds
                )
                if eval_files:
                    plot_diagnostics_eval_curve(
                        eval_files,
                        metric=args.metric,
                        title=f"Eval ({args.metric}) — {title_base}",
                        out_path=os.path.join(out_dir, f"eval_{args.metric}.png"),
                        ci=args.ci,
                        smooth=args.smooth,
                    )

            per_env_tables[env_name] = final_table
            any_ok = True
            print(f"Figures → {out_dir}")

    if len(per_env_tables) > 1:
        cross_out = args.out_dir or os.path.join(
            repo, "results", args.root, "_cross_env", "buffer_diagnostics"
        )
        os.makedirs(cross_out, exist_ok=True)
        plot_diagnostics_cross_env(
            per_env_tables,
            ["corr_r_rstar", "corr_segment_r_rstar", "rms_delta_r", "mean_sa_var"],
            title="Final buffer diagnostics across environments",
            out_path=os.path.join(cross_out, "cross_env_comparison.png"),
        )
        cross_csv = os.path.join(cross_out, "cross_env_summary.csv")
        pd.concat(per_env_tables.values(), ignore_index=True).to_csv(
            cross_csv, index=False, float_format="%.6g"
        )
        print(f"Saved {cross_csv}")

    if not any_ok:
        return 1
    print("\nDiagnostics plotting complete.")
    return 0


def main_ablation(args: argparse.Namespace) -> int:
    if args.metric is None:
        args.metric = default_metric_for_env(args.env)
    if args.teacher_betas is None:
        args.teacher_betas = default_teacher_betas_for_env(args.env, mode="ablation")
    apply_style()

    repo = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(repo, args.root, args.env)
    if not os.path.isdir(env_dir):
        print(f"Environment directory not found: {env_dir}")
        return 1

    base_out = args.out_dir or os.path.join(
        repo, "results", args.root, args.env, "enhancement_ablation"
    )
    if args.teacher_betas is not None:
        beta_slug = "b" + format_teacher_betas_tag(args.teacher_betas).replace(" ", "")
        base_out = os.path.join(base_out, beta_slug)

    variants = collect_variants(env_dir)
    if not variants:
        print(f"No ablation_* folders under {env_dir}")
        return 1

    available = discover_feedback_budgets(variants, teacher_betas=args.teacher_betas)
    print(f"Found {len(variants)} ablation variants in {env_dir}")
    if args.teacher_betas is not None:
        print(f"teacher_betas filter : {format_teacher_betas_tag(args.teacher_betas)}")
    else:
        print("teacher_betas filter : (none — all run folders)")
    print(f"Available max_feedback budgets: {available if available else '(none parsed)'}")

    if args.max_feedback is not None and len(args.max_feedback) == 0:
        budgets: List[Optional[int]] = available if available else [None]
    elif args.max_feedback:
        unknown = [b for b in args.max_feedback if available and b not in available]
        if unknown:
            print(f"Warning: requested budgets not found: {unknown}")
        budgets = list(args.max_feedback)
    elif len(available) > 1:
        budgets = available
        print(
            f"Multiple feedback folders detected; plotting each separately: {budgets}\n"
            f"  Tip: pass --max-feedback 20000 to plot only one."
        )
    elif len(available) == 1:
        budgets = available
    else:
        budgets = [None]

    any_ok = False
    for fb in budgets:
        if len(budgets) == 1 and fb is None:
            out_dir = base_out
        elif len(available) > 1:
            out_dir = os.path.join(base_out, f"max_feedback{fb}")
        else:
            out_dir = base_out

        ok = plot_one_feedback_budget(
            variants,
            max_feedback=fb,
            teacher_betas=args.teacher_betas,
            out_dir=out_dir,
            env=args.env,
            metric=args.metric,
            ci=args.ci,
            last_n=args.last_n,
            smooth=args.smooth,
            skip_reward=args.skip_reward,
            alphas_only=args.alphas_only,
        )
        any_ok = any_ok or ok

    if not any_ok:
        return 1
    print(f"\nDone. Base output: {base_out}")
    return 0


def main() -> int:
    args = parse_args()
    mode = resolve_plot_mode(args.root, args.mode)
    if mode == "diagnostics":
        return main_diagnostics(args)
    return main_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
