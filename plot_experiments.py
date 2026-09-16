#!/usr/bin/env python3
"""General plotting for TriTrust-PBRL / PEBBLE mixture experiment trees.

Expected on-disk layout (Hydra run folders)::

    <root>/
      <env>/
        max_feedbackN_feed_type..._b[1, 1, 1, -1]_m0_s0_e0/
          seedS/
            test/eval.csv
            reward/reward.csv   # optional (alphas, expert coefs)

Also supports SAC / GT-reward trees (no preference config folder)::

    <root>/
      <env>/
        tests/
          seedS/
            test/eval.csv

Works for ``exp_pebble_mixture_zero_last``, ``exp_sac``, and any similarly structured tree.
Incomplete seeds are NaN-padded so curves span the longest run (max step).
Figures go to ``results/<root>/<env>/b[1, 1, 1, -1]/`` (one folder per teacher β).

Examples
--------
  # One figure: every seed as a line + mean ± CI on the same axes
  python plot_experiments.py --root exp_pebble_mixture_zero_last_wk_sgd
  python plot_experiments.py --root exp_pebble_mixture_zero_last_wk_sgd --env sweep_into

  # Conclusion graphs (mean ± SEM only)
  python plot_experiments.py --series exp_pebble_mixture_zero_last_wk_sgd --env sweep_into

  # Overlay several experiments on one conclusion plot
  python plot_experiments.py \\
      --series zero_last:exp_pebble_mixture_zero_last \\
      --series wk_sgd:exp_pebble_mixture_zero_last_wk_sgd \\
      --env sweep_into

See ``PLOTTING.md`` for the full guide.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants / aliases
# ---------------------------------------------------------------------------

FEEDBACK_RE = re.compile(r"^max_feedback(?P<fb>\d+)_")
TEACHER_BETAS_RE = re.compile(r"_b\[(?P<betas>[^\]]+)\]_")
SEED_RE = re.compile(r"seed(?P<seed>\d+)")

ENV_ALIASES: Dict[str, str] = {
    "door_open": "metaworld_door-open-v2",
    "sweep_into": "metaworld_sweep-into-v2",
    "walker": "walker_walk",
    "cheetah": "cheetah_run",
}

DEFAULT_METRIC: Dict[str, str] = {
    "walker_walk": "true_episode_reward",
    "cheetah_run": "true_episode_reward",
    "metaworld_door-open-v2": "success_rate",
    "metaworld_sweep-into-v2": "success_rate",
}

SERIES_COLORS = [
    "#E45756",
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#EACA2B",
]
EXPERT_LINESTYLES = ["-", "--", "-.", ":"]


@dataclass(frozen=True)
class RunConfig:
    """One Hydra config folder under ``<root>/<env>/``."""

    path: str
    env: str
    max_feedback: int
    teacher_betas: Tuple[float, ...]
    label: str
    # True for SAC / GT trees with no ``_b[...]_`` folder — included under any β filter.
    beta_agnostic: bool = False


@dataclass(frozen=True)
class SeriesSpec:
    """A named experiment root to overlay."""

    name: str
    root: str
    color: str
    linestyle: str = "-"


# Alpha / trust plots are TTP-specific; skip reward.csv for other overlays.
TTP_SERIES_NAMES = frozenset({"TTP", "ttp"})
TTP_ROOT_BASENAMES = frozenset(
    {
        "exp_pebble_mixture_zero_last_wk_sgd",
        "exp_pebble_mixture_zero_last",
    }
)


def series_plots_alpha_trust(spec: SeriesSpec, *, multi_series: bool) -> bool:
    """Whether to load reward.csv and write α / w_k / logit figures for this series."""
    if not multi_series:
        return True
    if spec.name in TTP_SERIES_NAMES:
        return True
    base = os.path.basename(os.path.normpath(spec.root))
    return base in TTP_ROOT_BASENAMES


# ---------------------------------------------------------------------------
# Path / config parsing
# ---------------------------------------------------------------------------


def resolve_env(env: str) -> str:
    return ENV_ALIASES.get(env, env)


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def abs_under_repo(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root(), path)


def normalize_teacher_betas(betas: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(b) for b in betas)


def parse_teacher_betas_from_name(name: str) -> Optional[Tuple[float, ...]]:
    m = TEACHER_BETAS_RE.search(name)
    if not m:
        return None
    parts = [p.strip() for p in m.group("betas").split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def teacher_betas_match(name: str, teacher_betas: Sequence[float]) -> bool:
    parsed = parse_teacher_betas_from_name(name)
    if parsed is None:
        return False
    return parsed == normalize_teacher_betas(teacher_betas)


def _format_beta_value(b: float) -> str:
    ib = int(b)
    return str(ib) if b == ib else str(b)


def format_teacher_betas(betas: Sequence[float]) -> str:
    return "[" + ", ".join(_format_beta_value(b) for b in betas) + "]"


def teacher_betas_dirname(betas: Sequence[float]) -> str:
    """Folder name matching Hydra's ``_b[...]_`` fragment, e.g. ``b[1, 1, 1, -1]``."""
    return "b" + format_teacher_betas(betas)


def parse_seed_from_path(path: str) -> Optional[int]:
    m = SEED_RE.search(path)
    return int(m.group("seed")) if m else None


def default_metric_for_env(env: str) -> str:
    folder = resolve_env(env)
    if folder in DEFAULT_METRIC:
        return DEFAULT_METRIC[folder]
    if "metaworld" in folder.lower():
        return "success_rate"
    return "true_episode_reward"


def eval_metrics_for_env(env: str, primary: Optional[str] = None) -> List[str]:
    """Eval CSV columns to plot: primary metric plus return columns when present."""
    primary = primary or default_metric_for_env(env)
    metrics: List[str] = []
    for m in (primary, "true_episode_reward", "episode_reward"):
        if m not in metrics:
            metrics.append(m)
    return metrics


def pretty_metric(metric: str) -> str:
    return {
        "true_episode_reward": "True episode return",
        "episode_reward": "Episode return (learned reward)",
        "success_rate": "Success rate",
        "actor_loss": "Actor loss",
        "critic_loss": "Critic loss",
    }.get(metric, metric.replace("_", " ").title())


def run_label(max_feedback: int, teacher_betas: Sequence[float]) -> str:
    return f"fb={max_feedback}, β={format_teacher_betas(teacher_betas)}"


def discover_envs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    )


def has_seed_eval_runs(path: str) -> bool:
    """True if ``path`` contains ``seed*/test/eval.csv`` (SAC-style bucket)."""
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    for name in names:
        if not SEED_RE.match(name):
            continue
        seed_dir = os.path.join(path, name)
        if not os.path.isdir(seed_dir):
            continue
        if os.path.isfile(os.path.join(seed_dir, "test", "eval.csv")):
            return True
    return False


def list_run_configs(
    env_dir: str,
    env: str,
    *,
    teacher_betas: Optional[Sequence[float]] = None,
    max_feedback: Optional[int] = None,
) -> List[RunConfig]:
    """List Hydra config folders directly under an environment directory.

    Preference-based runs live in ``max_feedbackN_..._b[...]_/``. SAC / GT-reward
    runs live in a flat bucket such as ``tests/seedS/`` with no feedback or β in
    the path; those are returned as ``beta_agnostic`` and match any β / feedback
    filter so they can overlay on preference comparison plots.
    """
    configs: List[RunConfig] = []
    if not os.path.isdir(env_dir):
        return configs
    for name in sorted(os.listdir(env_dir)):
        path = os.path.join(env_dir, name)
        if not os.path.isdir(path):
            continue
        m = FEEDBACK_RE.match(name)
        if m:
            fb = int(m.group("fb"))
            if max_feedback is not None and fb != max_feedback:
                continue
            betas = parse_teacher_betas_from_name(name)
            if betas is None:
                continue
            if teacher_betas is not None and betas != normalize_teacher_betas(
                teacher_betas
            ):
                continue
            configs.append(
                RunConfig(
                    path=path,
                    env=env,
                    max_feedback=fb,
                    teacher_betas=betas,
                    label=run_label(fb, betas),
                )
            )
            continue

        # SAC / GT layout: <env>/tests/seedS/test/eval.csv (no max_feedback / β).
        if not has_seed_eval_runs(path):
            continue
        if teacher_betas is not None:
            betas = normalize_teacher_betas(teacher_betas)
        else:
            betas = ()
        configs.append(
            RunConfig(
                path=path,
                env=env,
                max_feedback=0,
                teacher_betas=betas,
                label="SAC" if name == "tests" else name,
                beta_agnostic=True,
            )
        )
    return configs


def find_csv_files(
    run_dir: str,
    csv_name: str,
    seeds: Optional[Sequence[int]] = None,
) -> List[str]:
    pattern = os.path.join(glob.escape(run_dir), "**", f"{csv_name}.csv")
    files = sorted(f for f in glob.glob(pattern, recursive=True) if os.path.getsize(f) > 0)
    if not seeds:
        return files
    seed_set = {int(s) for s in seeds}
    return [f for f in files if parse_seed_from_path(f) in seed_set]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def align_to_max_steps(
    xs: Sequence[np.ndarray],
    ys: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map seed series onto the union of step grids (max coverage). Missing → NaN.

    ``ys[i]`` is 1D ``(n_steps,)`` or 2D ``(n_steps, n_ch)``.
    Returns ``(x_union, Y)`` with ``Y`` shape ``(n_seeds, n_union[, n_ch])``.
    """
    if not xs:
        raise ValueError("No series to align")
    x_union = np.unique(np.concatenate([np.asarray(x, dtype=float) for x in xs]))
    if x_union.size == 0:
        raise ValueError("Seed runs have no x values")

    extra_shape = np.asarray(ys[0], dtype=float).shape[1:]
    Y = np.full((len(ys), x_union.size, *extra_shape), np.nan, dtype=float)
    pos = {float(v): j for j, v in enumerate(x_union)}
    for i, (x, y) in enumerate(zip(xs, ys)):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        for k, xv in enumerate(x):
            j = pos.get(float(xv))
            if j is not None:
                Y[i, j] = y[k]
    return x_union, Y


def load_seed_series(
    csv_files: Sequence[str], metric: str, x_col: str = "step"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, Y)`` with ``Y`` shape ``(n_seeds, n_steps)`` on a shared x grid."""
    x, Y, _ = load_seed_series_labeled(csv_files, metric, x_col=x_col)
    return x, Y


def resolve_eval_metric_column(df: pd.DataFrame, metric: str) -> Optional[str]:
    """Pick the CSV column for ``metric``.

    SAC MetaWorld logs ground-truth return as ``episode_reward`` only (no
    ``true_episode_reward``). Prefer the requested column; fall back so SAC can
    overlay on true-return comparison plots.
    """
    if metric in df.columns:
        return metric
    if metric == "true_episode_reward" and "episode_reward" in df.columns:
        return "episode_reward"
    return None


def load_seed_series_labeled(
    csv_files: Sequence[str], metric: str, x_col: str = "step"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return ``(x, Y, labels)`` with ``Y`` shape ``(n_seeds, n_steps)``.

    ``x`` spans the longest seed (union of steps). Shorter runs are NaN-padded
    so mean ± CI still cover the full horizon.
    """
    series: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    labels: List[str] = []
    for path in csv_files:
        df = pd.read_csv(path)
        if x_col not in df.columns:
            continue
        col = resolve_eval_metric_column(df, metric)
        if col is None:
            continue
        xs.append(df[x_col].to_numpy(dtype=float))
        series.append(df[col].to_numpy(dtype=float))
        seed = parse_seed_from_path(path)
        labels.append(f"seed{seed}" if seed is not None else os.path.basename(path))

    if not series:
        raise ValueError(f"No usable series for metric={metric!r}")

    x_union, Y = align_to_max_steps(xs, series)
    return x_union, Y, labels


def aggregate(Y: np.ndarray, ci: str = "sem") -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(Y, axis=0)
    n_valid = np.sum(np.isfinite(Y), axis=0).astype(float)
    with np.errstate(invalid="ignore"):
        std = np.nanstd(Y, axis=0, ddof=1)
    std = np.where(n_valid > 1, std, 0.0)
    if ci == "std":
        band = std
    elif ci == "sem":
        band = std / np.sqrt(np.maximum(n_valid, 1.0))
    elif ci == "none":
        band = np.zeros_like(mean)
    else:
        raise ValueError(f"Unknown ci={ci!r}; use sem|std|none")
    return mean, band


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y
    return (
        pd.Series(np.asarray(y, dtype=float))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def trailing_seed_scores(Y: np.ndarray, last_n: int) -> np.ndarray:
    """Last ``last_n`` *finite* eval points per seed (each seed's own horizon)."""
    scores = np.full(Y.shape[0], np.nan, dtype=float)
    for i in range(Y.shape[0]):
        finite = Y[i][np.isfinite(Y[i])]
        if finite.size == 0:
            continue
        n = min(last_n, int(finite.size))
        scores[i] = float(np.mean(finite[-n:]))
    return scores


def load_reward_channels(
    reward_files: Sequence[str],
    col_regex: str,
    max_points: int = 400,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List[str]]]:
    """Load aligned reward CSV channels → ``(x, Y[n_seeds,n_steps,n_ch], col_names, seed_labels)``."""
    seed_mats: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    labels: List[str] = []
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
            cols = [c for c in cols_ref if c in cols]
            if not cols:
                continue
        xs.append(df["step"].to_numpy(dtype=float))
        seed_mats.append(df[cols].to_numpy(dtype=float))
        seed = parse_seed_from_path(path)
        labels.append(f"seed{seed}" if seed is not None else os.path.basename(path))

    if not seed_mats or not cols_ref:
        return None

    try:
        x_union, Y = align_to_max_steps(xs, seed_mats)
    except ValueError:
        return None
    if x_union.size > max_points:
        idx = np.linspace(0, x_union.size - 1, max_points).astype(int)
        x_union = x_union[idx]
        Y = Y[:, idx]
    return x_union, Y, cols_ref, labels


def load_reward_scalar(
    reward_files: Sequence[str],
    column: str,
    max_points: int = 400,
    ci: str = "sem",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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


# ---------------------------------------------------------------------------
# Plotting
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


def save_fig(fig: plt.Figure, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


Curve = Tuple[np.ndarray, np.ndarray, np.ndarray]  # x, mean, band
SeedOverlay = Tuple[np.ndarray, np.ndarray, List[str]]  # x, Y[n_seeds,n_steps], seed labels
# x, Y[n_seeds,n_steps,n_ch], channel names, seed labels
ChannelPanel = Tuple[np.ndarray, np.ndarray, List[str], List[str]]


def plot_learning_curves(
    curves: Dict[str, Curve],
    *,
    metric: str,
    title: Optional[str] = None,
    out_path: str,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    xlabel: str = "Environment steps",
    seed_overlays: Optional[Dict[str, SeedOverlay]] = None,
    seed_colors: Optional[Dict[str, List[str]]] = None,
) -> None:
    if not curves and not seed_overlays:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    x_max = 0.0

    # Individual seed traces first (fainter), then mean ± CI.
    if seed_overlays:
        for g_i, (group, (sx, Y, labs)) in enumerate(seed_overlays.items()):
            if sx.size:
                x_max = max(x_max, float(np.nanmax(sx)))
            palette = (seed_colors or {}).get(group)
            for i in range(Y.shape[0]):
                color = (
                    palette[i % len(palette)]
                    if palette
                    else SERIES_COLORS[i % len(SERIES_COLORS)]
                )
                ok = np.isfinite(Y[i])
                ax.plot(
                    sx[ok],
                    Y[i][ok],
                    color=color,
                    linewidth=1.2,
                    alpha=0.55,
                    label=labs[i],
                )

    for i, (label, (x, mean, band)) in enumerate(curves.items()):
        if x.size:
            x_max = max(x_max, float(np.nanmax(x)))
        color = (colors or {}).get(label, "#222222")
        ls = (linestyles or {}).get(label, "-")
        ok = np.isfinite(mean)
        ax.plot(x[ok], mean[ok], color=color, linestyle=ls, linewidth=2.6, label=label)
        band_ok = ok & np.isfinite(band) & (band > 0)
        if np.any(band_ok):
            ax.fill_between(
                x,
                mean - band,
                mean + band,
                where=band_ok,
                color=color,
                alpha=0.18,
                linewidth=0,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(pretty_metric(metric))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if x_max > 0:
        ax.set_xlim(left=0.0, right=x_max)
    save_fig(fig, out_path)


def plot_final_bars(
    finals: Dict[str, np.ndarray],
    *,
    metric: str,
    title: str,
    out_path: str,
    last_n: int,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    if not finals:
        return
    labels = list(finals.keys())
    means = [float(np.nanmean(finals[k])) for k in labels]
    sems = []
    for k in labels:
        vals = np.asarray(finals[k], dtype=float)
        n_ok = int(np.sum(np.isfinite(vals)))
        if n_ok > 1:
            sems.append(float(np.nanstd(vals, ddof=1) / np.sqrt(n_ok)))
        else:
            sems.append(0.0)
    bar_colors = [
        (colors or {}).get(k, SERIES_COLORS[i % len(SERIES_COLORS)])
        for i, k in enumerate(labels)
    ]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.6))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=sems,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        alpha=0.9,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(pretty_metric(metric))
    ax.set_title(title)
    y_max = max(means[i] + sems[i] for i in range(len(means))) if means else 1.0
    for i, (mu, se) in enumerate(zip(means, sems)):
        ax.text(
            i,
            mu + se + 0.01 * max(y_max, 1e-8),
            f"{mu:.2g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_fig(fig, out_path)
    print(f"  (final bars average last {last_n} eval points per seed)")


def plot_cross_env_curves(
    env_curves: Dict[str, Dict[str, Curve]],
    *,
    metric_by_env: Dict[str, str],
    title: str,
    out_path: str,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
) -> None:
    """One subplot per environment; overlay series within each."""
    envs = [e for e, curves in env_curves.items() if curves]
    if not envs:
        return
    n = len(envs)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 4.2 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()

    for ax, env in zip(axes_flat, envs):
        curves = env_curves[env]
        x_max = 0.0
        for i, (label, (x, mean, band)) in enumerate(curves.items()):
            if x.size:
                x_max = max(x_max, float(np.nanmax(x)))
            color = (colors or {}).get(label, SERIES_COLORS[i % len(SERIES_COLORS)])
            ls = (linestyles or {}).get(label, "-")
            ok = np.isfinite(mean)
            ax.plot(x[ok], mean[ok], color=color, linestyle=ls, label=label)
            band_ok = ok & np.isfinite(band) & (band > 0)
            if np.any(band_ok):
                ax.fill_between(
                    x,
                    mean - band,
                    mean + band,
                    where=band_ok,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
        ax.set_title(env)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(pretty_metric(metric_by_env.get(env, "true_episode_reward")))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend(frameon=False, fontsize=8, loc="best")
        if x_max > 0:
            ax.set_xlim(left=0.0, right=x_max)

    for ax in axes_flat[len(envs) :]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    save_fig(fig, out_path)


REWARD_XLABEL = "Reward step"

# Larger typography for alpha / logit-coef style figures (readable in paper grids).
ALPHA_LABELSIZE = 16
ALPHA_TICKSIZE = 14
ALPHA_LEGENDSIZE = 12
ALPHA_OFFSETSIZE = 13

CHANNEL_SYMBOL = {
    "expert_logits_coef": r"\bar{\alpha}",
    "expert_coef": r"w",
    "alpha_tan": r"\tilde{\alpha}",
    "alpha": r"\alpha",
}


def _style_alpha_axes(ax: plt.Axes, *, xlabel: str, ylabel: str) -> None:
    """Apply larger fonts used by alpha / logit_coef / related reward plots."""
    ax.set_xlabel(xlabel, fontsize=ALPHA_LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=ALPHA_LABELSIZE)
    ax.tick_params(axis="both", labelsize=ALPHA_TICKSIZE)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(ALPHA_OFFSETSIZE)


def _channel_names(
    cols: Sequence[str], n_ch: int, channel_prefix: str
) -> List[str]:
    names = list(cols)
    while len(names) < n_ch:
        names.append(f"{channel_prefix}_{len(names)}")
    return names[:n_ch]


def _pretty_channel_label(name: str, k: Optional[int] = None) -> str:
    """Pretty math label for a reward channel, e.g. ``expert_coef_2`` → ``$w_2$``."""
    m = re.search(r"_(\d+)$", name)
    if k is None and m:
        k = int(m.group(1))
    stem = name[: m.start()] if m else name
    symbol = CHANNEL_SYMBOL.get(stem)
    if symbol is None:
        return name
    if k is None:
        return rf"${symbol}_k$"
    return rf"${symbol}_{{{k}}}$"


def _draw_runs_mean_band(
    ax: plt.Axes,
    x: np.ndarray,
    Y: np.ndarray,
    *,
    seed_colors: Sequence[str],
    ci: str,
) -> None:
    """Faint per-seed traces (distinct colors) + thick mean + CI/std band.

    No legend entries (seeds, mean, or band).
    ``Y`` has shape ``(n_seeds, n_steps)``.
    """
    for i in range(Y.shape[0]):
        y = Y[i]
        ok = np.isfinite(y)
        if not np.any(ok):
            continue
        color = seed_colors[i % len(seed_colors)]
        ax.plot(x[ok], y[ok], color=color, linewidth=1.2, alpha=0.4)

    mean, band = aggregate(Y, ci=ci)
    ok = np.isfinite(mean)
    ax.plot(x[ok], mean[ok], color="black", linewidth=2.6, zorder=3)
    band_ok = ok & np.isfinite(band) & (band > 0)
    if np.any(band_ok):
        ax.fill_between(
            x,
            mean - band,
            mean + band,
            where=band_ok,
            color="gray",
            alpha=0.2,
            linewidth=0,
            zorder=2,
        )


def plot_channel_panels(
    channel_curves: Dict[str, ChannelPanel],
    *,
    title: str,
    out_path: str,
    ylabel: str,
    channel_prefix: str,
    ci: str = "sem",
    show_seeds: bool = False,
) -> None:
    """One subplot per series; each expert in its own color with a thick mean.

    When ``show_seeds`` is true, faint per-seed traces are drawn in the same
    expert color under the mean ± CI (``--root`` / per-seed mode).
    """
    if not channel_curves:
        return
    labels = list(channel_curves.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False, sharex=True
    )
    axes_flat = axes.ravel()

    for ax, label in zip(axes_flat, labels):
        x, Y, cols, _seed_labs = channel_curves[label]
        n_ch = Y.shape[2]
        names = _channel_names(cols, n_ch, channel_prefix)
        for k in range(n_ch):
            color = SERIES_COLORS[k % len(SERIES_COLORS)]
            if show_seeds:
                for i in range(Y.shape[0]):
                    y_i = Y[i, :, k]
                    ok_i = np.isfinite(y_i)
                    if np.any(ok_i):
                        ax.plot(
                            x[ok_i],
                            y_i[ok_i],
                            color=color,
                            linewidth=1.0,
                            alpha=0.28,
                            zorder=2,
                        )
            mean_k, band_k = aggregate(Y[:, :, k], ci=ci)
            ok = np.isfinite(mean_k)
            ax.plot(
                x[ok],
                mean_k[ok],
                color=color,
                linestyle=EXPERT_LINESTYLES[k % len(EXPERT_LINESTYLES)],
                label=_pretty_channel_label(names[k], k),
                linewidth=2.6,
                zorder=3,
            )
            band_ok = ok & np.isfinite(band_k) & (band_k > 0)
            if np.any(band_ok):
                ax.fill_between(
                    x,
                    mean_k - band_k,
                    mean_k + band_k,
                    where=band_ok,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        _style_alpha_axes(ax, xlabel=REWARD_XLABEL, ylabel=ylabel)
        if x.size:
            ax.set_xlim(left=0.0, right=float(np.nanmax(x)))
        ax.legend(frameon=False, fontsize=ALPHA_LEGENDSIZE, loc="best")

    for ax in axes_flat[len(labels) :]:
        ax.axis("off")

    save_fig(fig, out_path)


def plot_channel_with_runs(
    channel_curves: Dict[str, ChannelPanel],
    *,
    out_dir: str,
    channel_prefix: str,
    ci: str = "sem",
) -> None:
    """One figure per expert: seed runs in separate colors + thick mean ± band.

    Same style as ``results/reward_alpha_0_with_runs.pdf``. Multiple series
    become side-by-side subplots on that expert's figure.
    """
    if not channel_curves:
        return
    labels = list(channel_curves.keys())
    n_ch = max(Y.shape[2] for _, Y, _, _ in channel_curves.values())
    first_cols = next(iter(channel_curves.values()))[2]
    names = _channel_names(first_cols, n_ch, channel_prefix)
    n_series = len(labels)
    seed_colors = SERIES_COLORS

    for k, name in enumerate(names):
        pretty = _pretty_channel_label(name, k)
        fig, axes = plt.subplots(
            1,
            n_series,
            figsize=(10.0 if n_series == 1 else 5.6 * n_series, 6.0),
            squeeze=False,
            sharey=True,
        )
        axes_flat = axes.ravel()
        for ax, label in zip(axes_flat, labels):
            x, Y, _cols, _seed_labs = channel_curves[label]
            if k >= Y.shape[2]:
                ax.set_visible(False)
                continue
            _draw_runs_mean_band(
                ax,
                x,
                Y[:, :, k],
                seed_colors=seed_colors,
                ci=ci,
            )
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
            _style_alpha_axes(ax, xlabel=REWARD_XLABEL, ylabel=pretty)
            if x.size:
                ax.set_xlim(left=0.0, right=float(np.nanmax(x)))

        out_path = os.path.join(out_dir, f"{name}_with_runs.png")
        save_fig(fig, out_path)


def plot_scalar_overlay(
    curves: Dict[str, Curve],
    *,
    title: str,
    out_path: str,
    ylabel: str,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x_max = 0.0
    for i, (label, (x, mean, band)) in enumerate(curves.items()):
        if x.size:
            x_max = max(x_max, float(np.nanmax(x)))
        color = (colors or {}).get(label, SERIES_COLORS[i % len(SERIES_COLORS)])
        ok = np.isfinite(mean)
        ax.plot(x[ok], mean[ok], color=color, label=label)
        band_ok = ok & np.isfinite(band) & (band > 0)
        if np.any(band_ok):
            ax.fill_between(
                x,
                mean - band,
                mean + band,
                where=band_ok,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
    _style_alpha_axes(ax, xlabel=REWARD_XLABEL, ylabel=ylabel)
    ax.legend(frameon=False, loc="best", fontsize=ALPHA_LEGENDSIZE)
    if x_max > 0:
        ax.set_xlim(left=0.0, right=x_max)
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_series_arg(raw: str, index: int) -> SeriesSpec:
    """Parse ``name:path`` or bare ``path`` (name = basename)."""
    if ":" in raw and not (raw.startswith("/") and raw.count(":") == 1 and "\\" not in raw):
        # Prefer split on first ':' that separates name from path.
        # Windows drive letters are rare here; still allow bare paths without name.
        name, path = raw.split(":", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Bad --series {raw!r}; expected name:path")
    else:
        path = raw
        name = os.path.basename(os.path.normpath(path))
    return SeriesSpec(
        name=name,
        root=path,
        color=SERIES_COLORS[index % len(SERIES_COLORS)],
        linestyle=["-", "--", "-.", ":"][index % 4],
    )


def collect_env_names(series_list: Sequence[SeriesSpec], requested: Sequence[str]) -> List[str]:
    if requested:
        return [resolve_env(e) for e in requested]
    envs: set = set()
    for spec in series_list:
        root = abs_under_repo(spec.root)
        envs.update(discover_envs(root))
    return sorted(envs)


def collect_teacher_betas(
    series_list: Sequence[SeriesSpec],
    env: str,
    *,
    teacher_betas: Optional[Sequence[float]] = None,
    max_feedback: Optional[int] = None,
) -> List[Optional[Tuple[float, ...]]]:
    """Teacher-beta groups to plot for one env (one output subfolder each)."""
    if teacher_betas is not None:
        return [normalize_teacher_betas(teacher_betas)]
    found: List[Tuple[float, ...]] = []
    seen = set()
    for spec in series_list:
        env_dir = os.path.join(abs_under_repo(spec.root), env)
        for cfg in list_run_configs(env_dir, env, max_feedback=max_feedback):
            if cfg.beta_agnostic:
                continue
            if cfg.teacher_betas not in seen:
                seen.add(cfg.teacher_betas)
                found.append(cfg.teacher_betas)
    return found if found else [None]


def build_curve_from_files(
    eval_files: Sequence[str],
    metric: str,
    *,
    ci: str,
    smooth: int,
) -> Optional[Tuple[Curve, np.ndarray, int]]:
    """Return ``((x, mean, band), seed_finals, n_seeds)`` or None."""
    if not eval_files:
        return None
    try:
        x, Y = load_seed_series(eval_files, metric)
    except ValueError as exc:
        print(f"  skip eval: {exc}")
        return None
    mean, band = aggregate(Y, ci=ci)
    if smooth > 1:
        mean = smooth_curve(mean, smooth)
        band = smooth_curve(band, smooth)
    return (x, mean, band), Y, Y.shape[0]


def group_label_for(
    spec: SeriesSpec,
    cfg: RunConfig,
    *,
    group_by: str,
    multi_series: bool,
    n_configs: int,
) -> str:
    if group_by == "series":
        return spec.name
    if group_by == "config":
        return cfg.label if not multi_series else f"{spec.name} | {cfg.label}"
    root_base = os.path.basename(os.path.normpath(spec.root))
    named = spec.name != root_base
    if multi_series:
        return spec.name if n_configs == 1 else f"{spec.name} | {cfg.label}"
    if n_configs > 1:
        return cfg.label
    return spec.name if named else cfg.label


def seed_curve_label(
    seed_lab: str,
    group_label: str,
    *,
    per_seed: bool,
    multi_series: bool,
    n_configs: int,
) -> str:
    if not per_seed:
        return group_label
    if multi_series or n_configs > 1:
        return f"{group_label} | {seed_lab}"
    return seed_lab


def plot_env(
    env: str,
    series_list: Sequence[SeriesSpec],
    *,
    out_dir: str,
    metric: Optional[str],
    teacher_betas: Optional[Sequence[float]],
    max_feedback: Optional[int],
    seeds: Optional[Sequence[int]],
    ci: str,
    last_n: int,
    smooth: int,
    skip_reward: bool,
    group_by: str,
    per_seed: bool = False,
) -> Optional[Dict[str, Curve]]:
    """Plot one environment across series / configs. Returns primary-metric curves.

    ``per_seed=True`` (``--root``): seed lines + mean ± CI on the same figure.
    ``per_seed=False`` (``--series``): mean ± CI conclusion plot only.
    """
    primary_metric = metric or default_metric_for_env(env)
    metrics = eval_metrics_for_env(env, primary_metric)
    os.makedirs(out_dir, exist_ok=True)
    mode_tag = "seeds + mean" if per_seed else "series mean±CI"
    print(
        f"\n{'=' * 72}\n{env}  [{mode_tag}]  metrics={metrics}  → {out_dir}\n{'=' * 72}"
    )

    curves_by_metric: Dict[str, Dict[str, Curve]] = {m: {} for m in metrics}
    finals_by_metric: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in metrics}
    seed_overlays_by_metric: Dict[str, Dict[str, SeedOverlay]] = {m: {} for m in metrics}
    seed_colors_by_metric: Dict[str, Dict[str, List[str]]] = {m: {} for m in metrics}
    colors: Dict[str, str] = {}
    linestyles: Dict[str, str] = {}
    alpha_panels: Dict[str, ChannelPanel] = {}
    coef_panels: Dict[str, ChannelPanel] = {}
    alpha_tan_panels: Dict[str, ChannelPanel] = {}
    logit_coef_panels: Dict[str, ChannelPanel] = {}
    abs_sum_curves: Dict[str, Curve] = {}
    summary_rows_by_metric: Dict[str, List[dict]] = {m: [] for m in metrics}

    multi_series = len(series_list) > 1

    for spec in series_list:
        root = abs_under_repo(spec.root)
        env_dir = os.path.join(root, env)
        configs = list_run_configs(
            env_dir,
            env,
            teacher_betas=teacher_betas,
            max_feedback=max_feedback,
        )
        if not configs:
            print(f"  [{spec.name}] no matching runs under {env_dir}")
            continue

        for cfg in configs:
            label = group_label_for(
                spec,
                cfg,
                group_by=group_by,
                multi_series=multi_series,
                n_configs=len(configs),
            )

            eval_files = find_csv_files(cfg.path, "eval", seeds=seeds)
            if not eval_files:
                print(f"  [{label}] no eval.csv")
            elif per_seed:
                for plot_metric in metrics:
                    try:
                        x, Y, seed_labs = load_seed_series_labeled(eval_files, plot_metric)
                    except ValueError as exc:
                        print(f"  [{label}] skip {plot_metric}: {exc}")
                        continue
                    Y_plot = np.stack(
                        [smooth_curve(Y[i], smooth) if smooth > 1 else Y[i] for i in range(Y.shape[0])]
                    )
                    mean, band = aggregate(Y, ci=ci)
                    if smooth > 1:
                        mean = smooth_curve(mean, smooth)
                        band = smooth_curve(band, smooth)
                    mean_label = "Mean" if not multi_series and len(configs) == 1 else f"{label} mean"
                    palette = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(Y.shape[0])]
                    overlay_labs = [
                        seed_curve_label(
                            seed_lab,
                            label,
                            per_seed=True,
                            multi_series=multi_series,
                            n_configs=len(configs),
                        )
                        for seed_lab in seed_labs
                    ]
                    seed_overlays_by_metric[plot_metric][label] = (x, Y_plot, overlay_labs)
                    seed_colors_by_metric[plot_metric][label] = palette
                    colors[mean_label] = spec.color if multi_series or len(configs) > 1 else "#222222"
                    linestyles[mean_label] = spec.linestyle
                    curves_by_metric[plot_metric][mean_label] = (x, mean, band)

                    seed_scores = trailing_seed_scores(Y, last_n)
                    n_ok = int(np.sum(np.isfinite(seed_scores)))
                    final_mu = float(np.nanmean(seed_scores)) if n_ok else float("nan")
                    final_sd = float(np.nanstd(seed_scores, ddof=1)) if n_ok > 1 else 0.0
                    for i, seed_lab in enumerate(seed_labs):
                        curve_label = overlay_labs[i]
                        finals_by_metric[plot_metric][curve_label] = np.array([float(seed_scores[i])])
                        colors.setdefault(curve_label, palette[i])
                        print(
                            f"  [{curve_label:40s}] {plot_metric:22s}  "
                            f"last-{last_n}={seed_scores[i]:.3g}"
                        )
                        summary_rows_by_metric[plot_metric].append(
                            {
                                "series": spec.name,
                                "env": env,
                                "label": curve_label,
                                "seed": seed_lab,
                                "max_feedback": cfg.max_feedback,
                                "teacher_betas": format_teacher_betas(cfg.teacher_betas),
                                "metric": plot_metric,
                                "n_seeds": 1,
                                "final_mean": float(seed_scores[i]),
                                "final_std": 0.0,
                                "last_n": last_n,
                            }
                        )
                    finals_by_metric[plot_metric][mean_label] = seed_scores
                    colors.setdefault(mean_label, "#222222")
                    print(
                        f"  [{mean_label:40s}] {plot_metric:22s} seeds={Y.shape[0]}  "
                        f"last-{last_n} mean={final_mu:.3g} ± {final_sd:.3g}"
                    )
                    summary_rows_by_metric[plot_metric].append(
                        {
                            "series": spec.name,
                            "env": env,
                            "label": mean_label,
                            "seed": "mean",
                            "max_feedback": cfg.max_feedback,
                            "teacher_betas": format_teacher_betas(cfg.teacher_betas),
                            "metric": plot_metric,
                            "n_seeds": int(Y.shape[0]),
                            "final_mean": final_mu,
                            "final_std": final_sd,
                            "last_n": last_n,
                        }
                    )
            else:
                colors[label] = spec.color
                linestyles[label] = spec.linestyle
                for plot_metric in metrics:
                    built = build_curve_from_files(
                        eval_files, plot_metric, ci=ci, smooth=smooth
                    )
                    if built is None:
                        continue
                    curve, Y, n_seeds = built
                    curves_by_metric[plot_metric][label] = curve
                    seed_scores = trailing_seed_scores(Y, last_n)
                    n_ok = int(np.sum(np.isfinite(seed_scores)))
                    final_mu = float(np.nanmean(seed_scores)) if n_ok else float("nan")
                    final_sd = float(np.nanstd(seed_scores, ddof=1)) if n_ok > 1 else 0.0
                    finals_by_metric[plot_metric][label] = seed_scores
                    print(
                        f"  [{label:40s}] {plot_metric:22s} seeds={n_seeds}  "
                        f"last-{last_n} mean={final_mu:.3g} ± {final_sd:.3g}"
                    )
                    summary_rows_by_metric[plot_metric].append(
                        {
                            "series": spec.name,
                            "env": env,
                            "label": label,
                            "max_feedback": cfg.max_feedback,
                            "teacher_betas": format_teacher_betas(cfg.teacher_betas),
                            "metric": plot_metric,
                            "n_seeds": n_seeds,
                            "final_mean": final_mu,
                            "final_std": final_sd,
                            "last_n": last_n,
                        }
                    )

            if skip_reward or not series_plots_alpha_trust(spec, multi_series=multi_series):
                continue

            reward_files = find_csv_files(cfg.path, "reward", seeds=seeds)
            if not reward_files:
                print(f"  [{label}] no reward.csv")
                continue

            def _add_channel_panels(
                panels: Dict[str, ChannelPanel],
                loaded: Optional[Tuple[np.ndarray, np.ndarray, List[str], List[str]]],
                kind: str,
            ) -> None:
                if loaded is None:
                    return
                x_ch, Y_ch, cols_ch, seed_labs = loaded
                panels[label] = (x_ch, Y_ch, cols_ch, seed_labs)
                print(f"  [{label:40s}] {kind} channels={cols_ch} seeds={seed_labs}")

            _add_channel_panels(
                alpha_panels, load_reward_channels(reward_files, r"alpha_\d+"), "alphas"
            )
            _add_channel_panels(
                alpha_tan_panels,
                load_reward_channels(reward_files, r"alpha_tan_\d+"),
                "alpha_tan",
            )
            _add_channel_panels(
                coef_panels,
                load_reward_channels(reward_files, r"expert_coef_\d+"),
                "expert_coef",
            )
            _add_channel_panels(
                logit_coef_panels,
                load_reward_channels(reward_files, r"expert_logits_coef_\d+"),
                "expert_logits_coef",
            )

            abs_sum = load_reward_scalar(reward_files, "alpha_abs_sum", ci=ci)
            if abs_sum is not None:
                abs_sum_curves[label] = abs_sum
                colors.setdefault(label, spec.color)

    has_eval = any(curves_by_metric[m] for m in metrics)
    if not has_eval and not alpha_panels:
        print(f"Nothing to plot for {env}.")
        return None

    for plot_metric in metrics:
        curves = curves_by_metric[plot_metric]
        finals = finals_by_metric[plot_metric]
        summary_rows = summary_rows_by_metric[plot_metric]
        if not curves:
            continue
        plot_learning_curves(
            curves,
            metric=plot_metric,
            out_path=os.path.join(out_dir, f"learning_curve_{plot_metric}.png"),
            colors=colors,
            linestyles=linestyles,
            seed_overlays=seed_overlays_by_metric.get(plot_metric) or None,
            seed_colors=seed_colors_by_metric.get(plot_metric) or None,
        )
        plot_final_bars(
            finals,
            metric=plot_metric,
            title=f"Final performance (last {last_n} evals)",
            out_path=os.path.join(out_dir, f"final_bar_{plot_metric}.png"),
            last_n=last_n,
            colors=colors,
        )
        if summary_rows:
            table = pd.DataFrame(summary_rows)
            table_path = os.path.join(out_dir, f"summary_{plot_metric}.csv")
            table.to_csv(table_path, index=False, float_format="%.4f")
            print(f"Saved {table_path}")
            print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    if alpha_panels:
        plot_channel_panels(
            alpha_panels,
            title=r"Trust parameters $\alpha_k$",
            out_path=os.path.join(out_dir, "alphas.png"),
            ylabel=r"$\alpha_k$",
            channel_prefix="alpha",
            ci=ci,
            show_seeds=per_seed,
        )
        plot_channel_with_runs(
            alpha_panels,
            out_dir=out_dir,
            channel_prefix="alpha",
            ci=ci,
        )
    if alpha_tan_panels:
        plot_channel_panels(
            alpha_tan_panels,
            title=r"Trust parameters $\tilde\alpha_k$ (tanh)",
            out_path=os.path.join(out_dir, "alpha_tan.png"),
            ylabel=r"$\tilde\alpha_k$",
            channel_prefix="alpha_tan",
            ci=ci,
            show_seeds=per_seed,
        )
        plot_channel_with_runs(
            alpha_tan_panels,
            out_dir=out_dir,
            channel_prefix="alpha_tan",
            ci=ci,
        )
    if coef_panels:
        plot_channel_panels(
            coef_panels,
            title="Expert coefficients",
            out_path=os.path.join(out_dir, "expert_coefficients.png"),
            ylabel=r"$w_k$",
            channel_prefix="expert_coef",
            ci=ci,
            show_seeds=per_seed,
        )
        plot_channel_with_runs(
            coef_panels,
            out_dir=out_dir,
            channel_prefix="expert_coef",
            ci=ci,
        )
    if logit_coef_panels:
        plot_channel_panels(
            logit_coef_panels,
            title=r"Expert logit coefficients $a_{bar}$",
            out_path=os.path.join(out_dir, "expert_logit_coefs.png"),
            ylabel=r"$\bar{\alpha}_k$",
            channel_prefix="expert_logits_coef",
            ci=ci,
            show_seeds=per_seed,
        )
        plot_channel_with_runs(
            logit_coef_panels,
            out_dir=out_dir,
            channel_prefix="expert_logits_coef",
            ci=ci,
        )
    if abs_sum_curves:
        plot_scalar_overlay(
            abs_sum_curves,
            title=r"$|\alpha|_1$",
            out_path=os.path.join(out_dir, "alpha_abs_sum.png"),
            ylabel=r"$|\alpha|_1$",
            colors=colors,
        )

    return curves_by_metric.get(primary_metric)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="General learning-curve / alpha plots for PEBBLE mixture runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Experiment root: one figure with every seed line plus mean ± CI. "
        "Example: --root exp_pebble_mixture_zero_last_wk_sgd. "
        "Ignored if --series is set.",
    )
    p.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="Conclusion plot: mean ± CI across seeds. Repeat to overlay roots. "
        "Example: --series zero_last:exp_pebble_mixture_zero_last "
        "--series wk_sgd:exp_pebble_mixture_zero_last_wk_sgd",
    )
    p.add_argument("--env", type=str, default=None, help="Single environment folder / alias")
    p.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment list (aliases ok). Default: discover all under root(s).",
    )
    p.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Eval CSV column. Default: success_rate for MetaWorld, "
        "true_episode_reward otherwise.",
    )
    p.add_argument(
        "--teacher-betas",
        type=float,
        nargs="+",
        default=None,
        help="Filter runs by teacher_betas encoded in the folder name",
    )
    p.add_argument(
        "--max-feedback",
        type=int,
        default=None,
        help="Filter runs by max_feedbackN prefix",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only include these seeds",
    )
    p.add_argument("--ci", choices=["sem", "std", "none"], default="sem")
    p.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Number of trailing eval points averaged for final bars / summary",
    )
    p.add_argument("--smooth", type=int, default=1, help="Moving-average window (≥1)")
    p.add_argument(
        "--skip-reward",
        action="store_true",
        help="Skip alpha / expert-coefficient plots from reward.csv",
    )
    p.add_argument(
        "--group-by",
        choices=["auto", "series", "config"],
        default="auto",
        help="How to label overlapping curves within an env",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: results/<root_basename>/<env>/b[...]/)",
    )
    p.add_argument(
        "--no-cross-env",
        action="store_true",
        help="Do not write the multi-environment overview figure",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_style()

    series_list: List[SeriesSpec] = []
    per_seed = False
    if args.series:
        for i, raw in enumerate(args.series):
            series_list.append(parse_series_arg(raw, i))
        per_seed = False
    elif args.root:
        series_list.append(parse_series_arg(args.root, 0))
        per_seed = True
    else:
        # Sensible default for this project: per-seed plots of zero_last.
        series_list.append(parse_series_arg("exp_pebble_mixture_zero_last", 0))
        per_seed = True

    for spec in series_list:
        root = abs_under_repo(spec.root)
        if not os.path.isdir(root):
            print(f"ERROR: experiment root not found: {root}")
            return 1

    requested: List[str] = []
    if args.env:
        requested.append(args.env)
    if args.envs:
        requested.extend(args.envs)
    env_names = collect_env_names(series_list, requested)
    if not env_names:
        print("No environments found.")
        return 1

    if args.out:
        out_root = abs_under_repo(args.out)
    elif len(series_list) == 1:
        out_root = os.path.join(
            repo_root(), "results", os.path.basename(os.path.normpath(series_list[0].root))
        )
    else:
        out_root = os.path.join(repo_root(), "results", "experiment_compare")

    print("Plot experiments")
    print(f"  mode   : {'seeds+mean (--root)' if per_seed else 'conclusion (--series)'}")
    print(f"  series : {[(s.name, s.root) for s in series_list]}")
    print(f"  envs   : {env_names}")
    print(f"  out    : {out_root}")
    if len(series_list) > 1 and not args.skip_reward:
        print("  alpha    : TTP only (reward.csv skipped for other series)")

    env_curves_by_beta: Dict[Tuple[float, ...], Dict[str, Dict[str, Curve]]] = {}
    metric_by_env: Dict[str, str] = {}
    any_ok = False

    for env in env_names:
        metric = args.metric or default_metric_for_env(env)
        metric_by_env[env] = metric
        beta_groups = collect_teacher_betas(
            series_list,
            env,
            teacher_betas=args.teacher_betas,
            max_feedback=args.max_feedback,
        )
        for betas in beta_groups:
            if betas is None:
                out_dir = os.path.join(out_root, env)
            else:
                out_dir = os.path.join(out_root, env, teacher_betas_dirname(betas))
            curves = plot_env(
                env,
                series_list,
                out_dir=out_dir,
                metric=metric,
                teacher_betas=betas,
                max_feedback=args.max_feedback,
                seeds=args.seeds,
                ci=args.ci,
                last_n=args.last_n,
                smooth=args.smooth,
                skip_reward=args.skip_reward,
                group_by=args.group_by,
                per_seed=per_seed,
            )
            if curves:
                env_curves_by_beta.setdefault(betas or (), {})[env] = curves
                any_ok = True

    if any_ok and not args.no_cross_env:
        for betas, env_curves in env_curves_by_beta.items():
            if len(env_curves) <= 1:
                continue
            # Color map: reuse first series color per label when possible.
            colors: Dict[str, str] = {}
            linestyles: Dict[str, str] = {}
            for spec in series_list:
                colors[spec.name] = spec.color
                linestyles[spec.name] = spec.linestyle
            seen_labels: List[str] = []
            for curves in env_curves.values():
                for label in curves:
                    if label not in seen_labels:
                        seen_labels.append(label)
            for i, label in enumerate(seen_labels):
                colors.setdefault(label, SERIES_COLORS[i % len(SERIES_COLORS)])

            beta_tag = teacher_betas_dirname(betas) if betas else None
            title = "Cross-environment overview"
            if beta_tag:
                title += f" ({beta_tag})"
                out_name = f"cross_env_learning_curves_{beta_tag}.png"
            else:
                out_name = "cross_env_learning_curves.png"
            plot_cross_env_curves(
                env_curves,
                metric_by_env=metric_by_env,
                title=title,
                out_path=os.path.join(out_root, out_name),
                colors=colors,
                linestyles=linestyles,
            )

    if not any_ok:
        print("No plots were generated.")
        return 1

    print(f"\nAll figures → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
