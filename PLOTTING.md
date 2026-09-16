# Plotting experiments

General plotting for Hydra PEBBLE / TriTrust-PBRL run trees. Works out of the box for **`exp_pebble_mixture_zero_last`** and any folder with the same layout.

For enhancement ablations (`ablation_t*_m*_w*`) and buffer diagnostics, use the specialized script instead:

```bash
python plot_enhancement_ablation.py --help
```

---

## Expected layout

```text
<root>/
  <env>/
    max_feedbackN_feed_type..._b[1, 1, 1, -1]_m0_s0_e0/
      seedS/
        test/eval.csv
        reward/reward.csv      # optional (α_k, expert coefs)
```

Example (`exp_pebble_mixture_zero_last`):

```text
exp_pebble_mixture_zero_last/
  walker_walk/
    max_feedback5000_..._b[1, 1, 1, -1]_.../
      seed12345/test/eval.csv
  metaworld_door-open-v2/
    max_feedback40000_..._b[1, 1, 1, -1]_.../
      seed12345/test/eval.csv
```

---

## Quick start (zero_last)

From the repo root (host or Docker container):

```bash
# Plot every env found under exp_pebble_mixture_zero_last
python plot_experiments.py --root exp_pebble_mixture_zero_last

# Same thing (this is also the script default if you omit --root/--series)
python plot_experiments.py
```

Figures land in:

```text
results/exp_pebble_mixture_zero_last/<env>/
  learning_curve_<metric>.png|.pdf   # primary + episode_reward + true_episode_reward
  final_bar_<metric>.png|.pdf
  alphas.png|.pdf                 # if reward.csv present
  expert_coefficients.png|.pdf
  alpha_abs_sum.png|.pdf
  summary_<metric>.csv
results/exp_pebble_mixture_zero_last/
  cross_env_learning_curves.png|.pdf   # when ≥2 envs plotted
```

By default each environment gets learning curves and final bars for the **primary metric** (MetaWorld: `success_rate`; DM Control: `true_episode_reward`) **plus** `episode_reward` and `true_episode_reward` whenever those columns exist in `eval.csv`.

---

## Common recipes

### One environment

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last --env walker_walk
```

Aliases work for MetaWorld / short names:

| Alias | Folder |
|---|---|
| `door_open` | `metaworld_door-open-v2` |
| `sweep_into` | `metaworld_sweep-into-v2` |
| `walker` | `walker_walk` |
| `cheetah` | `cheetah_run` |

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last --env door_open
```

### Several environments

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last \
  --envs walker_walk door_open sweep_into cheetah_run
```

### Choose the eval metric

Defaults:

- DM Control → `true_episode_reward`
- MetaWorld → `success_rate`

Override:

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last \
  --env walker_walk --metric episode_reward
```

Useful columns in `test/eval.csv`:

- `true_episode_reward` — ground-truth return
- `episode_reward` — return under the learned reward
- `success_rate` — MetaWorld success (0–1 in logs; plotted as stored)

### Filter by feedback budget / teachers / seeds

```bash
# Only max_feedback=5000
python plot_experiments.py --root exp_pebble_mixture_zero_last \
  --env walker_walk --max-feedback 5000

# Only teacher_betas = [1,1,1,-1]
python plot_experiments.py --root exp_pebble_mixture_zero_last \
  --teacher-betas 1 1 1 -1

# Subset of seeds
python plot_experiments.py --root exp_pebble_mixture_zero_last \
  --seeds 12345 23451 34512
```

### Error bands and smoothing

```bash
# Mean ± SEM across seeds (default)
python plot_experiments.py --root exp_pebble_mixture_zero_last --ci sem

# Mean ± std, or hide bands
python plot_experiments.py --root exp_pebble_mixture_zero_last --ci std
python plot_experiments.py --root exp_pebble_mixture_zero_last --ci none

# Moving-average window on the mean/band
python plot_experiments.py --root exp_pebble_mixture_zero_last --smooth 3
```

### Final-performance window

Final bars / summary CSV average the last `N` eval points per seed (default `10`):

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last --last-n 5
```

### Skip α / expert plots

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last --skip-reward
```

### Custom output directory

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last \
  --out results/my_zero_last_plots
```

---

## Comparing multiple experiment roots

Overlay two (or more) trees as named series:

```bash
python plot_experiments.py \
  --series zero_last:exp_pebble_mixture_zero_last \
  --series other:exp_some_other_method \
  --env walker_walk
```

Bare paths are also fine (series name = folder basename):

```bash
python plot_experiments.py \
  --series exp_pebble_mixture_zero_last \
  --series exp_pebble_mixture_ablation
```

Outputs go to `results/experiment_compare/` unless you pass `--out`.

Curve labels:

| `--group-by` | Behavior |
|---|---|
| `auto` (default) | Prefer series name when each root has one config; otherwise `series \| fb=…, β=…` |
| `series` | Always use the series name |
| `config` | Always use `fb=…, β=…` (prefix with series when comparing roots) |

---

## What each figure shows

| File | Description |
|---|---|
| `learning_curve_<metric>.png` | Mean metric vs env steps (± CI when >1 seed). Generated for primary metric plus `episode_reward` and `true_episode_reward`. |
| `final_bar_<metric>.png` | Last-`N` mean per series/config (same metrics as above) |
| `alphas.png` | Per-expert trust parameters \(\alpha_k\) from `reward.csv` |
| `expert_coefficients.png` | Softmax-style expert coefficients |
| `alpha_abs_sum.png` | \(\|\alpha\|_1\) over training |
| `summary_<metric>.csv` | Tabular final means / stds |
| `cross_env_learning_curves.png` | Multi-panel overview across envs |

PNG and PDF are written for every figure.

---

## CLI reference

```text
python plot_experiments.py --help
```

| Flag | Default | Meaning |
|---|---|---|
| `--root PATH` | `exp_pebble_mixture_zero_last` | Single experiment root |
| `--series NAME:PATH` | — | Named root(s) to overlay (repeatable) |
| `--env` / `--envs` | all discovered | Environment filter (aliases ok) |
| `--metric` | env-dependent | Column in `eval.csv` |
| `--teacher-betas` | all | Filter by `_b[...]_` in folder name |
| `--max-feedback` | all | Filter by `max_feedbackN_` prefix |
| `--seeds` | all | Seed filter |
| `--ci {sem,std,none}` | `sem` | Uncertainty band |
| `--last-n` | `10` | Trailing evals for finals |
| `--smooth` | `1` | MA window (1 = off) |
| `--skip-reward` | off | No α / coef plots |
| `--group-by {auto,series,config}` | `auto` | Curve labeling |
| `--out` | `results/<root>/` | Output directory |
| `--no-cross-env` | off | Skip overview figure |

---

## Dependencies

Uses the project stack already installed in Docker: `numpy`, `pandas`, `matplotlib`. No extra packages.

```bash
# inside the container
docker compose exec mixture_pbrl bash
cd /workspace
python plot_experiments.py --root exp_pebble_mixture_zero_last
```

---

## Tips

1. **One seed** (current zero_last runs): learning curves still plot; CI bands are zero-width; final bars show a single value.
2. **Multiple configs under one env** (different feedback or \(\beta\)): they appear as separate curves labeled `fb=…, β=…`.
3. **Ablation trees** with `ablation_tTrue_mTrue_wTrue/...` nesting are **not** this layout — use `plot_enhancement_ablation.py`.
4. Re-run the script anytime; it overwrites figures in the output folder.
