# TriTrust-PBRL (TTP)

Official implementation of **TriTrust-PBRL (TTP)** from *Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback*.

TTP jointly learns a shared reward model and expert-specific trust parameters from multi-expert preference feedback. During training, each trust parameter evolves toward:

- **positive** → trust (reliable expert)
- **near zero** → down-weight (noisy expert)
- **negative** → invert (systematically adversarial expert)

This lets the method recover useful signal from anti-aligned annotators instead of discarding them.

## Environments

Policy-learning experiments follow the paper and cover:

| Domain | Environment ID | Metric |
|---|---|---|
| DM Control | `cheetah_run` | true episode return |
| DM Control | `walker_walk` | true episode return |
| MetaWorld | `metaworld_door-open-v2` | success rate |
| MetaWorld | `metaworld_sweep-into-v2` | success rate |

Multi-expert teachers use B-Pref-style rationalities \(\beta_k \in \{-1, 0, 1\}\):

- `1`: reliable
- `0`: noisy / random
- `-1`: adversarial (preference flip)

Principal mixtures in the paper: adversarial `teacher_betas=[1,1,1,-1]` and noisy `teacher_betas=[1,1,1,0]`.

## Installation (Docker)

The recommended setup uses the prepared `dockerfile` and `docker-compose.yml` (PyTorch 1.11 + CUDA 11.3, MuJoCo 2.1.0, MetaWorld v2.0.0, and project dependencies).

### Requirements

- Docker
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU)

### Build and start

```bash
# from the repository root
docker compose build
docker compose up -d
docker compose exec mixture_pbrl bash
```

Inside the container the project is mounted at `/workspace` with `MUJOCO_GL=egl`.

Equivalent manual Docker run:

```bash
docker build -f dockerfile -t mixture_pbrl .
docker run -it --rm --gpus all --shm-size=16g \
  -e MUJOCO_GL=egl \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --name mixture_pbrl_container \
  mixture_pbrl bash
```

### Manual install (optional)

If you prefer a local environment (Python 3.8), mirror the Dockerfile steps: install MuJoCo 2.1.0 / `mujoco-py`, then:

```bash
pip install mujoco==2.3.5
pip install gym==0.25.2
pip install dm_control==1.0.12
pip install git+https://github.com/denisyarats/dmc2gym.git
pip install tensorboard termcolor pybullet scikit-image
pip install hydra-core==1.0.4
pip install transformers==4.32.0
pip install "cython<3"

wget https://github.com/Farama-Foundation/Metaworld/archive/refs/tags/v2.0.0.tar.gz
tar -xvzf v2.0.0.tar.gz
cd Metaworld-2.0.0
pip install .
```

### Hydra note

This repo uses Hydra `1.0.4` (B-Pref originally used `0.x`). Config paths and `@hydra.main` arguments were adjusted accordingly; see [Hydra config path changes](https://hydra.cc/docs/upgrades/0.11_to_1.0/config_path_changes/).

## How to run

Run commands from the repository root (inside the Docker container if using Docker). Prepared runner scripts live under `scripts/<env>/run_experiments.py`.

### TriTrust-PBRL (TTP)

Main entrypoint: `train_PEBBLE_mixture.py` (config `train_PEBBLE_mixture`).

#### MetaWorld Sweep-Into (adversarial)

```bash
python scripts/sweep_into/run_experiments.py --method pebble_mixture --preset 1,1,1,-1
# or a single seed:
python train_PEBBLE_mixture.py env=metaworld_sweep-into-v2 seed=12345 \
  agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 activation=tanh \
  num_unsup_steps=9000 num_train_steps=1000000 agent.params.batch_size=512 \
  double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
  diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 \
  reward_update=10 num_interact=5000 max_feedback=40000 reward_batch=50 \
  feed_type=6 teacher_betas=[1,1,1,-1]
```

#### MetaWorld Sweep-Into (noisy)

```bash
python scripts/sweep_into/run_experiments.py --method pebble_mixture --preset 1,1,1,0
```

#### MetaWorld Door-Open (adversarial)

```bash
python scripts/door_open/run_experiments.py --method pebble_mixture --preset 1,1,1,-1
```

#### DM Control Cheetah-Run (adversarial / noisy)

```bash
python scripts/cheetah_run/run_experiments.py --method pebble_mixture --preset 1,1,1,-1
python scripts/cheetah_run/run_experiments.py --method pebble_mixture --preset 1,1,1,0
```

#### DM Control Walker-Walk (feedback-budget study)

```bash
# example: 5000 feedback, mixture [1,1,1,-1,-1]
python scripts/walker_walk/run_experiments.py --method zero_last_wk_sgd --max-feedback 5000 --teacher-betas 1 1 1 -1 -1
```

### Baselines

| Method | Script / entrypoint |
|---|---|
| PEBBLE | `train_PEBBLE.py` / `scripts/<env>/run_experiments.py --method pebble` |
| MCP | `train_PEBBLE_mixup_mixture.py` / `scripts/<env>/run_experiments.py --method mixup_mixture` |
| RIME | `train_RIME_mixture.py` / `scripts/<env>/run_experiments.py --method rime_mixture` |
| Raw Mixture | `train_PEBBLE_mixture_raw.py` / `scripts/<env>/run_experiments.py --method mixture_raw` |
| Oracle SAC | `train_SAC.py` / `scripts/<env>/run_experiments.py --method sac` |

Example baseline runners for Sweep-Into under adversarial experts:

```bash
python scripts/sweep_into/run_experiments.py --method mixup_mixture --preset 1,1,1,-1   # MCP
python scripts/sweep_into/run_experiments.py --method rime_mixture --preset 1,1,1,-1    # RIME
```

## Plotting

For flat experiment trees such as `exp_pebble_mixture_zero_last`, see **[PLOTTING.md](PLOTTING.md)** and:

```bash
python plot_experiments.py --root exp_pebble_mixture_zero_last
```

Enhancement ablations / buffer diagnostics use `plot_enhancement_ablation.py` (see that file’s docstring).

## Acknowledgement

This implementation builds on the official codebases of [B-Pref](https://github.com/rll-research/BPref), [PEBBLE](https://github.com/rll-research/BPref), [SURF](https://github.com/alinlab/SURF), [RUNE](https://github.com/rll-research/rune), [MRN](https://github.com/RyanLiu112/MRN), [QPA](https://github.com/huxiao09/QPA), [RIME](https://github.com/CJReinforce/RIME_ICML2024), and [MCP](https://github.com/JongkookHeo/MCP).

## Citation

```bibtex
@article{hosseini2025tritrust,
  title={Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback},
  author={Hosseini, Seyed Amir and Abdolali, Maryam and Tavakkoli, Amirhosein and Ayar, Fardin and Javanmardi, Ehsan and Tsukada, Manabu and Javanmardi, Mahdi},
  year={2025}
}
```
