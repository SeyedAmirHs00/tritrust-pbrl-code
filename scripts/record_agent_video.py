#!/usr/bin/env python3
"""Roll out a saved SAC actor checkpoint and write an mp4.

Example (inside mixture_pbrl_container):

  python scripts/record_agent_video.py \\
    --ckpt_dir 'exp_pebble_mixture_zero_last_wk_sgd/cheetah_run/max_feedback4000_feed_type6_n50_l50_g1_b[1, 1, 1, -1]_m0_s0_e0/seed67890' \\
    --step 1000000
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

# Allow running as scripts/record_agent_video.py from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import utils


def parse_args():
    p = argparse.ArgumentParser(description="Record a policy rollout video from actor_*.pt")
    p.add_argument(
        "--ckpt_dir",
        required=True,
        help="Run directory containing actor_{step}.pt and .hydra/config.yaml",
    )
    p.add_argument("--step", type=int, default=1000000, help="Checkpoint step suffix")
    p.add_argument("--episodes", type=int, default=1, help="Number of eval episodes to record")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument(
        "--out",
        default=None,
        help="Output mp4 path (default: {ckpt_dir}/video_actor_{step}.mp4)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--actor_only",
        action="store_true",
        help="Load only actor_{step}.pt (skip critic weights)",
    )
    return p.parse_args()


def load_cfg(ckpt_dir: str, device: str):
    cfg_path = os.path.join(ckpt_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing Hydra config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    cfg.device = device
    return cfg


def build_agent(cfg, env):
    # Saved configs keep Hydra placeholders (???) for dims filled at train time.
    cfg.agent.params.obs_dim = int(env.observation_space.shape[0])
    cfg.agent.params.action_dim = int(env.action_space.shape[0])
    cfg.agent.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]
    cfg.agent.params.device = cfg.device
    OmegaConf.resolve(cfg)
    return hydra.utils.instantiate(cfg.agent)


def main():
    args = parse_args()
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or os.path.join(ckpt_dir, f"video_actor_{args.step}.mp4")

    actor_path = os.path.join(ckpt_dir, f"actor_{args.step}.pt")
    if not os.path.isfile(actor_path):
        raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")

    cfg = load_cfg(ckpt_dir, device)
    if "metaworld" in cfg.env:
        env = utils.make_metaworld_env(cfg)
        is_metaworld = True
    else:
        env = utils.make_env(cfg)
        is_metaworld = False

    agent = build_agent(cfg, env)
    if args.actor_only:
        agent.actor.load_state_dict(
            torch.load(actor_path, map_location=device)
        )
    else:
        agent.load(ckpt_dir, args.step)

    frames = []
    returns = []
    for ep in range(args.episodes):
        obs = env.reset()
        agent.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            if is_metaworld:
                frame = env.render(mode="rgb_array")
            else:
                frame = env.render(
                    mode="rgb_array",
                    height=args.height,
                    width=args.width,
                )
            frames.append(np.asarray(frame))
            obs, reward, done, _extra = env.step(action)
            ep_ret += float(reward)
        returns.append(ep_ret)
        print(f"episode {ep}: true return = {ep_ret:.1f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=args.fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(
        f"saved {out_path} "
        f"({len(frames)} frames, {args.episodes} ep, mean return {mean_ret:.1f})"
    )


if __name__ == "__main__":
    main()
