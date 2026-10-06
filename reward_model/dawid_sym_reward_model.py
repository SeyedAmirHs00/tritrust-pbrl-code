"""Dawid-Skene Symmetric (Dawid-Sym) crowdsourcing reward model for PBRL.

Formulation:
    Latent true preference between segments (σ_1, σ_2):
        P(z = 1) = σ(s * (r_2 - r_1)), where z = 1 means σ_2 ≻ σ_1.
        P(z = 0) = 1 - P(z = 1) = σ(s * (r_1 - r_2)).

    Annotator e with reliability q_e ∈ [0, 1] reporting label y ∈ {0, 1}:
        P(y = 1 | z = 1) = q_e,   P(y = 0 | z = 1) = 1 - q_e
        P(y = 0 | z = 0) = q_e,   P(y = 1 | z = 0) = 1 - q_e

    Marginal label likelihood:
        P(y = 1 | σ_1, σ_2, e) = q_e * P(z = 1) + (1 - q_e) * P(z = 0)
        P(y = 0 | σ_1, σ_2, e) = q_e * P(z = 0) + (1 - q_e) * P(z = 1)

Trainable parameters:
    - Reward network ensemble (learning rate: ``lr`` / ``reward_lr``)
    - Expert reliabilities q ∈ [0, 1]^K, initialized to q_init (default 1.0)
    - Temperature scale s > 0, initialized to s_init (default 1.0)
"""

import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.logger import Logger
from .mixture_reward_model_no_wk import (
    MixtureBufferDataset,
    MixtureRewardModel as _BaseMixtureRewardModel,
)
from .vanilla_reward_model import RewardModel, gen_net

device = "cuda" if torch.cuda.is_available() else "cpu"


class DawidSymRewardModel(_BaseMixtureRewardModel):
    """Dawid-Skene Symmetric crowdsourcing reward model with learnable q and s."""

    def __init__(
        self,
        reward_models: Union[List[RewardModel], None],
        ds: int,
        da: int,
        ensemble_size: int = 3,
        mb_size: int = 128,
        lr: float = 0.001,
        expert_lr: float = 0.005,
        alpha_lr: Optional[float] = None,
        q_init: float = 1.0,
        s_init: float = 1.0,
        size_segment: int = 1,
        env_maker=None,
        max_size: int = 100,
        activation: str = "tanh",
        capacity: float = 5e5,
        large_batch: int = 1,
        label_margin: float = 0.0,
        logger: Union[Logger, None] = None,
        entropy_coef: float = 0.05,
        **kwargs,
    ):
        self.expert_lr = alpha_lr if alpha_lr is not None else expert_lr
        self.q_init = float(q_init)
        self.s_init = float(s_init)

        super().__init__(
            reward_models=reward_models,
            ds=ds,
            da=da,
            ensemble_size=ensemble_size,
            mb_size=mb_size,
            lr=lr,
            size_segment=size_segment,
            env_maker=env_maker,
            max_size=max_size,
            activation=activation,
            capacity=capacity,
            large_batch=large_batch,
            label_margin=label_margin,
            logger=logger,
            entropy_coef=entropy_coef,
        )

    def construct_ensemble(self):
        """Construct reward network ensemble and initialize trainable q and s."""
        for _ in range(self.de):
            model = nn.Sequential(
                *gen_net(
                    in_size=self.ds + self.da,
                    out_size=1,
                    H=256,
                    n_layers=3,
                    activation=self.activation,
                )
            ).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        k = len(self.reward_models)
        # Trainable expert reliability vector q ∈ [0, 1]^K, initialized to q_init (default 1.0)
        self.q = nn.Parameter(
            torch.full((k,), self.q_init, dtype=torch.float32, device=device)
        )
        # Trainable inverse temperature scale parameter s, initialized to s_init (default 1.0)
        self.s = nn.Parameter(
            torch.tensor(self.s_init, dtype=torch.float32, device=device)
        )

        net_params = [p for p in self.paramlst]
        expert_params = [self.q, self.s]

        # Separate parameter groups for reward network vs expert parameters
        self.opt = torch.optim.SGD(
            [
                {"params": net_params, "lr": self.lr},
                {"params": expert_params, "lr": self.expert_lr},
            ]
        )
        print(
            f"DawidSymRewardModel: SGD optimizer | net_lr={self.lr} "
            f"expert_lr={self.expert_lr} | init q={self.q_init} (len {k}) | init s={self.s_init}"
        )

    def _train_reward_common(self, use_soft_loss: bool = False):
        dataset = MixtureBufferDataset(self.reward_models)
        dataloaders = [
            DataLoader(
                dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=2,
            )
            for _ in range(self.de)
        ]
        loaders = [iter(dl) for dl in dataloaders]

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.zeros(self.de, dtype=np.int64)
        total = np.zeros(self.de, dtype=np.int64)
        net_params = [p for p in self.paramlst]

        while True:
            self.opt.zero_grad()
            loss = 0.0
            is_finished = False

            for m, loader in enumerate(loaders):
                try:
                    seg1, seg2, labels, expert_inds, expert_data_counters = next(loader)
                    batch_size = labels.size(0)
                    total[m] += batch_size
                except StopIteration:
                    is_finished = True
                    break

                seg1 = seg1.to(device)
                seg2 = seg2.to(device)
                labels = labels.to(device).view(-1)
                expert_inds = expert_inds.to(device).view(-1)

                # Predict segment returns for ensemble member m
                r1 = self.ensemble[m](torch.cat((seg1,), dim=1)).sum(dim=1)  # [B, 1]
                r2 = self.ensemble[m](torch.cat((seg2,), dim=1)).sum(dim=1)  # [B, 1]

                # Latent return difference: delta = r2 - r1 (label 1 corresponds to seg2 ≻ seg1)
                delta = (r2 - r1).view(-1, 1)  # [B, 1]

                # Latent true preference probability under scale s:
                # P(z = 1) = sigmoid(s * (r2 - r1)), P(z = 0) = 1 - P(z = 1)
                p_z1 = torch.sigmoid(self.s * delta)
                p_z0 = 1.0 - p_z1

                # Gather reliability q_e for each query's expert
                q_batch = self.q[expert_inds].view(-1, 1)  # [B, 1]

                # Marginalized label probabilities:
                # P(y = 1) = q_e * P(z = 1) + (1 - q_e) * P(z = 0)
                # P(y = 0) = q_e * P(z = 0) + (1 - q_e) * P(z = 1)
                p_y1 = q_batch * p_z1 + (1.0 - q_batch) * p_z0
                p_y0 = q_batch * p_z0 + (1.0 - q_batch) * p_z1

                probs = torch.cat([p_y0, p_y1], dim=1).clamp(1e-6, 1.0 - 1e-6)  # [B, 2]
                log_probs = torch.log(probs)

                if use_soft_loss:
                    uniform_index = labels == -1
                    labels_mod = labels.clone()
                    labels_mod[uniform_index] = 0
                    target_onehot = torch.zeros_like(probs).scatter(
                        1, labels_mod.unsqueeze(1), self.label_target
                    )
                    target_onehot += self.label_margin
                    if uniform_index.sum() > 0:
                        target_onehot[uniform_index] = 0.5
                    cur_loss = -(target_onehot * log_probs).sum(dim=1).mean()
                    target_labels = labels_mod
                else:
                    cur_loss = F.nll_loss(log_probs, labels, reduction="none").mean()
                    target_labels = labels

                loss = loss + cur_loss
                ensemble_losses[m].append(cur_loss.item())

                _, preds = probs.max(dim=1)
                ensemble_acc[m] += (preds == target_labels).sum().item()

            if is_finished:
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_params, 10.0)
            self.opt.step()

            # Direct projections to satisfy model constraints:
            # - Expert reliability q must remain valid probabilities in [0, 1]
            # - Temperature scale s must remain strictly positive
            with torch.no_grad():
                self.q.data.clamp_(0.0, 1.0)
                self.s.data.clamp_min_(1e-4)

            self.total_epochs += 1
            if self.logger is not None:
                with torch.no_grad():
                    self.logger.log("reward/s", self.s.item(), self.total_epochs)
                    self.logger.log("reward/q_mean", self.q.mean().item(), self.total_epochs)
                    for i, q_i in enumerate(self.q):
                        self.logger.log(f"reward/q_{i}", q_i.item(), self.total_epochs)

        if self.logger is not None:
            self.logger.dump(self.total_epochs, ty="reward")
        ensemble_acc = ensemble_acc / np.maximum(total, 1)
        return ensemble_acc

    def save(self, work_dir: str, step: int):
        os.makedirs(work_dir, exist_ok=True)
        # Save ensemble models
        for idx, model in enumerate(self.ensemble):
            torch.save(
                model.state_dict(),
                os.path.join(work_dir, f"ensemble_{idx}_step_{step}.pt"),
            )
        # Save Dawid-Sym parameters
        torch.save(self.q.data, os.path.join(work_dir, f"q_step_{step}.pt"))
        torch.save(self.s.data, os.path.join(work_dir, f"s_step_{step}.pt"))

    @classmethod
    def load(cls, work_dir: str, step: int, ds: int, da: int, reward_models=None, **kwargs):
        obj = cls(reward_models, ds, da, **kwargs)
        for idx, model in enumerate(obj.ensemble):
            model_path = os.path.join(work_dir, f"ensemble_{idx}_step_{step}.pt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))

        q_path = os.path.join(work_dir, f"q_step_{step}.pt")
        if os.path.exists(q_path):
            obj.q.data.copy_(torch.load(q_path, map_location=device))

        s_path = os.path.join(work_dir, f"s_step_{step}.pt")
        if os.path.exists(s_path):
            obj.s.data.copy_(torch.load(s_path, map_location=device))

        return obj
