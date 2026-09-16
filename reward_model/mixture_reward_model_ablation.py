import os
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.logger import Logger

from .vanilla_reward_model import RewardModel, gen_net


device = "cuda"


class MixtureBufferDataset(Dataset):
    def __init__(self, reward_models):
        self.examples = []
        self.expert_data_counter = [0 for _ in range(len(reward_models))]
        self.total_data_counter = 0
        for expert_idx, rm in enumerate(reward_models):
            n = len(rm.buffer_label) if rm.buffer_full else rm.buffer_index
            self.expert_data_counter[expert_idx] = n
            self.total_data_counter += n
            for i in range(n):
                self.examples.append(
                    (rm.buffer_seg1[i], rm.buffer_seg2[i], rm.buffer_label[i], expert_idx)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seg1, seg2, label, expert_idx = self.examples[idx]
        return (
            torch.from_numpy(seg1).float(),
            torch.from_numpy(seg2).float(),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(expert_idx, dtype=torch.long),
            torch.tensor(self.expert_data_counter[expert_idx], dtype=torch.float),
        )


class MixtureRewardModel(RewardModel):
    """TTP mixture reward model with toggles for the three practical enhancements.

    Flags match Section ``cross-entropy-weight`` / Table ``enhancement-ablation``:
      - use_tanh:           ``\\tilde{\\alpha}_k = tanh(\\alpha_k)``
      - use_max_norm:       ``\\bar{\\alpha}_k = \\tilde{\\alpha}_k / max|\\tilde{\\alpha}|``
      - use_confidence_weight: ``w_k = K|\\tilde{\\alpha}_k| / sum|\\tilde{\\alpha}|``
      - use_confidence_weight_in_alpha: if False, detach ``w_k`` so it weights
        the reward CE loss but does not contribute gradients to ``\\alpha``
    """

    def __init__(
        self,
        reward_models: Union[List[RewardModel], None],
        ds,
        da,
        ensemble_size=3,
        mb_size=128,
        lr=3e-4,
        size_segment=1,
        env_maker=None,
        max_size=100,
        activation="tanh",
        capacity=5e5,
        large_batch=1,
        label_margin=0.0,
        logger: Union[Logger, None] = None,
        entropy_coef=0.05,
        init_trust=0.01,
        use_tanh=True,
        use_max_norm=True,
        use_confidence_weight=True,
        use_confidence_weight_in_alpha=True,
    ):
        if reward_models is None:
            reward_models = [
                RewardModel(
                    ds,
                    da,
                    ensemble_size=ensemble_size,
                    lr=lr,
                    mb_size=mb_size,
                    size_segment=size_segment,
                    env_maker=env_maker,
                    max_size=max_size,
                    activation=activation,
                    capacity=capacity,
                    large_batch=large_batch,
                    label_margin=label_margin,
                    teacher_beta=1,
                    teacher_gamma=1,
                )
                for _ in range(3)
            ] + [
                RewardModel(
                    ds,
                    da,
                    ensemble_size=ensemble_size,
                    lr=lr,
                    mb_size=mb_size,
                    size_segment=size_segment,
                    env_maker=env_maker,
                    max_size=max_size,
                    activation=activation,
                    capacity=capacity,
                    large_batch=large_batch,
                    label_margin=label_margin,
                    teacher_beta=-1,
                    teacher_gamma=1,
                )
            ]

        self.reward_models = reward_models
        self.init_trust = init_trust
        self.use_tanh = bool(use_tanh)
        self.use_max_norm = bool(use_max_norm)
        self.use_confidence_weight = bool(use_confidence_weight)
        self.use_confidence_weight_in_alpha = bool(use_confidence_weight_in_alpha)

        super().__init__(
            ds,
            da,
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
            teacher_beta=1,
            teacher_gamma=1,
        )
        self.env_maker = env_maker
        self.label_margin = label_margin
        self.entropy_coef = entropy_coef
        self.CEloss = nn.CrossEntropyLoss(reduction="none")
        self.label_target = 1 - 2 * self.label_margin
        self.total_epochs = 0
        self.logger = logger
        self.teacher_beta = -1

        for reward_model in self.reward_models:
            reward_model.ensemble = self.ensemble

    def change_batch(self, new_frac):
        super().change_batch(new_frac)
        for reward_model in self.reward_models:
            reward_model.change_batch(new_frac)

    def set_batch(self, new_batch):
        super().set_batch(new_batch)
        for reward_model in self.reward_models:
            reward_model.set_batch(new_batch)

    def construct_ensemble(self):
        for _ in range(self.de):
            model = (
                nn.Sequential(
                    *gen_net(
                        in_size=self.ds + self.da,
                        out_size=1,
                        H=256,
                        n_layers=3,
                        activation=self.activation,
                    )
                )
                .float()
                .to(device)
            )
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        alphas_tensor = self.init_trust * torch.ones(
            len(self.reward_models), dtype=torch.float32, device=device
        )
        self.alphas = nn.Parameter(alphas_tensor)
        self.paramlst.append(self.alphas)
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def add_data(self, obs, act, rew, done):
        for reward_model in self.reward_models:
            reward_model.add_data(obs, act, rew, done)

    def add_data_batch(self, obses, rewards):
        for reward_model in self.reward_models:
            reward_model.add_data_batch(obses, rewards)

    def save(self, work_dir, step):
        os.makedirs(work_dir, exist_ok=True)
        for idx, model in enumerate(self.ensemble):
            torch.save(model.state_dict(), os.path.join(work_dir, f"ensemble_{idx}_step_{step}.pt"))
        torch.save(self.alphas.data, os.path.join(work_dir, f"alphas_step_{step}.pt"))

    def uniform_sampling(self):
        return sum(rm.uniform_sampling() for rm in self.reward_models)

    def disagreement_sampling(self):
        return sum(rm.disagreement_sampling() for rm in self.reward_models)

    def shuffle_disagreement_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.reward_models[0].get_queries(
            mb_size=self.mb_size * self.large_batch * len(self.reward_models)
        )
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[: self.mb_size * len(self.reward_models)]
        top_k_index = np.random.permutation(top_k_index)
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        total_labels = 0
        for i, reward_model in enumerate(self.reward_models):
            sa_t_1_rm, sa_t_2_rm, r_t_1_rm, r_t_2_rm, labels_rm = reward_model.get_label(
                sa_t_1[i * self.mb_size : (i + 1) * self.mb_size],
                sa_t_2[i * self.mb_size : (i + 1) * self.mb_size],
                r_t_1[i * self.mb_size : (i + 1) * self.mb_size],
                r_t_2[i * self.mb_size : (i + 1) * self.mb_size],
            )
            if len(labels_rm) > 0:
                reward_model.put_queries(sa_t_1_rm, sa_t_2_rm, labels_rm)
                total_labels += len(labels_rm)
        return total_labels

    def entropy_sampling(self):
        return sum(rm.entropy_sampling() for rm in self.reward_models)

    def kcenter_sampling(self):
        return sum(rm.kcenter_sampling() for rm in self.reward_models)

    def kcenter_disagree_sampling(self):
        return sum(rm.kcenter_disagree_sampling() for rm in self.reward_models)

    def kcenter_entropy_sampling(self):
        return sum(rm.kcenter_entropy_sampling() for rm in self.reward_models)

    def _bound_trust(self, alphas):
        return torch.tanh(alphas) if self.use_tanh else alphas

    def _normalize_trust(self, tilde_alphas):
        if not self.use_max_norm:
            return tilde_alphas
        denom = tilde_alphas.abs().max().detach().clamp_min(1e-8)
        return tilde_alphas / denom

    def _confidence_weights(self, tilde_alphas, expert_inds):
        k = len(self.reward_models)
        abs_sum = tilde_alphas.abs().sum().clamp_min(1e-8)
        return tilde_alphas[expert_inds].abs() / abs_sum * k

    def _train_reward_common(self, use_soft_loss=False):
        dataset = MixtureBufferDataset(self.reward_models)
        dataloaders = [
            DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=2)
            for _ in range(self.de)
        ]
        loaders = [iter(dl) for dl in dataloaders]

        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.zeros(self.de, dtype=np.int64)
        total = np.zeros(self.de, dtype=np.int64)

        while True:
            self.opt.zero_grad()
            loss = 0.0
            is_finished = False

            for m, loader in enumerate(loaders):
                try:
                    seg1, seg2, labels, expert_inds, _ = next(loader)
                    batch_size = labels.size(0)
                    total[m] += batch_size
                except StopIteration:
                    is_finished = True
                    break

                seg1 = seg1.to(device)
                seg2 = seg2.to(device)
                labels = labels.to(device)
                expert_inds = expert_inds.to(device)

                r1 = self.ensemble[m](torch.cat((seg1,), dim=1)).sum(dim=1)
                r2 = self.ensemble[m](torch.cat((seg2,), dim=1)).sum(dim=1)
                logits = torch.stack([r1, r2], dim=1)

                tilde_all = self._bound_trust(self.alphas)
                if self.use_max_norm:
                    bar_m = self._normalize_trust(tilde_all)[expert_inds]
                else:
                    bar_m = self._bound_trust(self.alphas[expert_inds])
                logits = logits * bar_m.view(-1, 1, 1)

                if use_soft_loss:
                    uniform_index = labels == -1
                    labels_mod = labels.clone()
                    labels_mod[uniform_index] = 0
                    target_onehot = torch.zeros_like(logits).scatter(
                        1, labels_mod.unsqueeze(1), self.label_target
                    )
                    target_onehot += self.label_margin
                    if sum(uniform_index) > 0:
                        target_onehot[uniform_index] = 0.5
                    cur_loss = super().softXEnt_loss(logits, target_onehot)
                else:
                    cur_loss = self.CEloss(logits, labels)

                if self.use_confidence_weight:
                    w_m = self._confidence_weights(tilde_all, expert_inds)
                    # Detach so w_k scales reward CE without flowing grads into alpha.
                    if not self.use_confidence_weight_in_alpha:
                        w_m = w_m.detach()
                    cur_loss = cur_loss * w_m

                cur_loss = cur_loss.mean()
                loss += cur_loss
                ensemble_losses[m].append(loss.item())

                _, preds = logits.max(dim=1)
                if use_soft_loss:
                    ensemble_acc[m] += (preds == labels_mod).sum().item()
                else:
                    ensemble_acc[m] += (preds == labels).sum().item()

            if is_finished:
                break

            loss.backward()
            self.opt.step()

            self.total_epochs += 1
            with torch.no_grad():
                tilde = self._bound_trust(self.alphas)
                bar = self._normalize_trust(tilde)
                if self.logger is not None:
                    self.logger.log("reward/alpha_abs_sum", self.alphas.abs().sum().item(), self.total_epochs)
                    for i, alpha in enumerate(self.alphas):
                        self.logger.log(f"reward/alpha_{i}", alpha.item(), self.total_epochs)
                    for i, alpha in enumerate(tilde):
                        self.logger.log(f"reward/alpha_bound_{i}", alpha.item(), self.total_epochs)
                    for i, alpha in enumerate(bar):
                        self.logger.log(f"reward/expert_logits_coef_{i}", alpha.item(), self.total_epochs)
                    if self.use_confidence_weight:
                        w = tilde.abs() / tilde.abs().sum().clamp_min(1e-8) * len(self.reward_models)
                        for i, coef in enumerate(w):
                            self.logger.log(f"reward/expert_coef_{i}", coef.item(), self.total_epochs)

        if self.logger is not None:
            self.logger.dump(self.total_epochs, ty="reward")
        return ensemble_acc / total

    def train_reward(self):
        return self._train_reward_common(use_soft_loss=False)

    def train_soft_reward(self):
        return self._train_reward_common(use_soft_loss=True)
