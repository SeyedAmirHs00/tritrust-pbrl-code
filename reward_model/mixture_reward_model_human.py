"""Mixture TTP reward model with human preference labels + wk_sgd training.

Inherits SGD two-path w_k loss from ``mixture_reward_model_wk_sgd``:

    trust = tanh(α)
    coef  = trust / max|trust|          # max detached
    w_k   = K |trust| / Σ|trust|        # detached; reward path only

    loss_R = mean(w_k * CE(coef.detach() * logits, y))
    loss_A = mean(CE(coef * logits.detach(), y))

Human experts share trajectory/frame buffers but keep separate preference
buffers (``RewardModelHuman``).
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np

from utils.logger import Logger

from .mixture_reward_model_wk_sgd import MixtureRewardModel as _WkSgdMixtureRewardModel
from .vanilla_reward_model_human import RewardModelHuman


class MixtureRewardModelHuman(_WkSgdMixtureRewardModel):
    def __init__(
        self,
        reward_models: Optional[List[RewardModelHuman]],
        ds,
        da,
        ensemble_size=3,
        mb_size=128,
        lr=0.001,
        alpha_lr=0.005,
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
        video_record_path="pebble_mixture_videos",
        seed=12345,
        num_humans=4,
    ):
        if reward_models is None:
            reward_models = []
            for i in range(num_humans):
                reward_models.append(
                    RewardModelHuman(
                        ds,
                        da,
                        ensemble_size=ensemble_size,
                        lr=lr,
                        mb_size=mb_size,
                        size_segment=size_segment,
                        max_size=max_size,
                        activation=activation,
                        capacity=capacity,
                        large_batch=large_batch,
                        label_margin=label_margin,
                        teacher_beta=-1,
                        teacher_gamma=1,
                        video_record_path=os.path.join(
                            video_record_path, f"expert_{i}"
                        ),
                        seed=seed,
                    )
                )

        self.video_record_path = video_record_path
        self.seed = seed
        os.makedirs(self.video_record_path, exist_ok=True)

        # construct_ensemble() (SGD param groups) runs inside super().__init__.
        super().__init__(
            reward_models=reward_models,
            ds=ds,
            da=da,
            ensemble_size=ensemble_size,
            mb_size=mb_size,
            lr=lr,
            alpha_lr=alpha_lr,
            size_segment=size_segment,
            env_maker=env_maker,
            max_size=max_size,
            activation=activation,
            capacity=capacity,
            large_batch=large_batch,
            label_margin=label_margin,
            logger=logger,
            entropy_coef=entropy_coef,
            init_trust=init_trust,
        )
        self._sync_traj_buffers()

    # ------------------------------------------------------------------
    # Shared trajectory / frame buffer (owned by expert 0)
    # ------------------------------------------------------------------
    def _sync_traj_buffers(self) -> None:
        base = self.reward_models[0]
        for rm in self.reward_models[1:]:
            rm.inputs = base.inputs
            rm.targets = base.targets
            rm.frames = base.frames

    def get_inputs(self):
        return self.reward_models[0].inputs

    def get_targets(self):
        return self.reward_models[0].targets

    def get_frames(self):
        return self.reward_models[0].frames

    def flush_data(self) -> None:
        self.reward_models[0].flush_data()
        self._sync_traj_buffers()

    def add_data_with_frame(self, obs, action, reward, done, frame) -> None:
        self.reward_models[0].add_data_with_frame(obs, action, reward, done, frame)
        self._sync_traj_buffers()

    def add_data(self, obs, act, rew, done) -> None:
        self.reward_models[0].add_data(obs, act, rew, done)
        self._sync_traj_buffers()

    # ------------------------------------------------------------------
    # Human preference sampling
    # ------------------------------------------------------------------
    def uniform_sampling_human(self) -> int:
        self._sync_traj_buffers()
        cnt = [rm.uniform_sampling_human() for rm in self.reward_models]
        return int(sum(cnt))

    def disagreement_sampling_human(self) -> int:
        self._sync_traj_buffers()
        cnt = [rm.disagreement_sampling_human() for rm in self.reward_models]
        return int(sum(cnt))

    def entropy_sampling_human(self) -> int:
        self._sync_traj_buffers()
        cnt = [rm.entropy_sampling_human() for rm in self.reward_models]
        return int(sum(cnt))

    def shuffle_disagreement_sampling_human(self) -> int:
        """One large disagreement pool, then ask each human for a slice."""
        self._sync_traj_buffers()
        k = len(self.reward_models)
        base = self.reward_models[0]
        sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat = base.get_queries_with_frame(
            mb_size=self.mb_size * self.large_batch * k
        )
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[: self.mb_size * k]
        top_k_index = np.random.permutation(top_k_index)

        sa_t_1 = sa_t_1[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]
        r_t_1 = r_t_1[top_k_index]
        r_t_2 = r_t_2[top_k_index]
        f_cat = f_cat[top_k_index]

        total_labels = 0
        for i, reward_model in enumerate(self.reward_models):
            sl = slice(i * self.mb_size, (i + 1) * self.mb_size)
            sa1, sa2, r1, r2, _ = reward_model.get_label(
                sa_t_1[sl], sa_t_2[sl], r_t_1[sl], r_t_2[sl]
            )
            labels = reward_model.get_human_label(f_cat[sl])
            if len(labels) > 0:
                reward_model.put_queries(sa1, sa2, labels)
                total_labels += len(labels)
            reward_model.session += 1
        return int(total_labels)
