"""PEBBLE mixture-human with zero-last init, SGD reward optimizer, and two-path w_k loss.

Reward-model SGD param groups (same as ``train_PEBBLE_mixture``):
  - network (ensemble) lr = ``reward_lr`` (default 0.001)
  - trust α lr = ``alpha_lr`` (default 0.005)

Human labels replace synthetic teachers; equal preferences use soft CE.
"""
import os
os.environ["PATH"] += os.pathsep + '/root/.mujoco/mujoco210/bin'
# Headless EGL rendering (needed for DMC rgb_array videos in Docker).
os.environ.setdefault('MUJOCO_GL', 'egl')
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
from utils.replay_buffer import ReplayBuffer
from utils.logger import Logger
import hydra
from reward_model.vanilla_reward_model import zero_last_linear
from reward_model.vanilla_reward_model_human import RewardModelHuman
from reward_model.mixture_reward_model_human import MixtureRewardModelHuman
from collections import deque
import pickle as pkl
import warnings
import imageio
import cv2
warnings.filterwarnings('ignore')


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.device = torch.device(cfg.device)
        self.logger = Logger(log_dir=self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.env, self.log_success = (utils.make_metaworld_env(cfg), True) if 'metaworld' in cfg.env else (utils.make_env(cfg), False)
        self.cfg = cfg
        # DMC default render is 84x84; use a larger size for human preference videos.
        self.render_height = int(getattr(cfg, 'render_height', 256))
        self.render_width = int(getattr(cfg, 'render_width', 256))

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # Videos live under the Hydra run dir (cwd after @hydra.main).
        self.video_root = os.path.join(self.work_dir, 'pebble_mixture_videos')
        os.makedirs(self.video_root, exist_ok=True)
        print(f'human preference videos → {self.video_root}')

        reward_models = []
        for i, beta in enumerate(cfg.teacher_betas):
            reward_models.append(RewardModelHuman(
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                ensemble_size=cfg.ensemble_size,
                size_segment=cfg.segment,
                activation=cfg.activation,
                lr=cfg.reward_lr,
                mb_size=cfg.reward_batch,
                large_batch=cfg.large_batch,
                label_margin=cfg.label_margin,
                teacher_beta=beta,
                teacher_gamma=cfg.teacher_gamma,
                teacher_eps_mistake=cfg.teacher_eps_mistake,
                teacher_eps_skip=cfg.teacher_eps_skip,
                teacher_eps_equal=cfg.teacher_eps_equal,
                video_record_path=os.path.join(self.video_root, f'expert_{i}'),
                seed=cfg.seed))

        self.reward_model = MixtureRewardModelHuman(
            reward_models=reward_models,
            ds=self.env.observation_space.shape[0],
            da=self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            alpha_lr=cfg.alpha_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            logger=self.logger,
            video_record_path=self.video_root,
            seed=cfg.seed,
            num_humans=len(cfg.teacher_betas))

        # Stabilized init: zero last Linear so r_hat ≡ 0 until first preference train.
        if cfg.zero_last_layer:
            for model in self.reward_model.ensemble:
                zero_last_linear(model)
            print('zero_last_layer: zeroed last Linear of each reward ensemble member')

        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        for ep in range(self.cfg.num_eval_episodes):
            frames = []
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                if self.step in list(range(450000, 510000)) + (list(range(950000, 1000000))):
                    if not 'metaworld' in self.cfg.env:
                        frame = self.env.render(
                            mode='rgb_array',
                            height=self.render_height,
                            width=self.render_width,
                        )
                    else:
                        frame = self.env.render('rgb_array')
                    frames.append(frame)
                obs, reward, done, extra = self.env.step(action)
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

            if self.step in list(range(450000, 510000)) + (list(range(950000, 1000000))):
                frames = np.stack(frames)
                os.makedirs(f'{self.work_dir}/eval_videos', exist_ok=True)
                os.makedirs(f'{self.work_dir}/eval_videos_frame', exist_ok=True)
                writer = imageio.get_writer(f'{self.work_dir}/eval_videos/step_{self.step}_ep_{ep:02d}.mp4', fps=15)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                with open(f'{self.work_dir}/eval_videos_frame/frames_{self.step}_ep{ep}.pkl', 'wb') as fi:
                    pkl.dump(frames, fi)

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            self.logger.log('eval/true_episode_success', success_rate, self.step)
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=False):
        labeled_queries = 0
        print(f'Collecting human labels; videos under {self.video_root}')
        if first_flag:
            labeled_queries = self.reward_model.uniform_sampling_human()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling_human()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling_human()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling_human()
            elif self.cfg.feed_type == 6:
                labeled_queries = self.reward_model.shuffle_disagreement_sampling_human()
            elif self.cfg.feed_type in (3, 4, 5):
                raise NotImplementedError
            else:
                raise NotImplementedError

        self.total_feedback += len(self.reward_model.reward_models) * self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0.
        if self.labeled_feedback > 0:
            for epoch in range(self.cfg.reward_update):
                # humans may mark equal (-1) → soft CE
                train_acc = self.reward_model.train_soft_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break

        print('Reward function is updated!! ACC : ' + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        render_mode = True
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time()-start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step>self.cfg.num_seed_steps))

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                self.learn_reward(first_flag=1)
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                self.agent.reset_critic()
                self.agent.update_after_reset(self.replay_buffer, self.logger,
                                              self.step, gradient_update=self.cfg.reset_update,
                                              policy_update=True)
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        # each interact asks K humans × mb_size queries
                        feedback_cost = len(self.reward_model.reward_models) * self.reward_model.mb_size
                        if self.total_feedback + feedback_cost > self.cfg.max_feedback:
                            remaining = self.cfg.max_feedback - self.total_feedback
                            per_human = max(1, remaining // len(self.reward_model.reward_models))
                            self.reward_model.set_batch(per_human)

                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, gradient_update=1)

            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, gradient_update=1, K=self.cfg.topK)

            if self.total_feedback == self.cfg.max_feedback:
                render_mode = False

            if render_mode:
                if not 'metaworld' in self.cfg.env:
                    frame = self.env.render(
                        mode='rgb_array',
                        height=self.render_height,
                        width=self.render_width,
                    )
                else:
                    frame = self.env.render('rgb_array')
                    frame2 = self.env.render('rgb_array', camera_name='topview')
                    frame = cv2.resize(frame[150:450, 180:480], (144, 144))
                    frame2 = cv2.resize(frame2[100:], (144, 144))
                    frame = np.concatenate((frame, frame2))

            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            if render_mode:
                self.reward_model.add_data_with_frame(obs, action, reward, done, frame)
            else:
                self.reward_model.flush_data()
            self.replay_buffer.add(obs, action, reward_hat, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        with open(f'{self.work_dir}/inputs.pkl', 'wb') as f:
            pkl.dump(self.reward_model.get_inputs(), f)
        with open(f'{self.work_dir}/targets.pkl', 'wb') as f:
            pkl.dump(self.reward_model.get_targets(), f)

@hydra.main(config_path='config', config_name='train_PEBBLE_mixture_human', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
