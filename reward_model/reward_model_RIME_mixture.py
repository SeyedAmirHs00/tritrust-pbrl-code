import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup
from typing import List, Union
from .vanilla_reward_model_RIME import RewardModel, device as _global_device, set_device as _set_global_device
from utils.utils_RIME import RunningMeanStd

from .constants import RATIONAL_TEACHER


def set_device_RIME(dev):
    global _global_device
    _global_device = dev
    _set_global_device(dev)


class MixtureRIMERewardModel(RewardModel):
    def __init__(
            self, 
            ds, da,
            seed,
            k,
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            reward_models: Union[List[RewardModel], None] = None,
            threshold_variance='kl', 
            threshold_alpha=0.5,
            threshold_beta_init=3.0,
            threshold_beta_min=1.0,
            flipping_tau=0.001,
            num_warmup_steps=50,
            #base reward_model args
            ensemble_size=3, mb_size=128, lr=3e-4, size_segment=1,
            env_maker=None, max_size=100, activation='tanh', capacity=5e5,
            large_batch=1, label_margin=0.0,
            *args, **kwargs
        ):
        super().__init__(ds, da,
                         ensemble_size=ensemble_size, lr=lr, mb_size=mb_size,
                         size_segment=size_segment, env_maker=env_maker, max_size=max_size,
                         activation=activation, capacity=capacity, large_batch=large_batch,
                         label_margin=label_margin,
                         teacher_beta=1, teacher_gamma=1,
                        #  *args, **kwargs
                         )
        self.lr_schedule = None
        self.seed = seed
        self.device = device
        self.k = k
        self.KL_div = RunningMeanStd(mode='fixed', lr=0.1)
        assert threshold_variance.lower() in ['kl', 'prob']

        self.threshold_variance = threshold_variance
        self.threshold_alpha = threshold_alpha
        self.threshold_beta_init = threshold_beta_init
        self.threshold_beta_min = threshold_beta_min
        self.flipping_tau = flipping_tau
        self.num_warmup_steps = num_warmup_steps

        self.update_step = 0
        
        if reward_models is None:
            reward_models = [
                RewardModel(ds, da, ensemble_size=ensemble_size,
                            lr=lr, mb_size=mb_size, size_segment=size_segment,
                            env_maker=env_maker, max_size=max_size,
                            activation=activation, capacity=capacity, large_batch=large_batch,
                            label_margin=label_margin, teacher_beta=1, teacher_gamma=1),
                RewardModel(ds, da, ensemble_size=ensemble_size,
                            lr=lr, mb_size=mb_size, size_segment=size_segment,
                            env_maker=env_maker, max_size=max_size,
                            activation=activation, capacity=capacity, large_batch=large_batch,
                            label_margin=label_margin, teacher_beta=1, teacher_gamma=1),
                RewardModel(ds, da, ensemble_size=ensemble_size,
                            lr=lr, mb_size=mb_size, size_segment=size_segment,
                            env_maker=env_maker, max_size=max_size,
                            activation=activation, capacity=capacity, large_batch=large_batch,
                            label_margin=label_margin, teacher_beta=1, teacher_gamma=1),
                RewardModel(ds, da, ensemble_size=ensemble_size,
                            lr=lr, mb_size=mb_size, size_segment=size_segment,
                            env_maker=env_maker, max_size=max_size,
                            activation=activation, capacity=capacity, large_batch=large_batch,
                            label_margin=label_margin, teacher_beta=-1, teacher_gamma=1),
            ]
        self.reward_models = reward_models
        for rm in self.reward_models:
            rm.ensemble = self.ensemble
    def add_data(self, obs, act, rew, done):
        for reward_model in self.reward_models:
            reward_model.add_data(obs, act, rew, done)

    def add_data_batch(self, obses, rewards):
        for reward_model in self.reward_models:
            reward_model.add_data_batch(obses, rewards)

    def uniform_sampling(self):
        cnt_labels = [reward_model.uniform_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def disagreement_sampling(self):
        cnt_labels = [reward_model.disagreement_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def entropy_sampling(self):
        cnt_labels = [reward_model.entropy_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def kcenter_sampling(self):
        cnt_labels = [reward_model.kcenter_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def kcenter_disagree_sampling(self):
        cnt_labels = [reward_model.kcenter_disagree_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)
    
    def kcenter_entropy_sampling(self):
        cnt_labels = [reward_model.kcenter_entropy_sampling() for reward_model in self.reward_models]
        return sum(cnt_labels)

    def set_lr_schedule(self):
        self.lr_schedule = get_constant_schedule_with_warmup(self.opt, self.num_warmup_steps)

    def get_threshold_beta(self):
        return max(self.threshold_beta_min, -(self.threshold_beta_init-self.threshold_beta_min)/self.k * self.update_step + self.threshold_beta_init)

    def shuffle_disagreement_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.reward_models[0].get_queries(
            mb_size=self.mb_size*self.large_batch*len(self.reward_models))
        
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size*len(self.reward_models)]
        top_k_index = np.random.permutation(top_k_index)
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]  

        total_labels = 0
        for i, reward_model in enumerate(self.reward_models):
            sa_t_1_rm, sa_t_2_rm, r_t_1_rm, r_t_2_rm, labels_rm, GT_labels_rm = reward_model.get_label(
                sa_t_1[i*self.mb_size:(i+1)*self.mb_size], sa_t_2[i*self.mb_size:(i+1)*self.mb_size], r_t_1[i*self.mb_size:(i+1)*self.mb_size], r_t_2[i*self.mb_size:(i+1)*self.mb_size])
            if len(labels_rm) > 0:
                reward_model.put_queries(sa_t_1_rm, sa_t_2_rm, labels_rm, GT_labels_rm)
                total_labels += len(labels_rm)

        return total_labels
    
    def train_reward(self, debug=False, trust_sample=True, label_flipping=True):
        # -------- 0) Gather ALL examples from ALL experts --------
        all_pairs = []     # [(expert_idx, idx_in_that_expert), ...]
        seg1_list = []
        seg2_list = []
        labels_list = []
        for e, rm in enumerate(self.reward_models):
            max_len = rm.capacity if rm.buffer_full else rm.buffer_index
            for i in range(max_len):
                all_pairs.append((e, i))
                seg1_list.append(rm.buffer_seg1[i])
                seg2_list.append(rm.buffer_seg2[i])
                labels_list.append(rm.buffer_label[i])

        seg1_all = np.stack(seg1_list, axis=0)
        seg2_all = np.stack(seg2_list, axis=0)
        labels_all = np.array(labels_list, dtype=np.int64)

        # Inja Amir
        self.buffer_seg1 = seg1_all
        self.buffer_seg2 = seg2_all
        self.buffer_label = labels_all
        self.buffer_index = len(labels_all)
        # In?
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.buffer_index

        # compute trust samples
        p_hat_all = []
        with torch.no_grad():
            for member in range(self.de):
                r_hat1 = self.r_hat_member(self.buffer_seg1[:max_len], member=member)
                r_hat2 = self.r_hat_member(self.buffer_seg2[:max_len], member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (max_len, 2)
                p_hat_all.append(F.softmax(r_hat, dim=-1).cpu())
        
        # predict label for all ensemble members
        p_hat_all = torch.stack(p_hat_all)  # (de, max_len, 2)
        
        # compute KL divergence
        predict_label = p_hat_all.mean(0)  # (max_len, 2)
        if self.label_margin > 0 or self.teacher_eps_equal > 0:
            buffer_label = torch.tensor(self.buffer_label[:max_len].flatten()).long()
            target_label = torch.zeros_like(predict_label)
            temp_buffer_label = torch.clamp(buffer_label, min=0)
            target_label.scatter_(1, temp_buffer_label.unsqueeze(1), 1)
            mask = buffer_label == -1
            target_label[mask, :] = 0.5
        else:
            target_label = torch.zeros_like(predict_label).scatter(1, torch.from_numpy(self.buffer_label[:max_len].flatten()).long().unsqueeze(1), 1)
        
        KL_div = (-target_label * torch.log(predict_label)).sum(1)  # (max_len,)
        
        # filter trust samples
        x = self.KL_div.max
        baseline = -np.log(x + 1e-8) + self.threshold_alpha * x
        if self.threshold_variance == 'prob':
            uncertainty = self.get_threshold_beta() * predict_label[:, 0].std(0)
        else:
            uncertainty = min(self.get_threshold_beta() * self.KL_div.var, 3.0)
        trust_sample_bool_index = KL_div < baseline + uncertainty
        trust_sample_index = np.where(trust_sample_bool_index)[0]

        # label flipping
        flipping_threshold = -np.log(self.flipping_tau)
        flipping_sample_bool_index = KL_div > flipping_threshold
        flipping_sample_index = np.where(flipping_sample_bool_index)[0]
        
        # update KL divergence statistics of trust samples
        self.KL_div.update(KL_div[trust_sample_bool_index].numpy())

        if trust_sample and label_flipping:
            # temporarily flipping
            self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
            training_sample_index = np.concatenate([trust_sample_index, flipping_sample_index])
        elif not trust_sample and label_flipping:
            # temporarily flipping
            self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
            training_sample_index = np.arange(max_len)
        elif trust_sample and not label_flipping:
            training_sample_index = trust_sample_index
        else:
            training_sample_index = np.arange(max_len)
        
        max_len = len(training_sample_index)
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(training_sample_index))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                if self.label_margin > 0 or self.teacher_eps_equal > 0:
                    uniform_index = labels == -1
                    labels[uniform_index] = 0
                    target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                    target_onehot += self.label_margin
                    if uniform_index.int().sum().item() > 0:
                        target_onehot[uniform_index] = 0.5
                    curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                else:
                    curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        self.lr_schedule.step()
        
        # change back
        if label_flipping:
            self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
        
        ensemble_acc = ensemble_acc / total
        self.update_step += 1
        
        return ensemble_acc