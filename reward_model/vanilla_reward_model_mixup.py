import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from reward_model.vanilla_reward_model import RewardModel
device = 'cuda'

# def gen_net(in_size=1, out_size=1, hidden_size=128, n_layers=3, activation='tanh'):
#     net = []
#     for i in range(n_layers):
#         net.append(nn.Linear(in_size, hidden_size))
#         net.append(nn.LeakyReLU())
#         in_size = hidden_size
#     net.append(nn.Linear(in_size, out_size))
#     if activation == 'tanh':
#         net.append(nn.Tanh())
#     elif activation =='sig':
#         net.append(nn.Sigmoid())
#     else:
#         net.append(nn.ReLU())
    
#     return net

class RewardModelMixup(RewardModel):
    def __init__(self, obs_dim, action_dim,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, max_size=100,
                 activation='tanh', capacity=5e5, large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0,
                 teacher_eps_skip=0, teacher_eps_equal=0, mixup_alpha=0.1):
        
        super(RewardModelMixup, self).__init__(obs_dim, action_dim, ensemble_size, lr, mb_size, size_segment, max_size,
                                               activation, capacity, large_batch, label_margin, teacher_beta, teacher_gamma,
                                               teacher_eps_mistake, teacher_eps_skip, teacher_eps_equal)                
        
        # self.obs_dim = obs_dim
        # self.action_dim = action_dim
        # self.ensemble_size = ensemble_size
        # self.lr = lr
        # self.ensemble = []
        # self.paramlst = []
        # self.optimizer = None
        # self.max_size = max_size
        # self.activation = activation
        # self.size_segment = size_segment
        
        # self.capacity = int(capacity)
        # self.buffer_seg1 = np.empty((self.capacity, size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        # self.buffer_seg2 = np.empty((self.capacity, size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        # self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        # self.buffer_index = 0
        # self.buffer_full = False
        
        # self.construct_ensemble()
        # self.inputs = []
        # self.targets = []
        # self.mb_size = mb_size
        # self.origin_mb_size = mb_size
        # self.train_batch_size = 128
        # self.CELoss = nn.CrossEntropyLoss()
        # self.large_batch = large_batch
        
        # # new teacher
        # self.teacher_beta = teacher_beta
        # self.teacher_gamma = teacher_gamma
        # self.teacher_eps_mistake = teacher_eps_mistake
        # self.teacher_eps_equal = teacher_eps_equal
        # self.teacher_eps_skip = teacher_eps_skip
        # self.teacher_thres_skip = 0
        # self.teacher_thres_equal = 0
        
        # self.label_margin = label_margin
        # self.label_target = 1-2*self.label_margin
        self.mixup_alpha = mixup_alpha
        
        
    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index
    
    def mixup_batch(self, sa_t_1, sa_t_2, target_onehot):
        """_summary_

        Args:
            sa_t_1 (torch.Tensor): (batch_size, size_segment, obs_dim + action_dim)
            sa_t_2 (torch.Tensor): (batch_size, size_segment, obs_dim + action_dim)
            target_onehot (torch.Tensor): (batch_size, 2)
        return sa_t_1_m, sa_t_2_m, target_onehot_m
        """
        indices = torch.randperm(sa_t_1.size(0))
        lmda = torch.FloatTensor([np.random.beta(self.mixup_alpha, self.mixup_alpha)])
        
        sa_t_1_m = sa_t_1 * lmda + sa_t_1[indices] * (1 - lmda)
        sa_t_2_m = sa_t_2 * lmda + sa_t_2[indices] * (1 - lmda)
        target_onehot_m = target_onehot * lmda + target_onehot[indices] * (1 - lmda)
        return sa_t_1_m, sa_t_2_m, target_onehot_m
        
    def train_reward_mixup(self, num_iters, original_concat):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        
        total = 0
        start_idx = 0
        
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            last_index = start_idx + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                ## Original Data : sa_t_1, sa_t_2, target_onehot
                idxs = total_batch_index[member][start_idx:last_index]
                sa_t_1 = torch.FloatTensor(self.buffer_seg1[idxs]) # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = torch.FloatTensor(self.buffer_seg2[idxs]) # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long()
            
                if member == 0:
                    total += labels.size(0)
                
                uniform_index = labels == -1
                labels[uniform_index] = 0
                
                target_onehot = torch.zeros((labels.size(0), 2)).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5 #(batch_size, 2)
                
                ## Mix up Data : sa_t_1_m, sa_t_2_m, target_onehot_m
                sa_t_1_m, sa_t_2_m, target_onehot_m = self.mixup_batch(sa_t_1, sa_t_2, target_onehot)
                
                ## Concat
                concat_sa_t_1 = torch.cat([sa_t_1, sa_t_1_m], axis=0)
                concat_sa_t_2 = torch.cat([sa_t_2, sa_t_2_m], axis=0)
                concat_target_onehot = torch.cat([target_onehot, target_onehot_m], axis=0).to(device)
                
                ## Predict
                r_hat1 = self.r_hat_member(concat_sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(concat_sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1 = r_hat1.sum(axis=1) # (batch_size, 1)
                r_hat2 = r_hat2.sum(axis=1) # (batch_size, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                
                ## Calculate Loss
                if original_concat:
                    curr_loss = self.softXEnt_loss(r_hat, concat_target_onehot)
                else:
                    curr_loss = self.softXEnt_loss(r_hat[labels.size(0):], concat_target_onehot[labels.size(0):])
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                ## Compute Acc
                _, predicted = torch.max(r_hat[:labels.size(0)].data, 1)
                correct = (predicted == labels.to(device)).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
            start_idx += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_idx = 0
                
        ensemble_acc = ensemble_acc / total
        return ensemble_acc     


        
