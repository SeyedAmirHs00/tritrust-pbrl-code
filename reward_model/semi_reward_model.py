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

class RewardModelSemiDataAug(RewardModel):
    def __init__(self, obs_dim, action_dim,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, max_size=100,
                 activation='tanh', capacity=5e5, large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0,
                 teacher_eps_skip=0, teacher_eps_equal=0,
                 inv_label_ratio=10, threshold_u=0.95, lambda_u=1, mu=1, dataaug_window=5, crop_range=5):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.optimizer = None
        self.max_size = max_size
        self.activation = activation
        
        ## SURF
        self.dataaug_window = dataaug_window
        self.original_size_segment = size_segment
        self.size_segment = size_segment + 2*dataaug_window
        self.crop_range = crop_range
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CELoss = nn.CrossEntropyLoss()
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1-2*self.label_margin
        
        self.u_buffer_seg1 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.u_buffer_seg2 = np.empty((self.capacity, self.size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.u_buffer_index = 0
        self.u_buffer_full = False
        
        self.inv_label_ratio = inv_label_ratio
        self.threshold_u = threshold_u
        self.lambda_u = lambda_u
        self.mu = mu
        self.UCELoss = nn.CrossEntropyLoss(reduction='none')
    
    
    def put_unlabeled_queries(self, sa_t_1, sa_t_2):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        """
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.u_buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.u_buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.u_buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.u_buffer_seg2[0:remain], sa_t_2[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.u_buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.u_buffer_seg2[self.buffer_index:next_index], sa_t_2)
            self.u_buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        return label : (batch_size, 1)
        """
        sum_r_t_1 = np.sum(r_t_1[:, self.dataaug_window:-self.dataaug_window], axis=1)
        sum_r_t_2 = np.sum(r_t_2[:, self.dataaug_window:-self.dataaug_window], axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1] - 2 * self.dataaug_window
        temp_r_t_1 = r_t_1.copy()[:, self.dataaug_window:-self.dataaug_window]
        temp_r_t_2 = r_t_2.copy()[:, self.dataaug_window:-self.dataaug_window]
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
        
    def uniform_sampling(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size = self.mb_size)
        sa_t_2, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
            
        # get unlabeled samples:
        u_sa_t_1, u_sa_t_2, u_r_t_1, u_r_t_2 =  self.get_queries(mb_size=self.mb_size*self.inv_label_ratio)
        self.put_unlabeled_queries(u_sa_t_1, u_sa_t_2)
        return len(labels)
    
    def disagreement_sampling(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size = self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        # get unlabeled samples
        u_sa_t_1, u_sa_t_2, u_r_t_1, u_r_t_2 =  self.get_queries(mb_size=self.mb_size*self.inv_label_ratio)
        self.put_unlabeled_queries(u_sa_t_1, u_sa_t_2)
        return len(labels)
    
    def entropy_sampling(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(mb_size=self.mb_size*self.large_batch)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
            
        # get unlabeled samples
        u_sa_t_1, u_sa_t_2, u_r_t_1, u_r_t_2 =  self.get_queries(mb_size=self.mb_size*self.inv_label_ratio)
        self.put_unlabeled_queries(u_sa_t_1, u_sa_t_2)
        return len(labels)
    
    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index
    
    def get_cropping_mask(self, r_hat1, r_hat2):
        B, S, _ = r_hat1.shape
        length = np.random.randint(self.original_size_segment-self.crop_range, self.original_size_segment+self.crop_range+1, size=B)
        start_index_1 = np.random.randint(0, S+1-length)
        start_index_2 = np.random.randint(0, S+1-length)
        mask_1 = torch.zeros((B,S,1)).to(device)
        mask_2 = torch.zeros((B,S,1)).to(device)
        for b in range(B):
            mask_1[b, start_index_1[b]:start_index_1[b]+length[b]]=1
            mask_2[b, start_index_2[b]:start_index_2[b]+length[b]]=1

        return mask_1, mask_2
    
    def semi_train_reward(self, num_iters):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        u_max_len = self.capacity if self.u_buffer_full else self.u_buffer_index
        u_total_batch_index = self.shuffle_dataset(u_max_len)
        
        total = 0
        
        start_idx = 0
        u_start_idx = 0
        
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = start_idx + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            u_last_index = u_start_idx + self.train_batch_size * self.mu
            if u_last_index > u_max_len:
                u_last_index = u_max_len
            
            for member in range(self.ensemble_size):
                
                ## Original Data : sa_t_1, sa_t_2, target_onehot
                idxs = total_batch_index[member][start_idx:last_index]
                sa_t_1 = torch.FloatTensor(self.buffer_seg1[idxs]) # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = torch.FloatTensor(self.buffer_seg2[idxs]) # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                ## Predict
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                
                ## shifting & cropping time
                mask_1, mask_2 = self.get_cropping_mask(r_hat1, r_hat2)
                r_hat1 = (mask_1*r_hat1).sum(axis=1) # (batch_size, 1)
                r_hat2 = (mask_2*r_hat2).sum(axis=1) # (batch_size, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                
                curr_loss = self.CELoss(r_hat, labels)
                
                ## get random unlabeled batch
                u_idxs = u_total_batch_index[member][u_start_idx:u_last_index]
                u_sa_t_1 = torch.FloatTensor(self.u_buffer_seg1[u_idxs])
                u_sa_t_2 = torch.FloatTensor(self.u_buffer_seg2[u_idxs])
                
                ## get logits
                u_r_hat1 = self.r_hat_member(u_sa_t_1, member=member)
                u_r_hat2 = self.r_hat_member(u_sa_t_2, member=member)
                
                ## pseudo-labeling
                u_r_hat1_noaug = u_r_hat1[:, self.dataaug_window:-self.dataaug_window]
                u_r_hat2_noaug = u_r_hat2[:, self.dataaug_window:-self.dataaug_window]
                
                with torch.no_grad():
                    u_r_hat1_noaug = u_r_hat1_noaug.sum(axis=1)
                    u_r_hat2_noaug = u_r_hat2_noaug.sum(axis=1)
                    u_r_hat_noaug = torch.cat([u_r_hat1_noaug, u_r_hat2_noaug], axis=-1)

                    pred = torch.softmax(u_r_hat_noaug, dim=1)
                    pred_max = pred.max(1)
                    mask = (pred_max[0] >= self.threshold_u)
                    pseudo_labels = pred_max[1].detach()

                # shifting & cropping time
                u_mask_1, u_mask_2 = self.get_cropping_mask(u_r_hat1, u_r_hat2)
                
                u_r_hat1 = (u_mask_1*u_r_hat1).sum(axis=1)
                u_r_hat2 = (u_mask_2*u_r_hat2).sum(axis=1)
                u_r_hat = torch.cat([u_r_hat1, u_r_hat2], axis=-1)

                curr_loss += torch.mean(self.UCELoss(u_r_hat, pseudo_labels) * mask) * self.lambda_u
  
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1) # (batch_size, )
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
            start_idx += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_idx = 0
                
            u_start_idx += self.train_batch_size * self.mu
            if u_last_index == u_max_len:
                u_total_batch_index = self.shuffle_dataset(u_max_len)
                u_start_idx = 0
                
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def semi_train_soft_reward(self, num_iters):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        u_max_len = self.capacity if self.u_buffer_full else self.u_buffer_index
        u_total_batch_index = self.shuffle_dataset(u_max_len)
        
        total = 0
        
        start_idx = 0
        u_start_idx = 0
        
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = start_idx + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            u_last_index = u_start_idx + self.train_batch_size * self.mu
            if u_last_index > u_max_len:
                u_last_index = u_max_len
            
            for member in range(self.ensemble_size):
                
                ## Original Data : sa_t_1, sa_t_2, target_onehot
                idxs = total_batch_index[member][start_idx:last_index]
                sa_t_1 = torch.FloatTensor(self.buffer_seg1[idxs]) # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = torch.FloatTensor(self.buffer_seg2[idxs]) # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                ## Predict
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                
                ## shifting & cropping time
                mask_1, mask_2 = self.get_cropping_mask(r_hat1, r_hat2)
                r_hat1 = (mask_1*r_hat1).sum(axis=1) # (batch_size, 1)
                r_hat2 = (mask_2*r_hat2).sum(axis=1) # (batch_size, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                
                uniform_index = labels == -1
                labels[uniform_index] = 0
                
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5 #(batch_size, 2)   
                
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                
                ## get random unlabeled batch
                u_idxs = u_total_batch_index[member][u_start_idx:u_last_index]
                u_sa_t_1 = torch.FloatTensor(self.u_buffer_seg1[u_idxs])
                u_sa_t_2 = torch.FloatTensor(self.u_buffer_seg2[u_idxs])
                
                ## get logits
                u_r_hat1 = self.r_hat_member(u_sa_t_1, member=member)
                u_r_hat2 = self.r_hat_member(u_sa_t_2, member=member)
                
                ## pseudo-labeling
                u_r_hat1_noaug = u_r_hat1[:, self.dataaug_window:-self.dataaug_window]
                u_r_hat2_noaug = u_r_hat2[:, self.dataaug_window:-self.dataaug_window]
                
                with torch.no_grad():
                    u_r_hat1_noaug = u_r_hat1_noaug.sum(axis=1)
                    u_r_hat2_noaug = u_r_hat2_noaug.sum(axis=1)
                    u_r_hat_noaug = torch.cat([u_r_hat1_noaug, u_r_hat2_noaug], axis=-1)

                    pred = torch.softmax(u_r_hat_noaug, dim=1)
                    pred_max = pred.max(1)
                    mask = (pred_max[0] >= self.threshold_u)
                    pseudo_labels = pred_max[1].detach()

                # shifting & cropping time
                u_mask_1, u_mask_2 = self.get_cropping_mask(u_r_hat1, u_r_hat2)
                
                u_r_hat1 = (u_mask_1*u_r_hat1).sum(axis=1)
                u_r_hat2 = (u_mask_2*u_r_hat2).sum(axis=1)
                u_r_hat = torch.cat([u_r_hat1, u_r_hat2], axis=-1)

                curr_loss += torch.mean(self.UCELoss(u_r_hat, pseudo_labels) * mask) * self.lambda_u
  
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1) # (batch_size, )
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
            start_idx += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_idx = 0
                
            u_start_idx += self.train_batch_size * self.mu
            if u_last_index == u_max_len:
                u_total_batch_index = self.shuffle_dataset(u_max_len)
                u_start_idx = 0
                
        ensemble_acc = ensemble_acc / total
        return ensemble_acc