import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from reward_model.vanilla_reward_model import RewardModel
import cv2
import imageio
device = 'cuda'

def gen_net(in_size=1, out_size=1, hidden_size=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, hidden_size))
        net.append(nn.LeakyReLU())
        in_size = hidden_size
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation =='sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())
    
    return net

class RewardModelHuman(RewardModel):
    def __init__(self, obs_dim, action_dim,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, max_size=100,
                 activation='tanh', capacity=5e5, large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0,
                 teacher_eps_skip=0, teacher_eps_equal=0, video_record_path='pebble_videos', seed=12345):
        
        super(RewardModelHuman, self).__init__(
            obs_dim,
            action_dim,
            ensemble_size=ensemble_size,
            lr=lr,
            mb_size=mb_size,
            size_segment=size_segment,
            max_size=max_size,
            activation=activation,
            capacity=capacity,
            large_batch=large_batch,
            label_margin=label_margin,
            teacher_beta=teacher_beta,
            teacher_gamma=teacher_gamma,
            teacher_eps_mistake=teacher_eps_mistake,
            teacher_eps_skip=teacher_eps_skip,
            teacher_eps_equal=teacher_eps_equal,
        )

        # Parent stores dims as ds/da; human code uses obs_dim/action_dim.
        self.obs_dim = self.ds
        self.action_dim = self.da
        self.video_record_path = video_record_path
        self.frames = []
        self.session = 0
        self.seed = seed
        os.makedirs(self.video_record_path, exist_ok=True)
    
    def flush_data(self):
        self.inputs = []
        self.targets = []
        self.frames = []
        
    def add_data_with_frame(self, obs, action, reward, done, frame):
        sa_t = np.concatenate([obs, action], axis=-1)
        r_t = reward
        frame = np.array(frame, dtype='uint8')
        h, w, c = frame.shape
        frame = frame.reshape(1, h, w, c)
        
        flat_input = sa_t.reshape(1, self.obs_dim + self.action_dim)
        flat_target = np.array(r_t).reshape(1, 1)
        
        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            self.frames.append(frame)
            
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            self.frames[-1] = np.concatenate([self.frames[-1], frame])
            
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.frames = self.frames[1:]
            self.inputs.append([]) # new episode
            self.targets.append([])
            self.frames.append([])
        
        else:
            if len(self.inputs[-1]) == 0: #new episode
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
                self.frames[-1] = frame
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                self.frames[-1] = np.concatenate([self.frames[-1], frame])
    
    def get_queries_with_frame(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        #완전히 끝난 에피소드들
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        train_frames = np.array(self.frames[:max_len])
        
        #에피소드 인덱스
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        
        sa_t_1 = train_inputs[batch_index_1] # (batch_size, episode_len(T), obs_dim + action_dim)
        r_t_1 = train_targets[batch_index_1] # (batch_size, episode_len(T), 1)
        f_t_1 = train_frames[batch_index_1] # (batch_size, episode_len(T), h, w, c)
        
        sa_t_2 = train_inputs[batch_index_2] # (batch_size, episode_len(T), obs_dim + action_dim)
        r_t_2 = train_targets[batch_index_2] # (batch_size, episode_len(T), 1)
        f_t_2 = train_frames[batch_index_2] # (batch_size, episode_len(T), h, w, c)
        
        b, t, h, w, c = f_t_1.shape
        
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (batch_size * T, obs_dim + action_dim)
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (batch_size * T, 1)
        f_t_1 = f_t_1.reshape(-1, h, w, c) # (batch_size * T, h, w, c)
        
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (batch_size * T, obs_dim + action_dim)
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (batch_size * T, 1)
        f_t_2 = f_t_2.reshape(-1, h, w, c) # (batch_size * T, h, w, c)
        
        time_index = np.array([list(range(i*len_traj, i*len_traj+self.size_segment)) for i in range(mb_size)])
        # mb_size 128, len_traj 500, segment_len 50
        # time_index : [[0~49], [500~549], ..., [63500~63549]]
        
        time_index_1 = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
        time_index_2 = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
        # time_index 1 &2 -> (batch_size, size_segment)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        f_t_1 = np.take(f_t_1, time_index_1, axis=0) # Batch x size_seg x h x w x c
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
        f_t_2 = np.take(f_t_2, time_index_2, axis=0) # Batch x size_seg x h x w x c
        
        f_cat = np.concatenate((f_t_1, f_t_2), axis=3) # Batch x size_seg x h x 2w x c
        return sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        return label : (batch_size, 1)
        """
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
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
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
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
    
    def get_human_label(self, f_cat):
        for idx, f_cat_ins in enumerate(f_cat):
            os.makedirs(f'{self.video_record_path}/session{self.session:03d}', exist_ok=True)
            writer = imageio.get_writer(f'{self.video_record_path}/session{self.session:03d}/video{idx:03d}.mp4', fps=15)
            for frame in f_cat_ins:
                writer.append_data(frame)
            writer.close()
        
        labels = []
        label_dict = {'1':(0, 'left'), '2':(1, 'right'), '3':(-1, 'equal')}
        for idx, f_cat_ins in enumerate(f_cat):
            s = False
            while not s:
                reward = input(f'[Seed:{self.seed}] Put Preference session{self.session:03d}/video{idx:03d} (1 (left), 2 (right), 3 (equal)): ').strip()
                try:
                    label, s = label_dict[reward]
                    print(s)
                except:
                    s = False
            labels.append(label)
        labels = np.array(labels).reshape(-1, 1)
        return labels
    
    # def get_human_label(self, f_cat):
    #     for idx, f_cat_ins in enumerate(f_cat):
    #         os.makedirs(f'{self.video_record_path}/session{self.session:03d}', exist_ok=True)
    #         writer = imageio.get_writer(f'{self.video_record_path}/session{self.session:03d}/video{idx:03d}.mp4', fps=15)
    #         for frame in f_cat_ins:
    #             writer.append_data(frame)
    #         writer.close()
        
    #     labels = []
    #     label_dict = {'1':(0, 'left'), '2':(1, 'right'), '3':(-1, 'equal')}
    #     for idx, f_cat_ins in enumerate(f_cat):
    #         cap = cv2.VideoCapture(f'{self.video_record_path}/session{self.session:03d}/video{idx:03d}.mp4')
    #         print(f'Put Preference session{self.session:03d}/video{idx:03d} (1 (left), 2 (right), 3 (equal)): ')
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #                 continue    
                
    #             cv2.imshow('Video Player', frame)
    #             reward = str(cv2.waitKey(0))
    #             if reward in label_dict.keys():
    #                 label, s = label_dict[reward]
    #                 print(s)
    #                 break
                
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         labels.append(label)
    #     labels = np.array(labels).reshape(-1, 1)
    #     return labels
    
    def uniform_sampling_human(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat= self.get_queries_with_frame(mb_size = self.mb_size)
        sa_t_1, sa_t_2, r_t_1, r_t_2, gt_labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        labels = self.get_human_label(f_cat)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        self.session += 1
        return len(labels)
    
    def disagreement_sampling_human(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat = self.get_queries_with_frame(mb_size = self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        f_cat = f_cat[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, gt_labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        labels = self.get_human_label(f_cat)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        self.session += 1
        return len(labels)
    
    def entropy_sampling_human(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2, f_cat =  self.get_queries_with_frame(mb_size=self.mb_size*self.large_batch)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        f_cat = f_cat[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, gt_labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        labels = self.get_human_label(f_cat)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        self.session += 1
        return len(labels)
