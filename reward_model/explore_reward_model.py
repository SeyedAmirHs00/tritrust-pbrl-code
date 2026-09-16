import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from reward_model.vanilla_reward_model import RewardModel
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

class RewardModelExplore(RewardModel):
    def __init__(self, obs_dim, action_dim,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, max_size=100,
                 activation='tanh', capacity=5e5, large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0,
                 teacher_eps_skip=0, teacher_eps_equal=0):
        
        super(RewardModelExplore, self).__init__(obs_dim, action_dim, ensemble_size, lr, mb_size, size_segment, max_size,
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
    
    def r_hat_std(self, x):
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats), np.std(r_hats)
    
    def r_hat_std_batch(self, x):
        """
        x : (batch_size, obs_dim + action_dim) -> (ensemble_size, batch_size, 1) -> (batch_size, 1)
        """
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0), np.std(r_hats, axis=0)
