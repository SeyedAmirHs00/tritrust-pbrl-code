# logger_union.py
# Combines features from logger.py and logger_RIME.py

from collections import defaultdict
import json
import os
import csv
import shutil
import torch
import numpy as np
from termcolor import colored
from time import time

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# Common formats
COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('true_episode_reward', 'TR', 'float'), 
    ('total_feedback', 'TF', 'int'),
    ('labeled_feedback', 'LR', 'int'),
    ('noisy_feedback', 'NR', 'int'),
    ('duration', 'D', 'time'),
    ('total_duration', 'TD', 'time'),
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('true_episode_reward', 'TR', 'float'),
    ('true_episode_success', 'TS', 'float'),
]

REWARD_METRIC_FORMAT = [
    ('step', 'S', 'int'),
    ('reward', 'RW', 'float'),
    ('true_reward', 'TRW', 'float'),
    ('success', 'SC', 'float'),
]

AGENT_TRAIN_FORMAT = {
    'sac': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('bc_loss', 'BCLOSS', 'float'),
    ],
    'ppo': [
        ('batch_reward', 'BR', 'float'),
    ],
}

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)

class MetersGroup(object):
    def __init__(self, file_name, formating, allow_dynamic_keys=False):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        os.makedirs(os.path.dirname(self._csv_file_name), exist_ok=True)
        self._csv_file = open(self._csv_file_name, 'w')
        self._csv_writer = None
        self._allow_dynamic_keys = allow_dynamic_keys

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            # Remove prefix for train/eval/reward
            if key.startswith('train'):
                short_key = key[len('train') + 1:]
            elif key.startswith('eval'):
                short_key = key[len('eval') + 1:]
            elif key.startswith('reward'):
                short_key = key[len('reward') + 1:]
            else:
                short_key = key
            short_key = short_key.replace('/', '_')
            data[short_key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise Exception(f'invalid format type: {ty}')

    def _dump_to_console(self, data, prefix):
        prefix_col = 'yellow' if prefix == 'train' else ('green' if prefix == 'eval' else 'cyan')
        prefix_disp = colored(prefix, prefix_col)
        pieces = [f'| {prefix_disp: <14}']
        if self._allow_dynamic_keys and len(data) > 0:
            for key in sorted(data.keys()):
                value = data[key]
                if isinstance(value, int):
                    pieces.append(f'{key}: {value}')
                elif isinstance(value, float):
                    pieces.append(f'{key}: {value:.04f}')
                else:
                    pieces.append(f'{key}: {value}')
        else:
            for key, disp_key, ty in self._formating:
                value = data.get(key, 0)
                pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()

class Logger(object):
    def __init__(self,
                 log_dir,
                 save_tb=True,
                 log_frequency=10000,
                 agent='sac',
                 train_log_name='train',
                 eval_log_name='eval',
                 use_reward_mg=True):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        if save_tb and HAS_TENSORBOARD:
            tb_dir = os.path.join(log_dir, 'tb', str(time()).replace('.', '_'))
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except Exception:
                    print("logger_union.py warning: Unable to remove tb directory")
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train', train_log_name),
                                     formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'test', eval_log_name),
                                    formating=COMMON_EVAL_FORMAT)
        self._reward_mg = MetersGroup(os.path.join(log_dir, 'reward', 'reward'),
                                      formating=REWARD_METRIC_FORMAT,
                                      allow_dynamic_keys=True) if use_reward_mg else None

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        if not (key.startswith('train') or key.startswith('eval') or key.startswith('reward')):
            raise ValueError(f"Unknown log key prefix: {key}")
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._try_sw_log(key, value / n, step)
        if key.startswith('train'):
            mg = self._train_mg
        elif key.startswith('eval'):
            mg = self._eval_mg
        elif key.startswith('reward') and self._reward_mg is not None:
            mg = self._reward_mg
        else:
            return
        mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
            if self._reward_mg is not None:
                self._reward_mg.dump(step, 'reward', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        elif ty == 'reward' and self._reward_mg is not None:
            self._reward_mg.dump(step, 'reward', save)
        else:
            raise Exception(f'invalid log type: {ty}')
