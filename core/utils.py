'''
* @name: utils.py
* @description: Other functions.
'''


import os
import random
import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)  # 检测梯度正向与反向传播异常
    torch.backends.cudnn.benchmark = False
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def save_model(save_path, epoch, model, optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'epoch_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)
