import torch
import numpy as np
import random
import os
import json

def reproducible_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def load_logs(name):
    with open(f'logs/{name}.json', 'r') as f:
        logs = json.load(f)
    results = {}
    for metric in logs[0]:
        results[metric] = [log[metric] for log in logs]
    return results

def load_cfg(name):
    with open(f'cfgs/{name}.json', 'r') as f:
        cfg = json.load(f)
    return cfg


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1.0 / batch_size))
    return res
