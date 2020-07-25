#!/usr/bin/env python3


import numpy as np
import torch

from newnet.config import cfg


def construct_optimizer(model):

    return torch.optim.SGD(

        model.parameters(),
        lr=cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.SGD.MOMENTUM,
        weight_decay=cfg.OPTIM.SGD.WEIGHT_DECAY,
        dampening=cfg.OPTIM.SGD.DAMPENING,
        nesterov=cfg.OPTIM.SGD.NESTEROV
    )



def lr_fun_cos(cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    base_lr, max_epoch = cfg.OPTIM.BASE_LR, cfg.OPTIM.MAX_EPOCH
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / max_epoch))


def get_lr_fun():
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.OPTIM.LR_POLICY
    if lr_fun not in globals():
        raise NotImplementedError("Unknown LR policy:" + cfg.OPTIM.LR_POLICY)
    return globals()[lr_fun]


def get_epoch_lr(cur_epoch):

    """Retrieves the lr for the given epoch according to the policy."""
    lr = get_lr_fun()(cur_epoch)
    # Linear warmup
    if cur_epoch < cfg.OPTIM.BURNIN:
        alpha = cur_epoch / cfg.OPTIM.BURNIN
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
