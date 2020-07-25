#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Meters."""


import numpy as np
import torch
from newnet.config import cfg

def accuracy(preds, labels):
    _, index = preds.max(dim = 1)
    correct = (index == labels).sum().item()
    n_labels = len(labels)

    acc = correct/n_labels*100
    return acc

class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, start_epoch, epoch_iters):

        # if not resume then start at 0
        self.current_epoch = start_epoch+1

        self.max_iter = (cfg.OPTIM.MAX_EPOCH) * epoch_iters
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.current_iter = start_epoch*epoch_iters

        self.loss = []
        self.avg_loss = 0.0
        self.lr = None

        self.acc = []
        self.avg_acc = 0.0

    def reset(self, timer=False):

        self.loss = []
        self.avg_loss = 0.0
        self.lr = None

        self.acc = []

    def update_iter(self, loss, preds, labels):
        # Current minibatch stats
        self.loss.append(loss.item())
        self.avg_loss = np.mean(self.loss)
        self.current_iter +=1

        # ----------------------------------- metrics
        self.acc.append(accuracy(preds, labels))
        self.avg_acc = np.mean(self.acc)


    def get_current_stats_iter(self):

        stats = {

            # plus one because epoch start at 0
            "iters":self.current_iter,
            "train_batch_loss":self.avg_loss,

        }
        return stats

    def update_epoch(self, lr):
        self.current_epoch +=1
        self.lr = lr
        self.avg_loss = np.mean(self.loss)

        self.avg_acc = np.mean(self.acc)

    def get_current_stats_epoch(self):
        stats = {
            "epoch":self.current_epoch,
            "train_loss":self.avg_loss,
            'train_acc': self.avg_acc
        }
        return stats

    def print_iter(self):
        epoch_string = f"epoch: {self.current_epoch}/{self.max_epoch}"
        loss_string = f"loss: {self.avg_loss}"
        iter_string = f"iter: {self.current_iter}/{self.max_iter}"
        acc_string = f"acc: {self.avg_acc}"
        print(f"BATCH: {epoch_string} {loss_string} {iter_string} {acc_string}")

    def print_epoch(self):

        epoch_string = f"epoch: {self.current_epoch}/{self.max_epoch}"
        loss_string = f"loss: {self.avg_loss}"
        print(f"TRAIN: {epoch_string} {loss_string}")

    
class TestMeter(object):
    """Measures training stats."""

    def __init__(self, start_epoch):

        # if not resume then start at 0
        self.current_epoch = start_epoch+1

        self.loss = []
        self.avg_loss = 0.0

        self.acc = []
        self.avg_loss = 0.0

    def reset(self, timer=False):

        self.loss = []
        self.avg_loss = 0.0

        self.acc = []
        self.avg_loss = 0.0

    def update_iter(self, preds, labels):
        self.acc.append(accuracy(preds, labels))

    def update_epoch(self):
        self.current_epoch +=1
        self.avg_acc = np.mean(self.acc)

    def get_current_stats_epoch(self):
        stats = {
            "test_acc":self.avg_acc
        }
        return stats

    def print_epoch(self):

        acc_string = f"acc: {self.avg_acc}"
        print(f"TEST: {acc_string}")

