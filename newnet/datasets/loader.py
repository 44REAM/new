#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from newnet.config import cfg
from newnet.datasets.sampledata import SampleDataset2D
from newnet.datasets import TBDatasets, Cifar10

DATASET ={'tb': TBDatasets, "cifar10": Cifar10}

def _construct_loader(dataset, batch_size, drop_last , split, shuffle,transformer = None):
    """Constructs the data loader for the given dataset."""
    # err_str = "Dataset '{}' not supported".format(dataset_name)
    # assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    # Retrieve the data path for the dataset
    # Construct the dataset
    # Create a sampler for multi-process training
    # sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset(cfg.DATASET.PATH, split, transformer),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return loader

def construct_train_loader(transformer = None):
    """Train loader wrapper."""
    return _construct_loader(
        dataset=DATASET[cfg.DATASET.NAME],
        batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False,
        split = 'train', 
        shuffle=cfg.DATASET.SHUFFLE,
        transformer = transformer
    )

def construct_test_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset=DATASET[cfg.DATASET.NAME],
        batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False,
        split = 'test',
        shuffle=cfg.DATASET.SHUFFLE
    )

def construct_gradcam_loader():
    return _construct_loader(
        dataset=DATASET[cfg.DATASET.NAME],
        batch_size=1,
        drop_last=False,
        split = 'test',
        shuffle = False
    )

if __name__ == '__main__': 

    a = construct_train_loader()
    print(a)
    for i,j in a:
        print(i.shape)
    a = construct_test_loader()
    print(a)
    for i,j in a:
        print(i.shape)