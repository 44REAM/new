#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import os
import wandb
import numpy as np
import torch
from dotenv import load_dotenv

from newnet.config import cfg, set_config
from newnet.models import AnyNet
from newnet.datasets import loader
from newnet.datasets.transformers import Transforms
from newnet.meters import TrainMeter, accuracy, TestMeter
from newnet.builders import get_model
import newnet.logger as logger
import newnet.optimizer as optim



def setup_env():
    """Sets up environment for training or testing."""
    # load environment variable rom file
    load_dotenv()

    # set config variable
    set_config()

    # init wandb logger
    logger.init_logger()

    # save cfg to wandb
    logger.save_config()

    torch.cuda.empty_cache() 

    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)



def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""

    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)


    # Enable training mode
    model.train()

    for cur_iter, (inputs, labels) in enumerate(train_loader):

        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        
        # Perform the forward pass
        preds = model(inputs)
        
        # Compute the loss
        loss = loss_fun(preds, labels)

        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

        # calculate metric
        acc = accuracy(preds, labels)

        # update iter
        train_meter.update_iter(loss, preds, labels)
        train_meter.print_iter()

        # log iter and metrics
        logger.info('train_acc', acc)
        logger.log(train_meter.get_current_stats_iter())

    # update epoch
    train_meter.update_epoch(lr)
    train_meter.print_epoch()

    # log epoch
    logger.log(train_meter.get_current_stats_epoch())
    train_meter.reset()

    # Log epoch stats



def setup_model():
    model = get_model()

    print(model)
    

    err_str = f"Avaliable gpu: {torch.cuda.device_count()}"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str

    cur_device = torch.cuda.current_device()

    model = model.cuda(device = cur_device)
    
    # use multiple GPU
    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(
            module=model, device_ids=[i for i in range(cfg.NUM_GPUS)]
        )

    return model

def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    model = setup_model()
    loss_fun =  torch.nn.CrossEntropyLoss().cuda()
    optimizer  = optim.construct_optimizer(model)
    start_epoch = 0
    # no resume

    # train loader and meters
    train_loader = loader.construct_train_loader(Transforms(basic=True, elastic_transform=True))
    train_meter = TrainMeter(start_epoch, len(train_loader))

    # test loader and meters
    test_loader = loader.construct_test_loader()
    test_meter = TestMeter(start_epoch)

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # train one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)



        # test model
        test_epoch(test_loader, model, test_meter)

    logger.save_checkpoint(model, optimizer, cur_epoch+1)


@torch.no_grad()
def test_epoch(test_loader, model, test_meter):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)

        # Compute the predictions
        preds = model(inputs)

        # Update and log stats
        test_meter.update_iter(preds, labels)

    # update epoch stats
    test_meter.update_epoch()
    test_meter.print_epoch()

    # log epoch stats
    logger.log(test_meter.get_current_stats_epoch())
    test_meter.reset()

def test_gradcam():
    # set initial config
    set_config()

    # load config from wandb
    logger.load_last_wandb_config()

    # load last model from wandb
    model = setup_model()
    logger.load_last_wandb_checkpoint(model)

    # get test data
    test_loader = loader.construct_test_loader()

    gradcam = GradCAM(model, model.head, ["fc"])

    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)

        # Compute the predictions
    grad, preds = gradcam(inputs)
    print(grad[0].shape)


def test_model():
    # set initial config
    set_config()

    # load config from wandb
    logger.load_last_wandb_config()

    # load last model from wandb
    model = setup_model()
    logger.load_last_wandb_checkpoint(model)

    # get test data
    test_loader = loader.construct_test_loader()

# def test_model():

#     # Construct the model
#     model = setup_model()
#     # Load model weights
#     checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
#     logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
#     # Create data loaders and meters
#     test_loader = loader.construct_test_loader()
#     test_meter = meters.TestMeter(len(test_loader))
#     # Evaluate the model
#     test_epoch(test_loader, model, test_meter, 0)
    
    
if __name__ =='__main__':

    test_gradcam()



