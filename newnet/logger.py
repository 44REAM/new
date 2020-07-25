import collections
import os

import torch
import wandb
from yacs.config import CfgNode 

from newnet.builders import get_model
from newnet.config import cfg, set_config

INIT = 'INIT:'

api = wandb.Api()



def init_logger():

    wandb.init(project = cfg.WANDB.PROJECT, name = cfg.WANDB.NAME)
    print(f'{INIT} init wandb finish')

def save_config():
    wandb.config.cfg = cfg

def _log_dict(name, values):
    return {name:values}

def info(name, values):
    wandb.log(_log_dict(name, values))

def log(values):
    wandb.log(values)




    

def get_all_wandb_run():
    runs = api.runs(f"{cfg.WANDB.ENTITY}/{cfg.WANDB.PROJECT}")
    print(f"Found {len(runs)} runs in {cfg.WANDB.PROJECT}")
    return runs

def get_last_wandb_run():
    runs = get_all_wandb_run()
    run = runs[0]

    print(f"Open last run at: {cfg.WANDB.ENTITY}/{cfg.WANDB.PROJECT}/{run.id}")

    return run

def load_last_wandb_config():
    run = get_last_wandb_run()
    print(cfg.WANDB.NAME)
    print(cfg.ANYNET.WIDTH_ARRAY)
    
    run.config['cfg']['OPTIM']['SGD']['DAMPENING'] = 0.0

    cfg.merge_from_other_cfg(CfgNode(run.config['cfg']))
    print(cfg.WANDB.NAME)
    print(cfg.ANYNET.WIDTH_ARRAY)

def load_last_wandb_checkpoint():
    run = get_last_wandb_run()
    name = _get_save_name()

    root = 'wandb'
    run.file(name = name).download(root = root, replace = True)

    filepath = os.path.join(root, name)
    ckpt = torch.load(filepath)
    
    return ckpt


def _get_save_name():
    name = cfg.WANDB.NAME + ".pth"
    return name

def get_wandb_checkpoint():

    path = os.path.join(wandb.run.dir, _get_save_name()) 
    return path

def get_checkpoint():
    # TODO
    return None


def save_checkpoint(model, optim, epoch, wandb_path = True):
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    ckpt = {
        "epoch": epoch,
        "model_state": sd,
        "optim_state": optim.state_dict()
    }

    if wandb_path:
        path = get_wandb_checkpoint()
    else:
        path = get_checkpoint()
    
    # save to local path
    torch.save(ckpt, path)
    
    # save to wandb 
    # wandb.save(path)

    print(f"Save to {path}")

def load_wandb_last_checkpoint():
    run = get_wandb_last_run()

    model = get_model()

    
if __name__ =='__main__':
    set_config()

    load_last_wandb_checkpoint(5)