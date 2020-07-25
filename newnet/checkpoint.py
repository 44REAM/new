import os

import torch
import wandb
from newnet.config import cfg

api = wandb.Api()

def get_wandb_checkpoint():
    
    name = cfg.WANDB.NAME
    path = os.path.join(wandb.run.dir, name) + ".pth"
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


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if cfg.NUM_GPUS > 1 else model
    ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]

def get_all_run():
    runs = api.runs(f"{cfg.WANDB.ENTITY}/{cfg.WANDB.PROJECT}")
    print(f"Found {len(runs)} in {cfg.WANDB.PROJECT} project")
    