import wandb
wandb.init(project="preemptable", resume=True)

if wandb.run.resumed:
    # restore the best model
    pass
else:
    pass