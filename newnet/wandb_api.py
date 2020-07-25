import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
runs = api.runs("quacktab/MODEL")
print("Found %i" % len(runs))

for run in runs:
    # save the metrics for the run to a csv file
    metrics_dataframe = run.config
    print(metrics_dataframe)