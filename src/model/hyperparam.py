import model.train_binary_model as train
import numpy as np
import random
import os
import click
import sacred
import torch

hyperparam_ex = sacred.Experiment("hyperparam", ingredients=[
    train.train_ex
])

def uniformly_sample_dist(start, end, log_scale=False, log_base=10):
    """
    Returns a single number uniformly sampled between `start` and `end`. If
    `log_scale` is True, the number between `start` and `end` represent
    exponents (base `log_base`).
    """
    if start > end:
            start, end = end, start
    sample = np.random.uniform(start, end)
    if log_scale:
            return log_base ** sample
    return sample


def uniformly_sample_list(vals):
    """
    Returns a single value uniformly sampled from the list `vals`.
    """
    return random.choice(vals)


def deep_update(parent, update):
    """
    Updates the dictionary `parent` with keys and values in `update`, and does
    so recursively for values that are dictionaries themselves. This mutates
    `parent`.
    """
    for key, val in update.items():
        if key not in parent:
            parent[key] = val
        if type(parent[key]) is dict and type(val) is dict:
            deep_update(parent[key], update[key])
        else:
            parent[key] = val


@hyperparam_ex.capture
def launch_training(hparams, config, train_file, val_file):
    deep_update(hparams, config)
    config_updates = hparams
    config_updates["train_bed_path"] = train_file
    config_updates["val_bed_path"] = val_file
    train.train_ex.run("run_training", config_updates=config_updates)


@click.command()
@click.option(
    "--train-file", "-t", nargs=1, required=True,
    help="Path to gzipped training BED"
)
@click.option(
    "--val-file", "-v", nargs=1, required=True,
    help="Path to gzipped validation BED"
)
@click.option(
    "--num-runs", "-n", nargs=1, default=50, help="Number of runs for tuning"
)
@click.option(
    "--config-path", "-c", nargs=1, default=None,
    help="Path to a config JSON file for Sacred, may override hyperparameters"
)
def main(train_file, val_file, num_runs, config_path):
    def sample_hyperparams():
        np.random.seed()  # Re-seed to random number
        hparams = {
            "fc_drop_rate": uniformly_sample_dist(-1, -3, log_scale=True),
            "learning_rate": uniformly_sample_dist(-1, -6, log_scale=True),
            "att_prior_loss_weight": uniformly_sample_dist(-1, 2, log_scale=True),
            "att_prior_pos_weight": uniformly_sample_dist(-2, 2, log_scale=True),
            "dataset": {
                "batch_size": uniformly_sample_list([32, 64, 128, 256])
            }
        }
        return hparams

    if "MODEL_DIR" not in os.environ:
        print("Warning: using default directory to store model outputs")
        print("\tTo change, set the MODEL_DIR environment variable")
        ok = input("\tIs this okay? [y|N]: ")
        if ok.lower() not in ("y", "yes"):
            print("Aborted")
            return
    else:
        model_dir = os.environ["MODEL_DIR"]
        print("Using %s as directory to store model outputs" % model_dir)

    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    for i in range(num_runs):
        launch_training(
            sample_hyperparams(), config, train_file, val_file
        )
    
        
if __name__ == "__main__":
        main()   
