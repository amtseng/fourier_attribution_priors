import model.train_profile_model as train_profile_model
import model.train_binary_model as train_binary_model
import numpy as np
import random
import os
import click
import sacred
import json
import multiprocessing

hyperparam_ex = sacred.Experiment("hyperparam")

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


def run_train_command(config_updates, model_type):
    if model_type.lower() == "profile":
        train_profile_model.train_ex.run(
            "run_training", config_updates=config_updates
        )
    else:
        train_binary_model.train_ex.run(
            "run_training", config_updates=config_updates
        )


@click.command()
@click.option(
    "--model-type", "-t",
    type=click.Choice(["binary", "profile"], case_sensitive=False),
    required=True, help="Whether to train a binary model or profile model"
)
@click.option(
    "--file-specs-json-path", "-f", nargs=1, required=True,
    help="Path to file containing paths for training data"
)
@click.option(
    "--chrom-split-json-path", "-s", nargs=1, required=True,
    help="Path to JSON containing possible chromosome splits"
)
@click.option(
    "--chrom-split-key", "-k", nargs=1, required=True, type=str,
    help="Key to chromosome split JSON, denoting the desired split"
)
@click.option(
    "--num-runs", "-n", nargs=1, default=50, help="Number of runs for tuning"
)
@click.option(
    "--hyperparam-json-path", "-p", nargs=1, default=None,
    help="Path to JSON file containing specifications for how to tune hyperparameters"
)
@click.option(
    "--config-json-path", "-c", nargs=1, default=None,
    help="Path to a config JSON file for Sacred, may override hyperparameters"
)
@click.argument(
    "config_cli_tokens", nargs=-1
)
def main(
    model_type, file_specs_json_path, chrom_split_json_path, chrom_split_key,
    num_runs, hyperparam_json_path, config_json_path, config_cli_tokens
):
    """
    Launches hyperparameter tuning for a given number of runs. The model trained can be binary or profile. Below is a description of the parameters.

    Type:
        Either "binary" or "profile", which is the type of model to train.

    File specs JSON:
        For a binary model, this JSON should have two keys: `train_bed_path` and `val_bed_path`, referring to the paths to the BED file with training coordinates/values and the BED file with validation coordinates/values, respectively. For a profile model, this JSON should have the following keys: `peak_beds`, which is a list of paths to BED files containing the peak and summit locations (one per task); and `profile_hdf5`, an HDF5 that contains profiles separated by chromosome, with each dataset being an L x S x 2 array, where L is the chromosome length, and S is the set of profiles (and 2 for each strand); the first half of S is the profiles to predict, the second half of S is the control profiles.

    Chromosome splits JSON:
        Training is split into chromosomes, where specific chromosomes constitute the training set, validation set, and test set. This JSON must map split keys to dictionaries for a split, like such: {"1": {"train": ["chr1", "chr2"], "val": ["chr3"], "test": ["chr4"]}, "2": ...}. The splits key determines which split to use

    Hyperparameters specs JSON:
        An optional JSON that specifies hyperparameters to tune. The entries should either be under the `train` dictionary or `dataset` dictionary, and each entry can be a distribution sampler or list sampler. For distribution samplers, the entry should map to a pair of values, which are endpoints for random sampling. The `log_scale` argument, if specified as a third value, determines whether sampling should be on a log scale. For list samplers, the entry should map to a list of possible values to choose from. Example: {train: {learning_rate: {dist: [-3, -1, True]}, ...}, dataset: {batch_size: {list: [32, 64, 128]}, ...}}

    Config JSON:
        An optional JSON that specifies additional configuration options to override existing Sacred parameters or sampled hyperparameters; dataset parameters should be under the `dataset` key, and training parameters should be under the `train` key.

    Additional commandline arguments are also accepted as additional configuration options. For example, specify `dataset.batch_size=64` or `train.num_epochs=20`. These arguments will override existing Sacred parameters, sampled hyperparameters, or anything in the config JSON.
    """ 
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

    base_config = {}
    # Extract the file paths specified, and put them into the base config at
    # the top level; these will be filled into the training command by Sacred
    with open(file_specs_json_path, "r") as f:
        file_specs_json = json.load(f)
    for key in file_specs_json:
        base_config[key] = file_specs_json[key]

    # Extract the chromosome split to use and put them into the base config at
    # the top level; these will also be filled into the training command
    with open(chrom_split_json_path, "r") as f:
        chrom_split_json = json.load(f)
    split = chrom_split_json[chrom_split_key]
    base_config["train_chroms"] = split["train"]
    base_config["val_chroms"] = split["val"]
    base_config["test_chroms"] = split["test"]

    # Read in the hyperparameter specs dictionary
    if hyperparam_json_path:
        with open(hyperparam_json_path, "r") as f:
            hyperparam_specs = json.load(f)
    else:
        hyperparam_specs = {}
        
    def sample_hyperparams(hyperparam_specs):
        # From the hyperparameter specs dictionary, actually perform the
        # sampling and return a similarly structured dictionary with sampled
        # values
        samples = {}
        for exp_key, param_dict in hyperparam_specs.items():
            samples[exp_key] = {}
            for entry_key, entry_dict in param_dict.items():
                assert ("dist" in entry_dict) + ("list" in entry_dict) == 1
                if "dist" in entry_dict:
                    samples[exp_key][entry_key] = \
                        uniformly_sample_dist(*entry_dict["dist"])
                else:
                    samples[exp_key][entry_key] = \
                        uniformly_sample_list(entry_dict["list"])
        return samples

    # Load in the configuration options supplied as a file
    if config_json_path:
        with open(config_json_path, "r") as f:
            config = json.load(f)
        deep_update(base_config, config)

    # Add in the configuration options supplied to commandline, overwriting the
    # options in the config JSON (or file paths JSON) if needed
    for token in config_cli_tokens:
        key, val = token.split("=", 1)
        try:
            val = eval(val)
        except (NameError, SyntaxError):
            pass  # Keep as string
        d = base_config
        key_pieces = key.split(".")
        for key_piece in key_pieces[:-1]:
            if key_piece not in d:
                d[key_piece] = {}
            d = d[key_piece]
        d[key_pieces[-1]] = val

    for i in range(num_runs):
        # Sample hyperparameters and add in the base config dict (i.e. options
        # specified by the config JSON, file specs JSON, or commandline)
        # Anything in base config overrides sample hyperparameters
        config_updates = sample_hyperparams(hyperparam_specs)
        deep_update(config_updates, base_config)

        # Up until this point, all training parameters/options were under the
        # "train" subdictionary; now hoist up the subdictionary to the root,
        # since that's the command that's being run
        train_dict = config_updates["train"]
        del config_updates["train"]
        deep_update(config_updates, train_dict)

        proc = multiprocessing.Process(
            target=run_train_command, args=(config_updates, model_type)
        )
        proc.start()
        proc.join()  # Wait until the training process stops

        
if __name__ == "__main__":
    main()
