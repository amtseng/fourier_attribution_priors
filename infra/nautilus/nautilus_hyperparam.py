import click
import os
import subprocess
import shutil
import json
import sys

CEPH_MOUNT = "/ceph"

def copy_item(path, directory=False):
    """
    Copies an item at the given path from the Ceph to its final destination.
    The path given should be the path to copy to, beginning with
    `/users/amtseng/`. This item should exist in Ceph, at the exact same path
    (starting with `/ceph/users/amtseng/`). If `directory` is True, the item
    at the path is assumed to be a directory.
    """
    stem = "/users/amtseng/"
    path = os.path.normpath(path)  # Normalize
    assert path.startswith(stem)
    ceph_path = os.path.join(CEPH_MOUNT, path[1:])  # Append without "/"
    if directory:
        # shutil.copytree will create the destination directory
        shutil.copytree(ceph_path, path)
    else:
        # shutil.copy requires the directory to exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(ceph_path, os.path.dirname(path))


def copy_data(
    model_type, file_specs_json_path, chrom_split_json_path,
    hyperparam_json_path, config_json_path
):
    """
    Given the paths to various files needed for model training, this function
    copies the paths from the bucket and into the corresponding location on
    the running pod. Note that all these paths must be absolute paths starting
    with `/users/amtseng/`, and they will be copied to this location, as well.
    This will also copy genomic references and source code.
    """
    print("Copying configuration/specification JSONs...")
    sys.stdout.flush()
    # Copy the jsons
    for json_path in [
        file_specs_json_path, chrom_split_json_path, hyperparam_json_path,
        config_json_path
    ]:
        if json_path:
            # Paths like `hyperparam_json_path` could be None
            copy_item(json_path)

    print("Copying data...")
    sys.stdout.flush()
    # Within the file specs, copy all paths in the file specs; this JSON should
    # be in the right place now
    with open(file_specs_json_path) as f:
        file_specs_json = json.load(f)
    if model_type == "binary":
        file_paths = [
            file_specs_json["labels_hdf5"], file_specs_json["bin_labels_npy"],
            file_specs_json["peak_signals_npy"]
        ]
    else:
        file_paths = file_specs_json["peak_beds"]
        file_paths.append(file_specs_json["profile_hdf5"])
    for file_path in file_paths:
        copy_item(file_path)

    print("Copying genomic references...")
    sys.stdout.flush()
    # Copy the genomic references
    copy_item("/users/amtseng/genomes/", directory=True)
        
    print("Copying source code...")
    sys.stdout.flush()
    copy_item("/users/amtseng/att_priors/src/", directory=True)


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
    # First check that we are inside a container
    assert os.path.exists("/.dockerenv")

    # Copy over the data
    copy_data(
        model_type, file_specs_json_path, chrom_split_json_path,
        hyperparam_json_path, config_json_path
    )

    # Go to the right directory and run the `hyperparam.py` script, with the
    # same environment variables
    os.chdir("/users/amtseng/att_priors/src")
    comm = ["python", "-m", "model.hyperparam"]
    comm += ["-t", model_type]
    comm += ["-f", file_specs_json_path]
    comm += ["-s", chrom_split_json_path]
    comm += ["-k", chrom_split_key]
    comm += ["-n", "1"]  # Each time, only run once
    if hyperparam_json_path:
        comm += ["-p", hyperparam_json_path]
    if config_json_path:
        comm += ["-c", config_json_path]
    comm += config_cli_tokens
    env = os.environ.copy()
    
    local_model_path = env["MODEL_DIR"]

    for i in range(num_runs):
        print("Beginning run %d" % (i + 1))
        sys.stdout.flush()

        proc = subprocess.Popen(comm, env=env, stderr=subprocess.STDOUT)
        proc.wait()

        local_run_num = i + 1  # This was the run just created locally
        
        # Now get the maximum run num on Ceph
        ceph_model_path = os.path.join(CEPH_MOUNT, local_model_path[1:])
        ceph_run_nums = []
        if os.path.isdir(ceph_model_path):  # It may not exist
            for dirname in os.listdir(ceph_model_path):
                try:
                    ceph_run_nums.append(int(dirname))
                except ValueError:
                    pass
        ceph_run_num = max(ceph_run_nums) + 1 if ceph_run_nums else 1
        
        print("Copying training results from run %d into Ceph..." % (i + 1))
        sys.stdout.flush()
        local_run_path = os.path.join(local_model_path, str(local_run_num))
        ceph_run_path = os.path.join(ceph_model_path, str(ceph_run_num))
        shutil.copytree(local_run_path, ceph_run_path)

    print("Done!")

if __name__ == "__main__":
    main()
