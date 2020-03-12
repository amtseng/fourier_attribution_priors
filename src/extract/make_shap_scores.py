import extract.compute_shap as compute_shap
import extract.data_loading as data_loading
import model.profile_models as profile_models
import model.binary_models as binary_models
import model.util as model_util
import feature.make_binary_dataset as make_binary_dataset
import feature.make_profile_dataset as make_profile_dataset
import torch
import json
import tqdm
import numpy as np
import h5py
import click
import os

def make_shap_scores(
    model_path, model_type, files_spec_path, input_length, num_tasks, out_path,
    reference_fasta, chrom_sizes, task_index=None, profile_length=1000,
    controls=None, num_strands=2, chrom_set=None, batch_size=128
):
    """
    Computes SHAP scores over an entire dataset, and saves them as an HDF5 file.
    The SHAP scores are computed for all positive input sequences (i.e. peaks or
    positive bins).
    Arguments:
        `model_path`: path to saved model
        `model_type`: either "binary" or "profile"
        `files_spec_path`: path to files specs JSON
        `input_length`: length of input sequences
        `num_tasks`: number of tasks in the model
        `out_path`: path to HDF5 to save SHAP scores and input sequences
        `reference_fasta`: path to reference FASTA
        `chrom_sizes`: path to chromosome sizes TSV
        `task_index`: index of task to explain; if None, explain all tasks in
            aggregate
        `profile_length`: for profile models, the length of output profiles
        `controls`: for profile models, the kind of controls used: "matched",
            "shared", or None; this also determines the class of the model
        `chrom_set`: the set of chromosomes to compute SHAP scores for; if None,
            defaults to all chromosomes
        `batch_size`: batch size for SHAP score computation
    Creates/saves an HDF5 containing the SHAP scores and the input sequences.
    The HDF5 has the following keys:
        `coords_chrom`: an N-array of the coordinate chromosomes
        `coords_start`: an N-array of the coordinate starts
        `coords_end`: an N-array of the coordinate ends
        `one_hot_seqs`: an N x I x 4 array of one-hot encoded input sequences
        `hyp_scores`: an N x I x 4 array of hypothetical SHAP contribution
            scores
        `model`: path to the model, `model_path`
    """
    assert model_type in ("binary", "profile")
    
    # Determine the model class and import the model
    if model_type == "binary":
        model_class = binary_models.BinaryPredictor
    elif controls == "matched":
        model_class = profile_models.ProfilePredictorWithMatchedControls
    elif controls == "shared":
        model_class = profile_models.ProfilePredictorWithSharedControls
    elif controls is None:
        model_class = profile_models.ProfilePredictorWithoutControls
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_util.restore_model(model_class, model_path)
    model.eval()
    model = model.to(device)

    # Create the data loaders
    if model_type == "binary":
        input_func = data_loading.get_binary_input_func(
           files_spec_path, input_length, reference_fasta
        )
        pos_samples = data_loading.get_positive_binary_bins(
            files_spec_path, chrom_set=chrom_set
        )
    else:
        input_func = data_loading.get_profile_input_func(
            files_spec_path, input_length, profile_length, reference_fasta,
        )
        pos_samples = data_loading.get_positive_profile_coords(
            files_spec_path, chrom_set=chrom_set
        )

    num_pos = len(pos_samples)
    num_batches = int(np.ceil(num_pos / batch_size))

    # Allocate arrays to hold the results
    coords_chrom = np.empty(num_pos, dtype=str)
    coords_start = np.empty(num_pos, dtype=int)
    coords_end = np.empty(num_pos, dtype=int)
    status = np.empty(num_pos, dtype=int)
    one_hot_seqs = np.empty((num_pos, input_length, 4))
    hyp_scores = np.empty((num_pos, input_length, 4))

    # Create the explainer
    if model_type == "binary":
        explainer = compute_shap.create_binary_explainer(
            model, input_length, num_tasks, task_index=task_index
        )
    else:
        explainer = compute_shap.create_profile_explainer(
            model, input_length, profile_length, num_tasks, num_strands,
            controls, task_index=task_index
        )

    # Compute the importance scores
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)  
        # Compute scores
        if model_type == "binary":
            input_seqs, _, coords = input_func(pos_samples[batch_slice])
            scores = explainer(input_seqs, hide_shap_output=True)
        else:
            coords = pos_samples[batch_slice]
            input_seqs, profiles = input_func(coords)
            scores = explainer(
                input_seqs, profiles[:, num_tasks:], hide_shap_output=True
            )  # Regardless of the type of controls, we can always put this in

        # Fill in data
        coords_chrom[batch_slice] = coords[:, 0]
        coords_start[batch_slice] = coords[:, 1]
        coords_end[batch_slice] = coords[:, 2]
        one_hot_seqs[batch_slice] = input_seqs
        hyp_scores[batch_slice] = scores

    # Write to HDF5
    print("Saving result to HDF5...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("coords_chrom", data=coords_chrom.astype("S"))
        f.create_dataset("coords_start", data=coords_start)
        f.create_dataset("coords_end", data=coords_end)
        f.create_dataset("hyp_scores", data=hyp_scores)
        f.create_dataset("one_hot_seqs", data=one_hot_seqs)
        model = f.create_dataset("model", data=0)
        model.attrs["model"] = model_path


@click.command()
@click.argument("model_path")
@click.argument(
    "model_type", type=click.Choice(["profile", "binary"])
)
@click.argument("files_spec_path")
@click.option(
    "--num-tasks", "-n", required=True, type=int,
    help="Number of tasks in model"
)
@click.option(
    "--task-index", "-i", default=None, type=int,
    help="Index of task to explain; defaults to all in aggregate"
)
@click.option(
    "--out-path", "-o", required=True, help="Path to output HDF5"
)
@click.option(
    "--chrom-set", "-c", default=None,
    help="Comma-separated list of chromosomes to compute scores for; defaults to all chromosomes"
)
@click.option(
    "--input-length", "-il", default=None,
    help="Length of input sequence; defaults to 1000 for binary and 1346 for profile models"
)
@click.option(
    "--reference-fasta", "-f", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference FASTA"
)
@click.option(
    "--chrom-sizes", "-s",
    default="/users/amtseng/genomes/hg38.canon.chrom.sizes",
    help="Path to chromosome sizes"
)
@click.option(
    "--profile-length", "-pl", default=1000,
    help="For profile models, the length of output profiles"
)
@click.option(
    "--controls", "-co", default=None,
    help="Type of controls for profile models; can be None (default), 'matched', or 'shared'"
)
@click.option(
    "--num-strands", "-d", default=2, help="Number of strands in profile model"
)
@click.option(
    "--batch-size", "-b", default=128, help="Batch size for computation"
)
def main(
    model_path, model_type, files_spec_path, num_tasks, task_index, out_path,
    chrom_set, input_length, reference_fasta, chrom_sizes, profile_length,
    controls, num_strands, batch_size
):
    if not input_length:
        if model_type == "binary":
            input_length = 1000
        else:
            input_length = 1346

    if chrom_set:
        chrom_set = chrom_set.split(",")
    
    make_shap_scores(
        model_path, model_type, files_spec_path, input_length, num_tasks,
        out_path, reference_fasta, chrom_sizes, task_index, profile_length,
        controls, num_strands, chrom_set, batch_size
    )

if __name__ == "__main__":
    main()
