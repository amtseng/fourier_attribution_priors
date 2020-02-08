import model.util as model_util
import extract.data_loading as data_loading
import numpy as np
import torch
import tqdm

def get_input_grads_batch(
    model, model_type, coords_or_bin_inds, input_func, use_controls=True
):
    """
    Fetches the necessary data from the given coordinates or bin indices and
    runs it through a profile or binary model.
    Arguments:
        `model`: a trained `BinaryPredictor`, `ProfilePredictorWithControls`,
            or `ProfilePredictorWithoutControls`
        `model_type`: either "binary" or "profile"
        `coords_or_bin_inds`: either an N-array of bin indices or an N x 3 array
            of coordinates to compute gradients for
        `input_func`: a function that takes in `coords_or_bin_inds` and returns
            data needed for the model; for a binary model, this function must
            return the N x I x 4 one-hot encoded sequences, N x T output values,
            and N x 3 array of coordinates; for a profile model, this function
            must return the one-hot sequences and the N x (T or 2T) x O x 2
            profiles (perhaps with controls)
        `use_controls`: for a profile model, whether or not control tracks are
            used
    Returns an N x I x 4 array of input gradients and an N x I x 4 array of
    input sequences.
    """
    if model_type == "binary":
        input_seqs_np, _, _ = input_func(coords_or_bin_inds)
    else:
        input_seqs_np, profiles = input_func(coords_or_bin_inds)
        profiles = model_util.place_tensor(torch.tensor(profiles)).float()
        if use_controls:
            num_tasks = profiles.shape[1] // 2
            true_profs = profiles[:, :num_tasks, :, :]
            cont_profs = profiles[:, num_tasks:, :, :]
        else:
            true_profs, cont_profs = profiles, None
    input_seqs = model_util.place_tensor(torch.tensor(input_seqs_np)).float()

    model.zero_grad()
    
    # Run through the model
    input_seqs.requires_grad = True  # Set gradient required
    if model_type == "binary":
        output = model(input_seqs)
    else:
        output, _ = model(input_seqs, cont_profs)
    
    # Compute input gradients
    input_grads, = torch.autograd.grad(
        output, input_seqs,
        grad_outputs=model_util.place_tensor(torch.ones(output.size())),
        retain_graph=True, create_graph=True
    )
    input_grads_np = input_grads.detach().cpu().numpy()
    
    return input_grads_np, input_seqs_np


def get_input_grads(
    model, model_type, files_spec_path, input_length, reference_fasta,
    chrom_set=None, profile_length=None, use_controls=True, batch_size=128
):
    """
    Starting from an imported model, computes input gradients for all specified
    positive examples.
    Arguments:
        `model`: a trained `BinaryPredictor`, `ProfilePredictorWithControls`,
            or `ProfilePredictorWithoutControls`
        `model_type`: either "binary" or "profile"
        `input_length`: length of input sequence
        `reference_fasta`: path to reference fasta
        `chrom_set`: if given, limit the set of coordinates or bin indices to
            these chromosomes only
        `profile_length`: if profile model, length of output profiles
        `use_controls`: for a profile model, whether or not control tracks are
            used
        `batch_size`: batch size for computing the gradients
    For all N positive examples used, returns an N x I x 4 array of the input
    gradients, and an N x I x 4 array of input sequences.
    """
    input_func = data_loading.get_input_func(
        model_type, files_spec_path, input_length, reference_fasta,
        profile_length=profile_length
    )
    coords_or_bin_inds = data_loading.get_positive_inputs(
        model_type, files_spec_path, chrom_set=chrom_set
    )

    num_examples = len(coords_or_bin_inds)
    all_input_grads = np.empty((num_examples, input_length, 4))
    all_input_seqs = np.empty((num_examples, input_length, 4))
    num_batches = int(np.ceil(num_examples / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = coords_or_bin_inds[batch_slice]
        input_grads, input_seqs = get_input_grads_batch(
            model, model_type, batch, input_func, use_controls=use_controls
        )
        all_input_grads[batch_slice] = input_grads
        all_input_seqs[batch_slice] = input_seqs
    return all_input_grads, all_input_seqs


if __name__ == "__main__":
    import model.profile_models as profile_models

    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    input_length = 1346
    profile_length = 1000
    use_controls = True

    files_spec_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/profile/config/SPI1/SPI1_training_paths.json"
    model_class = profile_models.ProfilePredictorWithControls
    model_path = "/users/amtseng/att_priors/models/trained_models/profile_models/SPI1/1/model_ckpt_epoch_10.pt"
    chrom_set = ["chr1"]

    # Import model
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_util.restore_model(model_class, model_path)
    model.eval()
    model = model.to(device)

    input_grads, input_seqs = get_input_grads(
        model, "profile", files_spec_path, input_length, reference_fasta,
        chrom_set=chrom_set, profile_length=profile_length,
        use_controls=use_controls
    )
