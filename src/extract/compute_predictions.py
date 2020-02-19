import model.util as model_util
import model.profile_models as profile_models
import model.binary_models as binary_models
import extract.data_loading as data_loading
import numpy as np
import torch
import tqdm

def _get_profile_model_predictions_batch(
    model, coords, num_tasks, input_func, controls=None, return_gradients=False
):
    """
    Fetches the necessary data from the given coordinates or bin indices and
    runs it through a profile or binary model. This will perform computation
    in a single batch.
    Arguments:
        `model`: a trained `ProfilePredictorWithMatchedControls`,
            `ProfilePredictorWithSharedControls`, or
            `ProfilePredictorWithoutControls`
        `coords`: a B x 3 array of coordinates to compute outputs for
        `num_tasks`: number of tasks for the model
        `input_func`: a function that takes in `coords` and returns the
            B x I x 4 array of one-hot sequences and the
            B x (T or T + 1 or 2T) x O x S array of profiles (perhaps with
            controls)
        `controls`: the type of control profiles (if any) used in model; can be
            "matched" (each task has a matched control), "shared" (all tasks
            share a control), or None (no controls); must match the model class
        `return_gradients`: if True, compute/return the input gradients and
            sequences
    Returns the following NumPy arrays: true profile raw counts (B x T x O x S),
    predicted profile log probabilities (B x T x O x S), true total counts
    (B x T x S), and predicted log counts (B x T x S). If `return_gradients` is
    True, then also return the input gradients (B x I x 4) and input sequences
    (B x I x 4) after that.
    """
    input_seqs, profiles = input_func(coords)
    if return_gradients:
        input_seqs_np = input_seqs
        model.zero_grad()  # Zero out weights because we are computing gradients
    input_seqs = model_util.place_tensor(torch.tensor(input_seqs)).float()
    profiles = model_util.place_tensor(torch.tensor(profiles)).float()

    if controls is not None:
        tf_profs = profiles[:, :num_tasks, :, :]
        cont_profs = profiles[:, num_tasks:, :, :]  # Last half or just one
    else:
        tf_profs, cont_profs = profiles, None

    if return_gradients:
        input_seqs.requires_grad = True  # Set gradient required
        logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)

        # Subtract mean along output profile dimension; this wouldn't change
        # softmax probabilities, but normalizes the magnitude of gradients
        norm_logit_pred_profs = logit_pred_profs - \
            torch.mean(logit_pred_profs, dim=2, keepdims=True)

        # Weight by post-softmax probabilities, but do not take the
        # gradients of these probabilities; this upweights important regions
        # exponentially
        pred_prof_probs = profile_models.profile_logits_to_log_probs(
            logit_pred_profs
        ).detach()
        weighted_norm_logits = norm_logit_pred_profs * pred_prof_probs

        input_grads, = torch.autograd.grad(
            weighted_norm_logits, input_seqs,
            grad_outputs=model_util.place_tensor(
                torch.ones(weighted_norm_logits.size())
            ),
            retain_graph=True, create_graph=True
            # We'll be operating on the gradient itself, so we need to
            # create the graph
            # Gradients are summed across strands and tasks
        )
        input_grads_np = input_grads.detach().cpu().numpy()
        input_seqs.requires_grad = False  # Reset gradient required
    else:
        logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)
        status, input_grads = None, None

    true_profs = tf_profs.detach().cpu().numpy()
    true_counts = np.sum(true_profs, axis=2)
    logit_pred_profs_np = logit_pred_profs.detach().cpu().numpy()
    pred_prof_probs_np = profile_models.profile_logits_to_log_probs(
        logit_pred_profs_np
    )
    log_pred_counts_np = log_pred_counts.detach().cpu().numpy()

    if return_gradients:
        return true_profs, pred_prof_probs_np, true_counts, \
            log_pred_counts_np, input_grads_np, input_seqs_np
    else:
        return true_profs, logit_pred_profs_np, true_counts, log_pred_counts_np


def _get_binary_model_predictions_batch(
    model, bins, input_func, return_gradients=False
):
    """
    Arguments:
        `model`: a trained `BinaryPredictor`,
        `bins`: an N-array of bin indices to compute outputs for
        `input_func`: a function that takes in `bins` and returns the B x I x 4
            array of one-hot sequences, the B x T array of output values, and
            B x 3 array of underlying coordinates for the input sequence
        `return_gradients`: if True, compute/return the input gradients and
            sequences
    Returns the following NumPy arrays: true output values (B x T), predicted
    probabilities (B x T), and underlying sequence coordinates (B x 3 object
    array). If `return_gradients` is True, then also return the input gradients
    (B x I x 4) and input sequences (B x I x 4) after that.
    """
    input_seqs, output_vals, coords = input_func(bins)
    output_vals_np = output_vals
    if return_gradients:
        input_seqs_np = input_seqs
        model.zero_grad()
    input_seqs = model_util.place_tensor(torch.tensor(input_seqs)).float()
    output_vals = model_util.place_tensor(torch.tensor(output_vals)).float()

    if return_gradients:
        input_seqs.requires_grad = True  # Set gradient required
        logit_pred_vals = model(input_seqs)
        # Compute the gradients of the output with respect to the input
        input_grads, = torch.autograd.grad(
            logit_pred_vals, input_seqs,
            grad_outputs=model_util.place_tensor(
                torch.ones(logit_pred_vals.size())
            ),
            retain_graph=True, create_graph=True
            # We'll be operating on the gradient itself, so we need to
            # create the graph
            # Gradients are summed across tasks
        )
        input_grads_np = input_grads.detach().cpu().numpy()
        input_seqs.requires_grad = False  # Reset gradient required
    else:
        logit_pred_vals = model(input_seqs)
        status, input_grads = None, None

    logit_pred_vals_np = logit_pred_vals.detach().cpu().numpy()
    pred_vals = binary_models.binary_logits_to_probs(logit_pred_vals_np)

    if return_gradients:
        return output_vals_np, pred_vals, coords, input_grads_np, input_seqs_np
    else:
        return output_vals_np, pred_vals, coords


def get_profile_model_predictions(
    model, coords, num_tasks, input_func, controls=None, return_gradients=False,
    batch_size=128, show_progress=False
):
    """
    Fetches the necessary data from the given coordinates and runs it through a
    profile model.
    Arguments:
        `model`: a trained `ProfilePredictorWithMatchedControls`,
            `ProfilePredictorWithSharedControls`, or
            `ProfilePredictorWithoutControls`
        `coords`: a N x 3 array of coordinates to compute outputs for
        `num_tasks`: number of tasks for the model
        `input_func`: a function that takes in `coords` and returns the
            N x I x 4 array of one-hot sequences and the
            N x (T or T + 1 or 2T) x O x S array of profiles (perhaps with
            controls)
        `controls`: the type of control profiles (if any) used in model; can be
            "matched" (each task has a matched control), "shared" (all tasks
            share a control), or None (no controls); must match the model class
        `return_gradients`: if True, compute/return the input gradients and
            sequences
        `batch_size`: batch size to use for prediction
        `show_progress`: whether or not to show progress bar over batches
    Returns the following NumPy arrays: true profile raw counts (N x T x O x S),
    predicted profile log probabilities (N x T x O x S), true total counts
    (N x T x S), and predicted log counts (N x T x S). If `return_gradients` is
    True, then also return the input gradients (N x I x 4) and input sequences
    (N x I x 4) after that.
    """
    num_examples = len(coords)
    num_batches = int(np.ceil(num_examples / batch_size))
    t_iter = tqdm.trange(num_batches) if show_progress else range(num_batches)
    first_batch = True
    for i in t_iter:
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        coords_batch = coords[batch_slice]
        if return_gradients:
            true_profs, log_pred_profs, true_counts, log_pred_counts, \
                input_grads, input_seqs = _get_profile_model_predictions_batch(
                    model, coords_batch, num_tasks, input_func,
                    controls=controls, return_gradients=True
                )
        else:
            true_profs, log_pred_profs, true_counts, log_pred_counts = \
                _get_profile_model_predictions_batch(
                    model, coords_batch, num_tasks, input_func,
                    controls=controls, return_gradients=True
                )
        if first_batch:
            # Allocate arrays of the same size, but holding all examples
            all_true_profs = np.empty((num_examples,) + true_profs.shape[1:])
            all_log_pred_profs = np.empty(
                (num_examples,) + log_pred_profs.shape[1:]
            )
            all_true_counts = np.empty((num_examples,) + true_counts.shape[1:])
            all_log_pred_counts = np.empty(
                (num_examples,) + log_pred_counts.shape[1:]
            )
            all_coords = np.empty((num_examples, 3), dtype=object)
            if return_gradients:
                all_input_grads = np.empty(
                    (num_examples,) + input_grads.shape[1:]
                )
                all_input_seqs = np.empty(
                    (num_examples,) + input_seqs.shape[1:]
                )
            first_batch = False
        all_true_profs[batch_slice] = true_profs
        all_log_pred_profs[batch_slice] = log_pred_profs
        all_true_counts[batch_slice] = true_counts
        all_log_pred_counts[batch_slice] = log_pred_counts
        if return_gradients:
            all_input_grads[batch_slice] = input_grads
            all_input_seqs[batch_slice] = input_seqs
    if return_gradients:
        return all_true_profs, all_log_pred_profs, all_true_counts, \
            all_log_pred_counts, all_input_grads, all_input_seqs
    else:
        return all_true_profs, all_log_pred_profs, all_true_counts, \
            all_log_pred_counts


def get_binary_model_predictions(
    model, bins, input_func, return_gradients=False, batch_size=128,
    show_progress=False
):
    """
    Fetches the necessary data from the given bin indices and runs it through a
    binary model.
    Arguments:
        `model`: a trained `BinaryPredictor`,
        `bins`: an N-array of bin indices to compute outputs for
        `input_func`: a function that takes in `bins` and returns the B x I x 4
            array of one-hot sequences, the B x T array of output values, and
            B x 3 array of underlying coordinates for the input sequence
         `return_gradients`: if True, compute/return the input gradients and
            sequences
        `batch_size`: batch size to use for prediction
        `show_progress`: whether or not to show progress bar over batches
    Returns the following NumPy arrays: true output values (N x T), predicted
    probabilities (N x T), and underlying sequence coordinates (N x 3 object
    array). If `return_gradients` is True, then also return the input gradients
    (N x I x 4) and input sequences (N x I x 4) after that.
    """
    num_examples = len(bins)
    num_batches = int(np.ceil(num_examples / batch_size))
    t_iter = tqdm.trange(num_batches) if show_progress else range(num_batches)
    first_batch = True
    for i in t_iter:
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        bins_batch = bins[batch_slice]
        if return_gradients:
            true_vals, pred_vals, coords, input_grads, input_seqs = \
                _get_binary_model_predictions_batch(
                    model, bins_batch, input_func, return_gradients=True
                )
        else:
            true_vals, pred_vals, coords = _get_binary_model_predictions_batch(
                model, bins_batch, input_func, return_gradients=False
            )
        if first_batch:
            # Allocate arrays of the same size, but holding all examples
            all_true_vals = np.empty((num_examples,) + true_vals.shape[1:])
            all_pred_vals = np.empty((num_examples,) + pred_vals.shape[1:])
            all_coords = np.empty((num_examples, 3), dtype=object)
            if return_gradients:
                all_input_grads = np.empty(
                    (num_examples,) + input_grads.shape[1:]
                )
                all_input_seqs = np.empty(
                    (num_examples,) + input_seqs.shape[1:]
                )
            first_batch = False
        all_true_vals[batch_slice] = true_vals
        all_pred_vals[batch_slice] = pred_vals
        all_coords[batch_slice] = coords
        if return_gradients:
            all_input_grads[batch_slice] = input_grads
            all_input_seqs[batch_slice] = input_seqs
    if return_gradients:
        return all_true_vals, all_pred_vals, all_coords, all_input_grads, \
            all_input_seqs
    else:
        return all_true_vals, all_pred_vals, all_coords


if __name__ == "__main__":
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    chrom_set = ["chr21"]

    print("Testing profile model")
    input_length = 1346
    profile_length = 1000
    controls = "matched"
    num_tasks = 4
    
    files_spec_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/profile/config/SPI1/SPI1_training_paths.json"
    model_class = profile_models.ProfilePredictorWithMatchedControls
    model_path = "/users/amtseng/att_priors/models/trained_models/profile/SPI1/1/model_ckpt_epoch_1.pt"

    input_func = data_loading.get_profile_input_func(
        files_spec_path, input_length, profile_length, reference_fasta,
    )
    pos_coords = data_loading.get_positive_profile_coords(
        files_spec_path, chrom_set=chrom_set
    )

    print("Loading model...")
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_util.restore_model(model_class, model_path)
    model.eval()
    model = model.to(device)

    print("Running predictions...")
    x = get_profile_model_predictions(
        model, pos_coords, num_tasks, input_func, controls=controls,
        return_gradients=True, show_progress=True
    )
    
    print("")

    print("Testing binary model")
    input_length = 1000
    
    files_spec_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/binary/config/SPI1/SPI1_training_paths.json"
    model_class = binary_models.BinaryPredictor
    model_path = "/users/amtseng/att_priors/models/trained_models/binary/SPI1/1/model_ckpt_epoch_1.pt"

    input_func = data_loading.get_binary_input_func(
       files_spec_path, input_length, reference_fasta
    )
    pos_bins = data_loading.get_positive_binary_bins(
        files_spec_path, chrom_set=chrom_set
    )

    print("Loading model...")
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_util.restore_model(model_class, model_path)
    model.eval()
    model = model.to(device)

    print("Running predictions...")
    x = get_binary_model_predictions(
        model, pos_bins, input_func, return_gradients=True, show_progress=True
    )
