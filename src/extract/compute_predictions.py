import model.util as model_util
import model.profile_models as profile_models
import model.binary_models as binary_models
import extract.data_loading as data_loading
import numpy as np
import torch
import tqdm

def _get_profile_model_predictions_batch(
    model, coords, num_tasks, input_func, controls=None,
    fourier_att_prior_freq_limit=200, fourier_att_prior_freq_limit_softness=0.2,
    att_prior_grad_smooth_sigma=3, return_losses=False, return_gradients=False
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
        `fourier_att_prior_freq_limit`: limit for frequencies in Fourier prior
            loss
        `fourier_att_prior_freq_limit_softness`: degree of softness for limit
        `att_prior_grad_smooth_sigma`: width of smoothing kernel for gradients
        `return_losses`: if True, compute/return the loss values
        `return_gradients`: if True, compute/return the input gradients and
            sequences
    Returns a dictionary of the following structure:
        true_profs: true profile raw counts (B x T x O x S)
        log_pred_profs: predicted profile log probabilities (B x T x O x S)
        true_counts: true total counts (B x T x S)
        log_pred_counts: predicted log counts (B x T x S)
        prof_losses: profile NLL losses (B-array), if `return_losses` is True
        count_losses: counts MSE losses (B-array) if `return_losses` is True
        att_losses: prior losses (B-array), if `return_losses` is True
        input_seqs: one-hot input sequences (B x I x 4), if `return_gradients`
            is true
        input_grads: "hypothetical" input gradients (B x I x 4), if
            `return_gradients` is true
    """
    result = {}
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

    if return_losses or return_gradients:
        input_seqs.requires_grad = True  # Set gradient required
        logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)

        # Subtract mean along output profile dimension; this wouldn't change
        # softmax probabilities, but normalizes the magnitude of gradients
        norm_logit_pred_profs = logit_pred_profs - \
            torch.mean(logit_pred_profs, dim=2, keepdim=True)

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
    
    result["true_profs"] = tf_profs.detach().cpu().numpy()
    result["true_counts"] = np.sum(result["true_profs"], axis=2)
    logit_pred_profs_np = logit_pred_profs.detach().cpu().numpy()
    result["log_pred_profs"] = profile_models.profile_logits_to_log_probs(
        logit_pred_profs_np
    )
    result["log_pred_counts"] = log_pred_counts.detach().cpu().numpy()

    if return_losses:
        log_pred_profs = profile_models.profile_logits_to_log_probs(
            logit_pred_profs
        )
        num_samples = log_pred_profs.size(0)
        result["prof_losses"] = np.empty(num_samples)
        result["count_losses"] = np.empty(num_samples)
        result["att_losses"] = np.empty(num_samples)

        # Compute losses separately for each example
        for i in range(num_samples):
            _, prof_loss, count_loss = model.correctness_loss(
                tf_profs[i:i+1], log_pred_profs[i:i+1], log_pred_counts[i:i+1],
                1, return_separate_losses=True
            )
            att_loss = model.fourier_att_prior_loss(
                model_util.place_tensor(torch.ones(1)),
                input_grads[i:i+1], fourier_att_prior_freq_limit,
                fourier_att_prior_freq_limit_softness,
                att_prior_grad_smooth_sigma
            )
            result["prof_losses"][i] = prof_loss
            result["count_losses"][i] = count_loss
            result["att_losses"][i] = att_loss

    if return_gradients:
        result["input_seqs"] = input_seqs_np
        result["input_grads"] = input_grads_np

    return result


def _get_binary_model_predictions_batch(
    model, bins, input_func, fourier_att_prior_freq_limit=150,
    fourier_att_prior_freq_limit_softness=0.2, att_prior_grad_smooth_sigma=3,
    return_losses=False, return_gradients=False
):
    """
    Arguments:
        `model`: a trained `BinaryPredictor`,
        `bins`: an N-array of bin indices to compute outputs for
        `input_func`: a function that takes in `bins` and returns the B x I x 4
            array of one-hot sequences, the B x T array of output values, and
            B x 3 array of underlying coordinates for the input sequence
        `fourier_att_prior_freq_limit`: limit for frequencies in Fourier prior
            loss
        `fourier_att_prior_freq_limit_softness`: degree of softness for limit
        `att_prior_grad_smooth_sigma`: width of smoothing kernel for gradients
        `return_losses`: if True, compute/return the loss values
        `return_gradients`: if True, compute/return the input gradients and
            sequences
    Returns a dictionary of the following structure:
        true_vals: true binary values (B x T)
        pred_vals: predicted probabilities (B x T)
        coords: coordinates used for prediction (B x 3 object array)
        corr_losses: correctness losses (B-array) if `return_losses` is True
        att_losses: prior losses (B-array), if `return_losses` is True
        input_seqs: one-hot input sequences (B x I x 4), if `return_gradients`
            is True
        input_grads: "hypothetical" input gradients (B x I x 4), if
            `return_gradients` is true
    """
    result = {}
    input_seqs, output_vals, coords = input_func(bins)
    output_vals_np = output_vals
    if return_gradients:
        input_seqs_np = input_seqs
        model.zero_grad()
    input_seqs = model_util.place_tensor(torch.tensor(input_seqs)).float()
    output_vals = model_util.place_tensor(torch.tensor(output_vals)).float()

    if return_losses or return_gradients:
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

    result["true_vals"] = output_vals_np
    logit_pred_vals_np = logit_pred_vals.detach().cpu().numpy()
    result["pred_vals"] = binary_models.binary_logits_to_probs(
        logit_pred_vals_np
    )
    result["coords"] = coords

    if return_losses:
        num_samples = logit_pred_vals.size(0)
        result["corr_losses"] = np.empty(num_samples)
        result["att_losses"] = np.empty(num_samples)

        # Compute losses separately for each example
        for i in range(num_samples):
            corr_loss = model.correctness_loss(
                output_vals[i:i+1], logit_pred_vals[i:i+1], True
            )
            att_loss = model.fourier_att_prior_loss(
                model_util.place_tensor(torch.ones(1)),
                input_grads[i:i+1], fourier_att_prior_freq_limit,
                fourier_att_prior_freq_limit_softness,
                att_prior_grad_smooth_sigma
            )
            result["corr_losses"][i] = corr_loss
            result["att_losses"][i] = att_loss

    if return_gradients:
        result["input_seqs"] = input_seqs_np
        result["input_grads"] = input_grads_np

    return result


def get_profile_model_predictions(
    model, coords, num_tasks, input_func, controls=None, 
    fourier_att_prior_freq_limit=200, fourier_att_prior_freq_limit_softness=0.2,
    att_prior_grad_smooth_sigma=3, return_losses=False, return_gradients=False,
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
        `fourier_att_prior_freq_limit`: limit for frequencies in Fourier prior
            loss
        `fourier_att_prior_freq_limit_softness`: degree of softness for limit
        `att_prior_grad_smooth_sigma`: width of smoothing kernel for gradients
        `return_losses`: if True, compute/return the loss values
        `return_gradients`: if True, compute/return the input gradients and
            sequences
        `batch_size`: batch size to use for prediction
        `show_progress`: whether or not to show progress bar over batches
    Returns a dictionary of the following structure:
        true_profs: true profile raw counts (N x T x O x S)
        log_pred_profs: predicted profile log probabilities (N x T x O x S)
        true_counts: true total counts (N x T x S)
        log_pred_counts: predicted log counts (N x T x S)
        prof_losses: profile NLL losses (N-array), if `return_losses` is True
        count_losses: counts MSE losses (N-array) if `return_losses` is True
        att_loss: prior losses (N-array), if `return_losses` is True
        input_seqs: one-hot input sequences (N x I x 4), if `return_gradients`
            is true
        input_grads: "hypothetical" input gradients (N x I x 4), if
            `return_gradients` is true
    """
    result = {}
    num_examples = len(coords)
    num_batches = int(np.ceil(num_examples / batch_size))
    t_iter = tqdm.trange(num_batches) if show_progress else range(num_batches)
    first_batch = True
    for i in t_iter:
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        coords_batch = coords[batch_slice]
        batch_result = _get_profile_model_predictions_batch(
            model, coords_batch, num_tasks, input_func, controls=controls,
            fourier_att_prior_freq_limit=fourier_att_prior_freq_limit,
            fourier_att_prior_freq_limit_softness=fourier_att_prior_freq_limit_softness,
            att_prior_grad_smooth_sigma=att_prior_grad_smooth_sigma,
            return_losses=return_losses, return_gradients=return_gradients
        )
            
        if first_batch:
            # Allocate arrays of the same size, but holding all examples
            result["true_profs"] = np.empty(
                (num_examples,) + batch_result["true_profs"].shape[1:]
            )
            result["log_pred_profs"] = np.empty(
                (num_examples,) + batch_result["log_pred_profs"].shape[1:]
            )
            result["true_counts"] = np.empty(
                (num_examples,) + batch_result["true_counts"].shape[1:]
            )
            result["log_pred_counts"] = np.empty(
                (num_examples,) + batch_result["log_pred_counts"].shape[1:]
            )
            if return_losses:
                result["prof_losses"] = np.empty(num_examples)
                result["count_losses"] = np.empty(num_examples)
                result["att_losses"] = np.empty(num_examples)
            if return_gradients:
                result["input_seqs"] = np.empty(
                    (num_examples,) + batch_result["input_seqs"].shape[1:]
                )
                result["input_grads"] = np.empty(
                    (num_examples,) + batch_result["input_grads"].shape[1:]
                )
            first_batch = False
        result["true_profs"][batch_slice] = batch_result["true_profs"]
        result["log_pred_profs"][batch_slice] = batch_result["log_pred_profs"]
        result["true_counts"][batch_slice] = batch_result["true_counts"]
        result["log_pred_counts"][batch_slice] = batch_result["log_pred_counts"]
        if return_losses:
            result["prof_losses"][batch_slice] = batch_result["prof_losses"]
            result["count_losses"][batch_slice] = batch_result["count_losses"]
            result["att_losses"][batch_slice] = batch_result["att_losses"]
        if return_gradients:
            result["input_seqs"][batch_slice] = batch_result["input_seqs"]
            result["input_grads"][batch_slice] = batch_result["input_grads"]

    return result


def get_binary_model_predictions(
    model, bins, input_func, fourier_att_prior_freq_limit=150,
    fourier_att_prior_freq_limit_softness=0.2, att_prior_grad_smooth_sigma=3,
    return_losses=False, return_gradients=False, batch_size=128,
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
        `fourier_att_prior_freq_limit`: limit for frequencies in Fourier prior
            loss
        `fourier_att_prior_freq_limit_softness`: degree of softness for limit
        `att_prior_grad_smooth_sigma`: width of smoothing kernel for gradients
        `return_losses`: if True, compute/return the loss values
        `return_gradients`: if True, compute/return the input gradients and
            sequences
        `batch_size`: batch size to use for prediction
        `show_progress`: whether or not to show progress bar over batches
    Returns a dictionary of the following structure:
        true_vals: true binary values (N x T)
        pred_vals: predicted probabilities (N x T)
        coords: coordinates used for prediction (N x 3 object array)
        corr_losses: correctness losses (N-array) if `return_losses` is True
        att_losses: prior losses (N-array), if `return_losses` is True
        input_seqs: one-hot input sequences (N x I x 4), if `return_gradients`
            is true
        input_grads: "hypothetical" input gradients (N x I x 4), if
            `return_gradients` is true
    """
    result = {}
    num_examples = len(bins)
    num_batches = int(np.ceil(num_examples / batch_size))
    t_iter = tqdm.trange(num_batches) if show_progress else range(num_batches)
    first_batch = True
    for i in t_iter:
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        bins_batch = bins[batch_slice]
        batch_result = _get_binary_model_predictions_batch(
            model, bins_batch, input_func, 
            fourier_att_prior_freq_limit=fourier_att_prior_freq_limit,
            fourier_att_prior_freq_limit_softness=fourier_att_prior_freq_limit_softness,
            att_prior_grad_smooth_sigma=att_prior_grad_smooth_sigma,
            return_losses=return_losses, return_gradients=return_gradients
        )
        if first_batch:
            # Allocate arrays of the same size, but holding all examples
            result["true_vals"] = np.empty(
                (num_examples,) + batch_result["true_vals"].shape[1:]
            )
            result["pred_vals"] = np.empty(
                (num_examples,) + batch_result["pred_vals"].shape[1:]
            )
            result["coords"] = np.empty((num_examples, 3), dtype=object)
            if return_losses:
                result["corr_losses"] = np.empty(num_examples)
                result["att_losses"] = np.empty(num_examples)
            if return_gradients:
                result["input_seqs"] = np.empty(
                    (num_examples,) + batch_result["input_seqs"].shape[1:]
                )
                result["input_grads"] = np.empty(
                    (num_examples,) + batch_result["input_grads"].shape[1:]
                )
            first_batch = False
        result["true_vals"][batch_slice] = batch_result["true_vals"]
        result["pred_vals"][batch_slice] = batch_result["pred_vals"]
        result["coords"][batch_slice] = batch_result["coords"]
        if return_losses:
            result["corr_losses"][batch_slice] = batch_result["corr_losses"]
            result["att_losses"][batch_slice] = batch_result["att_losses"]
        if return_gradients:
            result["input_seqs"][batch_slice] = batch_result["input_seqs"]
            result["input_grads"][batch_slice] = batch_result["input_grads"]

    return result


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
        return_losses=True, return_gradients=True, show_progress=True
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
        model, pos_bins, input_func, return_losses=True, return_gradients=True,
        show_progress=True
    )
