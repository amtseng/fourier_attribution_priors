import numpy as np
import sacred
import torch
import math
import tqdm
import os
import model.util as util
import model.profile_models as profile_models
import model.profile_performance as profile_performance
import feature.make_profile_dataset as make_profile_dataset

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/users/amtseng/att_priors/models/trained_models/profile_models/misc/"
)

train_ex = sacred.Experiment("train", ingredients=[
    make_profile_dataset.dataset_ex,
    profile_performance.performance_ex
])
train_ex.observers.append(
    sacred.observers.FileStorageObserver.create(MODEL_DIR)
)

@train_ex.config
def config(dataset):
    # Number of dilating convolutional layers to apply
    num_dil_conv_layers = 7
    
    # Size of dilating convolutional filters to apply
    dil_conv_filter_sizes = [21] + ([3] * (num_dil_conv_layers - 1))

    # Stride for dilating convolutional layers
    dil_conv_stride = 1

    # Number of filters to use for each dilating convolutional layer (i.e.
    # number of channels to output)
    dil_conv_depths = [64] * num_dil_conv_layers

    # Dilation values for each of the dilating convolutional layers
    dil_conv_dilations = [2 ** i for i in range(num_dil_conv_layers)]

    # Size of filter for large profile convolution
    prof_conv_kernel_size = 75

    # Stride for large profile convolution
    prof_conv_stride = 1

    # Number of prediction tasks (2 outputs for each task: plus/minus strand)
    num_tasks = 4

    # Amount to weight the counts loss within the correctness loss
    counts_loss_weight = 100

    # Weight to use for attribution prior loss; set to 0 to not use att. priors
    att_prior_loss_weight = 1.0

    # Annealing factor for attribution prior loss weight: e^(-factor * epoch);
    # set to 0 for no annealing
    att_prior_loss_weight_anneal = 0.5

    # Smoothing amount for gradients before computing attribution prior loss;
    # Smoothing window size is 1 + (2 * sigma); set to 0 for no smoothing
    att_prior_grad_smooth_sigma = 3

    # Maximum frequency integer to consider for a positive attribution prior
    att_prior_pos_limit = 160

    # Weight for positives within the attribution prior loss
    att_prior_pos_weight = 1.0

    # Number of training epochs
    num_epochs = 10

    # Learning rate
    learning_rate = 0.004

    # Whether or not to use early stopping
    early_stopping = True

    # Number of epochs to save validation loss (set to 1 for one step only)
    early_stop_hist_len = 3

    # Minimum improvement in loss at least once over history to not stop early
    early_stop_min_delta = 0.001

    # Training seed
    train_seed = None

    # If set, ignore correctness loss completely
    att_prior_loss_only = False

    # Imported from make_profile_dataset
    batch_size = dataset["batch_size"]

    # Imported from make_profile_dataset
    revcomp = dataset["revcomp"]

    # Imported from make_profile_dataset
    input_length = dataset["input_length"]
    
    # Imported from make_profile_dataset
    input_depth = dataset["input_depth"]

    # Imported from make_profile_dataset
    profile_length = dataset["profile_length"]
    
    # Imported from make_profile_dataset
    negative_ratio = dataset["negative_ratio"]


@train_ex.capture
def create_model(
    input_length, input_depth, profile_length, num_tasks, num_dil_conv_layers,
    dil_conv_filter_sizes, dil_conv_stride, dil_conv_dilations, dil_conv_depths,
    prof_conv_kernel_size, prof_conv_stride
):
    """
    Creates a profile model using the configuration above.
    """
    prof_model = profile_models.ProfileTFBindingPredictor(
        input_length=input_length,
        input_depth=input_depth,
        profile_length=profile_length,
        num_tasks=num_tasks,
        num_dil_conv_layers=num_dil_conv_layers,
        dil_conv_filter_sizes=dil_conv_filter_sizes,
        dil_conv_stride=dil_conv_stride,
        dil_conv_dilations=dil_conv_dilations,
        dil_conv_depths=dil_conv_depths,
        prof_conv_kernel_size=prof_conv_kernel_size,
        prof_conv_stride=prof_conv_stride
    )

    return prof_model


@train_ex.capture
def model_loss(
    model, true_profs, log_pred_profs, log_pred_counts, status, input_grads,
    epoch_num, counts_loss_weight, att_prior_loss_weight,
    att_prior_loss_weight_anneal, att_prior_pos_limit, att_prior_pos_weight,
    att_prior_grad_smooth_sigma, att_prior_loss_only
):
    """
    Computes the loss for the model.
    Arguments:
        `model`: the model being trained
        `true_profs`: a B x T x O x 2 tensor, where B is the batch size, T is
            the number of tasks, and O is the length of the output profiles;
            this contains true profile read counts (unnormalized)
        `log_pred_profs`: a B x T x O x 2 tensor, consisting of the output
            profile predictions as logits
        `log_pred_counts`: a B x T x 2 tensor consisting of the log counts
            predictions
        `status`: a B-tensor, where B is the batch size; each entry is 1 if that
            that example is to be treated as a positive example, and 0 otherwise
        `input_grads`: a B x I x D tensor, where I is the input length and D is
            the input depth; this is the gradient of the output with respect to
            the input, times the input itself
        `epoch_num`: a 0-indexed integer representing the current epoch
    Returns a scalar Tensor containing the loss for the given batch, a pair
    consisting of the correctness loss and the attribution prior loss, a pair
    for the profile loss and the counts loss, and a pair for the positive
    attribution prior loss and negative attribution prior loss.
    If the attribution prior loss is not computed at all, then 0s will be in its
    place, instead.
    """
    corr_loss, prof_loss, count_loss = model.correctness_loss(
        true_profs, log_pred_profs, log_pred_counts, counts_loss_weight,
        return_separate_losses=True
    )
    
    if not att_prior_loss_weight:
        return corr_loss, (corr_loss, torch.zeros(1)), \
            (prof_loss, count_loss), (torch.zeros(1), torch.zeros(1))
    
    att_prior_loss, pos_loss, neg_loss = model.att_prior_loss(
        status, input_grads, att_prior_pos_limit, att_prior_pos_weight,
        att_prior_grad_smooth_sigma, return_separate_losses=True
    )
    weight = att_prior_loss_weight * \
        np.exp(-att_prior_loss_weight_anneal * epoch_num)
    if att_prior_loss_only:
        final_loss = att_prior_loss
    else:
        final_loss = corr_loss + (weight * att_prior_loss)
    return final_loss, (corr_loss, att_prior_loss), (prof_loss, count_loss), \
        (pos_loss, neg_loss)


@train_ex.capture
def run_epoch(
    data_loader, mode, model, epoch_num, num_tasks, att_prior_loss_weight,
    batch_size, revcomp, profile_length, optimizer=None, return_data=False
):
    """
    Runs the data from the data loader once through the model, to train,
    validate, or predict.
    Arguments:
        `data_loader`: an instantiated `DataLoader` instance that gives batches
            of data; each batch must yield the input sequences, profiles, and
            statuses; profiles must be such that the first half are prediction
            (target) profiles, and the second half are control profiles
        `mode`: one of "train", "eval"; if "train", run the epoch and perform
            backpropagation; if "eval", only do evaluation
        `model`: the current PyTorch model being trained/evaluated
        `epoch_num`: 0-indexed integer representing the current epoch
        `optimizer`: an instantiated PyTorch optimizer, for training mode
        `return_data`: if specified, returns the following as NumPy arrays:
            true profile counts, predicted profile log probabilities,
            true total counts, predicted log counts, input sequences, input
            gradients (if the attribution prior loss is not used, these will all
            be garbage), and coordinates used
    Returns lists of overall losses, correctness losses, attribution prior
    losses, profile losses, count losses, positive prior losses, and negative
    prior losses, where each list is over all batches. If the attribution prior
    loss is not computed, then the corresponding lists will be all 0s.
    If `return_data` is True, then more things will be returned after these.
    """
    assert mode in ("train", "eval")
    if mode == "train":
        assert optimizer is not None
    else:
        assert optimizer is None 

    data_loader.dataset.on_epoch_start()  # Set-up the epoch
    num_batches = len(data_loader.dataset)
    t_iter = tqdm.tqdm(
        data_loader, total=num_batches, desc="\tLoss: ---"
    )

    if mode == "train":
        model.train()  # Switch to training mode
        torch.set_grad_enabled(True)

    batch_losses, corr_losses, att_losses = [], [], []
    prof_losses, count_losses = [], []
    pos_losses, neg_losses = [], []
    if return_data:
        # Allocate empty NumPy arrays to hold the results
        num_samples_exp = num_batches * batch_size
        num_samples_exp *= 2 if revcomp else 1
        # Real number of samples can be smaller because of partial last batch
        profile_shape = (num_samples_exp, num_tasks, profile_length, 2)
        count_shape = (num_samples_exp, num_tasks, 2)
        all_log_pred_profs = np.empty(profile_shape)
        all_log_pred_counts = np.empty(count_shape)
        all_true_profs = np.empty(profile_shape)
        all_true_counts = np.empty(count_shape)
        all_input_seqs = np.empty((num_samples_exp, 1346, 4))
        all_input_grads = np.empty((num_samples_exp, 1346, 4))
        all_coords = np.empty((num_samples_exp, 3), dtype=object)
        num_samples_seen = 0  # Real number of samples seen

    for input_seqs, profiles, statuses, coords, peaks in t_iter:
        if return_data:
            input_seqs_np = input_seqs
        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        profiles = util.place_tensor(torch.tensor(profiles)).float()

        tf_profs = profiles[:, :num_tasks, :, :]
        cont_profs = profiles[:, num_tasks:, :, :]

        if mode == "train" and att_prior_loss_weight > 0:
            optimizer.zero_grad()  # Clear gradients from last batch

        if att_prior_loss_weight > 0:
            input_seqs.requires_grad = True  # Set gradient required
            logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)
            # Compute the gradients of the output with respect to the input
            input_grads, = torch.autograd.grad(
                logit_pred_profs, input_seqs,
                grad_outputs=util.place_tensor(
                    torch.ones(logit_pred_profs.size())
                ),
                retain_graph=True, create_graph=True
                # We'll be operating on the gradient itself, so we need to
                # create the graph
                # Gradients are summed across strands and tasks
            )
            if return_data:
                input_grads_np = input_grads.detach().cpu().numpy()
            input_grads = input_grads * input_seqs  # Gradient * input
            status = util.place_tensor(torch.tensor(statuses))
            status[status != 0] = 1  # Set to 1 if not true negative example
            input_seqs.requires_grad = False  # Reset gradient required
        else:
            logit_pred_profs, log_pred_counts = model(input_seqs, cont_profs)
            status, input_grads = None, None

        loss, (corr_loss, att_loss), (prof_loss, count_loss), \
            (pos_loss, neg_loss) = model_loss(
            model, tf_profs, logit_pred_profs, log_pred_counts, status,
            input_grads, epoch_num
        )

        if mode == "train":
            loss.backward()  # Compute gradient
            optimizer.step()  # Update weights through backprop

        batch_losses.append(loss.item())
        corr_losses.append(corr_loss.item())
        att_losses.append(att_loss.item())
        prof_losses.append(prof_loss.item())
        count_losses.append(count_loss.item())
        pos_losses.append(pos_loss.item())
        neg_losses.append(neg_loss.item())
        print("\n%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f" % (prof_losses[-1], count_losses[-1], pos_losses[-1], neg_losses[-1], corr_losses[-1], att_losses[-1], batch_losses[-1]))
        t_iter.set_description(
            "\tLoss: %6.4f" % loss.item()
        )

        if return_data:
            logit_pred_profs_np = logit_pred_profs.detach().cpu().numpy()
            log_pred_counts_np = log_pred_counts.detach().cpu().numpy()
            true_profs_np = tf_profs.detach().cpu().numpy()
            true_counts = np.sum(true_profs_np, axis=2)

            num_in_batch = true_counts.shape[0]
          
            # Turn logit profile predictions into log probabilities
            log_pred_profs = profile_models.profile_logits_to_log_probs(
                logit_pred_profs_np, axis=2
            )

            # Fill in the batch data/outputs into the preallocated arrays
            start, end = num_samples_seen, num_samples_seen + num_in_batch
            all_log_pred_profs[start:end] = log_pred_profs
            all_log_pred_counts[start:end] = log_pred_counts_np
            all_true_profs[start:end] = true_profs_np
            all_true_counts[start:end] = true_counts
            all_input_seqs[start:end] = input_seqs_np
            if att_prior_loss_weight:
                all_input_grads[start:end] = input_grads_np
            all_coords[start:end] = coords

            num_samples_seen += num_in_batch

    if return_data:
        # Truncate the saved data to the proper size, based on how many
        # samples actually seen
        all_log_pred_profs = all_log_pred_profs[:num_samples_seen]
        all_log_pred_counts = all_log_pred_counts[:num_samples_seen]
        all_true_profs = all_true_profs[:num_samples_seen]
        all_true_counts = all_true_counts[:num_samples_seen]
        all_input_seqs = all_input_seqs[:num_samples_seen]
        all_input_grads = all_input_grads[:num_samples_seen]
        all_coords = all_coords[:num_samples_seen]
        return batch_losses, corr_losses, att_losses, prof_losses, \
            count_losses, pos_losses, neg_losses, all_log_pred_profs, \
            all_log_pred_counts, all_true_profs, all_true_counts, \
            all_input_seqs, all_input_grads, all_coords
    else:
        return batch_losses, corr_losses, att_losses, prof_losses, \
            count_losses, pos_losses, neg_losses


@train_ex.capture
def train_model(
    train_loader, val_loader, summit_loader, peak_loader, num_epochs,
    learning_rate, early_stopping, early_stop_hist_len, early_stop_min_delta,
    train_seed, _run
):
    """
    Trains the network for the given training and validation data.
    Arguments:
        `train_loader` (DataLoader): a data loader for the training data, each
            batch giving the 1-hot encoded sequence, profiles, and statuses
        `val_loader` (DataLoader): a data loader for the validation data, each
            batch giving the 1-hot encoded sequence, profiles, and statuses
        `summit_loader` (DataLoader): a data loader for the validation data,
            with coordinates centered at summits, each batch giving the 1-hot
            encoded sequence, profiles, and statuses
        `peak_loader` (DataLoader): a data loader for the validation data,
            with coordinates tiled across peaks, each batch giving the 1-hot
            encoded sequence, profiles, and statuses
    """
    run_num = _run._id
    output_dir = os.path.join(MODEL_DIR, str(run_num))
    
    if train_seed:
        torch.manual_seed(train_seed)

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

    model = create_model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if early_stopping:
        val_epoch_loss_hist = []

    for epoch in range(num_epochs):
        if torch.cuda.is_available:
            torch.cuda.empty_cache()  # Clear GPU memory

        t_batch_losses, t_corr_losses, t_att_losses, t_prof_losses, \
            t_count_losses, t_pos_losses, t_neg_losses = run_epoch(
                train_loader, "train", model, epoch, optimizer=optimizer
        )
        train_epoch_loss = np.nanmean(t_batch_losses)
        print(
            "Train epoch %d: average loss = %6.10f" % (
                epoch + 1, train_epoch_loss
            )
        )
        _run.log_scalar("train_epoch_loss", train_epoch_loss)
        _run.log_scalar("train_batch_losses", t_batch_losses)
        _run.log_scalar("train_corr_losses", t_corr_losses)
        _run.log_scalar("train_att_losses", t_att_losses)
        _run.log_scalar("train_prof_corr_losses", t_prof_losses)
        _run.log_scalar("train_count_corr_losses", t_count_losses)
        _run.log_scalar("train_pos_att_losses", t_pos_losses)
        _run.log_scalar("train_neg_att_losses", t_neg_losses)

        v_batch_losses, v_corr_losses, v_att_losses, v_prof_losses, \
            v_count_losses, v_pos_losses, v_neg_losses = run_epoch(
                val_loader, "eval", model, epoch
        )
        val_epoch_loss = np.nanmean(v_batch_losses)
        print(
            "Valid epoch %d: average loss = %6.10f" % (
                epoch + 1, val_epoch_loss
            )
        )
        _run.log_scalar("val_epoch_loss", val_epoch_loss)
        _run.log_scalar("val_batch_losses", v_batch_losses)
        _run.log_scalar("val_corr_losses", v_corr_losses)
        _run.log_scalar("val_att_losses", v_att_losses)
        _run.log_scalar("val_prof_corr_losses", v_prof_losses)
        _run.log_scalar("val_count_corr_losses", v_count_losses)
        _run.log_scalar("val_pos_att_losses", v_pos_losses)
        _run.log_scalar("val_neg_att_losses", v_neg_losses)

        # Save trained model for the epoch
        savepath = os.path.join(
            output_dir, "model_ckpt_epoch_%d.pt" % (epoch + 1)
        )
        util.save_model(model, savepath)

        # If losses are both NaN, then stop
        if np.isnan(train_epoch_loss) and np.isnan(val_epoch_loss):
            break

        # Check for early stopping
        if early_stopping:
            if len(val_epoch_loss_hist) < early_stop_hist_len - 1:
                # Not enough history yet; tack on the loss
                val_epoch_loss_hist = [val_epoch_loss] + val_epoch_loss_hist
            else:
                # Tack on the new validation loss, kicking off the old one
                val_epoch_loss_hist = \
                    [val_epoch_loss] + val_epoch_loss_hist[:-1]
                best_delta = np.max(np.diff(val_epoch_loss_hist))
                if best_delta < early_stop_min_delta:
                    break  # Not improving enough

    # Compute evaluation metrics and log them
    for data_loader, prefix in [
        (summit_loader, "summit"), # (peak_loader, "peak"),
        # (val_loader, "genomewide")
    ]:
        print("Computing validation metrics, %s:" % prefix)
        batch_losses, corr_losses, att_losses, prof_losses, count_losses, \
            pos_losses, neg_losses, log_pred_profs, log_pred_counts, \
            true_profs, true_counts, input_seqs, input_grads, \
            coords = run_epoch(
                summit_loader, "eval", model, float("inf"), return_data=True
                # Don't use attribution prior loss when computing final loss
        )
        _run.log_scalar("%s_batch_losses" % prefix, batch_losses)
        _run.log_scalar("%s_corr_losses" % prefix, corr_losses)
        _run.log_scalar("%s_att_losses" % prefix, att_losses)
        _run.log_scalar("%s_prof_corr_losses" % prefix, prof_losses)
        _run.log_scalar("%s_count_corr_losses" % prefix, count_losses)
        _run.log_scalar("%s_pos_att_losses" % prefix, pos_losses)
        _run.log_scalar("%s_neg_att_losses" % prefix, neg_losses)

        metrics = profile_performance.compute_performance_metrics(
            true_profs, log_pred_profs, true_counts, log_pred_counts
        )
        profile_performance.log_performance_metrics(metrics, prefix,  _run)


@train_ex.command
def run_training(train_peak_beds, val_peak_beds, prof_bigwigs):
    train_loader = make_profile_dataset.create_data_loader(
        train_peak_beds, prof_bigwigs, "SamplingCoordsBatcher",
        return_coords=True
    )
    val_loader = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs, "SamplingCoordsBatcher",
        return_coords=True
    )
    summit_loader = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs, "SummitCenteringCoordsBatcher",
        return_coords=True, revcomp=False
    )
    peak_loader = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs, "PeakTilingCoordsBatcher"
    )
    train_model(train_loader, val_loader, summit_loader, peak_loader)


@train_ex.automain
def main():
    import json
    paths_json_path = "/users/amtseng/att_priors/data/processed/ENCODE/profile/config/SPI1/SPI1_training_paths.json"
    with open(paths_json_path, "r") as f:
        paths_json = json.load(f)

    train_peak_beds = paths_json["train_peak_beds"]
    val_peak_beds = paths_json["val_peak_beds"]
    prof_bigwigs = paths_json["prof_bigwigs"]

    run_training(train_peak_beds, val_peak_beds, prof_bigwigs)
