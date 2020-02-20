import numpy as np
import sacred
import torch
import math
import tqdm
import os
import model.util as util
import model.binary_models as binary_models
import model.binary_performance as binary_performance
import feature.make_binary_dataset as make_binary_dataset

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/users/amtseng/att_priors/models/trained_models/binary/misc/"
)

train_ex = sacred.Experiment("train", ingredients=[
    make_binary_dataset.dataset_ex
])
train_ex.observers.append(
    sacred.observers.FileStorageObserver.create(MODEL_DIR)
)

@train_ex.config
def config(dataset):
    # Number of convolutional layers to apply
    num_conv_layers = 3

    # Size of convolutional filter to apply
    conv_filter_sizes = [15, 15, 13]

    # Convolutional stride
    conv_stride = 1
    
    # Number of filters to use at each convolutional level (i.e. number of
    # channels to output)
    conv_depth = 64
    conv_depths = [conv_depth, conv_depth, conv_depth]

    # Size max pool filter
    max_pool_size = 40

    # Strides for max pool filter
    max_pool_stride = 40

    # Number of fully-connected layers to apply
    num_fc_layers = 2

    # Number of hidden nodes in each fully-connected layer
    fc_sizes = [50, 15]
    
    # Whether to apply batch normalization
    batch_norm = True

    # Convolutional layer dropout rate
    conv_drop_rate = 0.0
    
    # Fully-connected layer dropout rate
    fc_drop_rate = 0.0

    # Number of outputs to predict
    num_tasks = 4

    # Whether to average the positive and negative correctness losses
    avg_class_loss = True

    # Weight to use for attribution prior loss; set to 0 to not use att. priors
    att_prior_loss_weight = 1

    # Type of annealing; can be None (constant/no annealing), "inflate" (follows
    # `2/(1 + e^(-c*x)) - 1`), or "deflate" (follows `e^(-c * x)`)
    att_prior_loss_weight_anneal_type = None

    # Annealing factor for attribution prior loss weight, c
    if att_prior_loss_weight_anneal_type is None:
        att_prior_loss_weight_anneal_speed = None
    elif att_prior_loss_weight_anneal_type == "inflate":
        att_prior_loss_weight_anneal_speed = 1
    elif att_prior_loss_weight_anneal_type == "deflate":
        att_prior_loss_weight_anneal_speed = 0.3

    # Smoothing amount for gradients before computing attribution prior loss;
    # Smoothing window size is 1 + (2 * sigma); set to 0 for no smoothing
    att_prior_grad_smooth_sigma = 3

    # Maximum frequency integer to consider for a Fourier attribution prior
    fourier_att_prior_freq_limit = 200

    # Amount to soften the Fourier attribution prior loss limit; set to None
    # to not soften; softness decays like 1 / (1 + x^c) after the limit
    fourier_att_prior_freq_limit_softness = 0.2

    # Number of training epochs
    num_epochs = 10

    # Learning rate
    learning_rate = 0.001

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

    # Imported from make_binary_dataset
    negative_ratio = dataset["negative_ratio"]


@train_ex.capture
def create_model(
    input_length, input_depth, num_conv_layers, conv_filter_sizes, conv_stride,
    conv_depths, max_pool_size, max_pool_stride, num_fc_layers, fc_sizes,
    num_tasks, batch_norm, conv_drop_rate, fc_drop_rate
):
    """
    Creates a binary model using the configuration above.
    """
    bin_model = binary_models.BinaryPredictor(
        input_length=input_length,
        input_depth=input_depth,
        num_conv_layers=num_conv_layers,
        conv_filter_sizes=conv_filter_sizes,
        conv_stride=conv_stride,
        conv_depths=conv_depths,
        max_pool_size=max_pool_size,
        max_pool_stride=max_pool_stride,
        num_fc_layers=num_fc_layers,
        fc_sizes=fc_sizes,
        num_tasks=num_tasks,
        batch_norm=batch_norm,
        conv_drop_rate=conv_drop_rate,
        fc_drop_rate=fc_drop_rate
    )

    return bin_model


@train_ex.capture
def model_loss(
    model, true_vals, logit_pred_vals, epoch_num, avg_class_loss,
    att_prior_loss_weight, att_prior_loss_weight_anneal_type,
    att_prior_loss_weight_anneal_speed, att_prior_grad_smooth_sigma,
    fourier_att_prior_freq_limit, fourier_att_prior_freq_limit_softness,
    att_prior_loss_only, input_grads=None, status=None
):
    """
    Computes the loss for the model.
    Arguments:
        `model`: the model being trained
        `true_vals`: a B x T tensor, where B is the batch size and T is the
            number of output tasks, containing the true binary values
        `logit_pred_vals`: a B x T tensor containing the predicted logits
        `epoch_num`: a 0-indexed integer representing the current epoch
        `input_grads`: a B x I x D tensor, where I is the input length and D is
            the input depth; this is the gradient of the output with respect to
            the input, times the input itself; only needed when attribution
            prior loss weight is positive
        `status`: a B-tensor, where B is the batch size; each entry is 1 if that
            that example is to be treated as a positive example, 0 if negative,
            and -1 if ambiguous; only needed when attribution prior loss weight
            is positive
    Returns a scalar Tensor containing the loss for the given batch, and a pair
    consisting of the correctness loss and the attribution prior loss.
    If the attribution prior loss is not computed at all, then 0 will be in its
    place, instead.
    """
    corr_loss = model.correctness_loss(
        true_vals, logit_pred_vals, avg_class_loss
    )
    
    if not att_prior_loss_weight:
        return corr_loss, (corr_loss, torch.zeros(1))
   
    att_prior_loss = model.fourier_att_prior_loss(
        status, input_grads, fourier_att_prior_freq_limit,
        fourier_att_prior_freq_limit_softness, att_prior_grad_smooth_sigma
    )
    
    if att_prior_loss_weight_anneal_type is None:
        weight = att_prior_loss_weight
    elif att_prior_loss_weight_anneal_type == "inflate":
        exp = np.exp(-att_prior_loss_weight_anneal_speed * epoch_num)
        weight = att_prior_loss_weight * ((2 / (1 + exp)) - 1)
    elif att_prior_loss_weight_anneal_type == "deflate":
        exp = np.exp(-att_prior_loss_weight_anneal_speed * epoch_num)
        weight = att_prior_loss_weight * exp

    if att_prior_loss_only:
        final_loss = att_prior_loss
    else:
        final_loss = corr_loss + (weight * att_prior_loss)

    return final_loss, (corr_loss, att_prior_loss)


@train_ex.capture
def run_epoch(
    data_loader, mode, model, epoch_num, num_tasks, att_prior_loss_weight,
    batch_size, revcomp, input_length, input_depth, optimizer=None,
    return_data=False
):
    """
    Runs the data from the data loader once through the model, to train,
    validate, or predict.
    Arguments:
        `data_loader`: an instantiated `DataLoader` instance that gives batches
            of data; each batch must yield the input sequences, the output
            values, the status, and coordinates
        `mode`: one of "train", "eval"; if "train", run the epoch and perform
            backpropagation; if "eval", only do evaluation
        `model`: the current PyTorch model being trained/evaluated
        `epoch_num`: 0-indexed integer representing the current epoch
        `optimizer`: an instantiated PyTorch optimizer, for training mode
        `return_data`: if specified, returns the following as NumPy arrays:
            true binding values (0, 1, or -1) (N x T), predicted binding
            probabilities (N x T), the underlying sequence coordinates (N x 3
            object array), input gradients (N x I x 4), and the input sequences
            (N x I x 4); if the attribution prior is not used, the gradients
            will be garbage
    Returns lists of overall losses, correctness losses, and attribution prior
    losses, where each list is over all batches. If the attribution prior loss
    is not computed, then it will be all 0s. If `return_data` is True, then more
    things will be returned after these.
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
    if return_data:
        # Allocate empty NumPy arrays to hold the results
        num_samples_exp = num_batches * batch_size
        num_samples_exp *= 2 if revcomp else 1
        # Real number of samples can be smaller because of partial last batch
        all_true_vals = np.empty((num_samples_exp, num_tasks))
        all_pred_vals = np.empty((num_samples_exp, num_tasks))
        all_input_seqs = np.empty((num_samples_exp, input_length, input_depth))
        all_input_grads = np.empty((num_samples_exp, input_length, input_depth))
        all_coords = np.empty((num_samples_exp, 3), dtype=object)
        num_samples_seen = 0  # Real number of samples seen

    for input_seqs, output_vals, statuses, coords in t_iter:
        if return_data:
            input_seqs_np = input_seqs
            output_vals_np = output_vals
        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        output_vals = util.place_tensor(torch.tensor(output_vals)).float()

        # Clear gradients from last batch if training
        if mode == "train":
            optimizer.zero_grad()
        elif att_prior_loss_weight > 0:
            # Not training mode, but we still need to zero out weights because
            # we are computing the input gradients
            model.zero_grad()

        if att_prior_loss_weight > 0:
            input_seqs.requires_grad = True  # Set gradient required
            logit_pred_vals = model(input_seqs)
            # Compute the gradients of the output with respect to the input
            input_grads, = torch.autograd.grad(
                logit_pred_vals, input_seqs,
                grad_outputs=util.place_tensor(
                    torch.ones(logit_pred_vals.size())
                ),
                retain_graph=True, create_graph=True
                # We'll be operating on the gradient itself, so we need to
                # create the graph
                # Gradients are summed across tasks
            )
            if return_data:
                input_grads_np = input_grads.detach().cpu().numpy()
            input_grads = input_grads * input_seqs  # Gradient * input
            status = util.place_tensor(torch.tensor(statuses))
            input_seqs.requires_grad = False  # Reset gradient required
        else:
            logit_pred_vals = model(input_seqs)
            status, input_grads = None, None

        loss, (corr_loss, att_loss) = model_loss(
            model, output_vals, logit_pred_vals, epoch_num, status=status,
            input_grads=input_grads
        )

        if mode == "train":
            loss.backward()  # Compute gradient
            optimizer.step()  # Update weights through backprop

        batch_losses.append(loss.item())
        corr_losses.append(corr_loss.item())
        att_losses.append(att_loss.item())
        t_iter.set_description(
            "\tLoss: %6.4f" % loss.item()
        )

        if return_data:
            logit_pred_vals_np = logit_pred_vals.detach().cpu().numpy()
          
            # Turn logits into probabilities
            pred_vals = binary_models.binary_logits_to_probs(logit_pred_vals_np)
            num_in_batch = pred_vals.shape[0]

            # Fill in the batch data/outputs into the preallocated arrays
            start, end = num_samples_seen, num_samples_seen + num_in_batch
            all_true_vals[start:end] = output_vals_np
            all_pred_vals[start:end] = pred_vals
            all_input_seqs[start:end] = input_seqs_np
            if att_prior_loss_weight:
                all_input_grads[start:end] = input_grads_np
            all_coords[start:end] = coords

            num_samples_seen += num_in_batch

    if return_data:
        # Truncate the saved data to the proper size, based on how many
        # samples actually seen
        all_true_vals = all_true_vals[:num_samples_seen]
        all_pred_vals = all_pred_vals[:num_samples_seen]
        all_input_seqs = all_input_seqs[:num_samples_seen]
        all_input_grads = all_input_grads[:num_samples_seen]
        all_coords = all_coords[:num_samples_seen]
        return batch_losses, corr_losses, att_losses, all_true_vals, \
            all_pred_vals, all_coords, all_input_grads, all_input_seqs
    else:
        return batch_losses, corr_losses, att_losses


@train_ex.capture
def train_model(
    train_loader, val_loader, test_loader, num_epochs, learning_rate,
    early_stopping, early_stop_hist_len, early_stop_min_delta, train_seed,
    negative_ratio, _run
):
    """
    Trains the network for the given training and validation data.
    Arguments:
        `train_loader` (DataLoader): a data loader for the training data
        `val_loader` (DataLoader): a data loader for the validation data
        `test_loader` (DataLoader): a data loader for the test data
    Note that all data loaders are expected to yield the 1-hot encoded
    sequences, output values, source coordinates, and source peaks.
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

        t_batch_losses, t_corr_losses, t_att_losses = run_epoch(
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

        v_batch_losses, v_corr_losses, v_att_losses = run_epoch(
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
            if len(val_epoch_loss_hist) < early_stop_hist_len + 1:
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
    print("Computing test metrics:")
    batch_losses, corr_losses, att_losses, true_vals, pred_vals, coords, \
        input_grads, input_seqs = run_epoch(
            test_loader, "eval", model, 0, return_data=True
            # Don't use attribution prior loss when computing final loss
    )
    _run.log_scalar("test_batch_losses", batch_losses)
    _run.log_scalar("test_corr_losses", corr_losses)
    _run.log_scalar("test_att_losses", att_losses)

    metrics = binary_performance.compute_performance_metrics(
        true_vals, pred_vals, negative_ratio 
    )
    binary_performance.log_performance_metrics(metrics, "test", _run)


@train_ex.command
def run_training(
    labels_hdf5, bin_labels_npy, train_chroms, val_chroms, test_chroms,
    peak_qvals_npy=None
):
    bin_labels_array = np.load(bin_labels_npy, allow_pickle=True)
    peak_qvals_array = np.load(peak_qvals_npy) if peak_qvals_npy else None

    train_loader = make_binary_dataset.create_data_loader(
        labels_hdf5, bin_labels_array, return_coords=True,
        chrom_set=train_chroms, peak_qvals_npy_or_array=peak_qvals_array
    )
    val_loader = make_binary_dataset.create_data_loader(
        labels_hdf5, bin_labels_array, return_coords=True,
        chrom_set=val_chroms, peak_qvals_npy_or_array=peak_qvals_array

    )
    test_loader = make_binary_dataset.create_data_loader(
        labels_hdf5, bin_labels_array, return_coords=True,
        chrom_set=test_chroms, peak_qvals_npy_or_array=peak_qvals_array

    )
    train_model(train_loader, val_loader, test_loader)


@train_ex.automain
def main():
    import json
    paths_json_path = "/users/amtseng/att_priors/data/processed/ENCODE_TFChIP/binary/config/SPI1/SPI1_training_paths.json"
    with open(paths_json_path, "r") as f:
        paths_json = json.load(f)
    labels_hdf5 = paths_json["labels_hdf5"]
    bin_labels_npy = paths_json["bin_labels_npy"]
    peak_qvals_npy = paths_json["peak_qvals_npy"]
    
    splits_json_path = "/users/amtseng/att_priors/data/processed/chrom_splits.json"
    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)
    train_chroms, val_chroms, test_chroms = \
        splits_json["1"]["train"], splits_json["1"]["val"], \
        splits_json["1"]["test"]

    run_training(
        labels_hdf5, bin_labels_npy, train_chroms, val_chroms, test_chroms,
        peak_qvals_npy
    )
