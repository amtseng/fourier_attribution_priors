import numpy as np
import sacred
import torch
import math
import tqdm
import os
import model.util as util
import model.profile_models as profile_models
import model.performance as performance
import feature.make_profile_dataset as make_profile_dataset

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/users/amtseng/att_priors/models/trained_profile_models/"
)

train_ex = sacred.Experiment("train", ingredients=[
    make_profile_dataset.dataset_ex
])
train_ex.observers.append(
    sacred.observers.FileStorageObserver.create(MODEL_DIR)
)
logger = util.make_logger("train_logger")
train_ex.logger = logger

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
    att_prior_loss_weight = 0.0

    # Maximum frequency integer to consider for a positive attribution prior
    att_prior_pos_limit = 160

    # Weight for positives within the attribution prior loss
    att_prior_pos_weight = 0.1

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
    model, true_profs, log_pred_profs, log_pred_counts, input_grads,
    counts_loss_weight, att_prior_loss_weight, att_prior_pos_limit,
    att_prior_pos_weight, return_loss_parts=False
):
    """
    Computes the loss for the model.
    Arguments:
        `model`: the model being trained
        `true_profs`: a B x T x 2 x O tensor, where B is the batch size, T is
            the number of tasks, and O is the length of the output profiles;
            this contains true profile read counts (unnormalized)
        `log_pred_profs`: a B x T x 2 x O tensor, consisting of the output
            profile predictions as logits
        `log_pred_counts`: a B x T x 2 tensor consisting of the log counts
            predictions
    Returns a scalar Tensor containing the loss for the given batch, as well as
    a pair consisting of the correctness loss and the attribution prior loss.
    If the attribution prior loss is not computed at all, then 0 will be in its
    place, instead.
    """
    corr_loss = model.correctness_loss(
        true_profs, log_pred_profs, log_pred_counts, counts_loss_weight
    )
    
    return corr_loss, (corr_loss, torch.zeros(1))
    

@train_ex.capture
def train_epoch(
    train_loader, model, optimizer, num_tasks, att_prior_loss_weight
):
    """
    Runs the data from the training loader once through the model, and performs
    backpropagation. Returns a list of losses for the batches, a list of
    correction losses specifically for the batches, and a list of attribution
    prior losses specifically for the batches. If the attribution prior loss is
    not computed, then the list will have all 0s. Note that the data loader is
    expected to return profiles where the first half of the tasks are prediction
    profiles, and the second half are control profiles.
    """
    train_loader.dataset.on_epoch_start()  # Set-up the epoch
    num_batches = len(train_loader.dataset)
    t_iter = tqdm.tqdm(
        train_loader, total=num_batches, desc="\tTraining loss: ---"
    )

    model.train()  # Switch to training mode
    torch.set_grad_enabled(True)

    batch_losses, corr_losses, att_losses = [], [], []
    for input_seqs, profiles, statuses in t_iter:
        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        profiles = util.place_tensor(torch.tensor(profiles)).float()
        
        tf_profs = profiles[:, :num_tasks, :, :]
        cont_profs = profiles[:, num_tasks:, :, :]
        
        # Make channels come first in input
        input_seqs = torch.transpose(input_seqs, 1, 2)

        log_pred_profs, log_pred_counts = model(input_seqs, cont_profs)
        input_grads = None

        optimizer.zero_grad()  # Clear gradients from last batch
        loss, (corr_loss, att_loss) = model_loss(
            model, tf_profs, log_pred_profs, log_pred_counts, input_grads
        )

        loss.backward()  # Compute gradient
        optimizer.step()  # Update weights through backprop
        
        batch_losses.append(loss.item())
        corr_losses.append(corr_loss.item())
        att_losses.append(att_loss.item())
        t_iter.set_description(
            "\tTraining loss: %6.10f" % loss.item()
        )

    return batch_losses, corr_losses, att_losses


@train_ex.capture
def eval_epoch(val_loader, model, num_tasks, att_prior_loss_weight):
    """
    Runs the data from the validation loader once through the model, and
    saves the output results. Returns a list of losses for the batches, a list
    of correction losses specifically for the batches, and a list of attribution
    prior losses specifically for the batches. If the attribution prior loss is
    not computed, then the list will have all 0s. Note that the data loader is
    expected to return profiles where the first half of the tasks are prediction
    profiles, and the second half are control profiles.
    """ 
    val_loader.dataset.on_epoch_start()  # Set-up the epoch
    num_batches = len(val_loader.dataset)
    t_iter = tqdm.tqdm(
        val_loader, total=num_batches, desc="\tValidation loss: ---"
    )

    model.eval()  # Switch to evaluation mode

    batch_losses, corr_losses, att_losses = [], [], []

    for input_seqs, profiles, statuses in t_iter:
        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        profiles = util.place_tensor(torch.tensor(profiles)).float()
        
        tf_profs = profiles[:, :num_tasks, :, :]
        cont_profs = profiles[:, num_tasks:, :, :]
        
        # Make channels come first in input
        input_seqs = torch.transpose(input_seqs, 1, 2)
    
        torch.set_grad_enabled(False)
        model.zero_grad()  # Clear gradients from last batch

        log_pred_profs, log_pred_counts = model(input_seqs, cont_profs)
        input_grads = None

        loss, (corr_loss, att_loss) = model_loss(
            model, tf_profs, log_pred_profs, log_pred_counts, input_grads
        )

        batch_losses.append(loss.item())
        corr_losses.append(corr_loss.item())
        att_losses.append(att_loss.item())
        t_iter.set_description(
            "\tTraining loss: %6.10f" % loss.item()
        )

    return batch_losses, corr_losses, att_losses


@train_ex.capture
def train(
    train_loader, val_loader, num_epochs, learning_rate, early_stopping,
    early_stop_hist_len, early_stop_min_delta, train_seed, _run
):
    """
    Trains the network for the given training and validation data.
    Arguments:
        `train_loader` (DataLoader): a data loader for the training data, each
            batch giving the 1-hot encoded sequence and values
        `val_loader` (DataLoader): a data loader for the validation data, each
            batch giving the 1-hot encoded sequence and values
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

        t_batch_losses, t_corr_losses, t_att_losses = train_epoch(
            train_loader, model, optimizer
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

        v_batch_losses, v_corr_losses, v_att_losses = \
        eval_epoch(
            val_loader, model
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


@train_ex.command
def run_training(train_peak_beds, val_peak_beds, prof_bigwigs):
    train_loader = make_profile_dataset.data_loader_from_beds_and_bigwigs(
        train_peak_beds, prof_bigwigs
    )
    val_loader = make_profile_dataset.data_loader_from_beds_and_bigwigs(
        val_peak_beds, prof_bigwigs
    )
    train(train_loader, val_loader)


@train_ex.automain
def main():
    base_path = "/users/amtseng/att_priors/data/processed/ENCODE/profile/labels"

    train_peak_beds = [
        os.path.join(base_path, ending) for ending in [
            "SPI1/SPI1_ENCSR000BGQ_GM12878_train_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BGW_K562_train_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_train_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BUW_HL-60_train_peakints.bed.gz"
        ]
    ]

    val_peak_beds = [
        os.path.join(base_path, ending) for ending in [
            "SPI1/SPI1_ENCSR000BGQ_GM12878_holdout_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BGW_K562_holdout_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_holdout_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BUW_HL-60_holdout_peakints.bed.gz"
        ]
    ]
            
    prof_bigwigs = [
        (os.path.join(base_path, e_1), os.path.join(base_path, e_2)) \
        for e_1, e_2 in [
            ("SPI1/SPI1_ENCSR000BGQ_GM12878_neg.bw",
            "SPI1/SPI1_ENCSR000BGQ_GM12878_pos.bw"),
            ("SPI1/SPI1_ENCSR000BGW_K562_neg.bw",
            "SPI1/SPI1_ENCSR000BGW_K562_pos.bw"),
            ("SPI1/SPI1_ENCSR000BIJ_GM12891_neg.bw",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_pos.bw"),
            ("SPI1/SPI1_ENCSR000BUW_HL-60_neg.bw",
            "SPI1/SPI1_ENCSR000BUW_HL-60_pos.bw"),
            ("SPI1/control_ENCSR000BGG_K562_neg.bw",
            "SPI1/control_ENCSR000BGG_K562_pos.bw"),
            ("SPI1/control_ENCSR000BGH_GM12878_neg.bw",
            "SPI1/control_ENCSR000BGH_GM12878_pos.bw"),
            ("SPI1/control_ENCSR000BIH_GM12891_neg.bw",
            "SPI1/control_ENCSR000BIH_GM12891_pos.bw"),
            ("SPI1/control_ENCSR000BVU_HL-60_neg.bw",
            "SPI1/control_ENCSR000BVU_HL-60_pos.bw")
        ]
    ]

    run_training(train_peak_beds, val_peak_beds, prof_bigwigs)
