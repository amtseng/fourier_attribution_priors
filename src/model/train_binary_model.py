import numpy as np
import sacred
import torch
import math
import tqdm
import os
import model.util as util
import model.binary_models as binary_models
import model.performance as performance
import feature.make_binary_dataset as make_binary_dataset

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/users/amtseng/att_priors/binary_models/trained_models/"
)

train_ex = sacred.Experiment("train", ingredients=[
    make_binary_dataset.dataset_ex
])
train_ex.observers.append(
    sacred.observers.FileStorageObserver.create(MODEL_DIR)
)
logger = util.make_logger("train_logger")
train_ex.logger = logger

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
    conv_depths = [50, 50, 50]

    # Size max pool filter
    max_pool_size = 40

    # Strides for max pool filter
    max_pool_stride = 40

    # Number of fully-connected layers to apply
    num_fc_layers = 2

    # Number of hidden nodes in each fully-connected layer
    fc_sizes = [50, 15]

    # Number of outputs to predict
    num_outputs = 1

    # Whether to average the positive and negative correctness losses
    avg_class_loss = True

    # Weight to use for attribution prior loss; set to 0 to not use att. priors
    att_prior_loss_weight = 1.0

    # Maximum frequency integer to consider for a positive attribution prior
    att_prior_pos_limit = 160

    # Weight for positives within the attribution prior loss
    att_prior_pos_weight = 0.1

    # Whether to apply batch normalization
    batch_norm = True

    # Convolutional layer dropout rate
    conv_drop_rate = 0.0
    
    # Fully-connected layer dropout rate
    fc_drop_rate = 0.2

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
    train_seed = None # 20190905

    # Imported from make_binary_dataset
    input_length = dataset["input_length"]
    
    # Imported from make_binary_dataset
    input_depth = dataset["input_depth"]
    
    # Imported from make_binary_dataset
    val_neg_downsample = dataset["negative_stride"]


@train_ex.capture
def create_model(
    input_length, input_depth, num_conv_layers, conv_filter_sizes, conv_stride,
    conv_depths, max_pool_size, max_pool_stride, num_fc_layers, fc_sizes,
    num_outputs, batch_norm, conv_drop_rate, fc_drop_rate
):
    """
    Creates a binary model using the configuration above.
    """
    bin_model = binary_models.BinaryTFBindingPredictor(
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
        num_outputs=num_outputs,
        batch_norm=batch_norm,
        conv_drop_rate=conv_drop_rate,
        fc_drop_rate=fc_drop_rate
    )

    return bin_model


@train_ex.capture
def model_loss(
    model, true_vals, probs, input_grads, avg_class_loss, att_prior_loss_weight,
    att_prior_pos_limit, att_prior_pos_weight, return_loss_parts=False
):
    """
    Computes the loss for the model.
    Arguments:
        `model`: the model being trained
        `true_seqs`: a B x C tensor, where B is the batch size and C is the
            number of output tasks, containing the true binary values
        `probs`: a B x C tensor containing the predicted probabilities
        `input_grads`: a B x C x L x D tensor, where B is the batch size, C is
            the number of output tasks, L is the length of the input, and D is
            the dimensionality of each input base; this needs to be the
            gradients of the input with respect to each output task
    Returns a scalar Tensor containing the loss for the given batch, as well as
    a pair consisting of the correctness loss and the attribution prior loss.
    If the attribution prior loss is not computed at all, then 0 will be in its
    place, instead.
    """
    pred_loss = model.prediction_loss(true_vals, probs, avg_class_loss)
   
    if not att_prior_loss_weight:
        return pred_loss, (pred_loss, torch.zeros(1))
    
    att_prior_loss = model.att_prior_loss(
        true_vals, input_grads, att_prior_pos_limit, att_prior_pos_weight
    )
    final_loss = pred_loss + (att_prior_loss_weight * att_prior_loss)
    return final_loss, (pred_loss, att_prior_loss)


@train_ex.capture
def train_epoch(train_loader, model, optimizer, att_prior_loss_weight):
    """
    Runs the data from the training loader once through the model, and performs
    backpropagation. Returns a list of losses for the batches, a list of
    correction losses specifically for the batches, and a list of attribution
    prior losses specifically for the batches. If the attribution prior loss is
    not computed, then the list will have all 0s.
    """
    train_loader.dataset.on_epoch_start()  # Set-up the epoch
    num_batches = len(train_loader.dataset)
    t_iter = tqdm.tqdm(
        train_loader, total=num_batches, desc="\tTraining loss: ---"
    )

    model.train()  # Switch to training mode
    torch.set_grad_enabled(True)

    batch_losses, pred_losses, att_losses = [], [], []
    for input_seqs, output_vals in t_iter:
        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        output_vals = util.place_tensor(torch.tensor(output_vals)).float()
        
        # Make channels come first in input
        input_seqs = torch.transpose(input_seqs, 1, 2)

        if att_prior_loss_weight > 0:
            input_seqs.requires_grad = True  # Set gradient required
            probs = model(input_seqs)
            input_grads = []
            for i in range(probs.size(1)):
                model.zero_grad()  # Clear gradients
                mask = torch.zeros(probs.size())
                mask[:, i] = 1  # Only consider gradients for one task
                probs.backward(
                    util.place_tensor(mask),
                    retain_graph=True
                )
                grad = torch.unsqueeze(input_seqs.grad, dim=1)  # B x 1 x D x L
                input_grads.append(grad)
            input_grads = torch.cat(input_grads, dim=1)  # B x C x D x L
            input_grads = input_grads.transpose(2, 3)  # B x C x L x D
            input_seqs.requires_grad = False  # Reset gradient required
        else:
            probs = model(input_seqs)
            input_grads = None

        optimizer.zero_grad()  # Clear gradients from last batch
        loss, (pred_loss, att_loss) = model_loss(
            model, output_vals, probs, input_grads
        )
        loss.backward()  # Compute gradient
        optimizer.step()  # Update weights through backprop
        
        batch_losses.append(loss.item())
        pred_losses.append(pred_loss.item())
        att_losses.append(att_loss.item())
        t_iter.set_description(
            "\tTraining loss: %6.10f" % loss.item()
        )

    return batch_losses, pred_losses, att_losses


@train_ex.capture
def eval_epoch(val_loader, model, att_prior_loss_weight):
    """
    Runs the data from the validation loader once through the model, and
    saves the output results. Returns a list of losses for the batches, a list
    of correction losses specifically for the batches, a list of attribution
    prior losses specifically for the batches, a NumPy array of predicted
    probabilities, and a NumPy array of true values. If the attribution prior
    loss is not computed, then the list will have all 0s.
    """ 
    val_loader.dataset.on_epoch_start()  # Set-up the epoch
    num_batches = len(val_loader.dataset)
    t_iter = tqdm.tqdm(
        val_loader, total=num_batches, desc="\tValidation loss: ---"
    )

    model.eval()  # Switch to evaluation mode

    batch_losses, pred_losses, att_losses = [], [], []
    pred_val_arr, true_val_arr = [], []
    for input_seqs, output_vals in t_iter:
        true_val_arr.append(output_vals)

        input_seqs = util.place_tensor(torch.tensor(input_seqs)).float()
        output_vals = util.place_tensor(torch.tensor(output_vals)).float()

        # Make channels come first in input
        input_seqs = torch.transpose(input_seqs, 1, 2)

        if att_prior_loss_weight > 0:
            torch.set_grad_enabled(True)  # We actually do need grad here
            input_seqs.requires_grad = True  # Set gradient required
            probs = model(input_seqs)
            pred_val_arr.append(probs.detach().to("cpu").numpy())
            input_grads = []
            for i in range(probs.size(1)):
                model.zero_grad()  # Clear gradients
                mask = torch.zeros(probs.size())
                mask[:, i] = 1  # Only consider gradients for one task
                probs.backward(
                    util.place_tensor(mask),
                    retain_graph=True
                )
                grad = torch.unsqueeze(input_seqs.grad, dim=1)  # B x 1 x D x L
                input_grads.append(grad)
            input_grads = torch.cat(input_grads, dim=1)  # B x C x D x L
            input_grads = input_grads.transpose(2, 3)  # B x C x L x D
            input_seqs.requires_grad = False  # Reset gradient required
            torch.set_grad_enabled(False)  # Don't need grad to compute loss
        else:
            torch.set_grad_enabled(False)
            model.zero_grad()  # Clear gradients from last batch
            probs = model(input_seqs)
            pred_val_arr.append(probs.detach().to("cpu").numpy())
            input_grads = None

        loss, (pred_loss, att_loss) = model_loss(
            model, output_vals, probs, input_grads
        )

        batch_losses.append(loss.item())
        pred_losses.append(pred_loss.item())
        att_losses.append(att_loss.item())

        t_iter.set_description(
            "\tValidation loss: %6.10f" % loss.item()
        )

    pred_vals = np.concatenate(pred_val_arr)
    true_vals = np.concatenate(true_val_arr)

    return batch_losses, pred_losses, att_losses, pred_vals, true_vals


@train_ex.capture
def train(
    train_loader, val_loader, num_epochs, learning_rate, early_stopping,
    early_stop_hist_len, early_stop_min_delta, train_seed, val_neg_downsample,
    _run
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

        t_batch_losses, t_pred_losses, t_att_losses = train_epoch(
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
        _run.log_scalar("train_pred_losses", t_pred_losses)
        _run.log_scalar("train_att_losses", t_att_losses)

        v_batch_losses, v_pred_losses, v_att_losses, pred_vals, true_vals = \
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
        _run.log_scalar("val_pred_losses", v_pred_losses)
        _run.log_scalar("val_att_losses", v_att_losses)

        # Compute evaluation metrics and log them
        metrics = performance.compute_evaluation_metrics(
            true_vals, pred_vals, val_neg_downsample
        )
        performance.log_evaluation_metrics(metrics, logger, _run)

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
def run_training(train_bed_path, val_bed_path):
    print("Importing training data")
    train_loader = make_binary_dataset.data_loader_from_bedfile(train_bed_path)
    print("Importing validation data")
    val_loader = make_binary_dataset.data_loader_from_bedfile(val_bed_path)
    print("Training")
    train(train_loader, val_loader)


@train_ex.automain
def main():
    train_bedfile = "/users/amtseng/tfmodisco/data/processed/DREAM/tests/SPI1_test_2000.tsv.gz"
    val_bedfile = train_bedfile
    train_bedfile = "/users/amtseng/tfmodisco/data/raw/DREAM/ChIPseq/labels/SPI1.train.labels.tsv.gz"
    val_bedfile = "/users/amtseng/tfmodisco/data/raw/DREAM/ChIPseq/heldout_chr/training_celltypes/labels/SPI1.train.labels.tsv.gz"
    run_training(train_bedfile, val_bedfile)
