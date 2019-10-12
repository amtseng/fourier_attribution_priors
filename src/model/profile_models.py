import torch
import math
import numpy as np
from model.util import sanitize_sacred_arguments, convolution_size

def multinomial_log_probs(category_log_probs, trials, query_counts):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    Arguments:
        `category_log_probs`: a D x N tensor containing log probabilities of
            seeing each of the N classes/categories
        `trials`: a D-tensor containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N tensor containing the observed count of eac
            category in each distribution; the probability is computed for these
            observations
    Returns a D-tensor containing the log probabilities of each observed query
    with its corresponding distribution.
    Note that D can be replaced with any shape (i.e. only the last dimension is
    reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)

    log_n_fact = torch.lgamma(trials.float() + 1)
    log_counts_fact = torch.lgamma(query_counts.float() + 1)
    log_counts_fact_sum = torch.sum(log_counts_fact, dim=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise sum
    log_prob_pows_sum = torch.sum(log_prob_pows, dim=-1)

    return log_n_fact - log_counts_fact_sum + log_prob_pows_sum


class ProfileTFBindingPredictor(torch.nn.Module):

    def __init__(
        self, input_length, input_depth, profile_length, num_tasks,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
        prof_conv_stride
    ):
        """
        Creates a TF binding profile predictor from a DNA sequence.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `profile_length`: length of the predicted profiles; it must be
                consistent with the convolutional layers specified
            `num_tasks`: number of tasks that are to be predicted; there will be
                two profiles and two read counts predicted for each task
            `num_dil_conv_layers`: number of dilating convolutional layers
            `dil_conv_filter_sizes`: sizes of the initial dilating convolutional
                filters; must have `num_conv_layers` entries
            `dil_conv_stride`: stride used for each dilating convolution
            `dil_conv_dilations`: dilations used for each layer of the dilating
                convolutional layers
            `dil_conv_depths`: depths of the dilating convolutional filters;
                must have `num_conv_layers` entries
            `prof_conv_kernel_size`: size of the large convolutional filter used
                for profile prediction
            `prof_conv_stride`: stride used for the large profile convolution

        Creates a close variant of the BPNet architecture, as described here:
            https://www.biorxiv.org/content/10.1101/737981v1.full
        """
        super().__init__()
        self.creation_args = locals()
        del self.creation_args["self"]
        del self.creation_args["__class__"]
        self.creation_args = sanitize_sacred_arguments(self.creation_args)
        
        assert len(dil_conv_filter_sizes) == num_dil_conv_layers
        assert len(dil_conv_dilations) == num_dil_conv_layers
        assert len(dil_conv_depths) == num_dil_conv_layers

        # Save some parameters
        self.input_depth = input_depth
        self.input_length = input_length
        self.profile_length = profile_length
        self.num_tasks = num_tasks
        self.num_dil_conv_layers = num_dil_conv_layers
        
        # 7 convolutional layers with increasing dilations
        self.dil_convs = torch.nn.ModuleList()
        last_out_size = input_length
        for i in range(num_dil_conv_layers):
            kernel_size = dil_conv_filter_sizes[i]
            in_channels = input_depth if i == 0 else dil_conv_depths[i - 1]
            out_channels = dil_conv_depths[i]
            dilation = dil_conv_dilations[i]
            padding = int(dilation * (kernel_size - 1) / 2)  # "same" padding,
                                                             # for easy adding
            self.dil_convs.append(
                torch.nn.Conv1d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, dilation=dilation, padding=padding
                )
            )
          
            last_out_size = last_out_size - (dilation * (kernel_size - 1))
        # The size of the final dilated convolution output, if there _weren't_
        # any padding (i.e. "valid" padding)
        self.last_dil_conv_size = last_out_size

        # ReLU activation for the convolutional layers
        self.relu = torch.nn.ReLU()

        # Profile prediction:
        # Convolutional layer with large kernel
        # TODO: Transposed convolution?
        # self.prof_first_conv = torch.nn.ConvTranspose1d(
        #     in_channels=64, out_channels=num_tasks, kernel_size=75
        # )
        self.prof_large_conv = torch.nn.Conv1d(
            in_channels=dil_conv_depths[-1], out_channels=(num_tasks * 2),
            kernel_size=prof_conv_kernel_size
        )

        self.prof_pred_size = self.last_dil_conv_size - \
            (prof_conv_kernel_size - 1)

        assert self.prof_pred_size == profile_length, \
            "Prediction length is specified to be %d, but with the given " +\
            "input length of %d and the given convolutions, the computed " +\
            "prediction length is %d" % \
            (profile_length, input_length, self.prof_pred_size)

        # Length-1 convolution over the convolutional output and controls to
        # get the final profile
        self.prof_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * 4), out_channels=(num_tasks * 2),
            kernel_size=1, groups=num_tasks  # One set of filters over each task
        )
        
        # Counts prediction:
        # Global average pooling
        self.count_pool = torch.nn.AvgPool1d(
            kernel_size=self.last_dil_conv_size
        )

        # Dense layer to consolidate pooled result to small number of features
        self.count_dense = torch.nn.Linear(
            in_features=dil_conv_depths[-1], out_features=(num_tasks * 2)
        )

        # Dense layer over pooling features and controls to get the final
        # counts, implemented as grouped convolution with kernel size 1
        self.count_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * 4), out_channels=(num_tasks * 2),
            kernel_size=1
        )

        # For converting profile logits to profile log probabilities
        self.sigmoid = torch.nn.Sigmoid()

        # MSE Loss for counts
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def forward(self, input_seqs, cont_profs):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x D x I tensor, where B is the batch size, D is
                the number of channels in the input, and I is the input sequence
                length
            `cont_profs`: a B x T x 2 x O tensor, where T is the number of
                tasks, and O is the output sequence length
        Returns the predicted (normalized) profiles for each task (both plus
        and minus strands) (a B x T x 2 x O tensor), and the predicted log
        counts for each task (both plus and minus strands) (a B x T x 2) tensor.
        """
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(2)
        assert input_length == self.input_length
        num_tasks = cont_profs.size(1)
        assert num_tasks == self.num_tasks
        profile_length = cont_profs.size(3)
        assert profile_length == self.profile_length
        
        # 1. Perform dilated convolutions on the input, each layer's input is
        # the sum of all previous layers' outputs
        dil_conv_out_list = None
        dil_conv_sum = 0
        for i, dil_conv in enumerate(self.dil_convs):
            if i == 0:
                dil_conv_out = self.relu(dil_conv(input_seqs))
            else:
                dil_conv_out = self.relu(dil_conv(dil_conv_sum))

            if i != self.num_dil_conv_layers - 1:
                dil_conv_sum = dil_conv_out + dil_conv_sum

        # 2. Truncate the final dilated convolutional layer output so that it
        # only has entries that did not see padding; this is equivalent to
        # truncating it to the size it would be if no padding were ever added
        start = int((dil_conv_out.size(2) - self.last_dil_conv_size) / 2)
        end = start + self.last_dil_conv_size
        dil_conv_out_cut = dil_conv_out[:, :, start : end]

        # Branch A: profile prediction
        # A1. Perform convolution with a large kernel
        prof_large_conv_out = self.prof_large_conv(dil_conv_out_cut)

        # A2. Concatenate with the control profiles
        # Reshaping is necessary to ensure the tasks are paired adjacently
        prof_large_conv_out = prof_large_conv_out.view(
            batch_size, num_tasks, 2, -1
        )
        prof_with_cont = torch.cat([prof_large_conv_out, cont_profs], dim=2)
        prof_with_cont = prof_with_cont.view(batch_size, num_tasks * 4, -1)

        # A3. Perform length-1 convolutions over the concatenated profiles with
        # controls; there are T convolutions, each one is done over one pair of
        # prof_first_conv_out, and a pair of controls
        prof_one_conv_out = self.prof_one_conv(prof_with_cont)
        prof_pred = prof_one_conv_out.view(batch_size, num_tasks, 2, -1)
        
        # Branch B: read count prediction
        # B1. Global average pooling across the output of dilated convolutions
        count_pool_out = self.count_pool(dil_conv_out_cut)
        count_pool_out = torch.squeeze(count_pool_out, dim=2)

        # B2. Reduce pooling output to fewer features, a pair for each task
        count_dense_out = self.count_dense(count_pool_out)

        # B3. Concatenate with the control counts
        # Reshaping is necessary to ensure the tasks are paired adjacently
        cont_counts = torch.sum(cont_profs, dim=3)
        count_dense_out = count_dense_out.view(batch_size, num_tasks, 2)
        count_with_cont = torch.cat([count_dense_out, cont_counts], dim=2)
        count_with_cont = count_with_cont.view(batch_size, num_tasks * 4, -1)

        # B4. Dense layer over the concatenation with control counts; each set
        # of counts gets a different dense network (implemented as convolution
        # with kernel size 1)
        count_one_conv_out = self.count_one_conv(count_with_cont)
        count_pred = count_one_conv_out.view(batch_size, num_tasks, 2, -1)
        count_pred = torch.squeeze(count_pred, dim=3)

        return prof_pred, count_pred

    def correctness_loss(
        self, true_profs, logit_pred_profs, log_pred_counts, count_loss_weight
    ):
        """
        Returns the loss of the correctness off the predicted profiles and
        predicted read counts. This prediction correctness loss is split into a
        profile loss and a count loss. The profile loss is the -log probability
        of seeing the true profile read counts, given the multinomial
        distribution defined by the predicted profile count probabilities. The
        count loss is a simple mean squared error on the log counts.
        Arguments:
            `true_profs`: a B x T x 2 x O tensor containing true UNnormalized
                profile values, where B is the batch size, T is the number of
                tasks, and O is the profile length; the sum of a profile gives
                the raw read count for that task
            `logit_pred_profs`: a B x T x 2 x O tensor containing the predicted
                profile _logits_
            `log_pred_counts`: a B x T x 2 tensor containing the predicted log
                read counts
            `count_loss_weight`: amount to weight the portion of the loss for
                the counts
        Returns a scalar loss tensor.
        """
        assert true_profs.size() == logit_pred_profs.size()
        batch_size = true_profs.size(0)
        num_tasks = true_profs.size(1)

        # Reshape the inputs to be flat along the tasks dimension
        true_profs = true_profs.view(batch_size, num_tasks * 2, -1)
        logit_pred_profs = logit_pred_profs.view(batch_size, num_tasks * 2, -1)
        log_pred_counts = log_pred_counts.view(batch_size, num_tasks * 2)

        # Add the profiles together to get the raw counts
        true_counts = torch.sum(true_profs, dim=2)

        # 1. Profile loss
        # Compute the log probabilities based on multinomial distributions,
        # each one is based on predicted probabilities, one for each track

        # Convert logits to log probabilities and normalize
        sig_pred_profs = self.sigmoid(logit_pred_profs)
        sig_sums = torch.sum(sig_pred_profs, dim=-1).unsqueeze(-1).repeat(
            1, 1, sig_pred_profs.size(-1)
        )
        norm_sig_pred_profs = torch.div(sig_pred_profs, sig_sums)
        log_pred_profs = torch.log(norm_sig_pred_profs)  # Log probs

        # Compute probability of seeing true profile under distribution of log
        # predicted probs
        log_probs = multinomial_log_probs(
            log_pred_profs, true_counts, true_profs
        )
        batch_prof_loss = torch.mean(-log_probs, dim=1)  # Average across tasks
        prof_loss = torch.mean(batch_prof_loss)  # Average across batch

        # 2. Counts loss
        # Mean squared error on the log counts (with 1 added for stability)
        log_true_counts = torch.log(true_counts + 1)

        mse = self.mse_loss(log_pred_counts, log_true_counts)
        batch_count_loss = torch.mean(mse, dim=1)  # Average acorss tasks
        count_loss = torch.mean(batch_count_loss)  # average across batch

        return prof_loss + (count_loss_weight * count_loss)
