import torch
import math
import numpy as np
from model.util import sanitize_sacred_arguments, convolution_size, \
    place_tensor, smooth_tensor_1d
import scipy.special

def multinomial_log_probs(category_log_probs, trials, query_counts):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    Arguments:
        `category_log_probs`: a D x N tensor containing log probabilities (base
            e) of seeing each of the N classes/categories
        `trials`: a D-tensor containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N tensor containing the observed count of each
            category in each distribution; the probability is computed for these
            observations
    Returns a D-tensor containing the log probabilities (base e) of each
    observed query with its corresponding distribution. Note that D can be
    replaced with any shape (i.e. only the last dimension is reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    trials, query_counts = trials.float(), query_counts.float()
    log_n_fact = torch.lgamma(trials + 1)
    log_counts_fact = torch.lgamma(query_counts + 1)
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

        # ReLU activation for the convolutional layers and attribution prior
        self.relu = torch.nn.ReLU()

        # Profile prediction:
        # Convolutional layer with large kernel
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

        # MSE Loss for counts
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def forward(self, input_seqs, cont_profs):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is
                the input sequence length, and D is the number of input channels
            `cont_profs`: a B x T x O x 2 tensor, where T is the number of
                tasks, and O is the output sequence length
        Returns the predicted profiles (unnormalized logits) for each task (both
        plus and minus strands) (a B x T x O x 2 tensor), and the predicted log
        counts (base e) for each task (both plus and minus strands)
        (a B x T x 2) tensor.
        """
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(1)
        assert input_length == self.input_length
        num_tasks = cont_profs.size(1)
        assert num_tasks == self.num_tasks
        profile_length = cont_profs.size(2)
        assert profile_length == self.profile_length

        # PyTorch prefers convolutions to be channel first, so transpose the
        # input and control profiles
        input_seqs = input_seqs.transpose(1, 2)  # Shape: B x D x I
        cont_profs = cont_profs.transpose(2, 3)  # Shape: B x T x 2 x O

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
        # Transpose profile predictions to get B x T x O x 2
        prof_pred = prof_pred.transpose(2, 3)
        
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
        self, true_profs, logit_pred_profs, log_pred_counts, count_loss_weight,
        return_separate_losses=False
    ):
        """
        Returns the loss of the correctness off the predicted profiles and
        predicted read counts. This prediction correctness loss is split into a
        profile loss and a count loss. The profile loss is the -log probability
        of seeing the true profile read counts, given the multinomial
        distribution defined by the predicted profile count probabilities. The
        count loss is a simple mean squared error on the log counts.
        Arguments:
            `true_profs`: a B x T x O x 2 tensor containing true UNnormalized
                profile values, where B is the batch size, T is the number of
                tasks, and O is the profile length; the sum of a profile gives
                the raw read count for that task
            `logit_pred_profs`: a B x T x O x 2 tensor containing the predicted
                profile _logits_
            `log_pred_counts`: a B x T x 2 tensor containing the predicted log
                read counts (base e)
            `count_loss_weight`: amount to weight the portion of the loss for
                the counts
            `return_separate_losses`: if True, also return the profile and
                counts losses (scalar Tensors)
        Returns a scalar loss tensor, or perhaps 3 scalar loss tensors.
        """
        assert true_profs.size() == logit_pred_profs.size()
        batch_size = true_profs.size(0)
        num_tasks = true_profs.size(1)

        # Add the profiles together to get the raw counts
        true_counts = torch.sum(true_profs, dim=2)  # Shape: B x T x 2

        # Transpose and reshape the profile inputs from B x T x O x 2 to
        # B x 2T x O; all metrics will be computed for each individual profile,
        # then averaged across pooled tasks/strands, then across the batch
        true_profs = true_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * 2, -1
        )
        logit_pred_profs = logit_pred_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * 2, -1
        )
        # Reshape the counts from B x T x 2 to B x 2T
        true_counts = true_counts.view(batch_size, num_tasks * 2)
        log_pred_counts = log_pred_counts.view(batch_size, num_tasks * 2)

        # 1. Profile loss
        # Compute the log probabilities based on multinomial distributions,
        # each one is based on predicted probabilities, one for each track

        # Convert logits to log probabilities (along the O dimension)
        log_pred_profs = profile_logits_to_log_probs(logit_pred_profs, axis=2)

        # Compute probability of seeing true profile under distribution of log
        # predicted probs
        neg_log_likelihood = -multinomial_log_probs(
            log_pred_profs, true_counts, true_profs
        )  # Shape: B x 2T
        # Average across tasks/strands, and then across the batch
        batch_prof_loss = torch.mean(neg_log_likelihood, dim=1)
        prof_loss = torch.mean(batch_prof_loss)

        # 2. Counts loss
        # Mean squared error on the log counts (with 1 added for stability)
        log_true_counts = torch.log(true_counts + 1)
        mse = self.mse_loss(log_pred_counts, log_true_counts)

        # Average across tasks/strands, and then across the batch
        batch_count_loss = torch.mean(mse, dim=1)
        count_loss = torch.mean(batch_count_loss)

        final_loss = prof_loss + (count_loss_weight * count_loss)

        if return_separate_losses:
            return final_loss, prof_loss, count_loss
        else:
            return final_loss

    def att_prior_loss(
        self, status, input_grads, pos_limit, pos_weight,
        att_prior_grad_smooth_sigma, return_separate_losses=False
    ):
        """
        Computes an attribution prior loss for some given training examples.
        Arguments:
            `status`: a B-tensor, where B is the batch size; each entry is 1 if
                that example is to be treated as a positive example, and 0
                otherwise
            `input_grads`: a B x L x D tensor, where B is the batch size, L is
                the length of the input, and D is the dimensionality of each
                input base; this needs to be the gradients of the input with
                respect to the output (for multiple tasks, this gradient needs
                to be aggregated); this should be *gradient times input*
            `pos_limit`: the maximum integer frequency index, k, to consider for
                the positive loss; this corresponds to a frequency cut-off of
                pi * k / L; k should be less than L / 2
            `pos_weight`: the amount to weight the positive loss by, to give it
                a similar scale as the negative loss
            `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
                computing the loss
            `return_separate_losses`: if True, also return the positive and
                negative losses (scalar Tensors)
        Returns a single scalar Tensor consisting of the attribution loss for
        the batch, perhaps with the positive and negative losses (scalars), too.
        """
        max_rect_grads = torch.max(self.relu(input_grads), dim=2)[0]
        # Smooth the gradients
        max_rect_grads_smooth = smooth_tensor_1d(
            max_rect_grads, att_prior_grad_smooth_sigma
        )

        neg_grads = max_rect_grads_smooth[status == 0]
        pos_grads = max_rect_grads_smooth[status == 1]

        # Loss for positives
        if pos_grads.nelement():
            pos_grads_complex = torch.stack(
                [pos_grads, place_tensor(torch.zeros(pos_grads.size()))], dim=2
            )  # Convert to complex number format: a -> a + 0i
            # Magnitude of the Fourier coefficients, normalized:
            pos_fft = torch.fft(pos_grads_complex, 1)
            pos_mags = torch.norm(pos_fft, dim=2)
            pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
            pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
            pos_mags = pos_mags / pos_mag_sum
            # Cut off DC and high-frequency components:
            pos_mags = pos_mags[:, 1:pos_limit]
            pos_score = torch.sum(pos_mags, dim=1)
            pos_loss = 1 - pos_score
            pos_loss_mean = torch.mean(pos_loss)
        else:
            pos_loss_mean = torch.zeros(1)
        # Loss for negatives
        if neg_grads.nelement():
            neg_loss = torch.sum(neg_grads, dim=1)
            neg_loss_mean = torch.mean(neg_loss)
        else:
            neg_loss_mean = torch.zeros(1)

        final_loss = (pos_weight * pos_loss_mean) + neg_loss_mean

        if return_separate_losses:
            return final_loss, pos_loss_mean, neg_loss_mean
        else:
            return final_loss


def profile_logits_to_log_probs(logit_pred_profs, axis=2):
    """
    Converts the model's predicted profile logits into normalized probabilities
    via a softmax on the specified dimension (defaults to axis=2).
    Arguments:
        `logit_pred_profs`: a tensor/array containing the predicted profile
            logits
    Returns a tensor/array of the same shape, containing the predicted profiles
    as log probabilities by doing a log softmax on the specified dimension. If
    the input is a tensor, the output will be a tensor. If the input is a NumPy
    array, the output will be a NumPy array. Note that the  reason why this
    function returns log probabilities rather than raw probabilities is for
    numerical stability.
    """
    if type(logit_pred_profs) is np.ndarray:
        return logit_pred_profs - \
            scipy.special.logsumexp(logit_pred_profs, axis=axis, keepdims=True)
    else:
        return torch.log_softmax(logit_pred_profs, dim=axis)
