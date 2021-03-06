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


class ProfilePredictor(torch.nn.Module):
    def __init__(self):
        """
        Superclass containing loss functions for profile predictors.
        """
        super().__init__()

        # MSE Loss for counts
        self.mse_loss = torch.nn.MSELoss(reduction="none")

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
            `true_profs`: a B x T x O x S tensor containing true UNnormalized
                profile values, where B is the batch size, T is the number of
                tasks, O is the profile length, and S is the number of strands;
                the sum of a profile gives the raw read count for that task
            `logit_pred_profs`: a B x T x O x S tensor containing the predicted
                profile _logits_
            `log_pred_counts`: a B x T x S tensor containing the predicted log
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
        num_strands = true_profs.size(3)

        # Add the profiles together to get the raw counts
        true_counts = torch.sum(true_profs, dim=2)  # Shape: B x T x 2

        # Transpose and reshape the profile inputs from B x T x O x S to
        # B x ST x O; all metrics will be computed for each individual profile,
        # then averaged across pooled tasks/strands, then across the batch
        true_profs = true_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * num_strands, -1
        )
        logit_pred_profs = logit_pred_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * num_strands, -1
        )
        # Reshape the counts from B x T x S to B x ST
        true_counts = true_counts.view(batch_size, num_tasks * num_strands)
        log_pred_counts = log_pred_counts.view(
            batch_size, num_tasks * num_strands
        )

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

    def fourier_att_prior_loss(
        self, status, input_grads, freq_limit, limit_softness,
        att_prior_grad_smooth_sigma
    ):
        """
        Computes an attribution prior loss for some given training examples,
        using a Fourier transform form.
        Arguments:
            `status`: a B-tensor, where B is the batch size; each entry is 1 if
                that example is to be treated as a positive example, and 0
                otherwise
            `input_grads`: a B x L x D tensor, where B is the batch size, L is
                the length of the input, and D is the dimensionality of each
                input base; this needs to be the gradients of the input with
                respect to the output (for multiple tasks, this gradient needs
                to be aggregated); this should be *gradient times input*
            `freq_limit`: the maximum integer frequency index, k, to consider
                for the loss; this corresponds to a frequency cut-off of
                pi * k / L; k should be less than L / 2
            `limit_softness`: amount to soften the limit by, using a hill
                function; None means no softness
            `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
                computing the loss
        Returns a single scalar Tensor consisting of the attribution loss for
        the batch.
        """
        abs_grads = torch.sum(torch.abs(input_grads), dim=2)

        # Smooth the gradients
        grads_smooth = smooth_tensor_1d(
            abs_grads, att_prior_grad_smooth_sigma
        )

        # Only do the positives
        pos_grads = grads_smooth[status == 1]

        # Loss for positives
        if pos_grads.nelement():
            pos_fft = torch.rfft(pos_grads, 1)
            pos_mags = torch.norm(pos_fft, dim=2)
            pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
            pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
            pos_mags = pos_mags / pos_mag_sum

            # Cut off DC
            pos_mags = pos_mags[:, 1:]

            # Construct weight vector
            weights = place_tensor(torch.ones_like(pos_mags))
            if limit_softness is None:
                weights[:, freq_limit:] = 0
            else:
                x = place_tensor(
                    torch.arange(1, pos_mags.size(1) - freq_limit + 1)
                ).float()
                weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

            # Multiply frequency magnitudes by weights
            pos_weighted_mags = pos_mags * weights

            # Add up along frequency axis to get score
            pos_score = torch.sum(pos_weighted_mags, dim=1)
            pos_loss = 1 - pos_score
            return torch.mean(pos_loss)
        else:
            return place_tensor(torch.zeros(1))


class ProfilePredictorWithControls(ProfilePredictor):
    def __init__(
        self, input_length, input_depth, profile_length, num_tasks, num_strands,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
        prof_conv_stride, share_controls
    ):
        """
        Creates a profile predictor from a DNA sequence, using control profiles.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `profile_length`: length of the predicted profiles; it must be
                consistent with the convolutional layers specified
            `num_tasks`: number of tasks that are to be predicted; there will be
                two profiles and two read counts predicted for each task
            `num_strands`: number of strands for each profile, typically 1 or 2
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
            `share_controls`: if True, expect a single set of shared controls;
                otherwise, each task needs a matched control

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
        self.num_strands = num_strands
        self.num_dil_conv_layers = num_dil_conv_layers
        self.share_controls = share_controls
        
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
            in_channels=dil_conv_depths[-1],
            out_channels=(num_tasks * num_strands),
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
            in_channels=(num_tasks * 2 * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks  # One set of filters over each task
        )
        
        # Counts prediction:
        # Global average pooling
        self.count_pool = torch.nn.AvgPool1d(
            kernel_size=self.last_dil_conv_size
        )

        # Dense layer to consolidate pooled result to small number of features
        self.count_dense = torch.nn.Linear(
            in_features=dil_conv_depths[-1],
            out_features=(num_tasks * num_strands)
        )

        # Dense layer over pooling features and controls to get the final
        # counts, implemented as grouped convolution with kernel size 1
        self.count_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * 2 * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks
        )


    def forward(self, input_seqs, cont_profs):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is
                the input sequence length, and D is the number of input channels
            `cont_profs`: if `share_controls` is True, this must be a
                B x 1 x O x S tensor, where O is the output sequence length and
                S is the number of strands; otherwise, controls are matched, and
                this must be a B x T x O x S tensor, where T is the number of
                tasks
        Returns the predicted profiles (unnormalized logits) for each task and
        each strand (a B x T x O x S tensor), and the predicted log counts (base
        e) for each task and each strand (a B x T x S) tensor.
        """
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(1)
        assert input_length == self.input_length
        if self.share_controls:
            assert cont_profs.size(1) == 1
        else:
            assert cont_profs.size(1) == self.num_tasks
        profile_length = cont_profs.size(2)
        assert profile_length == self.profile_length
        num_strands = cont_profs.size(3)
        assert num_strands == self.num_strands

        # PyTorch prefers convolutions to be channel first, so transpose the
        # input and control profiles
        input_seqs = input_seqs.transpose(1, 2)  # Shape: B x D x I
        cont_profs = cont_profs.transpose(2, 3)  # Shape: B x T x S x O

        # Prepare the control tracks: profiles and counts
        cont_counts = torch.sum(cont_profs, dim=3)  # Shape: B x T x 2
        if self.share_controls:
            # Replicate the the profiles/counts from B x 1 x 2 x O/B x 1 x 2 to
            # B x T x 2 x O/B x T x 2
            cont_profs = torch.cat([cont_profs] * self.num_tasks, dim=1)
            cont_counts = torch.cat([cont_counts] * self.num_tasks, dim=1)

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
        # Shape: B x ST x O

        # A2. Concatenate with the control profiles
        # Reshaping is necessary to ensure the tasks are paired adjacently
        prof_large_conv_out = prof_large_conv_out.view(
            batch_size, self.num_tasks, num_strands, -1
        )
        prof_with_cont = torch.cat([prof_large_conv_out, cont_profs], dim=2)
        # Shape: B x T x 2S x O
        prof_with_cont = prof_with_cont.view(
            batch_size, self.num_tasks * 2 * num_strands, -1
        )

        # A3. Perform length-1 convolutions over the concatenated profiles with
        # controls; there are T convolutions, each one is done over one pair of
        # prof_first_conv_out, and a pair of controls
        prof_one_conv_out = self.prof_one_conv(prof_with_cont)
        # Shape: B x ST x O
        prof_pred = prof_one_conv_out.view(
            batch_size, self.num_tasks, num_strands, -1
        )
        # Transpose profile predictions to get B x T x O x S
        prof_pred = prof_pred.transpose(2, 3)
        
        # Branch B: read count prediction
        # B1. Global average pooling across the output of dilated convolutions
        count_pool_out = self.count_pool(dil_conv_out_cut)  # Shape: B x X x 1
        count_pool_out = torch.squeeze(count_pool_out, dim=2)

        # B2. Reduce pooling output to fewer features, a pair for each task
        count_dense_out = self.count_dense(count_pool_out)  # Shape: B x ST

        # B3. Concatenate with the control counts
        # Reshaping is necessary to ensure the tasks are paired adjacently
        count_dense_out = count_dense_out.view(
            batch_size, self.num_tasks, num_strands
        )
        count_with_cont = torch.cat([count_dense_out, cont_counts], dim=2)
        # Shape: B x T x 2S
        count_with_cont = count_with_cont.view(
            batch_size, self.num_tasks * 2 * num_strands, -1
        )  # Shape: B x 2ST x 1

        # B4. Dense layer over the concatenation with control counts; each set
        # of counts gets a different dense network (implemented as convolution
        # with kernel size 1)
        count_one_conv_out = self.count_one_conv(count_with_cont)
        # Shape: B x ST x 1
        count_pred = count_one_conv_out.view(
            batch_size, self.num_tasks, num_strands, -1
        )
        # Shape: B x T x S x 1
        count_pred = torch.squeeze(count_pred, dim=3)  # Shape: B x T x S

        return prof_pred, count_pred


class ProfilePredictorWithMatchedControls(ProfilePredictorWithControls):
    """
    Wrapper class for `ProfilePredictorWithControls`.
    """
    def __init__(
        self, input_length, input_depth, profile_length, num_tasks, num_strands,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
        prof_conv_stride, share_controls=False  # <- for compatibility
    ):
        super().__init__(
            input_length, input_depth, profile_length, num_tasks, num_strands,
            num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
            dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
            prof_conv_stride, share_controls=False
        )


class ProfilePredictorWithSharedControls(ProfilePredictorWithControls):
    """
    Wrapper class for `ProfilePredictorWithControls`.
    """
    def __init__(
        self, input_length, input_depth, profile_length, num_tasks, num_strands,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
        prof_conv_stride, share_controls=True  # <- for compatibility
    ):
        super().__init__(
            input_length, input_depth, profile_length, num_tasks, num_strands,
            num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
            dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
            prof_conv_stride, share_controls=True
        )


class ProfilePredictorWithoutControls(ProfilePredictor):
    def __init__(
        self, input_length, input_depth, profile_length, num_tasks, num_strands,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
        prof_conv_stride
    ):
        """
        Creates a profile predictor from a DNA sequence that does not take
        control profiles.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `profile_length`: length of the predicted profiles; it must be
                consistent with the convolutional layers specified
            `num_tasks`: number of tasks that are to be predicted; there will be
                two profiles and two read counts predicted for each task
            `num_strands`: number of strands for each profile, typically 1 or 2
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
        self.num_strands = num_strands
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
            in_channels=dil_conv_depths[-1],
            out_channels=(num_tasks * num_strands),
            kernel_size=prof_conv_kernel_size
        )

        self.prof_pred_size = self.last_dil_conv_size - \
            (prof_conv_kernel_size - 1)

        assert self.prof_pred_size == profile_length, \
            "Prediction length is specified to be %d, but with the given " +\
            "input length of %d and the given convolutions, the computed " +\
            "prediction length is %d" % \
            (profile_length, input_length, self.prof_pred_size)

        # Length-1 convolution over the convolutional output to get the final
        # profile
        self.prof_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks  # One set of filters over each task
        )
        
        # Counts prediction:
        # Global average pooling
        self.count_pool = torch.nn.AvgPool1d(
            kernel_size=self.last_dil_conv_size
        )

        # Dense layer to consolidate pooled result to small number of features
        self.count_dense = torch.nn.Linear(
            in_features=dil_conv_depths[-1],
            out_features=(num_tasks * num_strands)
        )

        # Dense layer over pooling features to get the final counts, implemented
        # as grouped convolution with kernel size 1
        self.count_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks
        )
 
    def forward(self, input_seqs, cont_profs=None):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is
                the input sequence length, and D is the number of input channels
            `cont_profs`: unused parameter, existing only for compatibility
        Returns the predicted profiles (unnormalized logits) for each task and
        each strand (a B x T x O x S tensor), and the predicted log
        counts (base e) for each task and each strand (a B x T x S) tensor.
        """
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(1)
        assert input_length == self.input_length

        # PyTorch prefers convolutions to be channel first, so transpose the
        # input
        input_seqs = input_seqs.transpose(1, 2)  # Shape: B x D x I

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
        # Shape: B x ST x O

        # A2. Perform length-1 convolutions over the profiles; there are T
        # convolutions, each one is done over one set of prof_large_conv_out
        prof_one_conv_out = self.prof_one_conv(prof_large_conv_out)
        # Shape: B x ST x O
        prof_pred = prof_one_conv_out.view(
            batch_size, self.num_tasks, self.num_strands, -1
        )
        # Transpose profile predictions to get B x T x O x S
        prof_pred = prof_pred.transpose(2, 3)
        
        # Branch B: read count prediction
        # B1. Global average pooling across the output of dilated convolutions
        count_pool_out = self.count_pool(dil_conv_out_cut)  # Shape: B x X x 1
        count_pool_out = torch.squeeze(count_pool_out, dim=2)

        # B2. Reduce pooling output to fewer features, a pair for each task
        count_dense_out = self.count_dense(count_pool_out)  # Shape: B x ST
        count_dense_out = count_dense_out.view(
            batch_size, self.num_strands * self.num_tasks, 1
        )

        # B3. Dense layer over the last layer's outputs; each set of counts gets
        # a different dense network (implemented as convolution with kernel size
        # 1)
        count_one_conv_out = self.count_one_conv(count_dense_out)
        # Shape: B x ST x 1
        count_pred = count_one_conv_out.view(
            batch_size, self.num_tasks, self.num_strands, -1
        )
        # Shape: B x T x S x 1
        count_pred = torch.squeeze(count_pred, dim=3)  # Shape: B x T x S

        return prof_pred, count_pred
    

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
