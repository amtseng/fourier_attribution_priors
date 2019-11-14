import numpy as np
import scipy.special
import sacred
from datetime import datetime

performance_ex = sacred.Experiment("performance")

@performance_ex.config
def config():
    # Bin sizes to try for count correlation
    prof_count_corr_bin_sizes = [1, 4, 10]

    # Bin sizes to try for computing auPRC
    auprc_bin_sizes = [1, 4, 10]

    # Minimum probability in true profile to be a positive for auPRC
    auprc_min_pos_prob = 0.15

    # Minimum number of reads in true profile to be a positive for auPRC
    auprc_min_pos_count = 3

    # Maximum probability in true profile to be a negative for auPRC
    auprc_max_neg_prob = 0.05


def multinomial_log_probs(category_log_probs, trials, query_counts):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    Arguments:
        `category_log_probs`: a D x N array containing log probabilities (base
            e) of seeing each of the N classes/categories
        `trials`: a D-array containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N array containing the observed count of each
            category in each distribution; the probability is computed for these
            observations
    Returns a D-array containing the log probabilities (base e) of each observed
    query with its corresponding distribution. Note that D can be replaced with
    any shape (i.e. only the last dimension is reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    log_n_fact = scipy.special.gammaln(trials + 1)
    log_counts_fact = scipy.special.gammaln(query_counts + 1)
    log_counts_fact_sum = np.sum(log_counts_fact, axis=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise
    log_prob_pows_sum = np.sum(log_prob_pows, axis=-1)

    return log_n_fact - log_counts_fact_sum + log_prob_pows_sum


def profile_multinomial_nll(true_profs, log_pred_profs, true_counts):
    """
    Computes the negative log likelihood of seeing the true profile, given the
    probabilities specified by the predicted profile. The NLL is computed
    separately for each sample, task, and strand, but the results are averaged
    across the strands.
    Arguments:
        `true_profs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `log_pred_profs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities
        `true_counts`: a N x T x 2 array, containing the true total counts
            for each task and strand
    Returns an N x T array, containing the strand-pooled multinomial NLL for
    each sample and task.
    """
    num_samples = true_profs.shape[0]
    num_tasks = true_profs.shape[1]

    # Swap axes on profiles to make them N x T x 2 x O
    true_profs = np.swapaxes(true_profs, 2, 3)
    log_pred_profs = np.swapaxes(log_pred_profs, 2, 3)

    nll = -multinomial_log_probs(log_pred_profs, true_counts, true_profs)
    return np.mean(nll, axis=2)  # Average strands


def _kl_divergence(probs1, probs2):
    """
    Computes the KL divergence in the last dimension of `probs1` and `probs2`
    as KL(P1 || P2). `probs1` and `probs2` must be the same shape. For example,
    if they are both A x B x L arrays, then the KL divergence of corresponding
    L-arrays will be computed and returned in an A x B array. Does not
    renormalize the arrays. If probs2[i] is 0, that value contributes 0.
    """
    quot = np.divide(
        probs1, probs2, out=np.ones_like(probs1),
        where=((probs1 != 0) & (probs2 != 0))
        # No contribution if P1 = 0 or P2 = 0
    )
    return np.sum(probs1 * np.log(quot), axis=-1)


def jensen_shannon_distance(probs1, probs2):
    """
    Computes the Jesnsen-Shannon distance in the last dimension of `probs1` and
    `probs2`. `probs1` and `probs2` must be the same shape. For example, if they
    are both A x B x L arrays, then the KL divergence of corresponding L-arrays
    will be computed and returned in an A x B array. This will renormalize the
    arrays so that each subarray sums to 1. If the sum of a subarray is 0, then
    the resulting JSD will be NaN.
    """
    # Renormalize both distributions, and if the sum is NaN, put NaNs all around
    probs1_sum = np.sum(probs1, axis=-1, keepdims=True)
    probs1 = np.divide(
        probs1, probs1_sum, out=np.full_like(probs1, np.nan),
        where=(probs1_sum != 0)
    )
    probs2_sum = np.sum(probs2, axis=-1, keepdims=True)
    probs2 = np.divide(
        probs2, probs2_sum, out=np.full_like(probs2, np.nan),
        where=(probs2_sum != 0)
    )

    mid = 0.5 * (probs1 + probs2)
    return 0.5 * (_kl_divergence(probs1, mid) + _kl_divergence(probs2, mid))


def profile_jsd(true_prof_probs, pred_prof_probs):
    """
    Computes the Jensen-Shannon divergence of the true and predicted profiles
    given their raw probabilities or counts. The inputs will be renormalized
    prior to JSD computation, so providing either raw probabilities or counts
    is sufficient.
    Arguments:
        `true_prof_probs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, O is the output profile length;
            contains the true profiles for each task and strand, as RAW
            PROBABILITIES or RAW COUNTS
        `pred_prof_probs`: N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as RAW PROBABILITIES or RAW
            COUNTS
    Returns an N x T array, where the JSD is computed across the profiles and
    averaged between the strands, for each sample/task.
    """
    # Transpose to N x T x 2 x O, so JSD is computed along last dimension
    true_prof_swap = np.swapaxes(true_prof_probs, 2, 3)
    pred_prof_swap = np.swapaxes(pred_prof_probs, 2, 3)
    jsd = jensen_shannon_distance(true_prof_swap, pred_prof_swap)
    return np.mean(jsd, axis=-1)  # Average over strands


def bin_array_max(arr, bin_size, pad_value=0):
    """
    Given a NumPy array, returns a binned version of the array along the last
    dimension, where each bin contains the maximum value of its constituent
    elements. If the array is not a length that is a multiple of the bin size,
    then the given pad will be used at the end.
    """
    num_bins = int(np.ceil(arr.shape[-1] / bin_size))
    pad_amount = (num_bins * bin_size) - arr.shape[-1]
    if pad_amount:
        arr = np.pad(
            arr, ([(0, 0)] * (arr.ndim - 1)) + [(0, pad_amount)],
            constant_values=pad_value
        )
    new_shape = arr.shape[:-1] + (num_bins, bin_size)
    return np.max(np.reshape(arr, new_shape), axis=-1)


def auprc_score(true_vals, pred_vals):
    """
    Computes the auPRC in the last dimension of `arr1` and `arr2`. `arr1` and
    `arr2` must be the same shape. For example, if they are both A x B x L
    arrays, then the auPRC of corresponding L-arrays will be computed and
    returned in an A x B array. `true_vals` should contain binary values; any
    values other than 0 or 1 will be ignored when computing auPRC. `pred_vals`
    should contain prediction values in the range [0, 1]. The behavior of this
    function is meant to match `sklearn.metrics.average_precision_score` in its
    calculation with regards to thresholding. If there are no true positives,
    the auPRC returned will be NaN.
    """
    # Sort true and predicted values in descending order
    sorted_inds = np.flip(np.argsort(pred_vals, axis=-1), axis=-1)
    pred_vals = np.take_along_axis(pred_vals, sorted_inds, -1)
    true_vals = np.take_along_axis(true_vals, sorted_inds, -1)

    # Compute the indices where a run of identical predicted values stops
    # In `thresh_inds`, there is a 1 wherever a run ends, and 0 otherwise
    diff = np.diff(pred_vals, axis=-1)
    diff[diff != 0] = 1  # Assign 1 to every nonzero diff
    thresh_inds = np.pad(
        diff, ([(0, 0)] * (diff.ndim - 1)) + [(0, 1)], constant_values=1
    ).astype(int)
    thresh_mask = thresh_inds == 1

    # Compute true positives and false positives at each location; this will
    # eventually be subsetted to only the threshold indices
    # Assign a weight of zero wherever the true value is not binary
    weight_mask = (true_vals == 0) | (true_vals == 1)
    true_pos = np.cumsum(true_vals * weight_mask, axis=-1)
    false_pos = np.cumsum((1 - true_vals) * weight_mask, axis=-1)

    # Compute precision array, but keep 0s wherever there isn't a threshold
    # index
    precis_denom = true_pos + false_pos
    precis = np.divide(
        true_pos, precis_denom,
        out=np.zeros(true_pos.shape),
        where=((precis_denom != 0) & thresh_mask)
    )

    # Compute recall array, but if there are no true positives, it's nan for the
    # entire subarray
    recall_denom = true_pos[..., -1:]
    recall = np.divide(
        true_pos, recall_denom,
        out=np.full(true_pos.shape, np.nan),
        where=(recall_denom != 0)
    )

    # Concatenate an initial value of 0 for recall; adjust `thresh_inds`, too
    thresh_inds = np.pad(
        thresh_inds, ([(0, 0)] * (thresh_inds.ndim - 1)) + [(1, 0)],
        constant_values=1
    )
    recall = np.pad(
        recall, ([(0, 0)] * (recall.ndim - 1)) + [(1, 0)], constant_values=0
    )
    # Concatenate an initial value of 1 for precision; technically, this initial
    # value won't be used for auPRC calculation, but it will be easier for later
    # steps to do this anyway
    precis = np.pad(
        precis, ([(0, 0)] * (precis.ndim - 1)) + [(1, 0)], constant_values=1
    )

    # We want the difference of the recalls, but only in buckets marked by
    # threshold indices; since the number of buckets can be different for each
    # subarray, we create a set of bucketed recalls and precisions for each
    # Each entry in `thresh_buckets` is an index mapping the thresholds to
    # consecutive buckets
    thresh_buckets = np.cumsum(thresh_inds, axis=-1) - 1
    # Set unused buckets to -1; won't happen if there are no unused buckets
    thresh_buckets[thresh_inds == 0] = -1
    # Place the recall values into the buckets into consecutive locations; any
    # unused recall values get placed (and may clobber) the last index
    recall_buckets = np.zeros_like(recall)
    np.put_along_axis(recall_buckets, thresh_buckets, recall, -1)
    # Do the same for precision
    precis_buckets = np.zeros_like(precis)
    np.put_along_axis(precis_buckets, thresh_buckets, precis, -1)

    # Compute the auPRC/average precision by computing the recall bucket diffs
    # and weighting by bucketed precision; note that when `precis` was made,
    # it is 0 wherever there is no threshold index, so all locations in
    # `precis_buckets` which aren't used (even the last index) have a 0
    recall_diffs = np.diff(recall_buckets, axis=-1)
    return np.sum(recall_diffs * precis_buckets[..., 1:], axis=-1)


@performance_ex.capture
def binned_profile_auprc(
    true_prof_counts, pred_prof_probs, true_total_counts, auprc_bin_sizes,
    auprc_min_pos_prob, auprc_min_pos_count, auprc_max_neg_prob,
    batch_size=50000
):
    """
    Binarizes the profile and computes auPRC for different bin sizes.
    Arguments:
        `true_prof_counts`: B x T x O x 2 array, where B is the batch size, T is
            the number of tasks, and O is the output profile length; contains
            the true profiles for each for each task and strand, as RAW COUNTS
        `pred_prof_probs`: a B x T x O x 2 array, containing the predicted
            profiles for each task and strand, as RAW PROBABILITIES
        `true_total_counts`: a B x T x 2 array, containing the true total counts
            for each task and strand
        `batch_size`: performs computation in a batch size of this many samples
    Returns an N x T x Z x 4 array containing the auPRCs for each sample and
    task, where each auPRC is computed across both strands (pooled), for each
    sample and task. This is done for every bin size in `auprc_bin_sizes`. For
    each sample, task, and bin size, the auPRC is reported, along with the
    number of positive, ambiguous, and negative positions.
    """
    # Normalize true profile (counts) into probabilities, keeping 0s when 0
    true_total_counts_dim = np.expand_dims(true_total_counts, axis=2)
    true_prof_probs = np.divide(
        true_prof_counts, true_total_counts_dim,
        out=np.zeros_like(true_prof_counts),
        where=(true_total_counts_dim != 0)
    )

    num_samples, num_tasks = true_total_counts.shape[:2]

    # Combine the profile length and strand dimensions (i.e. pool strands)
    new_shape = (num_samples, num_tasks, -1)
    true_prof_probs_flat = np.reshape(true_prof_probs, new_shape)
    true_prof_counts_flat = np.reshape(true_prof_counts, new_shape)
    pred_prof_probs_flat = np.reshape(pred_prof_probs, new_shape)
  
    result = np.zeros((num_samples, num_tasks, len(auprc_bin_sizes), 4))
    for i, bin_size in enumerate(auprc_bin_sizes):
        # Bin the values, taking the maximum for each bin
        true_prob_bins = bin_array_max(true_prof_probs_flat, bin_size)
        true_count_bins = bin_array_max(true_prof_counts_flat, bin_size)
        pred_prob_bins = bin_array_max(pred_prof_probs_flat, bin_size)

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            true_prob_batch = true_prob_bins[start:end, :, :]
            true_count_batch = true_count_bins[start:end, :, :]
            pred_prob_batch = pred_prob_bins[start:end, :, :]

            # Filter for the positives and negatives
            # A bin is positive if the maximum count inside it is at least a
            # minimum number of reads, and the maximum probability inside is at
            # least the minimum fraction of total reads
            pos_mask = (true_count_batch >= auprc_min_pos_count) & \
                (true_prob_batch >= auprc_min_pos_prob)
            neg_mask = true_prob_batch <= auprc_max_neg_prob

            num_pos = np.sum(pos_mask, axis=-1)
            num_neg = np.sum(neg_mask, axis=-1)
            num_ambi = true_prob_bins.shape[-1] - num_pos - num_neg

            true_vals = np.full(true_count_batch.shape, -1)
            true_vals[pos_mask] = 1
            true_vals[neg_mask] = 0
            pred_vals = pred_prob_batch
            auprc = auprc_score(true_vals, pred_vals)

            result[start:end, :, i, 0] = auprc
            result[start:end, :, i, 1] = num_pos
            result[start:end, :, i, 2] = num_ambi
            result[start:end, :, i, 3] = num_neg

    return result


def pearson_corr(arr1, arr2):
    """
    Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array.
    """
    mean1 = np.mean(arr1, axis=-1, keepdims=True)
    mean2 = np.mean(arr2, axis=-1, keepdims=True)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom = np.sqrt(var1 * var2)
   
    # Divide numerator by denominator, but use NaN where the denominator is 0
    return np.divide(
        numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0)
    )


def average_ranks(arr):
    """
    Computes the ranks of the elemtns of the given array along the last
    dimension. For ties, the ranks are _averaged_.
    Returns an array of the same dimension of `arr`. 
    """
    # 1) Generate the ranks for each subarray, with ties broken arbitrarily
    sorted_inds = np.argsort(arr, axis=-1)  # Sorted indices
    ranks, ranges = np.empty_like(arr), np.empty_like(arr)
    ranges = np.tile(np.arange(arr.shape[-1]), arr.shape[:-1] + (1,))
    # Put ranks by sorted indices; this creates an array containing the ranks of
    # the elements in each subarray of `arr`
    np.put_along_axis(ranks, sorted_inds, ranges, -1)
    ranks = ranks.astype(int)

    # 2) Create an array where each entry maps a UNIQUE element in `arr` to a
    # unique index for that subarray
    sorted_arr = np.take_along_axis(arr, sorted_inds, axis=-1)
    diffs = np.diff(sorted_arr, axis=-1)
    del sorted_arr  # Garbage collect
    # Pad with an extra zero at the beginning of every subarray
    pad_diffs = np.pad(diffs, ([(0, 0)] * (diffs.ndim - 1)) + [(1, 0)])
    del diffs  # Garbage collect
    # Wherever the diff is not 0, assign a value of 1; this gives a set of
    # small indices for each set of unique values in the sorted array after
    # taking a cumulative sum
    pad_diffs[pad_diffs != 0] = 1
    unique_inds = np.cumsum(pad_diffs, axis=-1).astype(int)
    del pad_diffs  # Garbage collect

    # 3) Average the ranks wherever the entries of the `arr` were identical
    # `unique_inds` contains elements that are indices to an array that stores
    # the average of the ranks of each unique element in the original array
    unique_maxes = np.zeros_like(arr)  # Maximum ranks for each unique index
    # Each subarray will contain unused entries if there are no repeats in that
    # subarray; this is a sacrifice made for vectorization; c'est la vie
    # Using `put_along_axis` will put the _last_ thing seen in `ranges`, which
    # result in putting the maximum rank in each unique location
    np.put_along_axis(unique_maxes, unique_inds, ranges, -1)
    # We can compute the average rank for each bucket (from the maximum rank for
    # each bucket) using some algebraic manipulation
    diff = np.diff(unique_maxes, prepend=-1, axis=-1)  # Note: prepend -1!
    unique_avgs = unique_maxes - ((diff - 1) / 2)
    del unique_maxes, diff  # Garbage collect

    # 4) Using the averaged ranks in `unique_avgs`, fill them into where they
    # belong
    avg_ranks = np.take_along_axis(
        unique_avgs, np.take_along_axis(unique_inds, ranks, -1), -1
    )

    return avg_ranks


def spearman_corr(arr1, arr2):
    """
    Computes the Spearman correlation in the last dimension of `arr1` and
    `arr2`. `arr1` and `arr2` must be the same shape. For example, if they are
    both A x B x L arrays, then the correlation of corresponding L-arrays will
    be computed and returned in an A x B array.
    """
    ranks1, ranks2 = average_ranks(arr1), average_ranks(arr2)
    return pearson_corr(ranks1, ranks2)


def mean_squared_error(arr1, arr2):
    """
    Computes the mean squared error in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the MSE of corresponding L-arrays will be computed
    and returned in an A x B array.
    """
    return np.mean(np.square(arr1 - arr2), axis=-1)


@performance_ex.capture
def binned_count_corr_mse(
    log_true_prof_counts, log_pred_prof_counts, prof_count_corr_bin_sizes,
    batch_size=50000
):
    """
    Returns the correlations of the true and predicted PROFILE counts (i.e.
    per base or per bin).
    Arguments:
        `log_true_prof_counts`: a N x T x O x 2 array, containing the true
            profile LOG COUNTS for each task and strand
        `log_pred_prof_counts`: a N x T x O x 2 array, containing the predicted
            profile LOG COUNTS for each task and strand
        `batch_size`: performs computation in a batch size of this many samples
    Returns 3 N x T x Z arrays, containing the Pearson correlation, Spearman
    correlation, and mean squared error of the profile predictions (as log
    counts). Correlations/MSE are computed for each sample/task, for each bin
    size in `prof_count_corr_bin_sizes` (strands are pooled together).
    """
    num_samples, num_tasks = log_true_prof_counts.shape[:2]
    num_bin_sizes = len(prof_count_corr_bin_sizes)
    pears = np.zeros((num_samples, num_tasks, num_bin_sizes))
    spear = np.zeros((num_samples, num_tasks, num_bin_sizes))
    mse = np.zeros((num_samples, num_tasks, num_bin_sizes))

    # Combine the profile length and strand dimensions (i.e. pool strands)
    new_shape = (num_samples, num_tasks, -1)
    log_true_prof_counts_flat = np.reshape(log_true_prof_counts, new_shape)
    log_pred_prof_counts_flat = np.reshape(log_pred_prof_counts, new_shape)

    for i, bin_size in enumerate(prof_count_corr_bin_sizes):
        true_count_binned = bin_array_max(log_true_prof_counts_flat, bin_size)
        pred_count_binned = bin_array_max(log_pred_prof_counts_flat, bin_size)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            true_batch = true_count_binned[start:end, :, :]
            pred_batch = pred_count_binned[start:end, :, :]
            pears[start:end, :, i] = pearson_corr(true_batch, pred_batch)
            spear[start:end, :, i] = spearman_corr(true_batch, pred_batch)
            mse[start:end, :, i] = mean_squared_error(true_batch, pred_batch)

    return pears, spear, mse


def total_count_corr_mse(log_true_total_counts, log_pred_total_counts):
    """
    Returns the correlations of the true and predicted TOTAL counts.
    Arguments:
        `log_true_total_counts`: a N x T x 2 array, containing the true total
            LOG COUNTS for each task and strand
        `log_pred_prof_counts`: a N x T x 2 array, containing the predicted
            total LOG COUNTS for each task and strand
    Returns 3 T-arrays, containing the Pearson correlation, Spearman
    correlation, and mean squared error of the profile predictions (as log
    counts). Correlations/MSE are computed for each task, over the samples and
    strands.
    """
    # Reshape inputs to be T x N * 2 (i.e. pool samples and strands)
    num_tasks = log_true_total_counts.shape[1]
    log_true_total_counts = np.reshape(
        np.swapaxes(log_true_total_counts, 0, 1), (num_tasks, -1)
    )
    log_pred_total_counts = np.reshape(
        np.swapaxes(log_pred_total_counts, 0, 1), (num_tasks, -1)
    )

    pears = pearson_corr(log_true_total_counts, log_pred_total_counts)
    spear = spearman_corr(log_true_total_counts, log_pred_total_counts)
    mse = mean_squared_error(log_true_total_counts, log_pred_total_counts)

    return pears, spear, mse


@performance_ex.capture
def compute_performance_metrics(
    true_profs, log_pred_profs, true_counts, log_pred_counts, print_updates=True
):
    """
    Computes some evaluation metrics on a set of positive examples, given the
    predicted profiles/counts, and the true profiles/counts.
    Arguments:
        `true_profs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `log_pred_profs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities 
        `true_counts`: a N x T x 2 array, containing the true total counts
            for each task and strand
        `log_pred_counts`: a N x T x 2 array, containing the predicted LOG total
            counts for each task and strand
        `print_updates`: if True, print out updates and runtimes
    Returns 2 dictionaries. The first dictionary contains:
        A N x T-array of the average negative log likelihoods for the profiles
            (given predicted probabilities, the likelihood for the true counts),
            for each sample/task (strands pooled)
        A N x T array of average Jensen-Shannon divergence between the predicted
            and true profiles (strands averaged)
        A N x T x Z x 4 array of the auPRCs for the binned and binarized
            profiles, and the class assignment counts for the binned and
            binarized profiles (positive, ambiguous, and negative counts); this
            is reported for Z bin sizes, for each sample/task (strands pooled)
        A N x T x Z array of the Pearson correlation of the predicted and true
            (log) counts, for each of the Z bin sizes, for each sample/task
            (strands pooled)
        A N x T x Z array of the Spearman correlation of the predicted and true
            (log) counts, for each of the Z bin sizes, for each sample/task
            (strands pooled)
        A N x T x Z array of the mean squared error of the predicted and true
            (log) counts, for each of the Z bin sizes, for each sample/task
            (strands pooled)
        A T-array of the Pearson correlation of the (log) total counts, over all
            strands and samples
        A T-array of the Spearman correlation of the (log) total counts, over
            all strands and samples
        A T-array of the mean squared error of the (log) total counts, over all
            strands and samples
    The second dictionary is computed in the same way as the first, but with
    order of the predicted samples randomized relative to the true samples (i.e.
    shuffled along the first dimension)
    """
    # Multinomial NLL
    if print_updates:
        print("\t\tComputing NLL... ", end="", flush=True)
        start = datetime.now()
    nll = profile_multinomial_nll(
        true_profs, log_pred_profs, true_counts
    )
    if print_updates:
        end = datetime.now()
        print("%ds" % (end - start).seconds)

    # Jensen-Shannon divergence
    # The true profile counts will be renormalized during JSD computation
    if print_updates:
        print("\t\tComputing JSD... ", end="", flush=True)
        start = datetime.now()
    pred_prof_probs = np.exp(log_pred_profs)
    jsd = profile_jsd(true_profs, pred_prof_probs)
    if print_updates:
        end = datetime.now()
        print("%ds" % (end - start).seconds)

    if print_updates:
        print("\t\tComputing auPRC... ", end="", flush=True)
        start = datetime.now()
    # Binned auPRC
    auprc = binned_profile_auprc(true_profs, pred_prof_probs, true_counts)
    if print_updates:
        end = datetime.now()
        print("%ds" % (end - start).seconds)

    del pred_prof_probs  # Garbage collect

    if print_updates:
        print("\t\tComputing correlations/MSE (binned)... ", end="", flush=True)
        start = datetime.now()
    # Binned profile count correlations/MSE
    log_true_profs = np.log(true_profs + 1)
    pears_bin, spear_bin, mse_bin = binned_count_corr_mse(
        log_true_profs, log_pred_profs
    )
    if print_updates:
        end = datetime.now()
        print("%ds" % (end - start).seconds)

    if print_updates:
        print("\t\tComputing correlations/MSE (total)... ", end="", flush=True)
        start = datetime.now()
    # Total count correlations/MSE
    log_true_counts = np.log(true_counts + 1)
    pears_tot, spear_tot, mse_tot = total_count_corr_mse(
        log_true_counts, log_pred_counts
    )
    if print_updates:
        end = datetime.now()
        print("%ds" % (end - start).seconds)

    return {
        "nll": nll,
        "jsd": jsd,
        "auprc_binned": auprc,
        "pearson_binned": pears_bin,
        "spearman_binned": spear_bin,
        "mse_binned": mse_bin,
        "pearson_total": pears_tot,
        "spearman_total": spear_tot,
        "mse_total": mse_tot
    }


@performance_ex.capture
def log_performance_metrics(
    metrics, prefix, _run, prof_count_corr_bin_sizes, auprc_bin_sizes,
    print_log=True
):
    """
    Given the metrics dictionary returned by `compute_performance_metrics`, logs
    them to a Sacred logging object (`_run`), and optionally prints out a log.
    When logging, `prefix` is prepended to each output key.
    """
    # Before logging, condense the metrics into averages over the samples (when
    # appropriate)
    nll = np.nanmean(metrics["nll"], axis=0)  # T
    jsd = np.nanmean(metrics["jsd"], axis=0)  # T
    auprc_bin = np.nanmean(metrics["auprc_binned"][:, :, :, 0], axis=0)  # T x Z
    pears_bin = np.nanmean(metrics["pearson_binned"], axis=0)  # T x Z
    spear_bin = np.nanmean(metrics["spearman_binned"], axis=0)  # T x Z
    mse_bin = np.nanmean(metrics["mse_binned"], axis=0)  # T x Z
    pears_tot = metrics["pearson_total"]  # T
    spear_tot = metrics["spearman_total"]  # T
    mse_tot = metrics["mse_total"]  # T
    # At this point, these metrics are all extracted from the dictionary and are
    # either T-arrays or T x Z arrays (where T is the number of tasks and Z is
    # the number of bin sizes for a metric)

    _run.log_scalar("%s_prof_nll" % prefix, list(nll))
    _run.log_scalar("%s_prof_jsd" % prefix, list(jsd))
    for i, bin_size in enumerate(auprc_bin_sizes):
        _run.log_scalar(
            "%s_prof_auprc_bin%d" % (prefix, bin_size), list(auprc_bin[:, i])
        )
    for i, bin_size in enumerate(prof_count_corr_bin_sizes):
        _run.log_scalar(
            "%s_prof_pearson_bin%d" % (prefix, bin_size), list(pears_bin[:, i])
        )
        _run.log_scalar(
            "%s_prof_spearman_bin%d" % (prefix, bin_size), list(spear_bin[:, i])
        )
        _run.log_scalar(
            "%s_prof_mse_bin%d" % (prefix, bin_size), list(mse_bin[:, i])
        )
    _run.log_scalar("%s_count_pearson" % prefix, list(pears_tot))
    _run.log_scalar("%s_count_spearman" % prefix, list(spear_tot))
    _run.log_scalar("%s_count_mse" % prefix, list(mse_tot))

    if print_log:
        print(("\t%s profile NLL: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in nll]
        ))
        print(("\t%s profile JSD: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in jsd]
        ))
        for i, bin_size in enumerate(auprc_bin_sizes):
            print(
                ("\t%s profile auPRC (bin size = %d): " % \
                (prefix, bin_size)) + \
                ", ".join([("%6.6f" % x) for x in auprc_bin[:, i]])
            )
        for i, bin_size in enumerate(prof_count_corr_bin_sizes):
            print(
                ("\t%s profile Pearson (bin size = %d): " % \
                (prefix, bin_size)) + \
                ", ".join([("%6.6f" % x) for x in pears_bin[:, i]])
            )
            print(
                ("\t%s profile Spearman (bin size = %d): " % \
                (prefix, bin_size)) + \
                ", ".join([("%6.6f" % x) for x in spear_bin[:, i]])
            )
            print(
                ("\t%s profile MSE (bin size = %d): " % (prefix, bin_size)) + \
                ", ".join([("%6.6f" % x) for x in mse_bin[:, i]])
            )
        print(("\t%s count Pearson: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in pears_tot]
        ))
        print(("\t%s count Spearman: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in spear_tot]
        ))
        print(("\t%s count MSE: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in mse_tot]
        ))
