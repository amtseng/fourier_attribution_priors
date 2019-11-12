import numpy as np
import scipy.special
import sacred
import sklearn.metrics
import scipy.stats
import scipy.spatial.distance
import warnings
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


def profile_multinomial_nll(
    true_prof_counts, pred_prof_log_probs, true_total_counts
):
    """
    Computes the multinomial negative log likelihood of the true profiles, given
    the probabilties of the predicted profile.
    Arguments:
        `true_prof_counts`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each task and for each
            strand, as RAW COUNTS
        `pred_prof_log_probs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG PROBABILITIES
        `true_total_counts`: a N x T x 2 array, containing the true total counts
            for each task and strand
    Returns an array of T items, containing the negative log likelihoods,
    averaged across the pooled strands and examples, for each task.
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)

    log_n_fact = scipy.special.gammaln(true_total_counts + 1)
    log_counts_fact = scipy.special.gammaln(true_prof_counts + 1)
    log_counts_fact_sum = np.sum(log_counts_fact, axis=2)
    log_prob_pows = pred_prof_log_probs * true_prof_counts  # Elementwise
    log_prob_pows_sum = np.sum(log_prob_pows, axis=2)

    nll = log_counts_fact_sum - log_n_fact - log_prob_pows_sum
    # Shape: N x T x 2
    nll = np.transpose(nll, axes=(1, 0, 2))  # Shape: T x N x 2
    nll = np.reshape(nll, (nll.shape[0], -1))  # Shape: T x 2N
    return np.mean(nll, axis=1)


def profile_jsd(true_prof_probs, pred_prof_probs):
    """
    Computes the Jensen-Shannon divergence of the true and predicted profiles
    given their log probabilities.
    Arguments:
        `true_prof_probs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, O is the output profile length;
            contains the true profiles for each task and strand, as RAW
            PROBABILITIES 
        `pred_prof_probs`: N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as RAW PROBABILITIES
    Returns an N x T array, where the JSD is computed across the profiles and
    averaged between the strands, for each sample/task.
    """
    num_samples, num_tasks = true_prof_probs.shape[:2]
    result = np.zeros((num_samples, num_tasks))
    for i in range(num_samples):
        for j in range(num_tasks):
            jsd_0 = scipy.spatial.distance.jensenshannon(
                true_prof_probs[i, j, :, 0], pred_prof_probs[i, j, :, 0]
            )
            jsd_1 = scipy.spatial.distance.jensenshannon(
                true_prof_probs[i, j, :, 1], pred_prof_probs[i, j, :, 1]
            )
            result[i][j] = np.mean(np.square([jsd_0, jsd_1]))
    return result


def bin_array_max(arr, bin_size, pad=0):
    """
    Given a NumPy array, returns a binned version of the array along the last
    dimension, where each bin contains the maximum value of its constituent
    elements. If the array is not a length that is a multiple of the bin size,
    then the given pad will be used at the end.
    """
    pad_amount = arr.shape[-1] % bin_size
    if pad_amount:
        arr = np.pad(arr, ([(0, 0)] * (arr.ndim - 1)) + [(0, pad_amount)])
    new_shape = arr.shape[:-1] + (arr.shape[-1] // bin_size, bin_size)
    return np.max(np.reshape(arr, new_shape), axis=-1)


@performance_ex.capture
def binned_profile_auprc(
    true_prof_counts, pred_prof_probs, true_total_counts, auprc_bin_sizes,
    auprc_min_pos_prob, auprc_min_pos_count, auprc_max_neg_prob
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
    for i in range(num_samples):
        for j in range(num_tasks):
            # Pool the strands together for the sample/task
            true_prob_slice = true_prof_probs_flat[i, j]
            true_count_slice = true_prof_counts_flat[i, j]
            pred_prob_slice = pred_prof_probs_flat[i, j]

            for k, bin_size in enumerate(auprc_bin_sizes):
                # Bin the values, taking the maximum for each bin
                true_prob_bins = bin_array_max(true_prob_slice, bin_size)
                true_count_bins = bin_array_max(true_count_slice, bin_size)
                pred_prob_bins = bin_array_max(pred_prob_slice, bin_size)

                # Filter for the positives and negatives
                # A bin is positive if the maximum count inside it is at least
                # a minimum number of reads, and the maximum probability inside
                # is at least the minimum fraction of total reads
                pos_mask = (true_count_bins >= auprc_min_pos_count) & \
                    (true_prob_bins >= auprc_min_pos_prob)
                neg_mask = true_prob_bins <= auprc_max_neg_prob

                num_pos, num_neg = np.sum(pos_mask), np.sum(neg_mask)
                num_ambi = true_prob_bins.size - num_pos - num_neg

                if num_pos:
                    true_vals = np.concatenate([
                        np.ones(num_pos), np.zeros(num_neg)
                    ])
                    pred_vals = np.concatenate([
                        pred_prob_bins[pos_mask], pred_prob_bins[neg_mask]
                    ])
                    auprc = sklearn.metrics.average_precision_score(
                        true_vals, pred_vals
                    )
                else:
                    auprc = np.nan

                result[i, j, k] = [auprc, num_pos, num_ambi, num_neg]

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
        A T-array of the average negative log likelihoods for the profiles
            (given predicted probabilities, the likelihood for the true counts)
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
    # Normalize true profile (counts) into probabilities, keeping 1s when the
    # sum is 0; upon renormalization, this would avoid dividing by 0
    if print_updates:
        print("\t\tComputing JSD... ", end="", flush=True)
        start = datetime.now()
    true_counts_dim = np.expand_dims(true_counts, axis=2)
    true_prof_probs = np.divide(
        true_profs, true_counts_dim,
        out=np.ones_like(true_profs),
        where=(true_counts_dim != 0)
    )
    pred_prof_probs = np.exp(log_pred_profs)
    jsd = profile_jsd(true_prof_probs, pred_prof_probs)
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
    nll = metrics["nll"]  # T
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


@performance_ex.command
def test1():
    from datetime import datetime
    np.random.seed(20191110)
    num_corrs, corr_len = 500, 1000
    arr1 = np.random.randint(100, size=(num_corrs, corr_len))
    arr2 = np.random.randint(100, size=(num_corrs, corr_len))

    print("Pearson correlation:")
    pears_scipy = np.empty(num_corrs)
    a = datetime.now()
    for i in range(num_corrs):
        pears_scipy[i] = scipy.stats.pearsonr(arr1[i], arr2[i])[0]
    b = datetime.now()
    print("\tTime to compute (Scipy): %ds" % (b - a).seconds)

    a = datetime.now()
    pears_vect = pearson_corr(arr1, arr2) 
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)
    print("\tSame result? %s" % np.allclose(pears_vect, pears_scipy))

    print("Spearman correlation:")
    spear_scipy = np.empty(num_corrs)
    a = datetime.now()
    for i in range(num_corrs):
        spear_scipy[i] = scipy.stats.spearmanr(arr1[i], arr2[i])[0]
    b = datetime.now()
    print("\tTime to compute (Scipy): %ds" % (b - a).seconds)

    a = datetime.now()
    spear_vect = spearman_corr(arr1, arr2) 
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)
    print("\tSame result? %s" % np.allclose(spear_vect, spear_scipy))


@performance_ex.command
def test2():
    from datetime import datetime
    np.random.seed(20191110)
    num_samples, num_tasks, profile_len = 500, 4, 1000
    bin_sizes = [1, 4, 10]
    arr1 = np.random.randint(100, size=(num_samples, num_tasks, profile_len, 2))
    arr2 = np.random.randint(100, size=(num_samples, num_tasks, profile_len, 2))
    arr3 = np.random.randint(100, size=(num_samples, num_tasks, 2))
    arr4 = np.random.randint(100, size=(num_samples, num_tasks, 2))

    print("Testing binned correlation and MSE...")
    a = datetime.now()
    # Combine the profile length and strand dimensions (i.e. pool strands)
    new_shape = (num_samples, num_tasks, -1)
    arr1_flat, arr2_flat = np.reshape(arr1, new_shape), np.reshape(arr2, new_shape)
    pears_scipy = np.empty((num_samples, num_tasks, len(bin_sizes)))
    spear_scipy = np.empty((num_samples, num_tasks, len(bin_sizes)))
    mse_scipy = np.empty((num_samples, num_tasks, len(bin_sizes)))
    for i in range(num_samples):
        for j in range(num_tasks):
            slice1, slice2 = arr1_flat[i, j], arr2_flat[i, j]
            for k, bin_size in enumerate(bin_sizes):
                # Bin the values, taking the maximum for each bin
                bins1 = bin_array_max(slice1, bin_size)
                bins2 = bin_array_max(slice2, bin_size)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Ignore warnings when computing correlations, to avoid
                    # warnings when input is constant
                    pears_scipy[i, j, k] = scipy.stats.pearsonr(bins1, bins2)[0]
                    spear_scipy[i, j, k] = scipy.stats.spearmanr(bins1, bins2)[0]
                    mse_scipy[i, j, k] = sklearn.metrics.mean_squared_error(bins1, bins2)
    b = datetime.now()
    print("\tTime to compute (SciPy): %ds" % (b - a).seconds)
    
    a = datetime.now()
    pears_vec, spear_vec, mse_vec = binned_count_corr_mse(arr1, arr2, [1, 4, 10])
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)

    print("\tSame Pearson result? %s" % np.allclose(pears_vec, pears_scipy))
    print("\tSame Spearman result? %s" % np.allclose(spear_vec, spear_scipy))
    print("\tSame MSE result? %s" % np.allclose(mse_vec, mse_scipy))


    print("Testing total correlation and MSE...")
    a = datetime.now()
    # Reshape inputs to be T x N * 2 (i.e. pool samples and strands)
    arr3_swap = np.reshape(np.swapaxes(arr3, 0, 1), (num_tasks, -1))
    arr4_swap = np.reshape(np.swapaxes(arr4, 0, 1), (num_tasks, -1))

    # For each task, compute the correlations/MSE
    pears_scipy, spear_scipy, mse_scipy = np.empty(num_tasks), np.empty(num_tasks), np.empty(num_tasks)
    for j in range(num_tasks):
        arr3_list, arr4_list = arr3_swap[j], arr4_swap[j]
        pears_scipy[j] = scipy.stats.pearsonr(arr3_list, arr4_list)[0]
        spear_scipy[j] = scipy.stats.spearmanr(arr3_list, arr4_list)[0]
        mse_scipy[j] = sklearn.metrics.mean_squared_error(arr3_list, arr4_list)
    b = datetime.now()
    print("\tTime to compute (SciPy): %ds" % (b - a).seconds)
    
    a = datetime.now()
    pears_vec, spear_vec, mse_vec = total_count_corr_mse(arr3, arr4)
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)

    print("\tSame Pearson result? %s" % np.allclose(pears_vec, pears_scipy))
    print("\tSame Spearman result? %s" % np.allclose(spear_vec, spear_scipy))
    print("\tSame MSE result? %s" % np.allclose(mse_vec, mse_scipy))
