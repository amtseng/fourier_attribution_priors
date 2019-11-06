import numpy as np
import scipy.special
import sacred
import sklearn.metrics
import scipy.stats

performance_ex = sacred.Experiment("performance")

@performance_ex.config
def config():
    # Bin sizes to try for count correlation
    bin_sizes = [1, 2, 4, 10]


def profile_multinomial_nll(
    true_prof_counts, pred_prof_log_probs, true_total_counts
):
    """
    Computes the multinomial negative log likelihood of the true profiles, given
    the probabilties of the predicted profile.
    Arguments:
        `true_prof_counts`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `pred_prof_log_probs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities 
        `true_total_counts`: a N x T x 2 array, containing the true counts for
            each task and strand
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


@performance_ex.capture
def profile_binary_auprc(
    true_prof_counts, pred_prof_probs, true_total_counts, bin_prof_min_count,
    bin_prof_pos_thresh, bin_prof_neg_thresh
):
    """
    Binarizes the profile and computes auPRC.
    Arguments:
        `true_prof_counts`: B x T x O x 2 array, where B is the batch size, T is
            the number of tasks, and O is the output profile length; contains
            the true profiles for each for each task and strand, as RAW counts
        `pred_prof_probs`: a B x T x O x 2 array, containing the predicted
            profiles for each task and strand, as PROBABILITIES
        `true_total_counts`: a B x T x 2 array, containing the true counts for
            each task and strand
    Returns an array of T items, containing the auPRCs for each task, where the
    auPRC is computed across the pooled strands and examples. Also returns a
    T x 3 array containing the total number of positive, ambiguous, and negative
    positions, also per task.
    """
    # Normalize true profile (counts) into probabilities, keeping 0s when 0
    true_total_counts_dim = np.expand_dims(true_total_counts, axis=2)
    true_prof_prob = np.divide(
        true_prof_counts, true_total_counts_dim,
        out=np.zeros_like(true_prof_counts),
        where=(true_total_counts_dim != 0)
    )
  
    # A position is positive if it has at least the minimum number of reads,
    # and it represents at least the minimum fraction of total reads
    pos_mask = (true_prof_counts >= bin_prof_min_count) & \
        (true_prof_prob >= bin_prof_pos_thresh)
    neg_mask = true_prof_prob <= bin_prof_neg_thresh

    auprcs, class_nums = [], []
    for task_ind in range(true_prof_prob.shape[1]):
        task_pos_mask = pos_mask[:, task_ind, :, :]
        task_neg_mask = neg_mask[:, task_ind, :, :]
        task_pred_prof_probs = pred_prof_probs[:, task_ind, :, :]

        num_pos, num_neg = np.sum(task_pos_mask), np.sum(task_neg_mask)
        num_ambi = task_pred_prof_probs.size - num_pos - num_neg
        true_vals = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
        pred_vals = np.concatenate([
            task_pred_prof_probs[task_pos_mask],
            task_pred_prof_probs[task_neg_mask]
        ])

        auprc = sklearn.metrics.average_precision_score(true_vals, pred_vals)
        auprcs.append(auprc)
        class_nums.append([num_pos, num_ambi, num_neg])

    return np.stack(auprcs), np.stack(class_nums)


def count_correlation(true_total_counts, pred_counts):
    """
    Returns the correlations of the true and predicted counts.
    Arguments:
        `true_total_counts`: a N x T x 2 array, containing the true counts for
            each task and strand
        `pred_counts`: a N x T x 2 array, containing the predicted counts for
            each task and strand
    Returns two arrays of T items, containing the Pearson and Spearman
    correlations, respectively, for each task.
    """
    pearson, spearman = [], []
    for task_ind in range(true_total_counts.shape[1]):
        task_true_total_counts = np.ravel(true_total_counts[:, task_ind, :])
        task_pred_counts = np.ravel(pred_counts[:, task_ind, :])

        # Avoid NaNs and infs
        finite_mask = np.isfinite(task_pred_counts)
        task_true_total_counts = task_true_total_counts[finite_mask]
        task_pred_counts = task_pred_counts[finite_mask]
        
        pearson.append(
            scipy.stats.pearsonr(task_true_total_counts, task_pred_counts)[0]
        )
        spearman.append(
            scipy.stats.spearmanr(task_true_total_counts, task_pred_counts)[0]
        )
    return np.stack(pearson), np.stack(spearman)


@performance_ex.capture
def compute_performance(
    true_prof_counts, pred_prof_log_probs, true_total_counts, pred_counts
):
    """
    Arguments:
        `true_prof_counts`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `pred_prof_log_probs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities 
        `true_total_counts`: a N x T x 2 array, containing the true counts for
            each task and strand
        `pred_counts`: a N x T x 2 array, containing the predicted counts for
            each task and strand
    Returns the following in a dictionary:
        A T-array of the negative log likelihoods for the profiles
        A T-array of the auPRCs for the binarized profiles
        A T x 3 array of the class assignment counts for the binarized profiles
            (positive, ambiguous, and negative counts)
        A T-array of the Pearson correlation of the counts
        A T-array of the Spearman correlation of the counts
    """
    nll = profile_multinomial_nll(
        true_prof_counts, pred_prof_log_probs, true_total_counts
    )
    pred_prof_probs = np.exp(pred_prof_log_probs)
    auprc, class_counts = profile_binary_auprc(
        true_prof_counts, pred_prof_probs, true_total_counts
    )
    pearson, spearman = count_correlation(true_total_counts, pred_counts)

    return {
        "nll": nll,
        "auprc": auprc,
        "class_counts": class_counts,
        "pearson": pearson,
        "spearman": spearman
    }


def log_performance(metrics, _run, print_log=True):
    """
    Given the metrics dictionary returned by `compute_performance`, logs them
    to a Sacred logging object (`_run`), and optionally prints out a log.
    """
    _run.log_scalar("val_prof_nll", list(metrics["nll"]))
    _run.log_scalar("val_prof_auprc", list(metrics["auprc"]))
    _run.log_scalar("val_count_pearson", list(metrics["pearson"]))
    _run.log_scalar("val_count_spearman", list(metrics["spearman"]))
   
    if print_log:
        print("Validation set performance:")
        print("\tProfile NLL: " + ", ".join(
            [("%6.6f" % x) for x in metrics["nll"]]
        ))
        print("\tProfile auPRC: " + ", ".join(
            [("%6.6f" % x) for x in metrics["auprc"]]
        ))
        print("\tCount Pearson: " + ", ".join(
            [("%6.6f" % x) for x in metrics["pearson"]]
        ))
        print("\tCount Spearman: " + ", ".join(
            [("%6.6f" % x) for x in metrics["spearman"]]
        ))
