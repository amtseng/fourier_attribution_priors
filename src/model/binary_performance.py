import numpy as np
import sklearn.metrics

def accuracies(true_vals, pred_vals_rounded):
    """
    Computes the accuracy for the given labels and rounded predictions.
    Arguments:
        `true_vals`: NumPy array of true binary values, with only 1s and 0s
        `pred_vals_rounded`: NumPy array of predictions that have been rounded
            to either 0 or 1
    Returns 3 scalars: the overall accuracy, the accuracy for only positives
    (i.e. where the true value is 1), and the accuracy for only negatives (i.e.
    where the true value is 0).
    """
    acc = np.sum(pred_vals_rounded == true_vals) / len(true_vals)

    pos_mask = true_vals == 1
    pred_vals_pos_rounded = pred_vals_rounded[pos_mask]
    pos_acc = np.sum(pred_vals_pos_rounded == 1) / len(pred_vals_pos_rounded)

    neg_mask = true_vals == 0
    pred_vals_neg_rounded = pred_vals_rounded[neg_mask]
    neg_acc = np.sum(pred_vals_neg_rounded == 0) / len(pred_vals_neg_rounded)

    return acc, pos_acc, neg_acc


def estimate_imbalanced_precision_recall(
    true_vals, pred_vals, neg_upsample_factor=1
):
    """
    Computes the precision and recall for the given labels and (unrounded)
    predictions. This function will correct for precision inflation due to
    downsampling the negatives during prediction. This is mean to return the
    same thing as `sklearn.metrics.precision_recall_curve` if
    `neg_upsample_factor` is 1.
    Arguments:
        `true_vals`: NumPy array of true binary values (1 is positive), with
            only 1s and 0s
        `pred_vals`: NumPy array of predictions between 0 and 1
        `neg_upsample_factor`: A positive number at least 1, this is the factor
            at which the negatives were downsampled for prediction, and it also
            the factor to upscale the false positive rate
    Returns a NumPy array of precision values corrected for downsampling
    negatives, a NumPy array of recall values, and a NumPy array of the
    thresholds (i.e. the sorted prediction values).
    """
    # Sort the true values in descending order of prediction values
    sort_inds = np.flip(np.argsort(pred_vals))
    sort_true_vals = true_vals[sort_inds]
    sort_pred_vals = pred_vals[sort_inds]

    num_vals = len(true_vals)
    num_neg_vals = np.sum(true_vals == 0)
    num_pos_vals = num_vals - num_neg_vals

    # Identify the thresholds as the locations where the sorted predicted values
    # differ; these are indices where the _next_ entry is different from the
    # current one
    thresh_inds = np.where(np.diff(sort_pred_vals) != 0)[0]
    # Tack on the last threshold, which is missed by `diff`
    thresh_inds = np.concatenate([thresh_inds, [num_vals - 1]])
    thresh = sort_pred_vals[thresh_inds]

    # Get the number of entries at each threshold point (i.e. at each threshold,
    # there are this many entries with predicted value at least this threshold)
    num_above = np.arange(1, num_vals + 1)[thresh_inds]

    # Compute the true positives at each threshold (i.e. at each threshold, this
    # this many entries with predicted value at least this threshold are truly
    # positives)
    tp = np.cumsum(sort_true_vals)[thresh_inds]

    # Compute the false positives at each threshold (i.e. at each threshold,
    # this many entries with predicted value at least this threshold are truly
    # negatives)
    fp = num_above - tp

    # Compute the false negatives at each threshold (i.e. at each threshold,
    # this many entries with predicted value below this threshold are truly
    # positives)
    fn = num_vals - num_above - (num_neg_vals - fp)

    # The precision is TP / (TP + FP); with `neg_upsample_factor` of 1, FP
    # remains the same; otherwise, there are presumably this many times more
    # true negatives above each threshold
    numer = tp
    denom = tp + (fp * neg_upsample_factor)
    denom[denom == 0] = 1  # When dividing, if there are no positives, keep 0
    precis = numer / denom

    # The recall is TP / (TP + FN); TP + FN is also the total number of true
    # positives
    recall = tp / (tp + fn)
    # Only NaN if no positives at all

    # Cut off the values after which the true positives has reached the maximum
    # (i.e. number of positives total); after this point, recall won't change
    max_ind = np.min(np.where(tp == num_pos_vals)[0])
    precis = precis[:max_ind + 1]
    recall = recall[:max_ind + 1]
    thresh = thresh[:max_ind + 1]

    # Flip the arrays, and concatenate final precision/recall values (i.e. when
    # there are no positives, precision is 1 and recall is 0)
    precis, recall, thresh = np.flip(precis), np.flip(recall), np.flip(thresh)
    precis = np.concatenate([precis, [1]])
    recall = np.concatenate([recall, [0]])
    return precis, recall, thresh


def precision_recall_scores(precis, recall, thresholds, pos_thresh=0.5):
    """
    From parallel NumPy arrays of precision, recall, and their increasing
    thresholds, returns the precision and recall scores, if the threshold to
    call a positive is `pos_thresh`.
    The inputs should be the result of `estimate_imbalanced_precision_recall`,
    or `sklearn.metrics.precision_recall_curve`.
    """
    assert np.all(np.diff(thresholds) >= 0)
    inds = np.where(thresholds >= pos_thresh)[0]
    if not inds.size:
        # If there are no predicted positives, then precision is 0
        return 0, 0
    # Index of the closest threshold at least pos_thresh:
    thresh_ind = min(inds)
    return precis[thresh_ind], recall[thresh_ind]


def compute_performance_metrics(true_values, pred_values, neg_upsample_factor):
    """
    For the given parallel 2D NumPy arrays, each of shape B x T, containing true
    and predicted values, computes various evaluation metrics and returns them
    as a dictionary of lists, where each list contains a metric computed on each
    of the tasks.
    """
    def single_task_metrics(true_vec, pred_vec):
        """
        Returns a set of metrics, but for a specific task. Inputs are parallel
        NumPy arrays of scalars.
        """
        # Ignore output values that are not 0 or 1
        mask = (true_vec == 0) | (true_vec == 1)
        true_vec, pred_vec = true_vec[mask], pred_vec[mask]

        # Overall accuracy, and accuracy for each class
        pred_vec_rounded = np.round(pred_vec)
        acc, pos_acc, neg_acc = accuracies(true_vec, pred_vec_rounded)

        # auROC
        auroc = sklearn.metrics.roc_auc_score(true_vec, pred_vec)

        # Precision, recall, auPRC
        precis, recall, thresh = estimate_imbalanced_precision_recall(
            true_vec, pred_vec
        )
        precis_score, recall_score = precision_recall_scores(
            precis, recall, thresh
        )
        auprc = sklearn.metrics.auc(recall, precis)

        # Precision, auPRC, corrected for downsampling
        c_precis, c_recall, c_thresh = estimate_imbalanced_precision_recall(
            true_vec, pred_vec, neg_upsample_factor=neg_upsample_factor
        )
        c_precis_score, c_recall_score = precision_recall_scores(
            c_precis, c_recall, c_thresh
        )
        c_auprc = sklearn.metrics.auc(c_recall, c_precis)

        return acc, pos_acc, neg_acc, auroc, precis_score, recall_score, \
            auprc, c_precis_score, c_recall_score, c_auprc

    result = []
    num_tasks = np.shape(true_values)[1]
    for task_num in range(num_tasks):
        result.append(
            single_task_metrics(
                true_values[:,task_num], pred_values[:,task_num]
            )
        )

    # Transpose `result` to get metric lists, each list has all the tasks
    metrics = map(list, zip(*result))
    labels = [
        "acc", "pos_acc", "neg_acc", "auroc", "precis_score", "recall_score",
        "auprc", "c_precis_score", "c_recall_score", "c_auprc"
    ]

    return dict(zip(labels, metrics))


def log_performance_metrics(metrics, prefix, _run, print_log=True):
    """
    Given the metrics dictionary returned by `compute_performance_metrics`, logs
    them to a Sacred logging object (`_run`), and optionally prints out a log.
    When logging, `prefix` is prepended to each output key.
    """
    _run.log_scalar("%s_acc" % prefix, metrics["acc"])
    _run.log_scalar("%s_pos_acc" % prefix, metrics["pos_acc"])
    _run.log_scalar("%s_neg_acc" % prefix, metrics["neg_acc"])
    _run.log_scalar("%s_auroc" % prefix, metrics["auroc"])
    _run.log_scalar("%s_precis_score" % prefix, metrics["precis_score"])
    _run.log_scalar("%s_recall_score" % prefix, metrics["recall_score"])
    _run.log_scalar("%s_auprc" % prefix, metrics["auprc"])
    _run.log_scalar("%s_corr_precis_score" % prefix, metrics["c_precis_score"])
    _run.log_scalar("%s_corr_recall_score" % prefix, metrics["c_recall_score"])
    _run.log_scalar("%s_corr_auprc" % prefix, metrics["c_auprc"])

    if print_log:
        print(("\t%s accuracy: " % prefix) + ", ".join(
            [("%6.2f%%" % (acc * 100)) for acc in metrics["acc"]]
        ))
        print(("\t%s + accuracy: " % prefix) + ", ".join(
            [("%6.2f%%" % (acc * 100)) for acc in metrics["pos_acc"]]
        ))
        print(("\t%s - accuracy: " % prefix) + ", ".join(
            [("%6.2f%%" % (acc * 100)) for acc in metrics["neg_acc"]]
        ))
        print(("\t%s auROC: " % prefix) + ", ".join(
            [("%6.6f" % auroc) for auroc in metrics["auroc"]]
        ))
        print(("\t%s precision score: " % prefix) + ", ".join(
            [("%6.6f" % precis) for precis in metrics["precis_score"]]
        ))
        print(("\t%s recall score: " % prefix) + ", ".join(
            [("%6.6f" % recall) for recall in metrics["recall_score"]]
        ))
        print(("\t%s auPRC: " % prefix) + ", ".join(
            [("%6.6f" % auprc) for auprc in metrics["auprc"]]
        ))
        print(("\t%s corrected precision score: " % prefix) + ", ".join(
            [("%6.6f" % precis) for precis in metrics["c_precis_score"]]
        ))
        print(("\t%s corrected recall score: " % prefix) + ", ".join(
            [("%6.6f" % recall) for recall in metrics["c_recall_score"]]
        ))
        print(("\t%s corrected auPRC: " % prefix) + ", ".join(
            [("%6.6f" % auprc) for auprc in metrics["c_auprc"]]
        )) 
