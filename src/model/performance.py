import numpy as np
import sklearn.metrics

def estimate_imbalanced_precision_recall(
    true_vals, pred_vals, neg_upsample_factor=1
):
    """
    Computes the precision and recall for the given labels and (unrounded)
    predictions. This function will correct for precision inflation due to
    downsampling the negatives during prediction.
    Arguments:
        `true_vals`: NumPy array of true binary values (1 is positive)
        `pred_vals`: NumPy array of predictions between 0 and 1
        `neg_upsample_factor`: A positive number at least 1, this is the factor
            at which the negatives were downsampled for prediction, and it also
            the factor to upscale the false positive rate
    Returns a NumPy array of precision values corrected for downsampling
    negatives, a NumPy array of recall values, and a NumPy array of the
    thresholds (i.e. the sorted prediction values).
    """
    # Sort the true values in order of prediction values
    sort_inds = np.argsort(pred_vals)
    sort_true_vals = true_vals[sort_inds]

    # The number of samples above each threshold and below each threshold
    # A threshold means breaking above it
    samp_above = np.arange(len(true_vals) - 1, -1, -1)
    samp_below = np.arange(1, len(true_vals) + 1)

    # The number of true positives above each prediction threshold
    tp_cumsum = np.cumsum(sort_true_vals)
    num_tp = tp_cumsum[-1]
    tp_above = np.concatenate([num_tp - tp_cumsum[:-1], [0]]).astype(float)

    # The number of true negatives below each prediction threshold
    tn_below = np.cumsum(1 - sort_true_vals)

    # The number of false positives above each threshold
    fp_above = samp_above - tp_above

    # The number of false negatives below each threshold
    fn_below = samp_below - tn_below

    # In case tp_above and fp_above are both zero, that precision is 1
    with np.errstate(divide="ignore", invalid="ignore"):
        precis_thresh = tp_above / (tp_above + (fp_above * neg_upsample_factor))
        precis_thresh[np.isnan(precis_thresh)] = 1
        recall_thresh = tp_above / (tp_above + fn_below)
        recall_thresh[np.isnan(recall_thresh)] = 1

    return precis_thresh, recall_thresh, pred_vals[sort_inds]


def average_precision(precis, recall):
    """
    From parallel NumPy arrays of precision and recall over increasing
    thresholds, returns the average precision.
    The inputs should be the result of `estimate_imbalanced_precision_recall`,
    or `sklearn.metrics.precision_recall_curve`.
    """
    return sum([
        (recall[i] - recall[i + 1]) * precis[i] \
        for i in range(0, len(precis) - 1)
    ])


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


def compute_evaluation_metrics(
    true_values, pred_values, val_neg_downsample, output_ignore_value
):
    """
    For the given parallel 2D NumPy arrays containing true and predicted values
    (shape num_samples x num_tasks), computes various evaluation metrics and
    returns them as a dictionary of lists, where each list is a metric computed
    once per each task.
    """
    def single_task_metrics(true_vec, pred_vec):
        """
        Returns a set of metrics, but for a specific task. Inputs are parallel
        NumPy arrays of scalars.
        """
        # Keep the output values where the true output isn't to be ignored
        mask = true_vec != output_ignore_value
        true_vec, pred_vec = true_vec[mask], pred_vec[mask]
        pred_vec_round = np.round(pred_vec)

        # Overall accuracy, and accuracy for each class
        acc = np.sum(pred_vec_round == true_vec) / len(true_vec)

        pos_mask = true_vec == 1
        pred_vec_pos_round = np.round(pred_vec[pos_mask])
        pos_acc = np.sum(pred_vec_pos_round == 1) / len(pred_vec_pos_round)

        neg_mask = true_vec == 0
        pred_vec_neg_round = np.round(pred_vec[neg_mask])
        neg_acc = np.sum(pred_vec_neg_round == 0) / len(pred_vec_neg_round)

        # auROC
        auroc = sklearn.metrics.roc_auc_score(true_vec, pred_vec)

        # Precision, recall, average precision, auPRC
        precis, recall, thresh = estimate_imbalanced_precision_recall(
            true_vec, pred_vec
        )
        pre, rec = precision_recall_scores(precis, recall, thresh)
        ap = average_precision(precis, recall)
        auprc = sklearn.metrics.auc(recall, precis)

        # Precision, average precision, auPRC, corrected for downsampling
        c_precis, recall, thresh = estimate_imbalanced_precision_recall(
            true_vec, pred_vec, neg_upsample_factor=val_neg_downsample
        )
        c_pre, _ = precision_recall_scores(c_precis, recall, thresh)
        c_ap = average_precision(c_precis, recall)
        c_auprc = sklearn.metrics.auc(recall, c_precis)

        return acc, pos_acc, neg_acc, auroc, pre, rec, ap, auprc, c_pre, c_ap, \
            c_auprc

    result = []
    num_tasks = np.shape(true_values)[1]
    for task_num in range(num_tasks):
        result.append(
            single_task_metrics(
                true_values[:,task_num], pred_values[:,task_num]
            )
        )

    # Transpose `result` to get metric lists, each list has all the tasks
    metrics = list(map(list, zip(*result)))
    labels = [
        "acc", "pos_acc", "neg_acc", "auroc", "pre", "rec", "ap", "auprc",
        "c_pre", "c_ap", "c_auprc"
    ]

    return dict(zip(labels, metrics))


def log_evaluation_metrics(metric_dict, logger, _run):
    """
    Given the result of `compute_evaluation_metrics`, logs them all to both
    the given logger and the given Sacred observer.
    """
    logger.info("Validation accuracies: " + ", ".join(
        [("%6.2f%%" % (acc * 100)) for acc in metric_dict["acc"]]
    ))
    logger.info("Validation POS accuracies: " + ", ".join(
        [("%6.2f%%" % (acc * 100)) for acc in metric_dict["pos_acc"]]
    ))
    logger.info("Validation NEG accuracies: " + ", ".join(
        [("%6.2f%%" % (acc * 100)) for acc in metric_dict["neg_acc"]]
    ))
    logger.info("Validation auROCs: " + ", ".join(
        [("%6.10f" % auroc) for auroc in metric_dict["auroc"]]
    ))
    logger.info("Validation precisions: " + ", ".join(
        [("%6.10f" % preci) for preci in metric_dict["pre"]]
    ))
    logger.info("Validation recalls: " + ", ".join(
        [("%6.10f" % recall) for recall in metric_dict["rec"]]
    ))
    logger.info("Validation average precisions: " + ", ".join(
        [("%6.10f" % preci) for preci in metric_dict["ap"]]
    ))
    logger.info("Validation auPRCs: " + ", ".join(
        [("%6.10f" % auprc) for auprc in metric_dict["auprc"]]
    ))
    logger.info("Validation precisions (corrected): " + ", ".join(
        [("%6.10f" % preci) for preci in metric_dict["c_pre"]]
    ))
    logger.info("Validation average precisions (corrected): " + ", ".join(
        [("%6.10f" % preci) for preci in metric_dict["c_ap"]]
    ))
    logger.info("Validation auPRCs (corrected): " + ", ".join(
        [("%6.10f" % auprc) for auprc in metric_dict["c_auprc"]]
    ))

    _run.log_scalar("val_acc", metric_dict["acc"])
    _run.log_scalar("val_pos_acc", metric_dict["pos_acc"])
    _run.log_scalar("val_neg_acc", metric_dict["neg_acc"])
    _run.log_scalar("val_auroc", metric_dict["auroc"])
    _run.log_scalar("val_precis", metric_dict["pre"])
    _run.log_scalar("val_recall", metric_dict["rec"])
    _run.log_scalar("val_avg_precis", metric_dict["ap"])
    _run.log_scalar("val_auprc", metric_dict["auprc"])
    _run.log_scalar("val_corr_precis", metric_dict["c_pre"])
    _run.log_scalar("val_corr_avg_precis", metric_dict["c_ap"])
    _run.log_scalar("val_corr_auprc", metric_dict["c_auprc"])
