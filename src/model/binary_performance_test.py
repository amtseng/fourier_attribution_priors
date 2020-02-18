import numpy as np
import sklearn.metrics
import model.binary_performance as binary_performance

def test_accuracies():
    np.random.seed(20200218)
    vec_size = 50
    true_vals = np.random.randint(2, size=vec_size)
    pred_vals = np.random.random(vec_size)

    print("Testing accuracies...")
    pred_vals_rounded = np.round(pred_vals)
    acc, pos_acc, neg_acc = binary_performance.accuracies(
        true_vals, pred_vals_rounded
    )

    num_pos, num_neg = 0, 0
    num_pos_right, num_neg_right = 0, 0
    for i in range(vec_size):
        if true_vals[i] == 1:
            num_pos += 1
            if pred_vals[i] >= 0.5:
                num_pos_right += 1
        else:
            num_neg += 1
            if pred_vals[i] < 0.5:
                num_neg_right += 1

    print("\tSame result? %s" % all([
        acc == (num_pos_right + num_neg_right) / vec_size,
        pos_acc == num_pos_right / num_pos,
        neg_acc == num_neg_right / num_neg
    ]))


def test_corrected_precision_auprc():
    np.random.seed(20200218)
    vec_size = 10000
    neg_upsample_factor = 5
    estimate_tolerance = 0.02

    def test_single_result(true_vec, pred_vec):
        """
        Tests similarity of precision/recall/auPRC computation to `sklearn`
        library, without any downsampling of negatives
        """
        precis, recall, thresh = \
            binary_performance.estimate_imbalanced_precision_recall(
                true_vec, pred_vec, neg_upsample_factor=1
            )
        precis_score, recall_score = binary_performance.precision_recall_scores(
            precis, recall, thresh
        )
        auprc = sklearn.metrics.auc(recall, precis)
        sk_precis, sk_recall, sk_thresh = \
            sklearn.metrics.precision_recall_curve(true_vec, pred_vec)
        sk_precis_score = sklearn.metrics.precision_score(
            true_vec, np.round(pred_vec)
        )
        sk_recall_score = sklearn.metrics.recall_score(
            true_vec, np.round(pred_vec)
        )
        sk_auprc = sklearn.metrics.auc(sk_recall, sk_precis)
        print("\tSame result? %s" % all([
            np.allclose(precis, sk_precis),
            np.allclose(recall, sk_recall),
            np.allclose(thresh, sk_thresh),
            precis_score == sk_precis_score,
            recall_score == sk_recall_score,
            auprc == sk_auprc
        ]))

    def test_neg_sampling_result(true_vec, pred_vec, neg_upsample_factor):
        """
        Tests that after down-sampling negatives and re-inflating them using
        `neg_upsample_factor`, the precision/recall/auPRC are roughly the same
        as if no down-sampling had occurred.
        """
        # Get results without downsampling negatives
        precis, recall, thresh = \
            binary_performance.estimate_imbalanced_precision_recall(
                true_vec, pred_vec, neg_upsample_factor=1
            )
        precis_score, recall_score = binary_performance.precision_recall_scores(
            precis, recall, thresh
        )
        auprc = sklearn.metrics.auc(recall, precis)

        # Subsample negatives
        pos_mask = true_vals == 1
        neg_mask = true_vals == 0
        sub_mask = np.random.choice(
            [True, False], size=vec_size,
            p=[1 / neg_upsample_factor, 1 - (1 / neg_upsample_factor)]
        )
        keep_mask = pos_mask | (neg_mask & sub_mask)
        c_precis, c_recall, c_thresh = \
            binary_performance.estimate_imbalanced_precision_recall(
                true_vec[keep_mask], pred_vec[keep_mask],
                neg_upsample_factor=neg_upsample_factor
            )
        c_precis_score, c_recall_score = \
            binary_performance.precision_recall_scores(
                c_precis, c_recall, c_thresh
            )
        c_auprc = sklearn.metrics.auc(c_recall, c_precis)
        print("\tAll within %s? %s" % (
            estimate_tolerance,
            all([
                abs(precis_score - c_precis_score) < estimate_tolerance,
                abs(recall_score - c_recall_score) < estimate_tolerance,
                abs(auprc - c_auprc) < estimate_tolerance
            ])
        ))


    true_vals = np.random.choice(2, size=vec_size)

    # Random predictions
    pred_vals = np.random.random(vec_size)

    print("Testing precision/recall/auPRC on random data without correction...")
    test_single_result(true_vals, pred_vals)

    print("Testing precision/recall/auPRC on random data with correction...")
    test_neg_sampling_result(true_vals, pred_vals, 5)

    # Predictions are a bit closer to being correct; make predictions Gaussian
    # noise centered as 0 or 1, and cut off anything outside [0, 1]
    pred_vals = np.empty(vec_size)
    rand = np.random.randn(vec_size) / 2
    pos_mask = true_vals == 1
    pred_vals[pos_mask] = rand[pos_mask] + 1
    neg_mask = true_vals == 0
    pred_vals[neg_mask] = rand[neg_mask]
    pred_vals[pred_vals < 0] = 0
    pred_vals[pred_vals > 1] = 1

    print("Testing precision/recall/auPRC on good data without correction...")
    test_single_result(true_vals, pred_vals)

    print("Testing precision/recall/auPRC on good data with correction...")
    test_neg_sampling_result(true_vals, pred_vals, 5)


class FakeLogger:
    def log_scalar(self, a, b):
        pass


def test_all_metrics_on_different_predictions():
    np.random.seed(20200218)
    batch_size, num_tasks  = 100, 2
    true_vals = np.random.randint(2, size=(batch_size, num_tasks))

    _run = FakeLogger()

    # Make some "perfect" predictions, which are identical to truth
    print("Testing all metrics on some perfect predictions...")
    pred_vals = true_vals
    metrics = binary_performance.compute_performance_metrics(
        true_vals, pred_vals, 1
    )
    binary_performance.log_performance_metrics(metrics, "Perfect", _run)

    # Make some "good" predictions, which are close to truth; make predictions
    # Gaussian noise centered as 0 or 1, and cut off anything outside [0, 1]
    print("Testing all metrics on some good predictions...")
    pred_vals = np.empty((batch_size, num_tasks))
    rand = np.random.randn(batch_size, num_tasks) / 2
    pos_mask = true_vals == 1
    pred_vals[pos_mask] = rand[pos_mask] + 1
    neg_mask = true_vals == 0
    pred_vals[neg_mask] = rand[neg_mask]
    pred_vals[pred_vals < 0] = 0
    pred_vals[pred_vals > 1] = 1
    metrics = binary_performance.compute_performance_metrics(
        true_vals, pred_vals, 1
    )
    binary_performance.log_performance_metrics(metrics, "Good", _run)

    # Make some "bad" predictions, which are just random
    print("Testing all metrics on some bad predictions...")
    pred_vals = np.random.random((batch_size, num_tasks))
    metrics = binary_performance.compute_performance_metrics(
        true_vals, pred_vals, 1
    )
    binary_performance.log_performance_metrics(metrics, "Bad", _run)


if __name__ == "__main__":
    test_accuracies()
    test_corrected_precision_auprc()
    test_all_metrics_on_different_predictions()
