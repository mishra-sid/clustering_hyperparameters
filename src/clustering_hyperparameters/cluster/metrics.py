from sklearn import metrics
import numpy as np


def adjusted_rand_score(target_labels, pred_labels):
    (tn, fp), (fn, tp) = metrics.pair_confusion_matrix(target_labels, pred_labels)
    (tn, fp), (fn, tp) = (np.float64(tn), np.float64(fp)), (np.float64(fn) , np.float64(tp))
    
    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))


def adjusted_mutual_info_score(target_labels, pred_labels):
    return metrics.adjusted_mutual_info_score(target_labels, pred_labels)


def completeness_score(target_labels, pred_labels):
    return metrics.completeness_score(target_labels, pred_labels)


def fowlkes_mallows_score(target_labels, pred_labels):
    return metrics.fowlkes_mallows_score(target_labels, pred_labels)


def homogeneity_score(target_labels, pred_labels):
    return metrics.homogeneity_score(target_labels, pred_labels)


def mutual_info_score(target_labels, pred_labels):
    return metrics.mutual_info_score(target_labels, pred_labels)


def normalized_mutual_info_score(target_labels, pred_labels):
    return metrics.normalized_mutual_info_score(target_labels, pred_labels)


def rand_score(target_labels, pred_labels):
    return metrics.rand_score(target_labels, pred_labels)


def v_measure_score(target_labels, pred_labels):
    return metrics.v_measure_score(target_labels, pred_labels)


def get_all_metrics(target_labels, pred_labels):
    return {
        "adjusted_rand_score": adjusted_rand_score(target_labels, pred_labels),
        "adjusted_mutual_info_score": adjusted_mutual_info_score(target_labels, pred_labels),
        "completeness_score": completeness_score(target_labels, pred_labels),
        "fowlkes_mallows_score": fowlkes_mallows_score(target_labels, pred_labels),
        "homogeneity_score": homogeneity_score(target_labels, pred_labels),
        "mutual_info_score": mutual_info_score(target_labels, pred_labels),
        "normalized_mutual_info_score": normalized_mutual_info_score(target_labels, pred_labels),
        "rand_score": rand_score(target_labels, pred_labels),
        "v_measure_score": v_measure_score(target_labels, pred_labels)
    }
