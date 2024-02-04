"""
Description:
-----------
This script hosts many helper functions to make notebooks cleaner.
 The hope is to not distract users with ugly code.

Alot of these were sourced from Dr. Lagerquist and found originally in his gewitter repo
 (https://github.com/thunderhoser/GewitterGefahr).

"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import sklearn

TOLERANCE = 1e-6


NUM_TRUE_POSITIVES_KEY = "num_true_positives"
NUM_FALSE_POSITIVES_KEY = "num_false_positives"
NUM_FALSE_NEGATIVES_KEY = "num_false_negatives"
NUM_TRUE_NEGATIVES_KEY = "num_true_negatives"


MIN_BINARIZATION_THRESHOLD = 0.0
MAX_BINARIZATION_THRESHOLD = 1.0 + TOLERANCE


DEFAULT_FORECAST_PRECISION = 1e-4


def get_contingency_table(forecast_labels, observed_labels):
    """分割表を作成する

    Args:
        forecast_labels (Array): See documentation for
                                 _check_forecast_and_observed_labels.
        observed_labels (Array): See doc for _check_forecast_and_observed_labels.

    Returns:
        dict: contingency_table_as_dict
            contingency_table_as_dict['num_true_positives']: Number of true positives.
            contingency_table_as_dict['num_false_positives']: Number of false positives.
            contingency_table_as_dict['num_false_negatives']: Number of false negatives.
            contingency_table_as_dict['num_true_negatives']: Number of true negatives.
    """

    true_positive_indices = np.where(
        np.logical_and(forecast_labels == 1, observed_labels == 1)
    )[0]
    false_positive_indices = np.where(
        np.logical_and(forecast_labels == 1, observed_labels == 0)
    )[0]
    false_negative_indices = np.where(
        np.logical_and(forecast_labels == 0, observed_labels == 1)
    )[0]
    true_negative_indices = np.where(
        np.logical_and(forecast_labels == 0, observed_labels == 0)
    )[0]

    return {
        NUM_TRUE_POSITIVES_KEY: len(true_positive_indices),
        NUM_FALSE_POSITIVES_KEY: len(false_positive_indices),
        NUM_FALSE_NEGATIVES_KEY: len(false_negative_indices),
        NUM_TRUE_NEGATIVES_KEY: len(true_negative_indices),
    }


def get_pod(contingency_table_as_dict):
    """
    捕捉率（probability of detection）TP/TP+FN  を計算する

    Args:
        contingency_table_as_dict: ictionary created by get_contingency_table

    Return:
        probability_of_detection: POD.
    """

    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY])
    return numerator / denominator


def get_sr(contingency_table_as_dict):
    """Success ratio TP/TP+FP を計算する

    Args:
        contingency_table_as_dict: ictionary created by get_contingency_table

    Returns:
        success_ratio: Success ratio.
    """

    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY])
    return numerator / denominator


def get_acc(contingency_table_as_dict):
    """
    accuracy (的中率) TP+PN/N を計算する

    Args:
        contingency_table_as_dict: ictionary created by get_contingency_table

    Returns:
        accuracy: accuracy.
    """

    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY]
        + contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    )
    numerator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    )

    return 100 * (numerator / denominator)


def csi_from_sr_and_pod(success_ratio_array, pod_array):
    """
    スレットスコア（CSI, critical success index）を success ratio と POD から求める

    Args:
        success_ratio_array: np array (any shape) of success ratios.
        pod_array: np array (same shape) of POD values.

    Returns:
        csi_array: np array (same shape) of CSI values.
    """

    return (success_ratio_array**-1 + pod_array**-1 - 1.0) ** -1


def frequency_bias_from_sr_and_pod(success_ratio_array, pod_array):
    """
    バイアススコアを success ratio と POD から求める

    Args:
        success_ratio_array: np array (any shape) of success ratios.
        pod_array: np array (same shape) of POD values.

    Returns:
        frequency_bias_array: np array (same shape) of frequency biases.

    """
    return pod_array / success_ratio_array


def get_far(contingency_table_as_dict):
    """
    FAR (false-alarm rate) を計算する

    Args:
        contingency_table_as_dict: dictionary created by get_contingency_table

    Returns:
        false_alarm_rate: FAR.
    """
    return 1.0 - get_sr(contingency_table_as_dict)


def get_pofd(contingency_table_as_dict):
    """
    POFD (probability of false detection) FP/FP+TN を計算する

    Args:
        contingency_table_as_dict: dictionary created by get_contingency_table

    Returns:
        probability_of_false_detection: POFD.
    """

    denominator = (
        contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY])
    return numerator / denominator


def get_points_in_roc_curve(
    forecast_probabilities=None,
    observed_labels=None,
    threshold_arg=None,
):
    """
    ROC曲線を計算する

    N = number of forecasts
    T = number of binarization thresholds

    Args:
        forecast_probabilities: See documentation for
                         `_check_forecast_probs_and_observed_labels`.
        observed_labels: See doc for
                        `_check_forecast_probs_and_observed_labels`.
        threshold_arg: See documentation for get_binarization_thresholds.
        forecast_precision: See doc for get_binarization_thresholds.

    Return:
        pofd_by_threshold: length-T np array of POFD values, to be plotted on the x-axis.
        pod_by_threshold: length-T np array of POD values, to be plotted on the y-axis.
    """

    binarization_thresholds = get_binarization_thresholds(threshold_arg=threshold_arg)

    num_thresholds = len(binarization_thresholds)
    pofd_by_threshold = np.full(num_thresholds, np.nan)
    pod_by_threshold = np.full(num_thresholds, np.nan)

    for i in range(num_thresholds):
        these_forecast_labels = binarize_forecast_probs(
            forecast_probabilities, binarization_thresholds[i]
        )
        this_contingency_table_as_dict = get_contingency_table(
            these_forecast_labels, observed_labels
        )

        pofd_by_threshold[i] = get_pofd(this_contingency_table_as_dict)
        pod_by_threshold[i] = get_pod(this_contingency_table_as_dict)

    return pofd_by_threshold, pod_by_threshold


def get_binarization_thresholds(threshold_arg):
    """
    Returns list of binarization thresholds.
    To understand the role of binarization thresholds, see
    binarize_forecast_probs.

    Args:
        threshold_arg: Main threshold argument.  May be in one of 2 formats.
            [1] 1-D np array.  In this case threshold_arg will be treated as an array
                of binarization thresholds.
            [2] Positive integer.  In this case threshold_arg will be treated as the
                number of binarization thresholds, equally spaced from 0...1.

    Return:
        binarization_thresholds: 1-D np array of binarization thresholds.

    Raises:
        ValueError: if threshold_arg cannot be interpreted.
    """

    if isinstance(threshold_arg, np.ndarray):
        binarization_thresholds = copy.deepcopy(threshold_arg)
    else:
        num_thresholds = copy.deepcopy(threshold_arg)

        binarization_thresholds = np.linspace(0, 1, num=num_thresholds, dtype=float)

    return _pad_binarization_thresholds(binarization_thresholds)


def _pad_binarization_thresholds(thresholds):
    """
    Pads an array of binarization thresholds. Specifically,
    this method ensures that the array contains 0 and a number slightly greater than 1.
    This ensures that:

    [1] For the lowest threshold, POD = POFD = 1, which is the top-right corner
        of the ROC curve.
    [2] For the highest threshold, POD = POFD = 0, which is the bottom-left
        corner of the ROC curve.

    Args:
        thresholds: 1-D np array of binarization thresholds.

    Returns:
        thresholds: 1-D np array of binarization thresholds (possibly with new elements).
    """

    thresholds = np.sort(thresholds)

    if thresholds[0] > MIN_BINARIZATION_THRESHOLD:
        thresholds = np.concatenate(
            (np.array([MIN_BINARIZATION_THRESHOLD]), thresholds)
        )

    if thresholds[-1] < MAX_BINARIZATION_THRESHOLD:
        thresholds = np.concatenate(
            (thresholds, np.array([MAX_BINARIZATION_THRESHOLD]))
        )

    return thresholds


def binarize_forecast_probs(forecast_probabilities, binarization_threshold):
    """
    Binarizes probabilistic forecasts, turning them into deterministic ones.
    N = number of forecasts

    Args:
        forecast_probabilities: length-N numpy array with forecast
            probabilities of some event (e.g., tornado).
        binarization_threshold: Binarization threshold (f*).
            All forecasts >= f* will be turned into "yes" forecasts;
            all forecasts < f* will be turned into "no".

    Returns:
        forecast_labels: length-N integer numpy array of deterministic
            forecasts (1 for "yes", 0 for "no").
    """

    forecast_labels = np.full(len(forecast_probabilities), 0, dtype=int)
    forecast_labels[forecast_probabilities >= binarization_threshold] = 1

    return forecast_labels


def get_area_under_roc_curve(pofd_by_threshold, pod_by_threshold):
    """
    Computes area under ROC curve. This calculation ignores NaN's.
    If you use `sklearn.metrics.auc` without this wrapper,
    if either input array contains any NaN, the result will be NaN.
    T = number of binarization thresholds

    Args:
        pofd_by_threshold: length-T numpy array of POFD values.
        pod_by_threshold: length-T numpy array of corresponding POD values.

    Returns:
        area_under_curve: Area under ROC curve.
    """

    sort_indices = np.argsort(-pofd_by_threshold)
    pofd_by_threshold = pofd_by_threshold[sort_indices]
    pod_by_threshold = pod_by_threshold[sort_indices]

    nan_flags = np.logical_or(np.isnan(pofd_by_threshold), np.isnan(pod_by_threshold))
    if np.all(nan_flags):
        return np.nan

    real_indices = np.where(np.invert(nan_flags))[0]

    return sklearn.metrics.auc(
        pofd_by_threshold[real_indices], pod_by_threshold[real_indices]
    )


def make_performance_diagram_axis(
    ax=None, figsize=(5, 5), CSIBOOL=True, FBBOOL=True, csi_cmap="Greys_r"
):
    """パフォーマンスダイアグラムのAxesオブジェクトを作成する"""

    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig.set_facecolor("w")
        ax = plt.gca()

    if CSIBOOL:
        sr_array = np.linspace(0.001, 1, 200)
        pod_array = np.linspace(0.001, 1, 200)
        X, Y = np.meshgrid(sr_array, pod_array)
        csi_vals = csi_from_sr_and_pod(X, Y)
        pm = ax.contourf(X, Y, csi_vals, levels=np.arange(0, 1.1, 0.1), cmap=csi_cmap)
        plt.colorbar(pm, ax=ax, label="CSI")

    if FBBOOL:
        fb = frequency_bias_from_sr_and_pod(X, Y)
        bias = ax.contour(
            X,
            Y,
            fb,
            levels=[0.25, 0.5, 1, 1.5, 2, 3, 5],
            linestyles="--",
            colors="Grey",
        )
        plt.clabel(
            bias,
            inline=True,
            inline_spacing=5,
            fmt="%.2f",
            fontsize=10,
            colors="LightGrey",
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("SR")
    ax.set_ylabel("POD")
    return ax


def get_mae(y, yhat):
    """Calcualte the mean absolute error"""
    return np.mean(np.abs(y - yhat))


def get_rmse(y, yhat):
    """Calcualte the root mean squared error"""
    return np.sqrt(np.mean((y - yhat) ** 2))


def get_bias(y, yhat):
    """Calcualte the mean bias (i.e., error)"""
    return np.mean(y - yhat)


def get_r2(y, yhat):
    """Calcualte the coef. of determination (R^2)"""
    ybar = np.mean(y)
    return 1 - (np.sum((y - yhat) ** 2)) / (np.sum((y - ybar) ** 2))
