import numpy as np

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    model_name,
    y_true,
    y_probability,
    threshold=0.50,
):
    """
    Evaluate a binary classifier from positive-class probabilities.

    Parameters
    ----------
    model_name : str
        Human-readable model name.

    y_true : array-like
        Ground-truth binary labels.

    y_probability : array-like
        Predicted probabilities for the positive class.

    threshold : float, default=0.50
        Probability threshold used to convert probabilities
        into binary predictions.

    Returns
    -------
    dict
        Threshold-independent and threshold-dependent
        classification metrics.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if y_probability.ndim != 1:
        raise ValueError(
            "y_probability must be one-dimensional."
        )

    if y_true.shape[0] != y_probability.shape[0]:
        raise ValueError(
            "y_true and y_probability must contain "
            "the same number of rows."
        )

    if y_true.shape[0] == 0:
        raise ValueError(
            "Evaluation arrays must not be empty."
        )

    if not np.isfinite(y_probability).all():
        raise ValueError(
            "Predicted probabilities contain "
            "non-finite values."
        )

    if (
        np.any(y_probability < 0.0)
        or np.any(y_probability > 1.0)
    ):
        raise ValueError(
            "Predicted probabilities must be "
            "between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    unique_targets = np.unique(y_true)

    if not np.all(
        np.isin(unique_targets, [0, 1])
    ):
        raise ValueError(
            "y_true must contain only binary "
            "labels 0 and 1."
        )

    if unique_targets.size < 2:
        raise ValueError(
            "y_true must contain both classes "
            "for ROC-AUC evaluation."
        )

    y_pred = (
        y_probability >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "model": model_name,
        "threshold": float(threshold),
        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "log_loss": log_loss(
            y_true,
            y_probability,
            labels=[0, 1],
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "actual_positive_rate": float(
            np.mean(y_true)
        ),
        "predicted_positive_rate": float(
            np.mean(y_pred)
        ),
    }