import numpy as np
import pytest

from src.evaluation.metrics import evaluate_binary_classifier


def test_evaluate_binary_classifier_returns_expected_metrics():
    y_true = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    y_probability = np.array(
        [0.10, 0.40, 0.35, 0.80],
        dtype=np.float32,
    )

    results = evaluate_binary_classifier(
        model_name="test_model",
        y_true=y_true,
        y_probability=y_probability,
        threshold=0.50,
    )

    assert results["model"] == "test_model"
    assert results["threshold"] == 0.50

    assert results["tn"] == 2
    assert results["fp"] == 0
    assert results["fn"] == 1
    assert results["tp"] == 1

    assert np.isclose(
        results["precision"],
        1.0,
    )

    assert np.isclose(
        results["recall"],
        0.5,
    )

    assert np.isclose(
        results["f1"],
        2 / 3,
    )

    assert np.isclose(
        results["roc_auc"],
        0.75,
    )

    assert np.isclose(
        results["pr_auc"],
        5 / 6,
    )

    assert np.isfinite(
        results["log_loss"]
    )


def test_rejects_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        evaluate_binary_classifier(
            model_name="test_model",
            y_true=np.array([0, 1]),
            y_probability=np.array([0.25]),
        )


def test_rejects_probability_outside_zero_one_range():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_binary_classifier(
            model_name="test_model",
            y_true=np.array([0, 1]),
            y_probability=np.array(
                [0.20, 1.10]
            ),
        )


def test_rejects_non_binary_targets():
    with pytest.raises(
        ValueError,
        match="binary labels 0 and 1",
    ):
        evaluate_binary_classifier(
            model_name="test_model",
            y_true=np.array([0, 2]),
            y_probability=np.array(
                [0.20, 0.80]
            ),
        )


def test_rejects_single_class_targets():
    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        evaluate_binary_classifier(
            model_name="test_model",
            y_true=np.array([0, 0, 0]),
            y_probability=np.array(
                [0.10, 0.20, 0.30]
            ),
        )