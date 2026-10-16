
# ============================================================
# Milestone 08 — Inference Contract Tests
# Synthetic data ONLY — Month 7 remains sealed
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fraud_inference import FraudInference


def main():

    print("MILESTONE 08 — INFERENCE CONTRACT TESTS")
    print("=" * 60)

    model = FraudInference(project_root=ROOT)
    config = model.feature_config

    # 1. Create 200 synthetic applications.
    n = 200

    data = {}

    for column in config["numeric_features"]:
        data[column] = np.linspace(
            0.1, 0.9, n
        ).astype(np.float32)

    for column in config["categorical_features"]:
        data[column] = ["UNSEEN_TEST_CATEGORY"] * n

    for column in config["binary_features"]:
        data[column] = np.zeros(
            n, dtype=np.float32
        )

    applications = pd.DataFrame(
        data,
        index=[f"application_{i}" for i in range(n)],
    )

    assert applications.shape == (200, 29)

    # 2. Check calibrated predictions.
    predictions = model.predict_proba(
        applications,
        batch_size=32,
    )

    assert predictions.shape == (200,)
    assert np.isfinite(predictions).all()
    assert np.all((predictions >= 0) & (predictions <= 1))

    print("Synthetic applications: PASS")
    print("Unknown categorical values: PASS")
    print("Valid calibrated probabilities: PASS")

    # 3. Confirm excluded fields cannot affect predictions.
    with_excluded = applications.copy()

    with_excluded["fraud_bool"] = 1
    with_excluded["month"] = 6
    with_excluded["device_fraud_count"] = 999

    excluded_predictions = model.predict_proba(
        with_excluded,
        batch_size=17,
    )

    np.testing.assert_allclose(
        predictions,
        excluded_predictions,
        rtol=1e-5,
        atol=1e-6,
    )

    print("Excluded feature protection: PASS")
    print("Different inference batch sizes: PASS")

    # 4. Missing required features must be rejected.
    missing_column = applications.drop(
        columns=["income"]
    )

    try:
        model.predict_proba(missing_column)
    except ValueError as error:
        assert "Missing required model features" in str(error)
    else:
        raise AssertionError(
            "Missing feature was not rejected"
        )

    print("Missing feature rejection: PASS")

    # 5. Verify investigation decisions.
    decisions = model.recommend_investigations(
        applications
    )

    assert decisions.index.equals(
        applications.index
    )

    assert decisions["investigate"].dtype == bool

    capacity = int(
        np.floor(n * model.max_fraction)
    )

    assert int(decisions["investigate"].sum()) <= capacity

    selected = decisions["investigate"].to_numpy()
    assert np.all(predictions[selected] > model.threshold)

    # Independently reproduce ranking and selections.
    eligible = np.flatnonzero(
        predictions > model.threshold
    )

    ranked = eligible[
        np.argsort(
            -predictions[eligible],
            kind="stable",
        )
    ]

    expected_selected = ranked[:capacity]

    actual_selected = np.flatnonzero(selected)

    assert np.array_equal(
        actual_selected,
        np.sort(expected_selected),
    )

    assert np.array_equal(
        np.sort(
            decisions.loc[
                decisions["investigate"],
                "review_priority",
            ].to_numpy()
        ),
        np.arange(1, len(actual_selected) + 1),
    )

    print("Investigation capacity: PASS")
    print("Economic threshold enforcement: PASS")
    print("Stable investigation ranking: PASS")

    # 6. Empty input should return empty predictions.
    empty = applications.iloc[:0]

    empty_predictions = model.predict_proba(empty)

    assert empty_predictions.shape == (0,)

    print("Empty input handling: PASS")

    # 7. Final report.
    print("\nVERIFICATION")
    print("-" * 60)

    print("Synthetic test records:", n)
    print("Required model features:", applications.shape[1])
    print("Investigation capacity:", capacity)
    print("Actual investigations:", len(actual_selected))
    print("Model retraining: NO")
    print("Calibration refitting: NO")
    print("Original dataset accessed: NO")
    print("Month 7: SEALED")

    print("\nStep 08.08: PASS")


if __name__ == "__main__":
    main()
