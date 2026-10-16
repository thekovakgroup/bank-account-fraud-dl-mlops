
# ============================================================
# Milestone 08 — Development-Only Inference Replay Test
# Reproduces Milestone 07 predictions without retraining
# Month 7: SEALED
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_inference import FraudInference


def main():

    print("MILESTONE 08 — INFERENCE REPLAY VERIFICATION")
    print("=" * 65)

    artifact_root = (
        PROJECT_ROOT / "artifacts" / "milestone_07"
    )

    snapshot = (
        artifact_root / "development_months_0_to_6.csv"
    )

    reference_predictions = (
        artifact_root
        / "calibration_policy_v01"
        / "mlp_calibrated_scores.npz"
    )

    reference_labels = (
        artifact_root
        / "restart_bundle_v01"
        / "predictions.npz"
    )

    assert snapshot.is_file()
    assert reference_predictions.is_file()
    assert reference_labels.is_file()

    # --------------------------------------------------------
    # 1. Load the verified standalone inference pipeline.
    # --------------------------------------------------------

    model = FraudInference(project_root=PROJECT_ROOT)

    print("Standalone inference module: PASS")
    print("Artifact checksum verification: PASS")

    # --------------------------------------------------------
    # 2. Read ONLY the development snapshot.
    # Never open data/raw/Base.csv.
    # --------------------------------------------------------

    month6_parts = []

    for chunk in pd.read_csv(
        snapshot,
        chunksize=125000,
        low_memory=False,
    ):

        assert "month" in chunk.columns

        allowed_months = set(range(7))

        observed_months = set(
            chunk["month"].dropna().unique()
        )

        assert observed_months.issubset(
            allowed_months
        ), "Unexpected month in development snapshot"

        month6 = chunk.loc[
            chunk["month"] == 6
        ]

        if not month6.empty:
            month6_parts.append(month6)

    assert month6_parts, "Month 6 not found"

    evaluation = pd.concat(
        month6_parts,
        ignore_index=True,
    )

    assert len(evaluation) == 108168

    y_actual = evaluation[
        "fraud_bool"
    ].to_numpy(dtype=np.int8)

    assert int(y_actual.sum()) == 1450

    print("Development snapshot: PASS")
    print("Evaluation records:", len(evaluation))
    print("Confirmed fraud cases:", int(y_actual.sum()))

    # --------------------------------------------------------
    # 3. Load frozen reference results.
    # --------------------------------------------------------

    with np.load(
        reference_predictions,
        allow_pickle=False,
    ) as saved:

        expected_probabilities = (
            saved["mlp_m6_sigmoid"].copy()
        )

    with np.load(
        reference_labels,
        allow_pickle=False,
    ) as saved:

        expected_labels = saved["y_m6"].copy()

    assert np.array_equal(
        y_actual,
        expected_labels,
    ), "Development snapshot row alignment failed"

    assert expected_probabilities.shape == (108168,)

    print("Saved label alignment: PASS")

    # --------------------------------------------------------
    # 4. Prepare exactly the original 29 model features.
    # No target, month, or device_fraud_count.
    # --------------------------------------------------------

    config = model.feature_config

    feature_columns = (
        config["numeric_features"]
        + config["categorical_features"]
        + config["binary_features"]
    )

    assert len(feature_columns) == 29

    applications = evaluation.loc[
        :, feature_columns
    ].copy()

    assert "fraud_bool" not in applications.columns
    assert "month" not in applications.columns
    assert "device_fraud_count" not in applications.columns

    # --------------------------------------------------------
    # 5. Generate standalone calibrated predictions.
    # --------------------------------------------------------

    actual_probabilities = model.predict_proba(
        applications
    )

    assert actual_probabilities.shape == (108168,)

    max_absolute_difference = float(
        np.max(
            np.abs(
                actual_probabilities
                - expected_probabilities
            )
        )
    )

    print("\nPREDICTION REPRODUCIBILITY")
    print("-" * 65)

    print(
        "Maximum absolute probability difference:",
        f"{max_absolute_difference:.10f}",
    )

    assert np.allclose(
        actual_probabilities,
        expected_probabilities,
        rtol=1e-5,
        atol=1e-6,
    ), "Standalone predictions differ from Milestone 07"

    print("Calibrated prediction replay: PASS")

    # --------------------------------------------------------
    # 6. Reproduce frozen 5% investigation policy.
    # --------------------------------------------------------

    decisions = model.recommend_investigations(
        applications
    )

    selected = decisions[
        "investigate"
    ].to_numpy(dtype=bool)

    investigations = int(selected.sum())

    fraud_caught = int(
        y_actual[selected].sum()
    )

    assert investigations == 5408
    assert fraud_caught == 705

    assert np.allclose(
        decisions["fraud_probability"].to_numpy(),
        actual_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )

    print("\nFROZEN POLICY REPLAY")
    print("-" * 65)

    print(
        "Economic threshold:",
        f"{model.threshold:.4%}",
    )

    print(
        "Investigation capacity:",
        f"{model.max_fraction:.0%}",
    )

    print("Investigations:", investigations)
    print("Fraud caught:", fraud_caught)

    print("Frozen policy replay: PASS")

    # --------------------------------------------------------
    # 7. Final verification.
    # --------------------------------------------------------

    print("\nVERIFICATION")
    print("-" * 65)

    print("Model retraining: NO")
    print("Preprocessor refitting: NO")
    print("Calibrator refitting: NO")
    print("Reference predictions reproduced: YES")
    print("Month 6: EXPLORATORY REPLAY")
    print("Month 7: SEALED")

    print("\nStep 08.07: PASS")


if __name__ == "__main__":
    main()
