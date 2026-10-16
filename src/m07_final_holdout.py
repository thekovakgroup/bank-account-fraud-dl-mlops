"""
Milestone 07 — Frozen Model One-Time Final Holdout Evaluation

Instructor-led use:
    1. Dry run using development month 6 ONLY:
       python -m src.m07_final_holdout --mode dry-run

    2. AFTER all dry-run and code reviews, authorize exactly one
       final month-7 evaluation separately (NOT part of the dry run):
       python -m src.m07_final_holdout --mode final --authorize "I AUTHORIZE ONE-TIME MONTH-7 EVALUATION"

WARNING: final mode opens the independent month-7 holdout and creates
an exclusive, persistent consumption marker BEFORE reading raw records.
No retraining, refitting, policy tuning, or threshold search is performed.
Final evaluation must NOT be rerun after seeing the results.
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.data.validation import validate_schema
from src.fraud_inference import FraudInference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "configs/milestone_07_candidate_policy_v01.json"
SPLIT_PATH = PROJECT_ROOT / "configs/milestone_07_split_config.json"
DEVELOPMENT_PATH = (
    PROJECT_ROOT / "artifacts/milestone_07/development_months_0_to_6.csv"
)
RAW_PATH = PROJECT_ROOT / "data/raw/Base.csv"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/milestone_07"
MODEL_EXCLUSIONS = ("fraud_bool", "month", "device_fraud_count")
AUTHORIZE_PHRASE = "I AUTHORIZE ONE-TIME MONTH-7 EVALUATION"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def verify_protocol():
    """Check the frozen policy and historical development split without holdout access."""
    policy = load_json(POLICY_PATH)
    split = load_json(SPLIT_PATH)

    expected = {
        "train": [0, 1, 2, 3, 4],
        "calibration": [5],
        "evaluation": [6],
        "holdout": [7],
    }
    for role, months in expected.items():
        if split[role]["months"] != months:
            raise RuntimeError(f"Unexpected {role} split")

    if not (
        split["holdout"]["status"] == "SEALED"
        and split["holdout"]["access_allowed"] is False
        and split["safeguards"]["prohibit_holdout_access"] is True
    ):
        raise RuntimeError("Development holdout protections changed")

    if not (
        policy["status"] == "FROZEN_PROVISIONAL_CANDIDATE"
        and policy["production_approved"] is False
        and policy["data_protocol"]["holdout_status"] == "SEALED"
        and policy["model"]["calibration"] == "sigmoid_platt"
        and policy["model"]["architecture"] == [55, 64, 32, 1]
        and policy["decision_rule"]["tie_breaker"]
        == "stable_input_order_provisional"
        and policy["decision_rule"]["capacity_count_rule"]
        == "floor(0.05 * application_count)"
        and policy["decision_rule"]["investigate_if"]
        == "calibrated_probability > threshold"
    ):
        raise RuntimeError("Frozen candidate policy mismatch")

    assumptions = policy["business_assumptions"]
    if assumptions["hypothetical"] is not True:
        raise RuntimeError("Financial assumptions must remain hypothetical")
    if not math.isclose(
        float(policy["decision_rule"]["maximum_investigation_fraction"]),
        0.05,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Capacity policy mismatch")
    if not math.isclose(
        float(policy["decision_rule"]["probability_threshold"]),
        0.04929577464788732,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Threshold policy mismatch")

    safeguards = split["safeguards"]
    for name in (
        "exclude_target_from_features",
        "exclude_month_from_features",
        "exclude_device_fraud_count",
    ):
        if safeguards[name] is not True:
            raise RuntimeError(f"Feature-leakage protection changed: {name}")

    return policy


def load_one_month(source_path, desired_month, allowed_months, expected_rows):
    """Read one selected month, preserving source order and validating every chunk."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Required source not found: {source_path}")

    # Verify full schema and column order, not just the feature subset.
    with source_path.open("r", encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file))
    schema = validate_schema(header)
    if not schema["is_valid"]:
        raise RuntimeError(f"Dataset schema mismatch: {schema}")

    selected_chunks = []
    total_selected = 0

    # Chunking avoids keeping all ~1 million raw records in memory.
    for chunk in pd.read_csv(source_path, chunksize=40_000):
        if chunk.columns.tolist() != header:
            raise RuntimeError("Dataset column order changed")
        if chunk["month"].isna().any():
            raise RuntimeError("Missing month field")
        if not chunk["month"].isin(allowed_months).all():
            raise RuntimeError("Unexpected month value")

        selected = chunk.loc[chunk["month"].eq(desired_month)]
        if not selected.empty:
            selected_chunks.append(selected.copy())
            total_selected += len(selected)

    if total_selected != expected_rows:
        raise RuntimeError(
            f"Unexpected month-{desired_month} rows: "
            f"{total_selected:,} (expected {expected_rows:,})"
        )

    result = pd.concat(selected_chunks, ignore_index=True)
    if result["month"].ne(desired_month).any():
        raise RuntimeError("Selected evaluation frame contains other months")
    if result["fraud_bool"].isna().any() or not result["fraud_bool"].isin([0, 1]).all():
        raise RuntimeError("Invalid binary fraud labels")
    return result


def expected_calibration_error(y, probabilities, bins=10):
    """Fixed-width 10-bin ECE, an exploratory descriptive calibration metric."""
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    positions = np.clip(np.digitize(probabilities, boundaries[1:-1]), 0, bins - 1)
    total_error = 0.0
    for bin_index in range(bins):
        mask = positions == bin_index
        if mask.any():
            total_error += mask.mean() * abs(y[mask].mean() - probabilities[mask].mean())
    return float(total_error)


def calculate_report(frame, engine, policy, mode):
    """Evaluate frozen model and one full-cohort policy; never fit any component."""
    excluded = set(MODEL_EXCLUSIONS)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if len(feature_columns) != 29 or any(c in feature_columns for c in excluded):
        raise RuntimeError("Expected exactly 29 leakage-free raw model features")

    applications = frame.loc[:, feature_columns].copy()
    y = frame["fraud_bool"].to_numpy(dtype=np.int64)

    # IMPORTANT: apply the capacity to the COMPLETE selected month, not to chunks.
    recommendations = engine.recommend_investigations(applications)
    probabilities = recommendations["fraud_probability"].to_numpy(dtype=np.float64)
    investigated = recommendations["investigate"].to_numpy(dtype=bool)
    priorities = recommendations["review_priority"].to_numpy(dtype=np.int32)

    if not (len(probabilities) == len(y) == len(investigated)):
        raise RuntimeError("Prediction-label alignment failed")
    if not np.isfinite(probabilities).all() or not np.all(
        (probabilities >= 0) & (probabilities <= 1)
    ):
        raise RuntimeError("Invalid calibrated scores")

    n = len(y)
    capacity = int(np.floor(0.05 * n))
    if investigated.sum() > capacity:
        raise RuntimeError("Frozen review capacity exceeded")
    if np.any(investigated & (probabilities <= engine.threshold)):
        raise RuntimeError("Frozen risk threshold violated")
    if sorted(priorities[investigated].tolist()) != list(range(1, int(investigated.sum()) + 1)):
        raise RuntimeError("Investigation priority sequence invalid")
    if np.any(priorities[~investigated] != 0):
        raise RuntimeError("Unexpected non-investigation priority")

    tp = int(np.sum(investigated & (y == 1)))
    fp = int(np.sum(investigated & (y == 0)))
    fn = int(np.sum(~investigated & (y == 1)))
    tn = int(np.sum(~investigated & (y == 0)))
    if tp + fp + fn + tn != n:
        raise RuntimeError("Confusion matrix does not reconcile")

    assumptions = policy["business_assumptions"]
    investigation_cost = float(assumptions["investigation_cost"])
    false_positive_cost = float(assumptions["additional_false_positive_cost"])
    loss_per_fraud = float(assumptions["fraud_loss_per_case"])
    prevention_rate = float(assumptions["prevention_rate"])

    prevented_loss = tp * loss_per_fraud * prevention_rate
    review_cost = int(investigated.sum()) * investigation_cost
    fp_cost = fp * false_positive_cost
    net_savings = prevented_loss - review_cost - fp_cost

    metrics = {
        "applications": int(n),
        "fraud_cases": int(y.sum()),
        "fraud_prevalence": float(y.mean()),
        "average_precision_pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "ece_10_equal_width_bins": expected_calibration_error(y, probabilities),
        "investigations": int(investigated.sum()),
        "capacity": capacity,
        "actual_investigation_fraction": float(investigated.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "recall_at_frozen_policy": float(tp / (tp + fn)) if tp + fn else None,
        "precision_at_frozen_policy": float(tp / (tp + fp)) if tp + fp else None,
        "false_positive_rate_at_frozen_policy": float(fp / (fp + tn)) if fp + tn else None,
        "hypothetical_prevented_fraud_loss_usd": float(prevented_loss),
        "hypothetical_investigation_cost_usd": float(review_cost),
        "hypothetical_additional_fp_cost_usd": float(fp_cost),
        "hypothetical_net_savings_vs_no_review_usd": float(net_savings),
    }

    if mode == "dry-run":
        frozen = policy["exploratory_month_6"]
        expectations = {
            "applications": frozen["applications"],
            "fraud_cases": frozen["fraud_cases"],
            "investigations": frozen["investigations"],
            "true_positives": frozen["fraud_caught"],
        }
        for key, expected in expectations.items():
            if metrics[key] != expected:
                raise RuntimeError(
                    f"Development dry-run mismatch for {key}: "
                    f"{metrics[key]} != {expected}"
                )
        if not math.isclose(
            net_savings,
            float(frozen["hypothetical_savings_usd"]),
            abs_tol=1e-8,
        ):
            raise RuntimeError("Development economics replay mismatch")

    return metrics


def write_aggregate_report(output_dir, mode, metrics, policy):
    """Persist aggregate metrics only. Never save individual scores or raw applications."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "aggregate_metrics_v01.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {report_path}")

    report = {
        "milestone": "07",
        "mode": mode,
        "evaluated_month": 6 if mode == "dry-run" else 7,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Frozen FraudMLP 55-64-32-1",
        "calibration": "sigmoid_platt",
        "policy_status": "FROZEN_PROVISIONAL_CANDIDATE",
        "production_approved": False,
        "frozen_threshold": float(policy["decision_rule"]["probability_threshold"]),
        "frozen_capacity_fraction": float(policy["decision_rule"]["maximum_investigation_fraction"]),
        "checkpoint_sha256": policy["model"]["checkpoint_sha256"],
        "calibrator_bundle_sha256": policy["model"]["calibrator_bundle_sha256"],
        "financial_assumptions_hypothetical": True,
        "final_holdout_consumed_once": mode == "final",
        "no_model_or_calibrator_fit": True,
        "no_policy_tuning": True,
        "metrics": metrics,
        "limitations": [
            "Capacity and policy are defined for the entire evaluation cohort.",
            "Money amounts use hypothetical inputs, not validated bank costs.",
            "Independent holdout results must not be used to tune this candidate.",
            "Stable input-order tie-breaker is provisional, not production ready.",
            "The model and policy are not approved for production.",
        ],
    }

    # Exclusive create also prevents inadvertent reruns from replacing results.
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print("Aggregate metrics written:", report_path.relative_to(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("dry-run", "final"))
    parser.add_argument("--authorize", default="")
    args = parser.parse_args()

    print("MILESTONE 07 — FROZEN FINAL EVALUATION")
    print("=" * 58)
    print("Mode:", args.mode)

    policy = verify_protocol()
    engine = FraudInference(project_root=PROJECT_ROOT)
    if not math.isclose(
        engine.threshold,
        float(policy["decision_rule"]["probability_threshold"]),
        abs_tol=1e-15,
    ):
        raise RuntimeError("Inference threshold differs from frozen policy")

    if args.mode == "dry-run":
        if args.authorize:
            raise RuntimeError("Do not supply final authorization for a dry run")
        source = DEVELOPMENT_PATH
        month, expected = 6, 108_168
        allowed_months = set(range(7))
        destination = ARTIFACT_ROOT / "final_holdout_dry_run_v01"
        print("Input: development months 0–6 ONLY")
        print("Month 7 accessed: NO")

    else:
        if args.authorize != AUTHORIZE_PHRASE:
            raise RuntimeError("Final holdout requires exact explicit authorization")
        source = RAW_PATH
        month, expected = 7, 96_843
        allowed_months = set(range(8))
        destination = ARTIFACT_ROOT / "final_holdout_once_v01"
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / "ONE_TIME_ACCESS_STARTED.lock"
        report_path = destination / "aggregate_metrics_v01.json"
        if report_path.exists():
            raise RuntimeError("Holdout evaluation result already exists; final is closed")

        # IMPORTANT: once this succeeds, DO NOT attempt another holdout run,
        # even if an error follows. Diagnose without opening holdout again.
        try:
            descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("Holdout access was already started; rerun prohibited") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(
                "Month-7 one-time evaluation initiated at "
                + datetime.now(timezone.utc).isoformat()
                + "\nNever rerun against month 7.\n"
            )
        print("One-time holdout access marker: CREATED")
        print("No second final run permitted")

    frame = load_one_month(source, month, allowed_months, expected)
    metrics = calculate_report(frame, engine, policy, args.mode)
    write_aggregate_report(destination, args.mode, metrics, policy)

    print("Rows:", f"{metrics['applications']:,}")
    print("Fraud cases:", f"{metrics['fraud_cases']:,}")
    print("Investigations:", f"{metrics['investigations']:,}")
    print("Fraud caught:", f"{metrics['true_positives']:,}")
    print("Average precision (PR-AUC):", f"{metrics['average_precision_pr_auc']:.6f}")
    print("Brier score:", f"{metrics['brier_score']:.6f}")
    print("Hypothetical net savings (USD):", f"{metrics['hypothetical_net_savings_vs_no_review_usd']:,.2f}")
    print("No training or tuning: PASS")
    if args.mode == "dry-run":
        print("Development month-6 replay: PASS")
        print("Month 7: SEALED")
        print("Step 07-FH development dry run: PASS")
    else:
        print("Month 7: CONSUMED ONCE — DO NOT REUSE FOR MODEL SELECTION")
        print("Step 07-FH final holdout: COMPLETE")


if __name__ == "__main__":
    main()
