
# ============================================================
# Milestone 06 — Historical Experiment Import into MLflow
# Preview by default; --commit writes MLflow runs.
# No model training or transaction data access.
# ============================================================

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "milestone_06"

TRACKING_URI = "sqlite:///artifacts/milestone_06/mlflow_tracking.db"
EXPERIMENT_NAME = "bank-account-fraud-m06-historical"


def load_json(path):
    assert path.is_file(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    assert path.is_file(), f"Missing artifact: {path}"
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_records():
    registry = load_json(
        ARTIFACT_DIR / "experiment_registry_v03.json"
    )

    m05 = load_json(
        PROJECT_ROOT
        / "artifacts"
        / "milestone_05_reference"
        / "milestone_05_mlp_metadata.json"
    )

    hgb = load_json(
        ARTIFACT_DIR / "m06_hgb_reproduction_v01.json"
    )

    assert registry["experiment_count"] == 4
    assert len(registry["experiments"]) == 4
    assert registry["holdout_status"] == "SEALED"

    assert m05["data"]["test_status"] == "SEALED"
    assert hgb["holdout_status"] == "SEALED"

    records = []

    # Four previously completed MLP experiments.
    for experiment in registry["experiments"]:

        experiment_id = experiment["experiment_id"]

        assert experiment["status"].startswith("completed")

        if experiment_id == "m05_fixed_budget_baseline":
            assert abs(
                experiment["validation_pr_auc"]
                - m05["validation_results"]["pr_auc"]
            ) < 1e-12

            assert m05["training"]["epochs"] == experiment["epochs"]

            source_file = (
                "milestone_05_mlp_metadata.json"
            )

            source_hash = sha256_file(
                PROJECT_ROOT
                / "artifacts"
                / "milestone_05_reference"
                / source_file
            )

            model_family = "pytorch_mlp"

        else:
            archive = (
                PROJECT_ROOT
                / Path(experiment["artifact_path"])
            )

            actual_hash = sha256_file(archive)

            assert actual_hash == experiment["artifact_sha256"], (
                f"Archive checksum mismatch: {archive.name}"
            )

            source_file = archive.name
            source_hash = actual_hash
            model_family = "pytorch_mlp"

        records.append({
            "id": experiment_id,
            "params": {
                "model_family": model_family,
                "device": experiment["device"],
                "seed": experiment["seed"],
                "epochs": experiment["epochs"],
                "learning_rate": experiment["learning_rate"],
                "dropout_rate": experiment["dropout_rate"],
                "imbalance_strategy": experiment["imbalance_strategy"],
                "training_months": "0,1,2,3,4,5",
                "validation_month": 6,
            },
            "metrics": {
                "validation_pr_auc": experiment["validation_pr_auc"],
                "validation_roc_auc": experiment["validation_roc_auc"],
                "validation_loss": experiment["validation_loss"],
            },
            "tags": {
                "source_record": source_file,
                "source_sha256": source_hash,
                "original_source": experiment["source"],
                "original_source_commit": experiment.get(
                    "source_commit", "not_recorded"
                ),
            },
        })

    # Independently reproduced HistGradientBoosting baseline.
    for file_key, hash_key in [
        ("model_file", "model_sha256"),
        ("predictions_file", "predictions_sha256"),
    ]:
        assert sha256_file(
            ARTIFACT_DIR / hgb[file_key]
        ) == hgb[hash_key]

    records.append({
        "id": "m06_hgb_reproduced_v01",
        "params": {
            "model_family": "hist_gradient_boosting",
            "learning_rate": hgb["parameters"]["learning_rate"],
            "max_iter": hgb["parameters"]["max_iter"],
            "max_leaf_nodes": hgb["parameters"]["max_leaf_nodes"],
            "early_stopping": hgb["parameters"]["early_stopping"],
            "random_state": hgb["parameters"]["random_state"],
            "training_months": "0,1,2,3,4,5",
            "validation_month": 6,
        },
        "metrics": {
            "validation_pr_auc": hgb["validation_pr_auc"],
            "fit_seconds": hgb["training_seconds"],
            "training_iterations": hgb["training_iterations"],
        },
        "tags": {
            "source_record": "m06_hgb_reproduction_v01.json",
            "source_sha256": sha256_file(
                ARTIFACT_DIR / "m06_hgb_reproduction_v01.json"
            ),
            "model_sha256": hgb["model_sha256"],
            "predictions_sha256": hgb["predictions_sha256"],
            "original_source": "milestone_04_baseline_reproduction",
            "original_source_commit": "not_recorded",
        },
    })

    assert len(records) == 5
    assert len({r["id"] for r in records}) == 5

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Write historical runs to the local MLflow database.",
    )
    args = parser.parse_args()

    assert Path.cwd().resolve() == PROJECT_ROOT, (
        "Run this script from the repository root."
    )

    assert (ARTIFACT_DIR / "mlflow_tracking.db").is_file()

    records = prepare_records()

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    assert experiment is not None, (
        "Expected MLflow experiment was not found."
    )

    expected_location = (
        ARTIFACT_DIR / "mlflow_artifacts"
    ).resolve().as_uri()

    assert experiment.artifact_location == expected_location

    existing_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    existing_keys = [
        run.data.tags.get("historical_import_id")
        for run in existing_runs
    ]

    assert len(existing_keys) == len(set(existing_keys)), (
        "Duplicate historical import IDs already exist."
    )

    print("MILESTONE 06 — MLFLOW HISTORICAL IMPORT")
    print("=" * 60)
    print("Mode:", "COMMIT" if args.commit else "PREVIEW")
    print("Experiment:", experiment.name)
    print("Verified source records:", len(records))

    for record in records:
        record_id = record["id"]

        if record_id in existing_keys:
            print("SKIP (already imported):", record_id)
            continue

        if not args.commit:
            print(
                "READY:",
                record_id,
                "| PR-AUC:",
                f"{record['metrics']['validation_pr_auc']:.6f}",
            )
            continue

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=record_id,
        ):
            mlflow.set_tags({
                "historical_import": "true",
                "historical_import_id": record_id,
                "milestone": "06",
                "evaluation_split": "validation_month_6",
                "holdout_status": "SEALED",
                **record["tags"],
            })

            mlflow.log_params({
                key: str(value)
                for key, value in record["params"].items()
            })

            mlflow.log_metrics(record["metrics"])

        existing_keys.append(record_id)

        print("IMPORTED:", record_id)

    print("\nTest data: NOT ACCESSED")
    print("Model retraining: NONE")

    if args.commit:
        print("MLflow historical import: COMPLETE")
    else:
        print("Preview complete — no MLflow runs created.")


if __name__ == "__main__":
    main()
