
# ============================================================
# MILESTONE 10 - MLFLOW MODEL REGISTRY
# Step 10.62 - Guarded Frozen Model Registration
# ============================================================

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import mlflow

from src.tracking.export_frozen_mlflow_model import EXPORT_ROOT
from src.tracking.prepare_frozen_bundle import (
    BUNDLE_ROOT,
    PROJECT_ROOT,
    prepare_inventory,
    sha256_file,
)


MODEL_NAME = "bank_account_fraud_weighted_mlp_frozen"

EXPERIMENT_NAME = "bank-account-fraud-m10-registry"

ARTIFACT_PATH = "frozen_fraud_pyfunc"

DB_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "milestone_06"
    / "mlflow_tracking.db"
)

TRACKING_URI = (
    "sqlite:///artifacts/milestone_06/mlflow_tracking.db"
)

EXPERIMENT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "milestone_10"
    / "mlflow_registry_runs"
)


def check_registry_read_only():
    """Inspect the existing registry without modifying it."""

    if not DB_PATH.is_file():
        raise FileNotFoundError(DB_PATH)

    connection = sqlite3.connect(
        DB_PATH.resolve().as_uri() + "?mode=ro",
        uri=True,
    )

    try:
        registered = connection.execute(
            "SELECT COUNT(*) FROM registered_models "
            "WHERE name = ?",
            (MODEL_NAME,),
        ).fetchone()[0]

        versions = connection.execute(
            "SELECT COUNT(*) FROM model_versions "
            "WHERE name = ?",
            (MODEL_NAME,),
        ).fetchone()[0]

    finally:
        connection.close()

    if registered or versions:
        raise RuntimeError(
            "Registration target already exists. "
            "Refusing to create a duplicate version."
        )

    return registered, versions


def verify_export():
    """Verify the frozen model and exported MLflow package."""

    source_files = prepare_inventory()


    # --------------------------------------------------------
    # Resolve the actual artifact path from MLflow metadata.
    # --------------------------------------------------------

    import yaml

    mlmodel_path = EXPORT_ROOT / "MLmodel"

    if not mlmodel_path.is_file():
        raise FileNotFoundError(
            "Missing exported MLflow model metadata."
        )

    metadata = yaml.safe_load(
        mlmodel_path.read_text(encoding="utf-8")
    )

    artifact_mapping = (
        metadata["flavors"]["python_function"]["artifacts"]
    )

    relative_bundle_path = artifact_mapping[
        "frozen_bundle"
    ]["path"]

    exported_bundle = (
        EXPORT_ROOT / relative_bundle_path
    ).resolve()

    # Ensure MLflow artifacts remain inside the export.
    if not exported_bundle.is_relative_to(
        EXPORT_ROOT.resolve()
    ):
        raise ValueError(
            "Export artifact path escapes MLflow package."
        )

    if not exported_bundle.is_dir():
        raise FileNotFoundError(
            f"Missing packaged frozen bundle: {exported_bundle}"
        )

    print(
        "MLflow artifact path:",
        relative_bundle_path,
    )

    for source in source_files:
        relative = source.relative_to(PROJECT_ROOT)

        bundle_copy = BUNDLE_ROOT / relative
        export_copy = exported_bundle / relative

        expected_hash = sha256_file(source)

        for packaged in (bundle_copy, export_copy):
            if not packaged.is_file():
                raise FileNotFoundError(packaged)

            if sha256_file(packaged) != expected_hash:
                raise ValueError(
                    f"Packaged checksum mismatch: {relative}"
                )

    policy = json.loads(
        (
            exported_bundle
            / "configs"
            / "milestone_07_candidate_policy_v01.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        policy["status"]
        == "FROZEN_PROVISIONAL_CANDIDATE"
    )

    assert policy["production_approved"] is False

    # Fingerprint the complete saved MLflow package.
    digest = hashlib.sha256()

    for path in sorted(EXPORT_ROOT.rglob("*")):
        if path.is_file():
            relative = path.relative_to(EXPORT_ROOT)

            digest.update(
                relative.as_posix().encode("utf-8")
            )

            digest.update(
                bytes.fromhex(sha256_file(path))
            )

    return len(source_files), digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--commit",
        action="store_true",
        help="Log and register the verified frozen model.",
    )

    args = parser.parse_args()

    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise RuntimeError(
            "Run this module from the repository root."
        )

    print("MILESTONE 10 - GUARDED MODEL REGISTRATION")
    print("=" * 55)
    print("Mode:", "COMMIT" if args.commit else "PREVIEW")

    # --------------------------------------------------------
    # 1. Verify private model artifacts and export.
    # --------------------------------------------------------

    file_count, fingerprint = verify_export()

    print("Verified frozen artifacts:", file_count)
    print("Export package SHA-256:", fingerprint)

    # --------------------------------------------------------
    # 2. Verify no registered model exists.
    # --------------------------------------------------------

    models, versions = check_registry_read_only()

    print("Existing target models:", models)
    print("Existing target versions:", versions)
    print("Registered model name:", MODEL_NAME)
    print("Experiment name:", EXPERIMENT_NAME)
    print("Production approved: FALSE")
    print("Month-7 holdout: CONSUMED - NOT ACCESSED")

    if not args.commit:
        print("Preview complete - no registry changes.")
        print("Step 10.62: PREVIEW PASS")
        return

    # --------------------------------------------------------
    # 3. Connect to the existing local MLflow backend.
    # --------------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        EXPERIMENT_ARTIFACT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(
                EXPERIMENT_ARTIFACT_ROOT.resolve().as_uri()
            ),
        )
    else:
        experiment_id = experiment.experiment_id

    # --------------------------------------------------------
    # 4. Store the exported model in an MLflow run.
    # --------------------------------------------------------

    tags = {
        "candidate_status": "FROZEN_PROVISIONAL_CANDIDATE",
        "production_approved": "false",
        "holdout_status": "CONSUMED",
        "model_type": "weighted_pytorch_mlp",
        "calibration": "platt",
        "input_features": "29",
        "research_only": "true",
        "export_sha256": fingerprint,
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="m10_frozen_model_registry_v01",
    ) as run:
        mlflow.set_tags(tags)

        mlflow.log_artifacts(
            str(EXPORT_ROOT),
            artifact_path=ARTIFACT_PATH,
        )

        run_id = run.info.run_id

    # --------------------------------------------------------
    # 5. Register the run-linked model.
    # --------------------------------------------------------

    model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        tags=tags,
    )

    print("Registered model:", registered.name)
    print("Registered version:", registered.version)
    print("MLflow run ID:", run_id)
    print("Model URI:", model_uri)
    print("Production approval: FALSE")
    print("Step 10.62: REGISTRATION CREATED")


if __name__ == "__main__":
    main()
