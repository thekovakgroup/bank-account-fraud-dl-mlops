
# ============================================================
# MILESTONE 10 - AUTOMATED MLFLOW INTEGRATION TESTS
# Step 10.71 - Research-Only Model Safeguards
# ============================================================

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlflow.types import DataType

from src.tracking import prepare_frozen_bundle as bundle
from src.tracking import register_frozen_model as registry
from src.tracking.fraud_mlflow_model import FrozenFraudPyFunc
from src.tracking.model_signature import (
    build_fraud_model_signature,
)


# ------------------------------------------------------------
# TEST 1 - Verify the 29-feature MLflow contract.
# ------------------------------------------------------------

def test_model_signature_has_expected_features():
    signature = build_fraud_model_signature()

    counts = Counter(
        column.type
        for column in signature.inputs.inputs
    )

    assert len(signature.inputs.inputs) == 29
    assert counts[DataType.double] == 18
    assert counts[DataType.string] == 5
    assert counts[DataType.long] == 6

    assert len(signature.outputs.inputs) == 1

    output = signature.outputs.inputs[0]

    assert output.name == "fraud_probability"
    assert output.type == DataType.double


# ------------------------------------------------------------
# TEST 2 - Reject duplicate input features.
# ------------------------------------------------------------

def test_model_signature_rejects_duplicates(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "preprocessing_config.json"
    )

    config = json.loads(
        source.read_text(encoding="utf-8")
    )

    config["numeric_features"][0] = (
        config["numeric_features"][1]
    )

    corrupted = tmp_path / "preprocessing_config.json"

    corrupted.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError):
        build_fraud_model_signature(corrupted)


# ------------------------------------------------------------
# TEST 3 - Verify the MLflow prediction interface.
# ------------------------------------------------------------

def test_pyfunc_returns_probability_dataframe():
    model = FrozenFraudPyFunc()

    model.engine = SimpleNamespace(
        predict_proba=lambda frame: np.array(
            [0.25, 0.75]
        )
    )

    frame = pd.DataFrame(
        {"synthetic_feature": [0, 1]},
        index=[10, 20],
    )

    result = model.predict(None, frame)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)

    assert list(result.columns) == [
        "fraud_probability"
    ]

    assert result.index.tolist() == [10, 20]

    np.testing.assert_allclose(
        result["fraud_probability"].to_numpy(),
        [0.25, 0.75],
    )


# ------------------------------------------------------------
# TEST 4 - Reject unsupported prediction inputs.
# ------------------------------------------------------------

def test_pyfunc_rejects_non_dataframe():
    model = FrozenFraudPyFunc()

    model.engine = SimpleNamespace(
        predict_proba=lambda frame: np.array([0.5])
    )

    with pytest.raises(TypeError):
        model.predict(None, [{"income": 0.5}])


# ------------------------------------------------------------
# FIXTURE - Build synthetic artifact files.
#
# These are dummy bytes, NOT private model artifacts.
# ------------------------------------------------------------

@pytest.fixture
def synthetic_frozen_tree(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_project"
    m07 = root / "artifacts" / "milestone_07"

    def write_file(relative, content):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(
                json.dumps(content),
                encoding="utf-8",
            )

        return path

    checkpoint = write_file(
        "artifacts/milestone_07/"
        "m07_weighted_mlp_m0to4_v01.pt",
        b"synthetic-checkpoint-only",
    )

    preprocessor = write_file(
        "artifacts/milestone_07/"
        "restart_bundle_v01/models.joblib",
        b"synthetic-preprocessor-only",
    )

    calibrator = write_file(
        "artifacts/milestone_07/"
        "calibration_policy_v01/mlp_calibrators.joblib",
        b"synthetic-calibrator-only",
    )

    checkpoint_hash = bundle.sha256_file(checkpoint)
    preprocessor_hash = bundle.sha256_file(preprocessor)
    calibrator_hash = bundle.sha256_file(calibrator)

    write_file(
        "configs/preprocessing_config.json",
        {},
    )

    write_file(
        "configs/milestone_07_candidate_policy_v01.json",
        {
            "status": "FROZEN_PROVISIONAL_CANDIDATE",
            "production_approved": False,
            "model": {
                "checkpoint_sha256": checkpoint_hash,
                "calibrator_bundle_sha256": calibrator_hash,
            },
        },
    )

    write_file(
        "artifacts/milestone_07/"
        "m07_weighted_mlp_m0to4_v01.json",
        {
            "checkpoint_file":
                "m07_weighted_mlp_m0to4_v01.pt",
            "checkpoint_sha256": checkpoint_hash,
        },
    )

    write_file(
        "artifacts/milestone_07/"
        "restart_bundle_v01/manifest.json",
        {
            "models_sha256": preprocessor_hash,
        },
    )

    write_file(
        "artifacts/milestone_07/"
        "calibration_policy_v01/manifest.json",
        {
            "file_checksums": {
                "mlp_calibrators.joblib": calibrator_hash,
            },
        },
    )

    monkeypatch.setattr(
        bundle, "PROJECT_ROOT", root
    )

    monkeypatch.setattr(
        bundle, "M07", m07
    )

    return root, checkpoint


# ------------------------------------------------------------
# TEST 5 - Verify valid synthetic artifact checksums.
# ------------------------------------------------------------

def test_artifact_inventory_accepts_valid_hashes(
    synthetic_frozen_tree,
):
    files = bundle.prepare_inventory()

    assert len(files) == 8
    assert all(path.is_file() for path in files)


# ------------------------------------------------------------
# TEST 6 - Detect a tampered frozen checkpoint.
# ------------------------------------------------------------

def test_artifact_inventory_detects_tampering(
    synthetic_frozen_tree,
):
    _, checkpoint = synthetic_frozen_tree

    checkpoint.write_bytes(
        b"corrupted-synthetic-checkpoint"
    )

    with pytest.raises(AssertionError):
        bundle.prepare_inventory()


# ------------------------------------------------------------
# TEST 7 - Reject duplicate model registrations.
# ------------------------------------------------------------

def test_registry_rejects_existing_model(
    tmp_path,
    monkeypatch,
):
    database = tmp_path / "test_registry.db"

    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE registered_models "
            "(name TEXT PRIMARY KEY)"
        )

        connection.execute(
            "CREATE TABLE model_versions "
            "(name TEXT, version TEXT)"
        )

        connection.execute(
            "INSERT INTO registered_models (name) "
            "VALUES (?)",
            (registry.MODEL_NAME,),
        )

        connection.execute(
            "INSERT INTO model_versions "
            "(name, version) VALUES (?, ?)",
            (registry.MODEL_NAME, "1"),
        )

    monkeypatch.setattr(
        registry, "DB_PATH", database
    )

    with pytest.raises(
        RuntimeError,
        match="already exists",
    ):
        registry.check_registry_read_only()


# ------------------------------------------------------------
# TEST 8 - Verify preview mode avoids registry writes.
# ------------------------------------------------------------

def test_registration_preview_makes_no_writes(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        registry,
        "PROJECT_ROOT",
        Path.cwd().resolve(),
    )

    monkeypatch.setattr(
        registry,
        "verify_export",
        lambda: (8, "a" * 64),
    )

    monkeypatch.setattr(
        registry,
        "check_registry_read_only",
        lambda: (0, 0),
    )

    def forbidden_write(*args, **kwargs):
        pytest.fail(
            "Preview attempted a registry write"
        )

    monkeypatch.setattr(
        registry.mlflow,
        "set_tracking_uri",
        forbidden_write,
    )

    monkeypatch.setattr(
        registry.mlflow,
        "create_experiment",
        forbidden_write,
    )

    monkeypatch.setattr(
        registry.mlflow,
        "start_run",
        forbidden_write,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["register_frozen_model"],
    )

    registry.main()

    output = capsys.readouterr().out

    assert "Mode: PREVIEW" in output
    assert "no registry changes" in output
    assert "PREVIEW PASS" in output
