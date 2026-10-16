
# ============================================================
# Milestone 08 — Reusable Fraud Inference Pipeline
# Artifact-verified, inference-only implementation
# ============================================================

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import logit

from src.features.preprocessing import (
    load_preprocessing_config,
    prepare_model_features,
)
from src.models.mlp import FraudMLP


def sha256_file(path):
    """Calculate SHA-256 for an existing file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_artifact(path, expected_hash):
    """Refuse to load an artifact with an incorrect hash."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")

    if sha256_file(path) != expected_hash:
        raise ValueError(f"Artifact checksum mismatch: {path}")


class FraudInference:
    """
    Loads the frozen Milestone 07 development candidate.

    Model: Weighted PyTorch MLP
    Calibration: Sigmoid / Platt

    No fitting, training, or dataset loading is performed.

    Only load trusted local serialized artifacts.
    """

    def __init__(self, project_root=None):

        self.root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

        artifact_dir = (
            self.root / "artifacts" / "milestone_07"
        )

        # ----------------------------------------------------
        # 1. Load configuration and artifact manifests.
        # ----------------------------------------------------

        def read_json(path):
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)

        self.policy = read_json(
            self.root
            / "configs/milestone_07_candidate_policy_v01.json"
        )

        self.feature_config = load_preprocessing_config(
            self.root / "configs/preprocessing_config.json"
        )

        mlp_manifest = read_json(
            artifact_dir / "m07_weighted_mlp_m0to4_v01.json"
        )

        recovery_manifest = read_json(
            artifact_dir
            / "restart_bundle_v01/manifest.json"
        )

        calibration_manifest = read_json(
            artifact_dir
            / "calibration_policy_v01/manifest.json"
        )

        # ----------------------------------------------------
        # 2. Enforce the frozen provisional policy.
        # ----------------------------------------------------

        if self.policy["status"] != "FROZEN_PROVISIONAL_CANDIDATE":
            raise ValueError("Unexpected policy status")

        if self.policy["production_approved"] is not False:
            raise ValueError("Expected provisional-only policy")

        if self.policy["data_protocol"]["holdout_status"] != "SEALED":
            raise ValueError("Unexpected holdout status")

        if mlp_manifest["architecture"] != [55, 64, 32, 1]:
            raise ValueError("Unexpected MLP architecture")

        # ----------------------------------------------------
        # 3. Identify and verify trusted local artifacts.
        # ----------------------------------------------------

        preprocessing_path = (
            artifact_dir
            / "restart_bundle_v01/models.joblib"
        )

        checkpoint_path = (
            artifact_dir / mlp_manifest["checkpoint_file"]
        )

        calibrator_path = (
            artifact_dir
            / "calibration_policy_v01/mlp_calibrators.joblib"
        )

        verify_artifact(
            preprocessing_path,
            recovery_manifest["models_sha256"],
        )

        verify_artifact(
            checkpoint_path,
            mlp_manifest["checkpoint_sha256"],
        )

        verify_artifact(
            calibrator_path,
            calibration_manifest["file_checksums"][
                "mlp_calibrators.joblib"
            ],
        )

        if (
            self.policy["model"]["checkpoint_sha256"]
            != mlp_manifest["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint does not match policy")

        if (
            self.policy["model"]["calibrator_bundle_sha256"]
            != calibration_manifest["file_checksums"][
                "mlp_calibrators.joblib"
            ]
        ):
            raise ValueError("Calibrator does not match policy")

        # ----------------------------------------------------
        # 4. Load verified fitted preprocessing and calibration.
        # ----------------------------------------------------

        self.preprocessor = joblib.load(
            preprocessing_path
        )["preprocessor"]

        self.calibrator = joblib.load(
            calibrator_path
        )["sigmoid"]

        # ----------------------------------------------------
        # 5. Reconstruct and restore the trained PyTorch MLP.
        # ----------------------------------------------------

        self.device = torch.device("cpu")

        self.model = FraudMLP(
            input_dim=55,
            dropout_rate=0.10,
        ).to(self.device)

        state = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state)
        self.model.eval()

        # Read the frozen economic decision rule.
        self.threshold = float(
            self.policy["decision_rule"]["probability_threshold"]
        )

        self.max_fraction = float(
            self.policy["decision_rule"][
                "maximum_investigation_fraction"
            ]
        )

        if not (0 < self.threshold < 1):
            raise ValueError("Invalid policy threshold")

        if not (0 < self.max_fraction <= 1):
            raise ValueError("Invalid investigation capacity")

    def predict_proba(self, applications, batch_size=4096):
        """
        Return sigmoid-calibrated fraud probabilities.

        applications: pandas DataFrame with required raw features.
        No model training or preprocessing refitting occurs.
        """

        if not isinstance(applications, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if len(applications) == 0:
            return np.empty(0, dtype=np.float64)

        # Apply the same feature preparation as Milestone 07.
        features = prepare_model_features(
            applications,
            config=self.feature_config,
        )

        X = self.preprocessor.transform(features)

        X = np.asarray(X, dtype=np.float32)
        X = np.ascontiguousarray(X)

        if X.shape != (len(applications), 55):
            raise ValueError(
                f"Unexpected transformed shape: {X.shape}"
            )

        if not np.isfinite(X).all():
            raise ValueError("Non-finite model features")

        # Batched CPU inference.
        raw_probabilities = []

        self.model.eval()

        with torch.inference_mode():

            for start in range(0, len(X), batch_size):

                batch = torch.from_numpy(
                    X[start:start + batch_size]
                )

                logits = self.model(batch)

                probabilities = torch.sigmoid(
                    logits
                ).cpu().numpy()

                raw_probabilities.append(probabilities)

        raw = np.concatenate(raw_probabilities)

        # Match Milestone 07 Platt calibration:
        # fit on logit(clipped MLP probabilities).
        clipped = np.clip(
            raw.astype(np.float64),
            1e-7,
            1.0 - 1e-7,
        )

        calibrated = self.calibrator.predict_proba(
            logit(clipped).reshape(-1, 1)
        )[:, 1]

        if not np.isfinite(calibrated).all():
            raise ValueError("Invalid calibrated predictions")

        return calibrated

    def recommend_investigations(self, applications):
        """
        Apply the frozen investigation policy to ONE complete
        decision cohort, not independent small batches.

        Stable input order is the provisional tie-breaker.
        """

        probabilities = self.predict_proba(applications)

        count = len(probabilities)

        capacity = int(
            np.floor(count * self.max_fraction)
        )

        eligible = np.flatnonzero(
            probabilities > self.threshold
        )

        ranked = eligible[
            np.argsort(
                -probabilities[eligible],
                kind="stable",
            )
        ]

        selected = ranked[:capacity]

        investigate = np.zeros(count, dtype=bool)
        investigate[selected] = True

        priority = np.zeros(count, dtype=np.int32)
        priority[selected] = np.arange(
            1,
            len(selected) + 1,
        )

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "investigate": investigate,
                "review_priority": priority,
            },
            index=applications.index,
        )
