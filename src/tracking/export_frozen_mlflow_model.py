
# ============================================================
# MILESTONE 10 - MLFLOW MODEL EXPORT
# Step 10.56 - Preview-First Frozen Model Packaging
# ============================================================

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import mlflow.pyfunc

from src.fraud_inference import FraudInference
from src.tracking.fraud_mlflow_model import FrozenFraudPyFunc
from src.tracking.model_signature import (
    build_fraud_model_signature,
)
from src.tracking.prepare_frozen_bundle import (
    BUNDLE_ROOT,
    PROJECT_ROOT,
    prepare_inventory,
    sha256_file,
)


EXPORT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "milestone_10"
    / "mlflow_export_v01"
)


def verify_frozen_bundle():
    """
    Confirm that the packaged files match their originals.

    No training, fitting, or test-set access.
    """

    source_files = prepare_inventory()

    for source in source_files:
        relative = source.relative_to(PROJECT_ROOT)
        packaged = BUNDLE_ROOT / relative

        if not packaged.is_file():
            raise FileNotFoundError(
                f"Missing packaged file: {relative}"
            )

        if sha256_file(source) != sha256_file(packaged):
            raise ValueError(
                f"Bundle checksum mismatch: {relative}"
            )

    policy_path = (
        BUNDLE_ROOT
        / "configs"
        / "milestone_07_candidate_policy_v01.json"
    )

    policy = json.loads(
        policy_path.read_text(encoding="utf-8")
    )

    assert (
        policy["status"]
        == "FROZEN_PROVISIONAL_CANDIDATE"
    )
    assert policy["production_approved"] is False

    # Reuse the already-verified inference loader.
    engine = FraudInference(project_root=BUNDLE_ROOT)

    assert engine.policy["production_approved"] is False
    assert not engine.model.training

    return len(source_files)


def build_requirements():
    """
    Capture local dependency versions.

    Strip PyTorch's local +cpu build suffix from the
    pip requirement. CPU-only checkpoint loading remains
    enforced by FraudInference.
    """

    packages = [
        "mlflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "cloudpickle",
    ]

    requirements = [
        f"{package}=={version(package)}"
        for package in packages
    ]

    torch_version = version("torch").split("+")[0]
    requirements.append(f"torch=={torch_version}")

    return requirements


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--commit",
        action="store_true",
        help="Save the private local MLflow model.",
    )

    args = parser.parse_args()

    print("MILESTONE 10 - MLFLOW MODEL EXPORT")
    print("=" * 50)
    print("Mode:", "COMMIT" if args.commit else "PREVIEW")

    # --------------------------------------------------------
    # 1. Verify complete frozen bundle.
    # --------------------------------------------------------

    file_count = verify_frozen_bundle()

    print("Frozen bundle files:", file_count)
    print("Artifact checksums: VERIFIED")
    print("Production approval: FALSE")

    # --------------------------------------------------------
    # 2. Build explicit model signature.
    # --------------------------------------------------------

    signature = build_fraud_model_signature(
        config_path=(
            BUNDLE_ROOT
            / "configs"
            / "preprocessing_config.json"
        )
    )

    assert len(signature.inputs.inputs) == 29
    assert len(signature.outputs.inputs) == 1

    print("Input features: 29")
    print("Output features: 1")

    # --------------------------------------------------------
    # 3. Prepare dependency requirements.
    # --------------------------------------------------------

    requirements = build_requirements()

    print("Package requirements:")

    for requirement in requirements:
        print(" ", requirement)

    # --------------------------------------------------------
    # 4. Define the private MLflow export path.
    # --------------------------------------------------------

    print("Export location:", EXPORT_ROOT)

    if EXPORT_ROOT.exists():
        raise FileExistsError(
            "MLflow export already exists; "
            "refusing to overwrite."
        )

    if not args.commit:
        print("Preview complete - no model exported.")
        print("MLflow registry: UNCHANGED")
        print("Step 10.56: PREVIEW PASS")
        return

    # --------------------------------------------------------
    # 5. Save a complete local MLflow model.
    # --------------------------------------------------------

    mlflow.pyfunc.save_model(
        path=str(EXPORT_ROOT),
        python_model=FrozenFraudPyFunc(),
        artifacts={
            "frozen_bundle": str(BUNDLE_ROOT),
        },
        code_paths=[
            str(PROJECT_ROOT / "src"),
        ],
        signature=signature,
        pip_requirements=requirements,
    )

    print("MLflow export: CREATED")
    print("Frozen model: UNCHANGED")
    print("Model retraining: NONE")
    print("Production approval: FALSE")
    print("MLflow registry: UNCHANGED")
    print("Step 10.56: EXPORT CREATED")


if __name__ == "__main__":
    main()
