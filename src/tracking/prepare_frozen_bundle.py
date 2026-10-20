
# ============================================================
# MILESTONE 10 - FROZEN MODEL ARTIFACT PACKAGING
# Step 10.48 - Preview-First Portable Bundle
# ============================================================

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

M07 = PROJECT_ROOT / "artifacts" / "milestone_07"

BUNDLE_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "milestone_10"
    / "frozen_bundle_v01"
)


def read_json(path):
    """Load a trusted local JSON manifest."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    """Calculate the SHA-256 hash of a file."""
    with path.open("rb") as source:
        return hashlib.file_digest(
            source, "sha256"
        ).hexdigest()


def prepare_inventory():
    """
    Inventory the exact files required for inference.

    Verify frozen artifact hashes before packaging.
    """

    # --------------------------------------------------------
    # 1. Load existing frozen manifests.
    # --------------------------------------------------------

    mlp = read_json(
        M07 / "m07_weighted_mlp_m0to4_v01.json"
    )

    recovery = read_json(
        M07 / "restart_bundle_v01/manifest.json"
    )

    calibration = read_json(
        M07 / "calibration_policy_v01/manifest.json"
    )

    policy = read_json(
        PROJECT_ROOT
        / "configs/milestone_07_candidate_policy_v01.json"
    )

    assert policy["status"] == (
        "FROZEN_PROVISIONAL_CANDIDATE"
    )

    assert policy["production_approved"] is False

    # --------------------------------------------------------
    # 2. Resolve checkpoint path safely.
    # --------------------------------------------------------

    checkpoint = (
        M07 / mlp["checkpoint_file"]
    ).resolve()

    assert checkpoint.is_relative_to(M07.resolve()), (
        "Checkpoint must remain inside Milestone 07"
    )

    # --------------------------------------------------------
    # 3. Define the complete inference file inventory.
    # --------------------------------------------------------

    files = [
        PROJECT_ROOT / "configs/preprocessing_config.json",
        PROJECT_ROOT / "configs/milestone_07_candidate_policy_v01.json",
        M07 / "m07_weighted_mlp_m0to4_v01.json",
        M07 / "restart_bundle_v01/manifest.json",
        M07 / "calibration_policy_v01/manifest.json",
        M07 / "restart_bundle_v01/models.joblib",
        M07 / "calibration_policy_v01/mlp_calibrators.joblib",
        checkpoint,
    ]

    # --------------------------------------------------------
    # 4. Verify all files exist inside the project.
    # --------------------------------------------------------

    for path in files:
        assert path.is_file(), f"Missing: {path}"

        assert path.resolve().is_relative_to(
            PROJECT_ROOT.resolve()
        ), "Artifact path outside project"

    # --------------------------------------------------------
    # 5. Verify the three core artifact checksums.
    # --------------------------------------------------------

    expected = {
        checkpoint: mlp["checkpoint_sha256"],
        M07 / "restart_bundle_v01/models.joblib":
            recovery["models_sha256"],
        M07 / "calibration_policy_v01/mlp_calibrators.joblib":
            calibration["file_checksums"][
                "mlp_calibrators.joblib"
            ],
    }

    for path, expected_hash in expected.items():
        assert sha256_file(path) == expected_hash, (
            f"Checksum mismatch: {path.name}"
        )

    assert policy["model"]["checkpoint_sha256"] == (
        mlp["checkpoint_sha256"]
    )

    assert policy["model"]["calibrator_bundle_sha256"] == (
        calibration["file_checksums"][
            "mlp_calibrators.joblib"
        ]
    )

    return files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--commit",
        action="store_true",
        help="Create the local frozen artifact bundle.",
    )

    args = parser.parse_args()

    files = prepare_inventory()

    print("MILESTONE 10 - FROZEN ARTIFACT BUNDLE")
    print("=" * 50)
    print("Mode:", "COMMIT" if args.commit else "PREVIEW")
    print("Required files:", len(files))

    for path in files:
        print("VERIFIED:", path.relative_to(PROJECT_ROOT))

    if not args.commit:
        print("Preview complete - no files copied.")
        return

    # --------------------------------------------------------
    # 6. Create a fresh, local, Git-ignored bundle.
    # --------------------------------------------------------

    if BUNDLE_ROOT.exists():
        raise FileExistsError(
            "Bundle already exists; refusing to overwrite it."
        )

    BUNDLE_ROOT.mkdir(parents=True)

    # --------------------------------------------------------
    # 7. Copy and verify the packaged files.
    # --------------------------------------------------------

    for source in files:
        relative_path = source.relative_to(PROJECT_ROOT)

        destination = BUNDLE_ROOT / relative_path

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(source, destination)

        assert sha256_file(destination) == (
            sha256_file(source)
        ), "Copied artifact checksum mismatch"

    print("Bundle location:", BUNDLE_ROOT)
    print("Copied files:", len(files))
    print("Frozen artifacts: UNCHANGED")
    print("Model retraining: NONE")
    print("Production approval: FALSE")
    print("Step 10.48: BUNDLE CREATED")


if __name__ == "__main__":
    main()
