
"""
Milestone 07 — Protected Development Dataset Builder

Creates a development-only CSV containing months 0–6.

Month 7 is excluded from the output and must never
be used for training, calibration, or model selection.

Run as:
    python -m src.data.m07_development_snapshot
"""

import csv
import json
from pathlib import Path

from src.data.validation import validate_schema


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = PROJECT_ROOT / "data/raw/Base.csv"

CONFIG_PATH = (
    PROJECT_ROOT / "configs/milestone_07_split_config.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts/milestone_07/development_months_0_to_6.csv"
)


def build_development_snapshot(
    source_path=SOURCE_PATH,
    config_path=CONFIG_PATH,
    output_path=OUTPUT_PATH,
):
    source_path = Path(source_path)
    config_path = Path(config_path)
    output_path = Path(output_path)

    # 1. Validate configuration before reading data.
    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    expected = {
        "train": [0, 1, 2, 3, 4],
        "calibration": [5],
        "evaluation": [6],
        "holdout": [7],
    }

    for role, months in expected.items():
        assert config[role]["months"] == months, (
            f"Unexpected {role} months"
        )

    assert config["holdout"]["status"] == "SEALED"
    assert config["holdout"]["access_allowed"] is False
    assert config["safeguards"]["prohibit_holdout_access"] is True

    allowed_months = set(range(7))

    assert source_path.is_file(), "Source dataset missing"

    # Never overwrite an existing development snapshot.
    assert not output_path.exists(), (
        "Development snapshot already exists. "
        "Stop rather than overwrite it."
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(".csv.tmp")

    assert not temporary_path.exists(), (
        "Temporary snapshot already exists"
    )

    month_counts = {month: 0 for month in range(7)}
    excluded_holdout_rows = 0

    try:
        with (
            source_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as source,
            temporary_path.open(
                "w", encoding="utf-8", newline=""
            ) as destination,
        ):
            reader = csv.reader(source)
            writer = csv.writer(destination)

            header = next(reader)

            schema = validate_schema(header)

            assert schema["is_valid"], (
                f"Unexpected dataset schema: {schema}"
            )

            month_index = header.index(
                config["temporal_field"]
            )

            writer.writerow(header)

            # Streaming processing avoids loading the
            # entire raw dataset into memory.
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(
                        "Invalid source CSV row width"
                    )

                month = int(row[month_index])

                if month == 7:
                    # Do not inspect or retain holdout
                    # features, targets, or records.
                    excluded_holdout_rows += 1
                    continue

                if month not in allowed_months:
                    raise ValueError(
                        f"Unexpected month: {month}"
                    )

                writer.writerow(row)
                month_counts[month] += 1

        total = sum(month_counts.values())

        assert total == 903157, (
            f"Unexpected development row count: {total}"
        )

        assert sum(
            month_counts[m] for m in range(6)
        ) == 794989, "Training-period count mismatch"

        assert month_counts[6] == 108168, (
            "Evaluation-period count mismatch"
        )

        assert excluded_holdout_rows == 96843, (
            "Unexpected holdout row count"
        )

        # Publish only after all checks pass.
        temporary_path.replace(output_path)

    finally:
        # Remove an incomplete temporary file on failure.
        if temporary_path.exists():
            temporary_path.unlink()

    print("MILESTONE 07 — DEVELOPMENT SNAPSHOT")
    print("=" * 55)

    for month, count in month_counts.items():
        print(f"Month {month}: {count:,} rows")

    print("Development records:", f"{total:,}")
    print("Month-7 rows excluded:", excluded_holdout_rows)
    print("Output:", output_path.relative_to(PROJECT_ROOT))
    print("Holdout data written: NO")
    print("Step 07.19: PASS")

    return output_path


if __name__ == "__main__":
    build_development_snapshot()
