from pathlib import Path
import json

import pandas as pd


DEFAULT_SPLIT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "split_config.json"
)


def load_split_config(config_path=DEFAULT_SPLIT_CONFIG_PATH):
    """Load the project split configuration from JSON."""

    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assign_split(df, config=None):
    """
    Assign deterministic split labels using the project split config.
    """

    if config is None:
        config = load_split_config()

    month_column = config["temporal_field"]

    if month_column not in df.columns:
        raise ValueError(
            f"Required temporal column '{month_column}' is missing."
        )

    train_months = set(config["train"]["months"])
    validation_months = set(config["validation"]["months"])
    test_months = set(config["test"]["months"])

    expected_months = (
        train_months
        | validation_months
        | test_months
    )

    observed_months = set(
        df[month_column]
        .dropna()
        .unique()
    )

    unexpected_months = observed_months - expected_months

    if unexpected_months:
        raise ValueError(
            f"Unexpected month values found: "
            f"{sorted(unexpected_months)}"
        )

    split = pd.Series(
        index=df.index,
        dtype="object",
        name="split",
    )

    split.loc[
        df[month_column].isin(train_months)
    ] = "train"

    split.loc[
        df[month_column].isin(validation_months)
    ] = "validation"

    split.loc[
        df[month_column].isin(test_months)
    ] = "test"

    if split.isna().any():
        raise ValueError(
            "Some rows could not be assigned to a split."
        )

    return split