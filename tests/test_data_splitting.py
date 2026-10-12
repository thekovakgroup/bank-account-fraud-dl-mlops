import pandas as pd
import pytest

from src.data.splitting import assign_split


def test_assign_split_correctly():
    df = pd.DataFrame({
        "month": [0, 2, 5, 6, 7]
    })

    result = assign_split(df)

    expected = [
        "train",
        "train",
        "train",
        "validation",
        "test",
    ]

    assert result.tolist() == expected


def test_all_expected_months_are_assigned():
    df = pd.DataFrame({
        "month": list(range(8))
    })

    result = assign_split(df)

    assert result.isna().sum() == 0
    assert set(result.unique()) == {
        "train",
        "validation",
        "test",
    }


def test_missing_month_column_fails():
    df = pd.DataFrame({
        "feature": [1, 2, 3]
    })

    with pytest.raises(
        ValueError,
        match="Required temporal column"
    ):
        assign_split(df)


def test_unexpected_month_fails():
    df = pd.DataFrame({
        "month": [0, 1, 8]
    })

    with pytest.raises(
        ValueError,
        match="Unexpected month values"
    ):
        assign_split(df)