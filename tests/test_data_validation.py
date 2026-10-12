from src.data.validation import EXPECTED_COLUMNS, validate_schema


def test_valid_schema_passes():
    result = validate_schema(EXPECTED_COLUMNS)

    assert result["is_valid"] is True
    assert result["missing_columns"] == []
    assert result["unexpected_columns"] == []
    assert result["correct_order"] is True


def test_missing_column_fails():
    columns = EXPECTED_COLUMNS[:-1]

    result = validate_schema(columns)

    assert result["is_valid"] is False
    assert "month" in result["missing_columns"]


def test_unexpected_column_fails():
    columns = EXPECTED_COLUMNS + ["unexpected_feature"]

    result = validate_schema(columns)

    assert result["is_valid"] is False
    assert "unexpected_feature" in result["unexpected_columns"]


def test_wrong_column_order_fails():
    columns = EXPECTED_COLUMNS.copy()

    columns[0], columns[1] = columns[1], columns[0]

    result = validate_schema(columns)

    assert result["is_valid"] is False
    assert result["correct_order"] is False