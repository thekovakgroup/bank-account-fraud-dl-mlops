import numpy as np
import pytest
import torch

from src.data.dataset import (
    arrays_to_tensor_dataset,
    build_dataloaders,
)


def test_arrays_to_tensor_dataset():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    y = np.array(
        [0.0, 1.0],
        dtype=np.float32,
    )

    dataset = arrays_to_tensor_dataset(X, y)

    X_tensor, y_tensor = dataset[0]

    assert len(dataset) == 2
    assert X_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.float32


def test_arrays_to_tensor_dataset_rejects_wrong_dtype():
    X = np.array(
        [[1.0, 2.0]],
        dtype=np.float64,
    )

    y = np.array(
        [0.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="X must be float32",
    ):
        arrays_to_tensor_dataset(X, y)


def test_arrays_to_tensor_dataset_rejects_row_mismatch():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    y = np.array(
        [0.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="row counts do not match",
    ):
        arrays_to_tensor_dataset(X, y)


def test_build_dataloaders():
    X_train = np.arange(
        40,
        dtype=np.float32,
    ).reshape(20, 2)

    y_train = np.arange(
        20,
        dtype=np.float32,
    )

    X_validation = np.arange(
        20,
        dtype=np.float32,
    ).reshape(10, 2)

    y_validation = np.arange(
        10,
        dtype=np.float32,
    )

    train_dataset = arrays_to_tensor_dataset(
        X_train,
        y_train,
    )

    validation_dataset = arrays_to_tensor_dataset(
        X_validation,
        y_validation,
    )

    train_loader, validation_loader = build_dataloaders(
        train_dataset,
        validation_dataset,
        batch_size=4,
        random_seed=42,
    )

    X_batch, y_batch = next(iter(train_loader))

    assert X_batch.shape == (4, 2)
    assert y_batch.shape == (4,)
    assert len(train_loader) == 5
    assert len(validation_loader) == 3