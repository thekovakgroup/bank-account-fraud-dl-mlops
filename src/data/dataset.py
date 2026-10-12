import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def arrays_to_tensor_dataset(X, y):
    """
    Convert NumPy feature and target arrays into a PyTorch TensorDataset.

    Expected dtypes:
        X -> float32
        y -> float32
    """
    if X.dtype != np.float32:
        raise ValueError(
            f"X must be float32, received {X.dtype}"
        )

    if y.dtype != np.float32:
        raise ValueError(
            f"y must be float32, received {y.dtype}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Feature and target row counts do not match."
        )

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    return TensorDataset(
        X_tensor,
        y_tensor,
    )


def build_dataloaders(
    train_dataset,
    validation_dataset,
    batch_size=1024,
    random_seed=42,
    num_workers=0,
):
    """
    Build deterministic training and validation DataLoaders.

    Training:
        shuffle=True

    Validation:
        shuffle=False
    """
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader