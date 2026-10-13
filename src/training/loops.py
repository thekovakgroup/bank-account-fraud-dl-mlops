
# ============================================================
# Bank Account Fraud Detection
# Milestone 05 — Reusable PyTorch Training Functions
# ============================================================
import numpy as np
import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
):
    """
    Train a PyTorch model for one complete epoch.

    Returns
    -------
    float
        Sample-weighted average training loss.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_features, batch_targets in dataloader:

        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        # Clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward propagation
        logits = model(batch_features)

        # Calculate loss
        loss = loss_fn(logits, batch_targets)

        if not torch.isfinite(loss).item():
            raise ValueError(
                "Non-finite training loss detected."
            )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Track sample-weighted loss
        batch_size = batch_targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # Verify complete dataset coverage
    if total_samples != len(dataloader.dataset):
        raise RuntimeError(
            "Training observations processed do not "
            "match the dataset size."
        )

    if total_samples == 0:
        raise ValueError("Training dataset is empty.")

    return total_loss / total_samples


def validate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device,
):
    """
    Evaluate a binary classification model without
    updating its weights.

    Returns
    -------
    tuple
        average_loss: float
        y_true: NumPy array
        y_probability: NumPy array
    """

    # Disable dropout and other training-specific behavior
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probabilities = []

    # Disable gradient computation
    with torch.inference_mode():

        for batch_features, batch_targets in dataloader:

            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            # Forward propagation
            logits = model(batch_features)

            # Calculate validation loss
            loss = loss_fn(logits, batch_targets)

            if not torch.isfinite(loss).item():
                raise ValueError(
                    "Non-finite validation loss detected."
                )

            # Convert logits to probabilities
            probabilities = torch.sigmoid(logits)

            # Accumulate sample-weighted loss
            batch_size = batch_targets.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Preserve targets and predictions
            all_targets.append(
                batch_targets.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    # Verify complete validation coverage
    if total_samples != len(dataloader.dataset):
        raise RuntimeError(
            "Validation observations processed do not "
            "match the dataset size."
        )

    if total_samples == 0:
        raise ValueError("Validation dataset is empty.")

    # Calculate average validation loss
    average_loss = total_loss / total_samples

    # Combine predictions from all batches
    y_true = np.concatenate(all_targets)
    y_probability = np.concatenate(all_probabilities)

    return average_loss, y_true, y_probability
