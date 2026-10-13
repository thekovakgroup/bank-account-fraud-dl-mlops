
# ============================================================
# Milestone 05 — Reusable Training Loop Tests
# ============================================================

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import FraudMLP
from src.training.loops import train_one_epoch, validate_one_epoch


def test_validation_preserves_model_weights():
    """
    Verify validation produces correct outputs
    without modifying model parameters.
    """

    # Create synthetic data: 17 observations, 55 features
    X = (
        torch.arange(17 * 55, dtype=torch.float32)
        .reshape(17, 55) / 1000.0
    )

    y = torch.tensor(
        [0.0, 1.0] * 8 + [1.0],
        dtype=torch.float32,
    )

    # Create DataLoader with an incomplete final batch
    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    # Create fresh model
    model = FraudMLP(
        input_dim=55,
        dropout_rate=0.10,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    # Preserve model parameters before validation
    weights_before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    # Run reusable validation function
    val_loss, labels, probabilities = validate_one_epoch(
        model=model,
        dataloader=loader,
        loss_fn=criterion,
        device=torch.device("cpu"),
    )

    # Verify output dimensions and target alignment
    assert labels.shape == (17,)
    assert probabilities.shape == (17,)

    np.testing.assert_array_equal(
        labels,
        y.numpy(),
    )

    # Verify probability validity
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    # Verify validation loss
    assert np.isfinite(val_loss)
    assert val_loss >= 0

    # Compare with loss calculated on all 17 samples
    model.eval()

    with torch.inference_mode():
        expected_logits = model(X)
        expected_loss = criterion(expected_logits, y)

    np.testing.assert_allclose(
        val_loss,
        expected_loss.item(),
        rtol=1e-6,
        atol=1e-7,
    )

    # Verify model weights remain unchanged
    for before, after in zip(
        weights_before,
        model.parameters(),
    ):
        assert torch.equal(
            before,
            after.detach(),
        )

    # Verify model remains in evaluation mode
    assert not model.training


def test_training_loop_updates_model_weights():
    """
    Verify one training epoch processes all batches
    and updates the model's trainable parameters.
    """

    # Isolate the test's random seed
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)

        # Create 17 synthetic observations
        X = torch.linspace(
            -1.0,
            1.0,
            steps=17 * 55,
            dtype=torch.float32,
        ).reshape(17, 55)

        y = torch.tensor(
            [0.0, 1.0] * 8 + [1.0],
            dtype=torch.float32,
        )

        dataset = TensorDataset(X, y)

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
        )

        # Create fresh MLP
        model = FraudMLP(
            input_dim=55,
            dropout_rate=0.10,
        )

        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

        # Preserve initial parameters
        parameters_before = [
            p.detach().clone()
            for p in model.parameters()
        ]

        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            loss_fn=criterion,
            device=torch.device("cpu"),
        )

        # Verify training loss
        assert np.isfinite(train_loss)
        assert train_loss >= 0

        # Verify expected batch count
        assert len(loader) == 3

        # Verify optimizer performed three updates
        first_parameter = next(model.parameters())

        optimizer_steps = int(
            optimizer.state[first_parameter]["step"].item()
        )

        assert optimizer_steps == 3

        # Verify model weights changed
        parameters_changed = any(
            not torch.equal(before, after.detach())
            for before, after in zip(
                parameters_before,
                model.parameters(),
            )
        )

        assert parameters_changed

        # Verify training mode is enabled
        assert model.training

        # Verify updated parameters remain finite
        assert all(
            torch.isfinite(p).all().item()
            for p in model.parameters()
        )
