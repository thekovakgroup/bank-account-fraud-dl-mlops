
# ============================================================
# Milestone 05 — PyTorch MLP Unit Tests
# ============================================================

import torch

from src.models.mlp import FraudMLP


def test_mlp_forward_shape_and_parameter_count():
    """
    Verify the initial MLP architecture produces
    valid logits and has the expected parameter count.
    """

    # Create the model
    model = FraudMLP(
        input_dim=55,
        dropout_rate=0.10,
    )

    # Create synthetic input data
    X = torch.zeros(8, 55, dtype=torch.float32)

    # Run inference
    model.eval()

    with torch.inference_mode():
        logits = model(X)

    # Verify output dimensions
    assert logits.shape == (8,)

    # Verify numerical validity
    assert torch.isfinite(logits).all().item()

    # Count trainable parameters
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count == 5697

def test_mlp_backpropagation_updates_parameters():
    """
    Verify that backpropagation produces valid gradients
    and that an optimizer updates model parameters.
    """

    # Use a reproducible random state for this test
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)

        # Create a new model
        model = FraudMLP(
            input_dim=55,
            dropout_rate=0.10,
        )

        model.train()

        # Synthetic training data
        X = torch.randn(16, 55, dtype=torch.float32)

        y = torch.tensor(
            [0.0, 1.0] * 8,
            dtype=torch.float32,
        )

        # Define loss and optimizer
        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

        # Save parameters before training
        parameters_before = [
            p.detach().clone()
            for p in model.parameters()
        ]

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)

        assert torch.isfinite(loss).item()

        # Backpropagation
        loss.backward()

        # Verify gradients exist and are finite
        gradients = [
            p.grad for p in model.parameters()
        ]

        assert all(g is not None for g in gradients)

        assert all(
            torch.isfinite(g).all().item()
            for g in gradients
        )

        assert any(
            torch.any(g != 0).item()
            for g in gradients
        )

        # Update model parameters
        optimizer.step()

        # Verify weights changed
        parameters_changed = any(
            not torch.equal(before, after.detach())
            for before, after in zip(
                parameters_before,
                model.parameters(),
            )
        )

        assert parameters_changed

def test_mlp_checkpoint_round_trip(tmp_path):
    """
    Verify model weights can be saved, reloaded,
    and used to reproduce identical predictions.
    """

    # Create original model
    original_model = FraudMLP(
        input_dim=55,
        dropout_rate=0.10,
    )

    original_model.eval()

    # Create synthetic input observations
    X = torch.linspace(
        -1.0,
        1.0,
        steps=8 * 55,
        dtype=torch.float32,
    ).reshape(8, 55)

    # Generate original predictions
    with torch.inference_mode():
        original_logits = original_model(X)

    # Save model weights to temporary file
    checkpoint_path = tmp_path / "test_mlp.pt"

    torch.save(
        original_model.state_dict(),
        checkpoint_path,
    )

    assert checkpoint_path.is_file()

    # Reconstruct a separate model
    reloaded_model = FraudMLP(
        input_dim=55,
        dropout_rate=0.10,
    )

    # Load saved weights
    saved_weights = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    reloaded_model.load_state_dict(
        saved_weights,
        strict=True,
    )

    reloaded_model.eval()

    # Generate predictions from reloaded model
    with torch.inference_mode():
        reloaded_logits = reloaded_model(X)

    # Verify output dimensions
    assert reloaded_logits.shape == (8,)

    # Verify predictions match exactly
    torch.testing.assert_close(
        original_logits,
        reloaded_logits,
        rtol=0,
        atol=0,
    )

    # Verify predictions are finite
    assert torch.isfinite(reloaded_logits).all().item()
