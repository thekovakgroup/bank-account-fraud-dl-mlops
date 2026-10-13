
# ============================================================
# Bank Account Fraud Detection
# Milestone 05 — Reusable PyTorch MLP Architecture
# ============================================================

from torch import nn


class FraudMLP(nn.Module):
    """
    Multilayer Perceptron for binary fraud classification.

    Architecture:
        55 -> 64 -> 32 -> 1

    Returns raw logits for BCEWithLogitsLoss.
    """

    def __init__(self, input_dim=55, dropout_rate=0.10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
