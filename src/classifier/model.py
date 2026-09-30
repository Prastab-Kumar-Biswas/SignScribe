"""
M4 — BiLSTM gesture classifier model definition.

Runs LOCALLY (CPU).  A 2-layer bidirectional LSTM that takes a temporal
window of hand landmarks (30, 126) and classifies it into one of the
target gesture classes.

Architecture (from synopsis):
    Input:  (batch, 30, 126)
    → 2-layer BiLSTM, hidden_size=128
    → Take final hidden states from both directions
    → Concatenate → (batch, 256)
    → Dropout(0.3) → Linear(256, num_classes)

Expected parameter count: ~680K (will be verified at import time).
"""

import torch
import torch.nn as nn

from src.config import (
    FEATURE_DIM,
    WINDOW_SIZE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_BIDIRECTIONAL,
    LSTM_DROPOUT,
    NUM_CLASSES,
)


class GestureClassifier(nn.Module):
    """Bidirectional LSTM classifier for temporal gesture sequences.

    Parameters
    ----------
    input_dim : int
        Feature dimension per frame (default: 126).
    hidden_size : int
        LSTM hidden size (default: 128).
    num_layers : int
        Number of stacked LSTM layers (default: 2).
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout rate between LSTM layers and before the classifier head.
    bidirectional : bool
        Whether to use bidirectional LSTM (default: True).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = LSTM_BIDIRECTIONAL,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Layer normalisation on the input features for training stability
        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Classification head: takes the concatenated final hidden states
        # from both directions → (batch, hidden_size * num_directions)
        classifier_input_dim = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, window_size, feature_dim) = (B, 30, 126).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        # Normalise input features
        x = self.input_norm(x)

        # LSTM: output shape (batch, seq_len, hidden * num_directions)
        # h_n shape: (num_layers * num_directions, batch, hidden)
        lstm_out, (h_n, _) = self.lstm(x)

        # Extract final hidden states from the last layer.
        # For bidirectional: h_n[-2] is forward, h_n[-1] is backward
        if self.num_directions == 2:
            h_forward = h_n[-2]   # (batch, hidden)
            h_backward = h_n[-1]  # (batch, hidden)
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1]     # (batch, hidden)

        # Classify
        logits = self.classifier(h_final)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes: int | None = None) -> GestureClassifier:
    """Factory function to create and sanity-check the classifier.

    Parameters
    ----------
    num_classes : int or None
        Override the default NUM_CLASSES from config (useful when the
        vocab is still being populated and NUM_CLASSES is < 96).

    Returns
    -------
    GestureClassifier
        The instantiated model.
    """
    if num_classes is None:
        num_classes = NUM_CLASSES

    model = GestureClassifier(num_classes=num_classes)
    param_count = model.count_parameters()

    print(f"GestureClassifier created:")
    print(f"  Input shape  : (batch, {WINDOW_SIZE}, {FEATURE_DIM})")
    print(f"  Num classes  : {num_classes}")
    print(f"  Parameters   : {param_count:,}")

    # Sanity check with a dummy forward pass
    dummy = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, num_classes), (
        f"Expected output shape (1, {num_classes}), got {out.shape}"
    )
    print(f"  Output shape : {out.shape} OK")

    return model


if __name__ == "__main__":
    # Quick verification
    model = build_model()
