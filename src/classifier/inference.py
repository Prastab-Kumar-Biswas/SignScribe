"""
M4 — Classifier inference module for real-time prediction.

Runs LOCALLY (CPU).  Loads the trained classifier checkpoint and
provides a simple predict() function that takes a (30, 126) window
and returns the predicted class label + confidence.

Used by the live demo (M7) and the evaluation module (M6).
"""

import time
from pathlib import Path

import numpy as np
import torch

from src.config import (
    CLASSIFIER_WEIGHTS,
    WINDOW_SIZE,
    FEATURE_DIM,
    BUDGET_CLASSIFIER_MS,
)
from src.classifier.model import GestureClassifier


class GesturePredictor:
    """Loads a trained classifier and runs inference on gesture windows.

    Attributes
    ----------
    label_names : list[str]
        Ordered class labels matching the model's output indices.
    """

    def __init__(
        self,
        checkpoint_path: Path = CLASSIFIER_WEIGHTS,
    ):
        """
        Parameters
        ----------
        checkpoint_path : Path
            Path to the saved .pt checkpoint from training.

        Raises
        ------
        FileNotFoundError
            If the checkpoint doesn't exist.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No classifier checkpoint found at {checkpoint_path}. "
                "Train the model first: python -m src.classifier.train"
            )

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        self.label_names: list[str] = checkpoint["label_names"]
        num_classes = checkpoint["num_classes"]

        self.model = GestureClassifier(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(
            f"GesturePredictor loaded: {num_classes} classes, "
            f"trained epoch {checkpoint.get('epoch', '?')}, "
            f"val_acc {checkpoint.get('val_acc', '?'):.4f}"
        )

    @torch.no_grad()
    def predict(
        self, window: np.ndarray
    ) -> tuple[str, float, float]:
        """Predict the gesture class for a single window.

        Parameters
        ----------
        window : np.ndarray
            Shape (WINDOW_SIZE, FEATURE_DIM) = (30, 126).

        Returns
        -------
        label : str
            Predicted class label.
        confidence : float
            Softmax probability of the predicted class.
        inference_ms : float
            Time taken for inference in milliseconds.
        """
        assert window.shape == (WINDOW_SIZE, FEATURE_DIM), (
            f"Expected ({WINDOW_SIZE}, {FEATURE_DIM}), got {window.shape}"
        )

        # Convert to tensor and add batch dimension
        x = torch.from_numpy(window).unsqueeze(0)  # (1, 30, 126)

        t_start = time.perf_counter()
        logits = self.model(x)                       # (1, num_classes)
        t_end = time.perf_counter()

        inference_ms = (t_end - t_start) * 1000.0

        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

        label = self.label_names[pred_idx.item()]
        return label, confidence.item(), inference_ms

    @torch.no_grad()
    def predict_top_k(
        self, window: np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """Return top-k predictions with confidences.

        Parameters
        ----------
        window : np.ndarray
            Shape (WINDOW_SIZE, FEATURE_DIM).
        k : int
            Number of top predictions to return.

        Returns
        -------
        list of (label, confidence) tuples, sorted by confidence descending.
        """
        x = torch.from_numpy(window).unsqueeze(0)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        top_k = torch.topk(probs, min(k, len(self.label_names)))
        results = []
        for prob, idx in zip(top_k.values, top_k.indices):
            results.append((self.label_names[idx.item()], prob.item()))
        return results
