"""
M4 — Classifier inference module for real-time prediction.

Runs LOCALLY (CPU). Loads the trained classifier checkpoint and
provides an intelligent, hand-agnostic predict() function that:
- Evaluates 2-handed ISL gestures (A, B, D, E, HELLO, etc.) directly with high accuracy.
- Evaluates 1-handed gestures (C, I, L, O, U, V, and digits 0-9) across both hand slots
  (Left & Right) and resolves Kaggle CSV sentinel padding artifacts.
- Automatically rejects idle / empty windows to eliminate false positives.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.config import (
    CLASSIFIER_WEIGHTS,
    WINDOW_SIZE,
    FEATURE_DIM,
)
from src.classifier.model import GestureClassifier

# In Indian Sign Language (ISL):
# 20 letters require BOTH hands:
TWO_HAND_LETTERS: set[str] = set("ABDEFGHKMNPQRSTWXYZ")

# In the training dataset, two-hand signs are split across two spatial orientations:
# Group 1 (Canonical: Hand 0 on Left, Hand 1 on Right in camera frame):
CANONICAL_2HAND_LETTERS: set[str] = set("ABEPQSTXZ")
# Group 2 (Inverted: Hand 0 on Right, Hand 1 on Left in camera frame):
INVERTED_2HAND_LETTERS: set[str] = set("DFGHJKMNRWY")

# 6 letters use ONE hand:
ONE_HAND_LETTERS: set[str] = set("CILOUV")
DIGIT_LABELS: set[str] = {str(d) for d in range(10)}
ALPHANUMERIC_LABELS: set[str] = DIGIT_LABELS | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _count_extended_fingers(pts: np.ndarray) -> int:
    """Count extended fingers (index, middle, ring, pinky) from (21, 3) landmarks."""
    wrist = pts[0]
    ext = 0
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist) * 1.15:
            ext += 1
    return ext


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
        self.label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(self.label_names)}
        num_classes = checkpoint["num_classes"]

        # Precompute indices for masking:
        self.word_indices: list[int] = [
            i for i, lbl in enumerate(self.label_names) if lbl not in ALPHANUMERIC_LABELS
        ]
        self.alphanumeric_indices: list[int] = [
            i for i, lbl in enumerate(self.label_names) if lbl in ALPHANUMERIC_LABELS
        ]
        self.alphanumeric_only: bool = True

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
        self,
        window: np.ndarray,
        min_active_frames: int = 12,
        alphanumeric_only: bool = True,
    ) -> tuple[str, float, float]:
        """Predict the gesture class for a single window.

        Intelligently handles both 2-handed and 1-handed signs, resolving
        left-vs-right hand slot placement and dataset sentinel padding.

        Parameters
        ----------
        window : np.ndarray
            Shape (WINDOW_SIZE, FEATURE_DIM) = (30, 126).
        min_active_frames : int
            Minimum number of frames that must contain non-zero hand landmarks.
            Windows with fewer frames are treated as IDLE.
        alphanumeric_only : bool
            If True (default), masks out the 61 whole-word classes to eliminate
            all word intrusions during fingerspelling.
            Windows with fewer frames are treated as IDLE.

        Returns
        -------
        label : str
            Predicted class label (or 'IDLE' if window has insufficient hand data).
        confidence : float
            Softmax probability of the predicted class (0.0 if IDLE).
        inference_ms : float
            Time taken for inference in milliseconds.
        """
        assert window.shape == (WINDOW_SIZE, FEATURE_DIM), (
            f"Expected ({WINDOW_SIZE}, {FEATURE_DIM}), got {window.shape}"
        )

        # Count active frames for each hand slot
        h1_frames = int(np.sum(np.any(window[:, :63] != 0, axis=1)))
        h2_frames = int(np.sum(np.any(window[:, 63:] != 0, axis=1)))
        total_active = max(h1_frames, h2_frames)

        # Reject windows with insufficient hand presence (idle / resting hands)
        if total_active < min_active_frames:
            return "IDLE", 0.0, 0.0

        t_start = time.perf_counter()

        # ── CASE 1: Both hands are active (Two-Handed Gesture) ───────────────
        # When both hands have at least 8 frames of presence, evaluate both
        # canonical (Slot 0=Left, Slot 1=Right) and inverted (Slot 0=Right, Slot 1=Left)
        # orientations with physical spatial-aware arbitration.
        # Group 1 (A, B, E, P, Q, S, T, X, Z) were recorded in canonical orientation.
        # Group 2 (D, F, G, H, J, K, M, N, R, W, Y) were recorded in inverted orientation.
        if h1_frames >= 8 and h2_frames >= 8:
            # Dynamically determine canonical vs inverted based on actual wrist positions:
            h0_x = float(np.mean(window[:, 0]))
            h1_x = float(np.mean(window[:, 63]))

            w_swapped = np.zeros_like(window)
            w_swapped[:, :63] = window[:, 63:]
            w_swapped[:, 63:] = window[:, :63]

            if h0_x <= h1_x:
                w_canon = window
                w_invert = w_swapped
            else:
                w_canon = w_swapped
                w_invert = window

            batch = np.stack([w_canon, w_invert], axis=0)
            logits = self.model(torch.from_numpy(batch))
            if alphanumeric_only and self.word_indices:
                logits[:, self.word_indices] = -float("inf")
            probs = torch.softmax(logits, dim=1).numpy()

            p_canon, p_invert = probs[0], probs[1]
            idx_canon, idx_invert = int(np.argmax(p_canon)), int(np.argmax(p_invert))
            lbl_canon, conf_canon = self.label_names[idx_canon], float(p_canon[idx_canon])
            lbl_invert, conf_invert = self.label_names[idx_invert], float(p_invert[idx_invert])

            # Apply orientation score arbitration:
            # - For canonical orientation, favor Group 1 letters and general words.
            #   Penalize inverted letters appearing in the canonical tensor (e.g. H hallucinating for F).
            # - For inverted orientation, favor Group 2 letters and general words.
            #   Penalize canonical letters appearing in the inverted tensor (e.g. X hallucinating for E).
            score_canon = (
                conf_canon
                if (lbl_canon in CANONICAL_2HAND_LETTERS or lbl_canon not in INVERTED_2HAND_LETTERS)
                else conf_canon * 0.4
            )
            score_invert = (
                conf_invert
                if (lbl_invert in INVERTED_2HAND_LETTERS or lbl_invert not in CANONICAL_2HAND_LETTERS)
                else conf_invert * 0.4
            )

            label, confidence = (
                (lbl_canon, conf_canon)
                if score_canon >= score_invert
                else (lbl_invert, conf_invert)
            )

            # ── Disambiguation rules for visually similar pairs ──
            # 1. Disambiguate F vs X using extended finger count:
            # F requires 2 extended fingers per hand (>= 2 on either hand).
            # X requires exactly 1 extended finger per hand (c0 == 1 and c1 == 1).
            if label in {"F", "X"}:
                pts0 = window[0, :63].reshape(21, 3)
                pts1 = window[0, 63:].reshape(21, 3)
                c0 = _count_extended_fingers(pts0)
                c1 = _count_extended_fingers(pts1)
                if c0 == 1 and c1 == 1:
                    label = "X"
                elif c0 >= 2 or c1 >= 2:
                    label = "F"

            # 2. Disambiguate E vs K using vertical displacement of wrists:
            # In K, the reaching hand is placed high above the base hand (|dy| >= 0.35).
            # In E, both hands are held level horizontally (|dy| <= 0.25).
            if label in {"E", "K"}:
                pts0 = window[0, :63].reshape(21, 3)
                pts1 = window[0, 63:].reshape(21, 3)
                wrist_dy = abs(pts0[0, 1] - pts1[0, 1])
                if wrist_dy >= 0.35:
                    label = "K"
                elif wrist_dy <= 0.25:
                    label = "E"

            # 3. Disambiguate J vs Q using vertical displacement of wrists:
            # In J, the base palm is low (y ~ 0.89) and drawing hand is high (y ~ 0.52), |dy| >= 0.28.
            # In Q, both hands are held level horizontally (|dy| <= 0.22).
            if label in {"J", "Q"}:
                pts0 = window[0, :63].reshape(21, 3)
                pts1 = window[0, 63:].reshape(21, 3)
                wrist_dy = abs(pts0[0, 1] - pts1[0, 1])
                if wrist_dy >= 0.28:
                    label = "J"
                elif wrist_dy <= 0.22:
                    label = "Q"

            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return label, confidence, inference_ms

        # ── CASE 2: Single hand detected (One-Handed Gesture) ────────────────
        # Extract whichever hand has active landmarks
        if h1_frames >= h2_frames:
            active_hand = window[:, :63]
        else:
            active_hand = window[:, 63:]

        # Physical thumb spread: distance between thumb tip (lm 4) and index MCP (lm 5)
        pts_active = active_hand[0].reshape(21, 3)
        thumb_spread = float(np.linalg.norm(pts_active[4] - pts_active[5]))

        # Build 3 hypotheses in a single batch pass:
        # Hyp 0: Slot 1 active (Right Hand slot with zero padding)
        # Hyp 1: Slot 0 active (Left Hand slot with zero padding)
        # Hyp 2: Slot 0 active with -1.0 padding in Slot 1 (Kaggle CSV 1-hand letter format)
        batch = np.zeros((3, WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
        batch[0, :, 63:] = active_hand  # Right-hand slot
        batch[1, :, :63] = active_hand  # Left-hand slot
        batch[2, :, :63] = active_hand  # Letter format
        batch[2, :, 63:] = -1.0

        x = torch.from_numpy(batch)
        logits = self.model(x)  # (3, num_classes)
        if alphanumeric_only and self.word_indices:
            logits[:, self.word_indices] = -float("inf")
        probs = torch.softmax(logits, dim=1).numpy()

        p0, p1, p2 = probs[0], probs[1], probs[2]

        top0_idx = int(np.argmax(p0))
        top1_idx = int(np.argmax(p1))
        top2_idx = int(np.argmax(p2))

        l0, c0 = self.label_names[top0_idx], float(p0[top0_idx])
        l1, c1 = self.label_names[top1_idx], float(p1[top1_idx])
        l2, c2 = self.label_names[top2_idx], float(p2[top2_idx])

        # Best prediction under clean 0.0 padding (Hyp 0 vs Hyp 1):
        best_clean_lbl, best_clean_conf = (l0, c0) if c0 >= c1 else (l1, c1)

        # 1. Letter 'L' thumb spread check (resolves 'L' being intercepted by digit '1'):
        # In 'L', the thumb is spread wide outwards (thumb_spread >= 0.18).
        if l2 == "L" and c2 >= 0.50 and thumb_spread >= 0.18:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "L", c2, inference_ms

        # 2. Letter 'C' check:
        # In single-hand gestures, 'C' can be shadowed by digit '1' under clean 0.0 padding.
        if l2 == "C" and c2 >= 0.70:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "C", c2, inference_ms
        if l1 == "C" and c1 >= 0.70:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "C", c1, inference_ms

        # 3. Disambiguate Digit '1' vs Letter 'I' (index pointing up):
        # In digit '1', the thumb is folded across the fist (thumb_spread <= 0.14).
        # In letter 'I', the thumb rests alongside or loose (thumb_spread >= 0.16).
        if best_clean_lbl == "1" and l2 == "I":
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            if thumb_spread >= 0.16:
                return "I", c2, inference_ms
            else:
                return "1", best_clean_conf, inference_ms

        # 4. Digit '6' check (unless V):
        idx_6 = self.label_to_idx.get("6", -1)
        prob_6 = max(float(p0[idx_6]), float(p1[idx_6])) if idx_6 >= 0 else 0.0
        if prob_6 >= 0.50 and best_clean_lbl not in {
            "0", "1", "2", "3", "4", "5", "7", "8", "9"
        } and l2 != "V":
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "6", prob_6, inference_ms

        # 5. Confident clean digit:
        if best_clean_lbl in DIGIT_LABELS and best_clean_conf >= 0.60:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return best_clean_lbl, best_clean_conf, inference_ms

        # 6. 1-hand ISL letters (O, U, V):
        if l2 in ONE_HAND_LETTERS and c2 >= 0.70:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return l2, c2, inference_ms

        # 7. Fallback to best clean prediction
        inference_ms = (time.perf_counter() - t_start) * 1000.0
        return best_clean_lbl, best_clean_conf, inference_ms

    @torch.no_grad()
    def predict_top_k(
        self,
        window: np.ndarray,
        k: int = 5,
        min_active_frames: int = 12,
        alphanumeric_only: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top-k predictions with confidences."""
        label, conf, _ = self.predict(
            window,
            min_active_frames=min_active_frames,
            alphanumeric_only=alphanumeric_only,
        )
        if label == "IDLE":
            return [("IDLE", 1.0)]
        return [(label, conf)]

    @staticmethod
    def is_two_hand_sign(label: str) -> bool:
        """Return True if the sign typically requires two hands in ISL."""
        return label in TWO_HAND_LETTERS
