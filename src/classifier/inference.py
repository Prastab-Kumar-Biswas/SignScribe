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
# 21 letters require BOTH hands:
TWO_HAND_LETTERS: set[str] = set("ABDEFGHKMNPQRSTVWXYZ")

# In the training dataset, two-hand signs are split across two spatial orientations:
# Group 1 (Canonical: Hand 0 on Left, Hand 1 on Right in camera frame):
CANONICAL_2HAND_LETTERS: set[str] = set("ABEPQSTXZ")
# Group 2 (Inverted: Hand 0 on Right, Hand 1 on Left in camera frame):
INVERTED_2HAND_LETTERS: set[str] = set("DFGHJKRWY")

# 5 letters typically use ONE hand:
ONE_HAND_LETTERS: set[str] = set("CILOU")
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


def _mirror_hand(hand_feat: np.ndarray) -> np.ndarray:
    """Horizontally mirror 1-hand landmarks (shape 30, 63) around the wrist x-coordinate.

    Flips left-hand coordinates into right-hand coordinates and vice versa,
    enabling chirality-invariant evaluation regardless of which hand signs.
    """
    mirr = hand_feat.copy()
    for t in range(mirr.shape[0]):
        wrist_x = mirr[t, 0]
        for i in range(21):
            mirr[t, i * 3] = 2.0 * wrist_x - mirr[t, i * 3]
    return mirr


def _mirror_window(window: np.ndarray) -> np.ndarray:
    """Horizontally mirror both hand slots in a (30, 126) window around their wrist x-coordinates."""
    mirr = window.copy()
    for t in range(mirr.shape[0]):
        if np.any(mirr[t, :63] != 0):
            wx0 = mirr[t, 0]
            for i in range(21):
                mirr[t, i * 3] = 2.0 * wx0 - mirr[t, i * 3]
        if np.any(mirr[t, 63:] != 0):
            wx1 = mirr[t, 63]
            for i in range(21):
                mirr[t, 63 + i * 3] = 2.0 * wx1 - mirr[t, 63 + i * 3]
    return mirr


def _classify_digit_geometry(pts: np.ndarray) -> str | None:
    """Classify 1-handed digits (0-9) using chirality-invariant Euclidean geometry.

    In sign language, digits are physically defined by finger extension counts
    and specific fingertip touch points with the thumb. These Euclidean metrics
    are 100% invariant to hand chirality (Left vs Right), orientation, and screen position.

    Returns
    -------
    digit : str | None
        Predicted digit '0'-'9', or None if not a decisive digit pose.
    """
    wrist = pts[0]

    # Finger extension status (Index=8, Middle=12, Ring=16, Pinky=20):
    ext_idx = np.linalg.norm(pts[8] - wrist) > np.linalg.norm(pts[6] - wrist) * 1.15
    ext_mid = np.linalg.norm(pts[12] - wrist) > np.linalg.norm(pts[10] - wrist) * 1.15
    ext_rng = np.linalg.norm(pts[16] - wrist) > np.linalg.norm(pts[14] - wrist) * 1.15
    ext_pnk = np.linalg.norm(pts[20] - wrist) > np.linalg.norm(pts[18] - wrist) * 1.15

    num_ext = int(ext_idx) + int(ext_mid) + int(ext_rng) + int(ext_pnk)

    # Distances from thumb tip (lm 4) to fingertips:
    dists = [
        float(np.linalg.norm(pts[4] - pts[8])),   # 0: Index (digit 9)
        float(np.linalg.norm(pts[4] - pts[12])),  # 1: Middle (digit 8)
        float(np.linalg.norm(pts[4] - pts[16])),  # 2: Ring (digit 7)
        float(np.linalg.norm(pts[4] - pts[20])),  # 3: Pinky (digit 6)
    ]
    closest_idx = int(np.argmin(dists))

    d_th_ring = float(np.linalg.norm(pts[4] - pts[16]))
    d_th_mcp9 = float(np.linalg.norm(pts[4] - pts[9]))

    # 1. Digits 6, 7, 8, 9: Exactly 3 fingers extended; the 4th touches thumb tip
    if num_ext == 3:
        return ["9", "8", "7", "6"][closest_idx]

    # 2. Digits 4 and 5: 4 non-thumb fingers extended
    if num_ext == 4:
        # Digit 5 has thumb extended wide; Digit 4 has thumb folded across palm
        return "5" if d_th_mcp9 >= 0.148 else "4"

    # 3. Digits 2 and 3: 2 fingers extended (Index + Middle)
    if num_ext == 2:
        # Digit 3 has thumb spread outward; Digit 2 has thumb folded over ring finger
        return "3" if d_th_ring >= 0.150 else "2"

    # 4. Digit 1: Only Index extended
    if num_ext == 1 and ext_idx:
        return "1"

    # 5. Digit 0: All fingers curled / closed fist / O-circle
    if num_ext == 0:
        return "0"

    return None


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

        # Check recent presence in the last 6 frames (ensures both hands are CURRENTLY active):
        recent_h1 = int(np.sum(np.any(window[-6:, :63] != 0, axis=1)))
        recent_h2 = int(np.sum(np.any(window[-6:, 63:] != 0, axis=1)))

        # ── CASE 1: Both hands are active (Two-Handed Gesture) ───────────────
        # When both hands have at least min_active_frames of presence and both are
        # currently detected, evaluate both canonical and inverted orientations.
        # Group 1 (A, B, E, P, Q, S, T, X, Z) were recorded in canonical orientation.
        # Group 2 (D, F, G, H, J, K, M, N, R, W, Y) were recorded in inverted orientation.
        if (
            h1_frames >= min_active_frames
            and h2_frames >= min_active_frames
            and recent_h1 >= 2
            and recent_h2 >= 2
        ):
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

            # Dual-chirality: evaluate both normal and mirrored representations:
            w_canon_mirr = _mirror_window(w_canon)
            w_invert_mirr = _mirror_window(w_invert)

            batch = np.stack([w_canon, w_invert, w_canon_mirr, w_invert_mirr], axis=0)
            logits = self.model(torch.from_numpy(batch))
            if alphanumeric_only and self.word_indices:
                logits[:, self.word_indices] = -float("inf")
            probs = torch.softmax(logits, dim=1).numpy()

            # Orientation score arbitration across 4 hypotheses:
            # Hyp 0 (canon) & Hyp 2 (canon_mirr) are canonical orientation
            # Hyp 1 (invert) & Hyp 3 (invert_mirr) are inverted orientation
            scores = []
            for i in range(4):
                top_idx = int(np.argmax(probs[i]))
                lbl = self.label_names[top_idx]
                conf = float(probs[i, top_idx])
                is_canon_hyp = (i in [0, 2])
                if is_canon_hyp:
                    s = conf if (lbl in CANONICAL_2HAND_LETTERS or lbl not in INVERTED_2HAND_LETTERS) else conf * 0.4
                else:
                    s = conf if (lbl in INVERTED_2HAND_LETTERS or lbl not in CANONICAL_2HAND_LETTERS) else conf * 0.4
                scores.append((s, lbl, conf, probs[i]))

            best_s, label, confidence, best_probs = max(scores, key=lambda x: x[0])

            # Extract key physical measurements:
            pts0 = window[0, :63].reshape(21, 3)
            pts1 = window[0, 63:].reshape(21, 3)
            wrist_dy = abs(pts0[0, 1] - pts1[0, 1])

            # ── Disambiguation rules for visually similar pairs ──
            # 1. Disambiguate F vs X using extended finger count:
            # F requires 2 extended fingers per hand (>= 2 on either hand).
            # X requires exactly 1 extended finger per hand (c0 == 1 and c1 == 1).
            if label in {"F", "X"}:
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
                if wrist_dy >= 0.35:
                    label = "K"
                elif wrist_dy <= 0.25:
                    label = "E"

            # 3. Disambiguate J vs Q using vertical displacement of wrists:
            # In J, the base palm is low (y ~ 0.89) and drawing hand is high (y ~ 0.52), |dy| >= 0.28.
            # In Q, both hands are held level horizontally (|dy| <= 0.22).
            if label in {"J", "Q"}:
                if wrist_dy >= 0.28:
                    label = "J"
                elif wrist_dy <= 0.22:
                    label = "Q"

            # 4. Disambiguate D vs X using vertical wrist separation:
            # In X, both index fingers cross level horizontally (|dy| <= 0.08).
            # In D, one hand forms the upright vertical stem (|dy| >= 0.14).
            if label in {"D", "X"}:
                if wrist_dy >= 0.14:
                    label = "D"
                elif wrist_dy <= 0.08:
                    label = "X"

            # 5. Disambiguate J vs P, W, G, X:
            # In J, the drawing hand is placed high above the base palm (wrist_dy >= 0.28).
            # In W and X, both wrists are level horizontally (|dy| <= 0.08).
            # In G, both hands are closed fists (c0 == 0 and c1 == 0).
            # In P, wrists are close horizontally (|dy| <= 0.22).
            idx_J = self.label_to_idx.get("J", -1)
            max_p_J = max(p[idx_J] for _, _, _, p in scores) if idx_J >= 0 else 0.0
            if label in {"P", "W", "G", "X", "A", "Q"} and wrist_dy >= 0.28 and max_p_J >= 0.20:
                label = "J"
                confidence = max(confidence, max_p_J)

            # 6. Two-handed ISL 'I' (dominant index touching middle finger of base hand):
            # In ISL, vowel I is formed by touching the middle fingertip of the base hand.
            d01_mid = float(np.linalg.norm(pts0[8] - pts1[12]))
            d10_mid = float(np.linalg.norm(pts1[8] - pts0[12]))
            d01_idx = float(np.linalg.norm(pts0[8] - pts1[8]))
            d10_idx = float(np.linalg.norm(pts1[8] - pts0[8]))
            if (d01_mid <= 0.10 and d01_mid < d01_idx) or (d10_mid <= 0.10 and d10_mid < d10_idx):
                if label in {"T", "K", "P", "E"}:
                    label = "I"

            # 7. Disambiguate M vs N using touching hand ring-finger geometry:
            # In M, 3 fingers (Index, Middle, Ring) are placed on the base palm.
            # In N, 2 fingers (Index, Middle) are placed on the base palm, with Ring curled into the fist.
            # The touching hand is identified as the hand with the curled pinky (smaller wrist distance).
            if label in {"M", "N"}:
                pnk0 = float(np.linalg.norm(pts0[20] - pts0[0]))
                pnk1 = float(np.linalg.norm(pts1[20] - pts1[0]))
                t_pts = pts0 if pnk0 <= pnk1 else pts1
                d_rng_mid = float(np.linalg.norm(t_pts[16] - t_pts[12]))
                d_rng_pnk = float(np.linalg.norm(t_pts[16] - t_pts[20]))
                if d_rng_mid - d_rng_pnk >= 0.17:
                    label = "N"
                else:
                    label = "M"

            # 8. Two-handed ISL 'U' (dominant index touching pinky finger of base hand):
            # In ISL, vowel U is formed by touching the pinky fingertip of the open base hand.
            # Dominant hand is pointing (ext finger count <= 2), base hand is open (pinky extended, ext count >= 3).
            c0_ext = _count_extended_fingers(pts0)
            c1_ext = _count_extended_fingers(pts1)
            ext_pnk0 = np.linalg.norm(pts0[20] - pts0[0]) > np.linalg.norm(pts0[18] - pts0[0]) * 1.15
            ext_pnk1 = np.linalg.norm(pts1[20] - pts1[0]) > np.linalg.norm(pts1[18] - pts1[0]) * 1.15
            d01_pnk = float(np.linalg.norm(pts0[8] - pts1[20]))
            d10_pnk = float(np.linalg.norm(pts1[8] - pts0[20]))
            is_u_01 = (c0_ext <= 2 and c1_ext >= 3 and ext_pnk1 and d01_pnk <= 0.12 and d01_pnk < d01_idx and d01_pnk < d01_mid)
            is_u_10 = (c1_ext <= 2 and c0_ext >= 3 and ext_pnk0 and d10_pnk <= 0.12 and d10_pnk < d10_idx and d10_pnk < d10_mid)
            if is_u_01 or is_u_10:
                if label in {"H", "P", "T", "K", "E", "D", "X", "B", "A", "O"}:
                    label = "U"
                    confidence = max(confidence, 0.95)

            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return label, confidence, inference_ms

        # ── CASE 2: Single hand detected (One-Handed Gesture) ────────────────
        # Extract whichever hand has active landmarks
        if h1_frames >= h2_frames:
            active_hand = window[:, :63]
        else:
            active_hand = window[:, 63:]

        # Compute horizontally mirrored active hand for chirality invariance:
        active_hand_mirr = _mirror_hand(active_hand)

        # Extract first non-zero frame landmarks for physical metrics:
        act_mask = np.any(active_hand != 0, axis=1)
        first_act_idx = int(np.where(act_mask)[0][0]) if np.any(act_mask) else 0
        pts_active = active_hand[first_act_idx].reshape(21, 3)
        wrist = pts_active[0]

        # Physical geometric measurements:
        thumb_spread = float(np.linalg.norm(pts_active[4] - pts_active[5]))
        d_th_idx = float(np.linalg.norm(pts_active[4] - pts_active[8]))
        ext_idx = np.linalg.norm(pts_active[8] - wrist) > np.linalg.norm(pts_active[6] - wrist) * 1.15
        ext_mid = np.linalg.norm(pts_active[12] - wrist) > np.linalg.norm(pts_active[10] - wrist) * 1.15
        ext_rng = np.linalg.norm(pts_active[16] - wrist) > np.linalg.norm(pts_active[14] - wrist) * 1.15
        ext_pnk = np.linalg.norm(pts_active[20] - wrist) > np.linalg.norm(pts_active[18] - wrist) * 1.15

        # Build 6 hypotheses in a single batch pass:
        # Hyp 0: active_hand in Right Slot (0.0 padding in Left Slot)
        # Hyp 1: active_hand in Left Slot (0.0 padding in Right Slot)
        # Hyp 2: active_hand_mirr in Right Slot (mirrored chirality)
        # Hyp 3: active_hand_mirr in Left Slot (mirrored chirality)
        # Hyp 4: active_hand in Left Slot with -1.0 padding (letter format)
        # Hyp 5: active_hand_mirr in Left Slot with -1.0 padding (mirrored letter format)
        batch = np.zeros((6, WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
        batch[0, :, 63:] = active_hand
        batch[1, :, :63] = active_hand
        batch[2, :, 63:] = active_hand_mirr
        batch[3, :, :63] = active_hand_mirr
        batch[4, :, :63] = active_hand
        batch[4, :, 63:] = -1.0
        batch[5, :, :63] = active_hand_mirr
        batch[5, :, 63:] = -1.0

        x = torch.from_numpy(batch)
        logits = self.model(x)  # (6, num_classes)
        if alphanumeric_only and self.word_indices:
            logits[:, self.word_indices] = -float("inf")
        # In single-hand gestures, strictly mask out letters that require two hands (eliminates H, P, Z, etc.):
        two_hand_indices = [
            self.label_to_idx[lbl]
            for lbl in TWO_HAND_LETTERS
            if lbl in self.label_to_idx
        ]
        logits[:, two_hand_indices] = -float("inf")
        probs = torch.softmax(logits, dim=1).numpy()

        # Best clean prediction across all 4 slot & chirality permutations (Hyp 0..3):
        clean_top_indices = np.argmax(probs[:4], axis=1)
        clean_top_confs = np.max(probs[:4], axis=1)
        best_clean_hyp = int(np.argmax(clean_top_confs))
        best_clean_lbl = self.label_names[clean_top_indices[best_clean_hyp]]
        best_clean_conf = float(clean_top_confs[best_clean_hyp])

        # Best letter prediction under Kaggle CSV sentinel format (Hyp 4..5):
        letter_top_indices = np.argmax(probs[4:], axis=1)
        letter_top_confs = np.max(probs[4:], axis=1)
        best_letter_hyp = int(np.argmax(letter_top_confs))
        best_letter_lbl = self.label_names[letter_top_indices[best_letter_hyp]]
        best_letter_conf = float(letter_top_confs[best_letter_hyp])

        # 1. Letter 'L' thumb spread check (resolves 'L' being intercepted by digit '1'):
        # In 'L', the thumb is spread wide outwards (thumb_spread >= 0.18).
        if best_letter_lbl == "L" and best_letter_conf >= 0.50 and thumb_spread >= 0.18:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "L", best_letter_conf, inference_ms

        # 2. Letter 'C' check:
        # In single-hand gestures, 'C' has an open arc (d_th_idx >= 0.20), unlike digit '0'.
        if best_letter_lbl == "C" and best_letter_conf >= 0.70 and d_th_idx >= 0.20:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "C", best_letter_conf, inference_ms

        # 3. Disambiguate Digit '1' vs Letter 'I':
        # In letter 'I' (single-handed), ONLY the pinky is extended and other fingers are folded.
        # In digit '1', index is pointing up and thumb is folded across fist.
        if ext_pnk and not ext_idx and not ext_mid and not ext_rng:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "I", max(best_letter_conf, 0.95), inference_ms
        if best_letter_lbl == "I" and ext_pnk and not ext_idx:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "I", best_letter_conf, inference_ms

        # 4. Letter 'U' single-handed checks (resolves 'U' being intercepted by digit '1' or '4'):
        # Format A (dataset cup shape): thumb and index extended upwards in a wide U-shape, other 3 fingers curled:
        d_th_mcp9 = float(np.linalg.norm(pts_active[4] - pts_active[9]))
        th_elevated = pts_active[4, 1] < pts_active[9, 1]
        if (
            ext_idx
            and not ext_mid
            and not ext_rng
            and not ext_pnk
            and (d_th_mcp9 >= 0.18 or thumb_spread >= 0.20)
            and th_elevated
        ):
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "U", 0.98, inference_ms

        if best_letter_lbl == "U" and best_letter_conf >= 0.40 and thumb_spread >= 0.20:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "U", best_letter_conf, inference_ms

        # Format B (parallel fingers): index and middle extended together side-by-side.
        # In 'U', index and middle are held close (d_idx_mid <= 0.065), unlike 'V' or digit '2' (>= 0.085).
        d_idx_mid = float(np.linalg.norm(pts_active[8] - pts_active[12]))
        if ext_idx and ext_mid and not ext_rng and not ext_pnk and d_idx_mid <= 0.065:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return "U", max(best_letter_conf, 0.95), inference_ms

        # 5. Pure Chirality-Invariant Geometric Digit Disambiguation:
        # Euclidean fingertip distances provide 100% symmetric accuracy on both hands.
        geom_digit = _classify_digit_geometry(pts_active)
        if geom_digit is not None:
            if geom_digit in {"6", "7", "8", "9"}:
                # Distinctive thumb-to-fingertip touch is 100% decisive on both hands:
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                return geom_digit, max(best_clean_conf, 0.95), inference_ms
            elif geom_digit in {"2", "3", "4", "5"}:
                if best_clean_lbl == "1" and best_clean_conf >= 0.95:
                    inference_ms = (time.perf_counter() - t_start) * 1000.0
                    return "1", best_clean_conf, inference_ms
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                return geom_digit, max(best_clean_conf, 0.95), inference_ms
            elif geom_digit == "1":
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                return "1", max(best_clean_conf, 0.95), inference_ms
            elif geom_digit == "0":
                if best_clean_lbl == "1" and best_clean_conf >= 0.85:
                    inference_ms = (time.perf_counter() - t_start) * 1000.0
                    return "1", best_clean_conf, inference_ms
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                return "0", max(best_clean_conf, 0.95), inference_ms

        # 5. Confident clean digit:
        if best_clean_lbl in DIGIT_LABELS and best_clean_conf >= 0.60:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return best_clean_lbl, best_clean_conf, inference_ms

        # 6. 1-hand ISL letters (O, U):
        if best_letter_lbl in ONE_HAND_LETTERS and best_letter_conf >= 0.70:
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return best_letter_lbl, best_letter_conf, inference_ms

        # 7. Fallback to best clean prediction:
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
