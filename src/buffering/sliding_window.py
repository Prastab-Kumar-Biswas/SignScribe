"""
M3 — Sliding-window buffer for temporal gesture sequences.

Runs LOCALLY.  Accumulates per-frame 126-dim landmark vectors into
fixed-size windows of shape (WINDOW_SIZE, FEATURE_DIM) = (30, 126).

Includes hand-presence gating:
- Automatically resets when no hands are detected for several frames (idle detection).
- Rejects windows that have insufficient hand presence (prevents hallucinating signs on empty frames).
"""

from __future__ import annotations

import numpy as np

from src.config import FEATURE_DIM, WINDOW_SIZE


class SlidingWindowBuffer:
    """Accumulates frame-level features into fixed-size temporal windows.

    Parameters
    ----------
    window_size : int
        Number of frames per window (default: 30).
    feature_dim : int
        Dimensionality of each frame's feature vector (default: 126).
    overlap : int
        How many frames to keep after emitting a window.
        Higher overlap = smoother transitions but more frequent
        classifications.  Default is window_size // 2 (50% overlap).
    min_active_ratio : float
        Minimum fraction of frames in a window that must contain active hand
        landmarks (default: 0.6). Windows with fewer active frames are rejected.
    max_idle_frames : int
        Number of consecutive frames without hands before the buffer resets
        to prevent contamination between separate gestures (default: 6).
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        feature_dim: int = FEATURE_DIM,
        overlap: int | None = None,
        min_active_ratio: float = 0.6,
        max_idle_frames: int = 6,
    ):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.overlap = overlap if overlap is not None else window_size // 2
        self.min_active_ratio = min_active_ratio
        self.max_idle_frames = max_idle_frames

        if self.window_size <= 0 or self.feature_dim <= 0:
            raise ValueError("window_size and feature_dim must be positive")
        if not 0 <= self.overlap < self.window_size:
            raise ValueError(
                "overlap must be at least 0 and smaller than window_size"
            )

        # Internal storage
        self._buffer: list[np.ndarray] = []
        self._idle_count: int = 0

    @property
    def current_length(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when the buffer has enough frames to emit a window."""
        return len(self._buffer) >= self.window_size

    def add_frame(
        self, features: np.ndarray, has_hands: bool | None = None
    ) -> np.ndarray | None:
        """Add a single frame's features and return a window if ready.

        Parameters
        ----------
        features : np.ndarray
            Shape (FEATURE_DIM,) = (126,).
        has_hands : bool or None
            Whether hands were detected in this frame. If None, detected by
            checking if any feature is non-zero.

        Returns
        -------
        np.ndarray or None
            If the buffer has reached `window_size` frames and passes the
            active-hand ratio check, returns the window as shape (window_size, feature_dim)
            and slides by keeping the last `overlap` frames. Otherwise returns None.
        """
        assert features.shape == (self.feature_dim,), (
            f"Expected feature vector of shape ({self.feature_dim},), "
            f"got {features.shape}"
        )

        if has_hands is None:
            has_hands = bool(np.any(features != 0))

        if not has_hands:
            self._idle_count += 1
            if self._idle_count >= self.max_idle_frames:
                # Reset buffer on prolonged silence so the next gesture starts fresh
                self.reset()
            return None

        # Hands are present; reset idle counter and append frame
        self._idle_count = 0
        self._buffer.append(features.copy())

        if self.is_ready:
            candidate = np.stack(self._buffer[: self.window_size], axis=0)

            # Slide buffer forward
            self._buffer = self._buffer[self.window_size - self.overlap :]

            # Hand-presence quality check: count frames with non-zero features
            active_frames = np.sum(np.any(candidate != 0, axis=1))
            if active_frames / self.window_size >= self.min_active_ratio:
                return candidate

        return None

    def reset(self):
        """Clear the buffer entirely."""
        self._buffer.clear()
        self._idle_count = 0

    def get_partial_window(self) -> np.ndarray:
        """Return whatever frames are buffered, zero-padded to window_size."""
        window = np.zeros(
            (self.window_size, self.feature_dim), dtype=np.float32
        )
        n = min(len(self._buffer), self.window_size)
        if n > 0:
            window[:n] = np.stack(self._buffer[:n], axis=0)
        return window
