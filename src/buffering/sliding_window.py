"""
M3 — Sliding-window buffer for temporal gesture sequences.

Runs LOCALLY.  Accumulates per-frame 126-dim landmark vectors into
fixed-size windows of shape (WINDOW_SIZE, FEATURE_DIM) = (30, 126).

The buffer emits a "ready" window once it has accumulated WINDOW_SIZE
frames.  It uses a sliding approach: after emitting, it keeps the last
`overlap` frames to provide temporal continuity between consecutive
classifications.

This module is used by:
  - The data recorder (for saving labeled training sequences)
  - The live demo (for feeding the classifier in real time)
"""

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
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        feature_dim: int = FEATURE_DIM,
        overlap: int | None = None,
    ):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.overlap = overlap if overlap is not None else window_size // 2

        if self.window_size <= 0 or self.feature_dim <= 0:
            raise ValueError("window_size and feature_dim must be positive")
        if not 0 <= self.overlap < self.window_size:
            raise ValueError(
                "overlap must be at least 0 and smaller than window_size"
            )

        # Internal storage — list of 1-D arrays, flushed after emission
        self._buffer: list[np.ndarray] = []

    @property
    def current_length(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when the buffer has enough frames to emit a window."""
        return len(self._buffer) >= self.window_size

    def add_frame(self, features: np.ndarray) -> np.ndarray | None:
        """Add a single frame's features and return a window if ready.

        Parameters
        ----------
        features : np.ndarray
            Shape (FEATURE_DIM,) = (126,).

        Returns
        -------
        np.ndarray or None
            If the buffer has reached `window_size` frames, returns the
            window as shape (window_size, feature_dim) and slides by
            keeping the last `overlap` frames.  Otherwise returns None.
        """
        assert features.shape == (self.feature_dim,), (
            f"Expected feature vector of shape ({self.feature_dim},), "
            f"got {features.shape}"
        )
        self._buffer.append(features.copy())

        if self.is_ready:
            # Stack into (window_size, feature_dim)
            window = np.stack(self._buffer[: self.window_size], axis=0)

            # Slide: keep the last `overlap` frames for continuity
            self._buffer = self._buffer[self.window_size - self.overlap :]

            return window

        return None

    def reset(self):
        """Clear the buffer entirely."""
        self._buffer.clear()

    def get_partial_window(self) -> np.ndarray:
        """Return whatever frames are buffered, zero-padded to window_size.

        Useful for visualising the current buffer state or for a
        "best-effort" classification when the user stops mid-gesture.

        Returns
        -------
        np.ndarray
            Shape (window_size, feature_dim), zero-padded at the end.
        """
        window = np.zeros(
            (self.window_size, self.feature_dim), dtype=np.float32
        )
        n = min(len(self._buffer), self.window_size)
        if n > 0:
            window[:n] = np.stack(self._buffer[:n], axis=0)
        return window
