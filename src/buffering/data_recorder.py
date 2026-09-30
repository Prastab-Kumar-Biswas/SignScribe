"""
M3 — Interactive dataset recording tool for gesture sequences.

Runs LOCALLY.  Opens the webcam, displays the live hand-tracking feed,
and lets you record labeled gesture samples by pressing keys.

Each recorded sample is a (WINDOW_SIZE, FEATURE_DIM) = (30, 126) numpy
array saved as a .npy file under:
    data/raw/<label>/<label>_<signer>_<rep_number>.npy

Usage:
    python -m src.buffering.data_recorder

Workflow:
    1. The tool starts in IDLE mode showing the live webcam feed.
    2. Press 'r' to start RECORDING — a countdown begins, then
       WINDOW_SIZE frames (1.2 s at 25 fps) are captured.
    3. After capture, the sample is saved and you return to IDLE.
    4. Press 'n' to change the current label (cycles through vocab
       or type a new one).
    5. Press 's' to change the signer ID.
    6. Press 'q' to quit.  A session summary is printed.

Controls:
    r  — start recording (with 3-second countdown)
    n  — set label (will prompt in console)
    s  — set signer ID (will prompt in console)
    q  — quit
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    DATA_RAW_DIR,
    FEATURE_DIM,
    TARGET_FPS,
    WINDOW_SIZE,
    VOCAB,
)
from src.capture.hand_tracker import HandTracker


class DataRecorder:
    """Interactive tool for recording labeled gesture sequences.

    Saves each sample as a .npy file with shape (WINDOW_SIZE, FEATURE_DIM).
    Files are organised as:
        data/raw/<LABEL>/<LABEL>_<SIGNER>_<NNN>.npy

    The recorder tracks how many samples exist per label so that
    numbering continues across sessions (you can quit and resume later).
    """

    def __init__(
        self,
        output_dir: Path = DATA_RAW_DIR,
        signer_id: str = "signer_01",
    ):
        self.output_dir = Path(output_dir)
        self.signer_id = signer_id
        self.current_label: str = VOCAB[0] if VOCAB else "UNKNOWN"

        # Session stats
        self.session_recorded: dict[str, int] = {}

    def _get_next_index(self, label: str) -> int:
        """Find the next available index for a label/signer combo."""
        label_dir = self.output_dir / label
        if not label_dir.exists():
            return 1
        existing = list(label_dir.glob(f"{label}_{self.signer_id}_*.npy"))
        if not existing:
            return 1
        # Parse the highest existing index
        indices = []
        for f in existing:
            try:
                idx = int(f.stem.split("_")[-1])
                indices.append(idx)
            except ValueError:
                pass
        return max(indices, default=0) + 1

    def save_sample(self, label: str, window: np.ndarray) -> Path:
        """Save a recorded window to disk.

        Parameters
        ----------
        label : str
            Gesture class label (e.g., "A", "HELLO").
        window : np.ndarray
            Shape (WINDOW_SIZE, FEATURE_DIM).

        Returns
        -------
        Path
            The path to the saved .npy file.
        """
        assert window.shape == (WINDOW_SIZE, FEATURE_DIM), (
            f"Expected shape ({WINDOW_SIZE}, {FEATURE_DIM}), "
            f"got {window.shape}"
        )

        label_dir = self.output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        idx = self._get_next_index(label)
        filename = f"{label}_{self.signer_id}_{idx:03d}.npy"
        filepath = label_dir / filename

        np.save(filepath, window)

        # Update session stats
        self.session_recorded[label] = (
            self.session_recorded.get(label, 0) + 1
        )

        return filepath

    def get_dataset_summary(self) -> dict[str, int]:
        """Count total samples per label on disk."""
        summary: dict[str, int] = {}
        if self.output_dir.exists():
            for label_dir in sorted(self.output_dir.iterdir()):
                if label_dir.is_dir():
                    count = len(list(label_dir.glob("*.npy")))
                    if count > 0:
                        summary[label_dir.name] = count
        return summary


def run_recorder():
    """Main recording loop with webcam + MediaPipe + key controls."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Is it connected?")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker()
    recorder = DataRecorder()
    frame_interval = 1.0 / TARGET_FPS

    # ── State machine ───────────────────────────────────────────
    # States: IDLE, COUNTDOWN, RECORDING
    state = "IDLE"
    countdown_start: float = 0.0
    countdown_duration = 3.0  # seconds before recording begins
    recording_frames: list[np.ndarray] = []

    print("=" * 60)
    print("  SignScribe — Data Recorder (Phase 2)")
    print("=" * 60)
    print(f"  Output dir  : {recorder.output_dir}")
    print(f"  Signer ID   : {recorder.signer_id}")
    print(f"  Window size : {WINDOW_SIZE} frames ({WINDOW_SIZE/TARGET_FPS:.1f} s)")
    print(f"  Vocab size  : {len(VOCAB)} classes")
    print()
    print("  Controls:")
    print("    r — record a sample (3 s countdown, then 1.2 s capture)")
    print("    n — change label (prompts in console)")
    print("    s — change signer ID (prompts in console)")
    print("    d — show dataset summary")
    print("    q — quit")
    print()
    print(f"  Current label: {recorder.current_label}")
    print("=" * 60)

    try:
        prev_frame_time = time.perf_counter()

        while True:
            # ── Throttle ────────────────────────────────────────
            now = time.perf_counter()
            elapsed = now - prev_frame_time
            if elapsed < frame_interval:
                sleep_time = frame_interval - elapsed - 0.002
                if sleep_time > 0:
                    time.sleep(sleep_time)
                while time.perf_counter() - prev_frame_time < frame_interval:
                    pass
            prev_frame_time = time.perf_counter()

            # ── Capture + process ───────────────────────────────
            ret, frame_bgr = cap.read()
            if not ret:
                print("WARNING: Failed to read frame.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            features, results = tracker.extract_landmarks(frame_rgb)
            frame_bgr = tracker.draw_landmarks(frame_bgr, results)

            num_hands = (
                len(results.multi_hand_landmarks)
                if results.multi_hand_landmarks
                else 0
            )

            # ── State machine logic ─────────────────────────────
            if state == "COUNTDOWN":
                time_left = countdown_duration - (
                    time.perf_counter() - countdown_start
                )
                if time_left <= 0:
                    state = "RECORDING"
                    recording_frames = []
                    print(f"  ▶ RECORDING '{recorder.current_label}'...")
                else:
                    # Draw countdown overlay
                    cv2.putText(
                        frame_bgr,
                        f"GET READY: {time_left:.1f}s",
                        (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        3,
                    )

            elif state == "RECORDING":
                recording_frames.append(features)
                progress = len(recording_frames)

                # Progress bar overlay
                bar_width = int(
                    (progress / WINDOW_SIZE) * 400
                )
                cv2.rectangle(
                    frame_bgr, (120, 400), (520, 430), (50, 50, 50), -1
                )
                cv2.rectangle(
                    frame_bgr,
                    (120, 400),
                    (120 + bar_width, 430),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    frame_bgr,
                    f"RECORDING: {progress}/{WINDOW_SIZE}",
                    (150, 390),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                if progress >= WINDOW_SIZE:
                    # Save the sample
                    window = np.stack(recording_frames[:WINDOW_SIZE], axis=0)
                    filepath = recorder.save_sample(
                        recorder.current_label, window
                    )
                    print(f"  ✅ Saved: {filepath.name}  "
                          f"({recorder.session_recorded[recorder.current_label]} "
                          f"this session)")
                    state = "IDLE"
                    recording_frames = []

            # ── HUD overlay ─────────────────────────────────────
            # Status bar at the top
            status_color = {
                "IDLE": (200, 200, 200),
                "COUNTDOWN": (0, 255, 255),
                "RECORDING": (0, 0, 255),
            }[state]

            cv2.putText(
                frame_bgr,
                f"Label: {recorder.current_label}  |  "
                f"Signer: {recorder.signer_id}  |  "
                f"Hands: {num_hands}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )
            cv2.putText(
                frame_bgr,
                f"State: {state}  |  "
                f"[r]ecord [n]ame [s]igner [d]ataset [q]uit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )

            cv2.imshow("SignScribe — Data Recorder", frame_bgr)

            # ── Key handling ────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r") and state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.perf_counter()
                print(f"\n  ⏱ Countdown started for '{recorder.current_label}'...")

            elif key == ord("n") and state == "IDLE":
                # Prompt in console for new label
                print("\n  Available vocab preview (first 20):")
                for i, v in enumerate(VOCAB[:20]):
                    print(f"    {i:3d}: {v}")
                if len(VOCAB) > 20:
                    print(f"    ... and {len(VOCAB) - 20} more")
                label_input = input(
                    "  Enter label (or index number): "
                ).strip()
                if label_input.isdigit():
                    idx = int(label_input)
                    if 0 <= idx < len(VOCAB):
                        recorder.current_label = VOCAB[idx]
                    else:
                        print(f"  ⚠️  Index {idx} out of range.")
                elif label_input:
                    recorder.current_label = label_input.upper()
                print(f"  → Label set to: {recorder.current_label}")

            elif key == ord("s") and state == "IDLE":
                signer = input("  Enter signer ID: ").strip()
                if signer:
                    recorder.signer_id = signer
                print(f"  → Signer ID set to: {recorder.signer_id}")

            elif key == ord("d") and state == "IDLE":
                summary = recorder.get_dataset_summary()
                print("\n  === Dataset Summary ===")
                if summary:
                    total = 0
                    for label, count in summary.items():
                        print(f"    {label:20s}: {count:4d} samples")
                        total += count
                    print(f"    {'TOTAL':20s}: {total:4d} samples")
                else:
                    print("    (no data recorded yet)")
                print()

    finally:
        # ── Session summary ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("  Session Summary")
        print("=" * 60)
        if recorder.session_recorded:
            for label, count in sorted(recorder.session_recorded.items()):
                print(f"    {label:20s}: {count:3d} new samples")
            print(
                f"    {'TOTAL':20s}: "
                f"{sum(recorder.session_recorded.values()):3d} new samples"
            )
        else:
            print("    No samples recorded this session.")

        # Overall dataset
        summary = recorder.get_dataset_summary()
        if summary:
            total = sum(summary.values())
            print(f"\n  Total on disk: {total} samples across "
                  f"{len(summary)} labels")
        print("=" * 60)

        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recorder()
