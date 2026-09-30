"""
M1 + M2 — Webcam capture with MediaPipe hand landmark extraction.

Runs LOCALLY.  Opens the default webcam, extracts 21 landmarks × 3 coords
× 2 hands = 126-dim feature vector per frame, throttled to the project's
TARGET_FPS (25 fps).  Overlays landmarks on the video and prints per-frame
latency so you can verify the 15 ms MediaPipe budget from the synopsis.

Usage:
    python -m src.capture.hand_tracker

Controls:
    q  — quit
"""

import time

import cv2
import mediapipe as mp
import numpy as np

# Import project constants from the central config
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.config` resolves
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    NUM_LANDMARKS,
    NUM_COORDS,
    NUM_HANDS,
    FEATURE_DIM,
    TARGET_FPS,
    BUDGET_MEDIAPIPE_MS,
)


class HandTracker:
    """Wraps MediaPipe Hands for real-time landmark extraction.

    Design note: we use `model_complexity=0` (lite model) because it's
    the fastest option on CPU.  `max_num_hands=2` matches our 2-hand
    feature vector design.

    `process_scale` controls optional frame downscaling before feeding
    MediaPipe.  Since landmarks are normalized 0-1, downscaling doesn't
    affect feature quality — it just reduces the pixel count the palm
    detector has to scan, which can significantly cut latency.
    """

    def __init__(
        self,
        max_hands: int = NUM_HANDS,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        process_scale: float = 1.0,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.process_scale = process_scale

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       # video mode → faster tracking
            max_num_hands=max_hands,
            model_complexity=0,            # 0 = lite (fastest)
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def extract_landmarks(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray, object | None]:
        """Extract the 126-dim feature vector from a single RGB frame.

        Returns:
            features: np.ndarray of shape (FEATURE_DIM,) = (126,).
                      If fewer than 2 hands are detected, the missing
                      hand's 63 values are filled with zeros.
            results:  The raw MediaPipe results object (for drawing).
        """
        # Optionally downscale for faster processing
        if self.process_scale < 1.0:
            h, w = frame_rgb.shape[:2]
            small = cv2.resize(
                frame_rgb,
                (int(w * self.process_scale), int(h * self.process_scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            results = self.hands.process(small)
        else:
            results = self.hands.process(frame_rgb)

        # Initialise with zeros — missing hands stay as zero-vectors
        features = np.zeros(FEATURE_DIM, dtype=np.float32)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(
                results.multi_hand_landmarks[:NUM_HANDS]
            ):
                offset = hand_idx * NUM_LANDMARKS * NUM_COORDS  # 0 or 63
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    base = offset + lm_idx * NUM_COORDS
                    features[base] = lm.x
                    features[base + 1] = lm.y
                    features[base + 2] = lm.z

        return features, results

    def draw_landmarks(
        self, frame_bgr: np.ndarray, results: object
    ) -> np.ndarray:
        """Draw detected hand landmarks and connections on the frame."""
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


def run_live_demo():
    """Open webcam, extract landmarks, overlay, and print latency."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam (index 0). Is it connected?")
        return

    # Try to set webcam resolution (optional — falls back gracefully)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker()
    frame_interval = 1.0 / TARGET_FPS  # 0.04 s = 40 ms at 25 fps

    print(f"SignScribe — Hand Tracker (Phase 1)")
    print(f"  Feature dim : {FEATURE_DIM}")
    print(f"  Target FPS  : {TARGET_FPS}")
    print(f"  MediaPipe budget: {BUDGET_MEDIAPIPE_MS} ms")
    print(f"  Press 'q' to quit.\n")

    # Running stats for latency
    latencies: list[float] = []

    try:
        prev_frame_time = time.perf_counter()

        while True:
            # ── Throttle to TARGET_FPS ──────────────────────────────
            now = time.perf_counter()
            elapsed = now - prev_frame_time
            if elapsed < frame_interval:
                # Sleep most of the remaining time, then spin-wait
                sleep_time = frame_interval - elapsed - 0.002
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Spin-wait for precise timing
                while time.perf_counter() - prev_frame_time < frame_interval:
                    pass

            prev_frame_time = time.perf_counter()

            # ── Capture frame ───────────────────────────────────────
            ret, frame_bgr = cap.read()
            if not ret:
                print("WARNING: Failed to read frame. Exiting.")
                break

            # ── MediaPipe processing (timed) ────────────────────────
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t_start = time.perf_counter()
            features, results = tracker.extract_landmarks(frame_rgb)
            t_end = time.perf_counter()

            mediapipe_ms = (t_end - t_start) * 1000.0
            latencies.append(mediapipe_ms)

            # ── Draw landmarks on frame ─────────────────────────────
            frame_bgr = tracker.draw_landmarks(frame_bgr, results)

            # ── On-screen HUD ───────────────────────────────────────
            num_hands = (
                len(results.multi_hand_landmarks)
                if results.multi_hand_landmarks
                else 0
            )
            budget_color = (
                (0, 255, 0)
                if mediapipe_ms <= BUDGET_MEDIAPIPE_MS
                else (0, 0, 255)
            )

            cv2.putText(
                frame_bgr,
                f"MediaPipe: {mediapipe_ms:.1f} ms  |  Hands: {num_hands}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                budget_color,
                2,
            )
            cv2.putText(
                frame_bgr,
                f"Budget: {BUDGET_MEDIAPIPE_MS} ms  |  FPS target: {TARGET_FPS}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            # ── Show frame ──────────────────────────────────────────
            cv2.imshow("SignScribe — Hand Tracker", frame_bgr)

            # ── Print feature vector summary to console ─────────────
            nonzero = np.count_nonzero(features)
            print(
                f"\rHands: {num_hands}  |  "
                f"MediaPipe: {mediapipe_ms:5.1f} ms  |  "
                f"Non-zero features: {nonzero}/{FEATURE_DIM}  |  "
                f"Vec[0:6]: {features[:6]}",
                end="",
            )

            # ── Quit on 'q' ─────────────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # ── Print latency summary ───────────────────────────────────
        print("\n")
        if latencies:
            arr = np.array(latencies)
            print("=== MediaPipe Latency Summary ===")
            print(f"  Frames processed : {len(arr)}")
            print(f"  Mean             : {arr.mean():.2f} ms")
            print(f"  Median           : {np.median(arr):.2f} ms")
            print(f"  95th percentile  : {np.percentile(arr, 95):.2f} ms")
            print(f"  Max              : {arr.max():.2f} ms")
            print(f"  Within budget    : {(arr <= BUDGET_MEDIAPIPE_MS).sum()}/{len(arr)} frames")
            if arr.mean() > BUDGET_MEDIAPIPE_MS:
                print(
                    f"  ⚠️  Mean ({arr.mean():.1f} ms) EXCEEDS the "
                    f"{BUDGET_MEDIAPIPE_MS} ms budget!"
                )
            else:
                print(f"  ✅ Mean is within the {BUDGET_MEDIAPIPE_MS} ms budget.")

        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_demo()
