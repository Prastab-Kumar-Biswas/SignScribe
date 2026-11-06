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


def _hand_wrist_and_palm(hand_landmarks) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extract wrist (lm 0) and palm center coordinates."""
    lms = hand_landmarks.landmark
    wrist = (lms[0].x, lms[0].y)
    palm_x = (lms[0].x + lms[5].x + lms[9].x + lms[13].x + lms[17].x) / 5.0
    palm_y = (lms[0].y + lms[5].y + lms[9].y + lms[13].y + lms[17].y) / 5.0
    return wrist, (palm_x, palm_y)


def _deduplicate_hands(
    detected_hands: list[object],
    handedness: list[object] | None,
    iou_thresh: float = 0.08,
) -> tuple[list[object], list[object] | None]:
    """Remove duplicate overlapping detections on the same physical hand."""
    if len(detected_hands) <= 1:
        return detected_hands, handedness

    h0, h1 = detected_hands[0], detected_hands[1]
    (w0_x, w0_y), (p0_x, p0_y) = _hand_wrist_and_palm(h0)
    (w1_x, w1_y), (p1_x, p1_y) = _hand_wrist_and_palm(h1)

    d_wrist = ((w0_x - w1_x) ** 2 + (w0_y - w1_y) ** 2) ** 0.5
    d_palm = ((p0_x - p1_x) ** 2 + (p0_y - p1_y) ** 2) ** 0.5

    # If both wrists and palms virtually coincide, they are duplicate boxes on 1 hand:
    if d_wrist < iou_thresh and d_palm < iou_thresh:
        if handedness and len(handedness) >= 2:
            score0 = (
                handedness[0].classification[0].score
                if handedness[0].classification
                else 0.5
            )
            score1 = (
                handedness[1].classification[0].score
                if handedness[1].classification
                else 0.5
            )
            if score1 > score0:
                return [h1], [handedness[1]]
            else:
                return [h0], [handedness[0]]
        return [h0], (handedness[:1] if handedness else None)

    return detected_hands, handedness


class HandTracker:
    """Wraps MediaPipe Hands for real-time landmark extraction.

    Design note: we use `model_complexity=1` (Full model) with
    spatial continuity tracking to prevent slot-flipping jitter,
    spatial deduplication to eliminate duplicate detections on a single hand,
    and temporal persistence ONLY during confirmed two-handed occlusions.
    `max_num_hands=2` matches our 2-hand feature vector design.

    `process_scale` controls optional frame downscaling before feeding
    MediaPipe.  Since landmarks are normalized 0-1, downscaling doesn't
    affect feature quality — it just reduces the pixel count the palm
    detector has to scan, which can significantly cut latency.
    """

    def __init__(
        self,
        max_hands: int = NUM_HANDS,
        model_complexity: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.55,
        process_scale: float = 1.0,
        max_persistence_frames: int = 2,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.process_scale = process_scale
        self.max_persistence_frames = max_persistence_frames

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       # video mode → faster tracking
            max_num_hands=max_hands,
            model_complexity=model_complexity,  # 1 = full model (occlusion robust)
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # Spatial continuity tracking & persistence state:
        self._tracked_wrists: list[tuple[float, float] | None] = [None] * max_hands
        self._last_landmarks: list[np.ndarray | None] = [None] * max_hands
        self._missing_counts: list[int] = [0] * max_hands
        self._two_hands_confirmed_count: int = 0
        self._last_ordered_hands: list[object | None] = [None] * max_hands
        self._num_active_hands: int = 0

    @property
    def num_active_hands(self) -> int:
        """Number of genuine active hands confirmed in the last frame (0, 1, or 2)."""
        return self._num_active_hands

    def reset(self) -> None:
        """Reset internal temporal tracking and persistence state."""
        self._tracked_wrists = [None] * NUM_HANDS
        self._last_landmarks = [None] * NUM_HANDS
        self._missing_counts = [0] * NUM_HANDS
        self._two_hands_confirmed_count = 0
        self._last_ordered_hands = [None] * NUM_HANDS
        self._num_active_hands = 0

    @staticmethod
    def _encode_hand_to_features(hand_landmarks, features: np.ndarray, slot: int) -> None:
        """Serialize 21 hand landmarks into the 63 floats of the given hand slot."""
        offset = slot * NUM_LANDMARKS * NUM_COORDS  # 0 or 63
        for lm_idx, lm in enumerate(hand_landmarks.landmark):
            base = offset + lm_idx * NUM_COORDS
            features[base] = lm.x
            features[base + 1] = lm.y
            features[base + 2] = lm.z

    def extract_landmarks(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray, object | None]:
        """Extract the 126-dim feature vector from a single RGB frame.

        Enforces strict spatial consistency:
        - When 1 hand is present: strictly encodes 1 hand (other slot is 0.0),
          eliminating all phantom ghost hands and flickering.
        - Deduplicates overlapping bounding boxes when MediaPipe detects 2 hands
          on the same physical hand.
        - Persists occluded hands ONLY when two hands were genuinely both active.

        Returns:
            features: np.ndarray of shape (FEATURE_DIM,) = (126,).
            results:  The raw MediaPipe results object.
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

        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        ordered_hands: list[object | None] = [None] * NUM_HANDS

        raw_hands = (
            list(results.multi_hand_landmarks[:NUM_HANDS])
            if (results and results.multi_hand_landmarks)
            else []
        )
        raw_handedness = (
            list(results.multi_handedness[:NUM_HANDS])
            if (results and getattr(results, "multi_handedness", None))
            else []
        )

        # 1. Non-Maximum Suppression (Deduplication) for duplicate boxes on the same hand:
        valid_hands, valid_handedness = _deduplicate_hands(raw_hands, raw_handedness)
        num_detected = len(valid_hands)

        # ── Case A: No hands detected ─────────────────────────────────
        if num_detected == 0:
            self._tracked_wrists = [None] * NUM_HANDS
            self._last_landmarks = [None] * NUM_HANDS
            self._missing_counts = [0] * NUM_HANDS
            self._two_hands_confirmed_count = 0
            self._last_ordered_hands = [None] * NUM_HANDS
            self._num_active_hands = 0
            return features, results

        # ── Case B: Both hands clearly detected (>= 2) ─────────────────
        if num_detected >= 2:
            hA, hB = valid_hands[0], valid_hands[1]
            (wA_x, wA_y), _ = _hand_wrist_and_palm(hA)
            (wB_x, wB_y), _ = _hand_wrist_and_palm(hB)

            # Spatial tracking: match to previously tracked slots if available:
            prev0, prev1 = self._tracked_wrists[0], self._tracked_wrists[1]
            if prev0 is not None and prev1 is not None:
                cost_normal = (
                    ((wA_x - prev0[0]) ** 2 + (wA_y - prev0[1]) ** 2) ** 0.5
                    + ((wB_x - prev1[0]) ** 2 + (wB_y - prev1[1]) ** 2) ** 0.5
                )
                cost_swapped = (
                    ((wB_x - prev0[0]) ** 2 + (wB_y - prev0[1]) ** 2) ** 0.5
                    + ((wA_x - prev1[0]) ** 2 + (wA_y - prev1[1]) ** 2) ** 0.5
                )
                if cost_normal <= cost_swapped:
                    hand0, hand1 = hA, hB
                else:
                    hand0, hand1 = hB, hA
            else:
                lblA = (
                    valid_handedness[0].classification[0].label.lower()
                    if valid_handedness and len(valid_handedness) > 0 and valid_handedness[0].classification
                    else ""
                )
                lblB = (
                    valid_handedness[1].classification[0].label.lower()
                    if valid_handedness and len(valid_handedness) > 1 and valid_handedness[1].classification
                    else ""
                )
                if lblA == "left" and lblB == "right":
                    hand0, hand1 = hA, hB
                elif lblA == "right" and lblB == "left":
                    hand0, hand1 = hB, hA
                else:
                    # In mirrored selfie view: smaller x is Left (slot 0), larger x is Right (slot 1)
                    if wA_x <= wB_x:
                        hand0, hand1 = hA, hB
                    else:
                        hand0, hand1 = hB, hA

            ordered_hands[0] = hand0
            ordered_hands[1] = hand1
            self._encode_hand_to_features(hand0, features, slot=0)
            self._encode_hand_to_features(hand1, features, slot=1)

            (w0_x, w0_y), _ = _hand_wrist_and_palm(hand0)
            (w1_x, w1_y), _ = _hand_wrist_and_palm(hand1)
            self._tracked_wrists[0] = (w0_x, w0_y)
            self._tracked_wrists[1] = (w1_x, w1_y)
            self._last_landmarks[0] = features[:63].copy()
            self._last_landmarks[1] = features[63:].copy()
            self._missing_counts = [0, 0]
            self._two_hands_confirmed_count = min(self._two_hands_confirmed_count + 1, 10)
            self._last_ordered_hands = ordered_hands
            self._num_active_hands = 2
            return features, results

        # ── Case C: Exactly one hand detected (num_detected == 1) ──────
        cur_hand = valid_hands[0]
        (cur_wx, cur_wy), _ = _hand_wrist_and_palm(cur_hand)

        # Check if this is a momentary occlusion during a confirmed two-handed sign:
        is_occlusion_case = (
            self._two_hands_confirmed_count >= 3
            and self._last_landmarks[0] is not None
            and self._last_landmarks[1] is not None
            and self._tracked_wrists[0] is not None
            and self._tracked_wrists[1] is not None
            and self.max_persistence_frames > 0
        )

        if is_occlusion_case:
            d0 = (
                (cur_wx - self._tracked_wrists[0][0]) ** 2
                + (cur_wy - self._tracked_wrists[0][1]) ** 2
            ) ** 0.5
            d1 = (
                (cur_wx - self._tracked_wrists[1][0]) ** 2
                + (cur_wy - self._tracked_wrists[1][1]) ** 2
            ) ** 0.5
            detected_slot = 0 if d0 <= d1 else 1
            occluded_slot = 1 - detected_slot

            if self._missing_counts[occluded_slot] < self.max_persistence_frames:
                ordered_hands[detected_slot] = cur_hand
                self._encode_hand_to_features(cur_hand, features, slot=detected_slot)
                self._tracked_wrists[detected_slot] = (cur_wx, cur_wy)
                self._last_landmarks[detected_slot] = features[
                    detected_slot * 63 : (detected_slot + 1) * 63
                ].copy()
                self._missing_counts[detected_slot] = 0

                # Forward-fill occluded slot
                occ_offset = occluded_slot * 63
                features[occ_offset : occ_offset + 63] = self._last_landmarks[occluded_slot]
                self._missing_counts[occluded_slot] += 1

                self._last_ordered_hands = ordered_hands
                self._num_active_hands = 2
                return features, results
            else:
                # Occlusion window expired: drop occluded hand
                self._two_hands_confirmed_count = 0
                self._last_landmarks[occluded_slot] = None
                self._tracked_wrists[occluded_slot] = None
                self._missing_counts[occluded_slot] = 0

        # Normal Single-Hand Tracking: STRICTLY 1 ACTIVE HAND
        self._two_hands_confirmed_count = 0

        # Spatial continuity: maintain previous slot if within reasonable radius (< 0.22)
        target_slot = None
        if self._tracked_wrists[0] is not None:
            d0 = (
                (cur_wx - self._tracked_wrists[0][0]) ** 2
                + (cur_wy - self._tracked_wrists[0][1]) ** 2
            ) ** 0.5
            if d0 < 0.22:
                target_slot = 0
        if target_slot is None and self._tracked_wrists[1] is not None:
            d1 = (
                (cur_wx - self._tracked_wrists[1][0]) ** 2
                + (cur_wy - self._tracked_wrists[1][1]) ** 2
            ) ** 0.5
            if d1 < 0.22:
                target_slot = 1

        if target_slot is None:
            lbl = (
                valid_handedness[0].classification[0].label.lower()
                if valid_handedness and len(valid_handedness) > 0 and valid_handedness[0].classification
                else ""
            )
            if lbl == "left":
                target_slot = 0
            elif lbl == "right":
                target_slot = 1
            else:
                target_slot = 0 if cur_wx < 0.5 else 1

        other_slot = 1 - target_slot

        ordered_hands[target_slot] = cur_hand
        self._encode_hand_to_features(cur_hand, features, slot=target_slot)
        self._tracked_wrists[target_slot] = (cur_wx, cur_wy)
        self._last_landmarks[target_slot] = features[
            target_slot * 63 : (target_slot + 1) * 63
        ].copy()
        self._missing_counts[target_slot] = 0

        # Explicitly zero-out the other slot: NO GHOST HANDS!
        self._tracked_wrists[other_slot] = None
        self._last_landmarks[other_slot] = None
        self._missing_counts[other_slot] = 0
        features[other_slot * 63 : (other_slot + 1) * 63] = 0.0

        self._last_ordered_hands = ordered_hands
        self._num_active_hands = 1
        return features, results

    def draw_landmarks(
        self, frame_bgr: np.ndarray, results: object = None
    ) -> np.ndarray:
        """Draw detected hand landmarks, connections, and handedness tags."""
        h, w = frame_bgr.shape[:2]

        # Draw only validated, active hands from this frame:
        for slot in range(NUM_HANDS):
            hand_landmarks = self._last_ordered_hands[slot]
            if hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Draw handedness tag near the wrist
                wrist = hand_landmarks.landmark[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                label = "Left" if slot == 0 else "Right"
                color = (255, 180, 0) if slot == 0 else (0, 220, 255)
                cv2.putText(
                    frame_bgr,
                    label,
                    (wx - 20, wy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
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
