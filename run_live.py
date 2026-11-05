"""
SignScribe — High-Performance Native Live Desktop Runner.

Runs LOCALLY via native OpenCV. Provides a silky-smooth, zero-latency (30-60 FPS)
sign language to text generator desktop application.

Controls:
    [q]         — Quit
    [Space]     — Complete current word / Add space
    [Backspace] — Delete last letter or word
    [c]         — Clear generated text

Usage:
    python run_live.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import TARGET_FPS, WINDOW_SIZE
from src.capture.hand_tracker import HandTracker
from src.buffering.sliding_window import SlidingWindowBuffer
from src.classifier.inference import GesturePredictor
from src.text_generator.generator import SignTextGenerator


def run():
    print("=" * 65)
    print("  SignScribe — High-Speed Native Desktop Generator")
    print("=" * 65)
    print("Loading models (CPU)...")

    try:
        classifier = GesturePredictor()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    tracker = HandTracker(
        model_complexity=1,
        detection_confidence=0.5,
        tracking_confidence=0.35,
        max_persistence_frames=4,
    )
    window_buffer = SlidingWindowBuffer(
        overlap=WINDOW_SIZE // 2,
        min_active_ratio=0.6,
        max_idle_frames=6,
    )
    text_gen = SignTextGenerator()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        return

    print("\nControls:")
    print("  [q]         Quit")
    print("  [Space]     Add word space")
    print("  [Backspace] Delete last")
    print("  [c]         Clear text")
    print("  [m]         Toggle mode (Letters & Numbers vs All 97 Signs)\n")
    print("Ready! Starting video feed...")

    last_label = "IDLE"
    last_conf = 0.0
    consecutive_same = 0
    stability_threshold = 2
    confidence_threshold = 0.60
    last_committed = ""
    alphanumeric_only = True

    prev_time = time.perf_counter()
    fps = 0.0

    cv2.namedWindow("SignScribe — Sign to Text", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror webcam for natural selfie view & correct MediaPipe handedness
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, results = tracker.extract_landmarks(frame_rgb)

            num_hands = len(results.multi_hand_landmarks) if (results and results.multi_hand_landmarks) else 0
            has_active_hands = (num_hands > 0) or bool(np.any(features != 0))

            # Hand-gated window accumulation
            window = window_buffer.add_frame(features, has_hands=has_active_hands)

            if window is not None and has_active_hands:
                label, conf, _ = classifier.predict(
                    window, min_active_frames=12, alphanumeric_only=alphanumeric_only
                )

                if label != "IDLE" and conf >= confidence_threshold:
                    if label == last_label:
                        consecutive_same += 1
                    else:
                        consecutive_same = 1
                        last_label = label

                    if consecutive_same >= stability_threshold:
                        if label != last_committed:
                            text_gen.add_token(label)
                            last_committed = label
                            print(f"\r[Text] {text_gen.get_text()}", end="", flush=True)
                        consecutive_same = 0
                else:
                    if label == "IDLE":
                        last_label = "IDLE"
                        last_conf = 0.0

                last_conf = conf

            elif num_hands == 0:
                last_label = "IDLE"
                last_conf = 0.0
                consecutive_same = 0
                last_committed = ""

            # Draw MediaPipe hand landmarks
            frame = tracker.draw_landmarks(frame, results)

            # Draw HUD Overlay
            # Top Banner for detection status
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 30), -1)

            # Bottom Banner for generated text
            cv2.rectangle(overlay, (0, h - 90), (w, h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Status line
            if num_hands == 2:
                status_text = f"FPS: {fps:.1f} | Hands: 2 (Two-Hand Mode)"
                hand_color = (0, 255, 0)
            elif num_hands == 1:
                status_text = f"FPS: {fps:.1f} | Hands: 1 (1-Hand: C,I,L,O,U,V, 0-9 | A,B,D... need 2 hands)"
                hand_color = (0, 200, 255)
            else:
                status_text = f"FPS: {fps:.1f} | Hands: 0 (Idle)"
                hand_color = (180, 180, 180)

            cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 2)

            if last_label != "IDLE" and last_conf > 0:
                is_2h = classifier.is_two_hand_sign(last_label)
                sign_type_tag = "[2-Hand]" if is_2h else "[1-Hand]"
                pred_text = f"Sign: {last_label} ({last_conf:.0%}) {sign_type_tag}"
                cv2.putText(frame, pred_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Waiting for sign...", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 140, 160), 1)

            # Bottom Text Display
            current_text = text_gen.get_text()
            display_text = current_text if current_text else "Sign to generate text here..."
            text_color = (255, 255, 255) if current_text else (130, 130, 150)
            cv2.putText(frame, display_text, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2)
            mode_str = "Chars Only" if alphanumeric_only else "All 97 Signs"
            cv2.putText(
                frame,
                f"[{mode_str}] [m] Toggle  [Space] Space  [Bksp] Delete  [c] Clear  [q] Quit",
                (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 160, 160),
                1,
            )

            cv2.imshow("SignScribe — Sign to Text", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                alphanumeric_only = not alphanumeric_only
                print(f"\n[Mode Switched] Alphanumeric only: {alphanumeric_only}")
            elif key == ord(" "):
                text_gen.add_space()
                print(f"\r[Text] {text_gen.get_text()}", end="", flush=True)
            elif key == 8:  # Backspace
                text_gen.backspace()
                print(f"\r[Text] {text_gen.get_text()}", end="", flush=True)
            elif key == ord("c"):
                text_gen.reset()
                last_committed = ""
                print("\n[Text Cleared]")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("\n\nFinal generated text:\n" + text_gen.get_text())


if __name__ == "__main__":
    run()

