"""
Video-to-landmarks converter for the Kaggle ISL video dataset.

Processes MP4 video clips through our MediaPipe pipeline and saves
the output as (30, 126) .npy files in data/raw/<LABEL>/, matching
the exact format the data recorder produces.

Dataset: kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset
Expected structure after download+extract:
    <dataset_root>/
        Bear/
            Bear_1.mp4
            Bear_2.mp4
            ...
        Crocodile/
            Crocodile_1.mp4
            ...
        ...

Usage:
    python scripts/convert_video_dataset.py --input <path_to_extracted_dataset>

    Example:
    python scripts/convert_video_dataset.py --input C:/Users/ratna/Downloads/ISL_Videos
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    DATA_RAW_DIR,
    FEATURE_DIM,
    WINDOW_SIZE,
    TARGET_FPS,
)
from src.capture.hand_tracker import HandTracker

# Mapping from dataset folder names to our vocabulary labels.
# The dataset uses title-case folder names; our vocab uses UPPER_CASE.
LABEL_MAP: dict[str, str] = {
    "Bear": "BEAR",
    "Break": "BREAK",
    "Brinjal": "BRINJAL",
    "Budget": "BUDGET",
    "Busy": "BUSY",
    "Cabbage": "CABBAGE",
    "Carrot": "CARROT",
    "Cauliflower": "CAULIFLOWER",
    "Chilli": "CHILLI",
    "Clean": "CLEAN",
    "Close": "CLOSE",
    "Come": "COME",
    "Cook": "COOK",
    "Crocodile": "CROCODILE",
    "Cry": "CRY",
    "Cucumber": "CUCUMBER",
    "Deer": "DEER",
    "Drink": "DRINK",
    "Elephant": "ELEPHANT",
    "Exam": "EXAM",
    "Fedup": "FEDUP",
    "Fever": "FEVER",
    "Giraffe": "GIRAFFE",
    "Give": "GIVE",
    "Good Morning": "GOOD_MORNING",
    "Good afternoon": "GOOD_AFTERNOON",
    "Hello": "HELLO",
    "Hug": "HUG",
    "Injury": "INJURY",
    "Interview": "INTERVIEW",
    "Jump": "JUMP",
    "Karnataka": "KARNATAKA",
    "Key": "KEY",
    "Knife": "KNIFE",
    "Lemon": "LEMON",
    "Lion": "LION",
    "Man": "MAN",
    "Maths": "MATHS",
    "Maybe": "MAYBE",
    "Monkey": "MONKEY",
    "Onion": "ONION",
    "Peacock": "PEACOCK",
    "Pigeon": "PIGEON",
    "Pour": "POUR",
    "Radish": "RADISH",
    "Sparrow": "SPARROW",
    "Still": "STILL",
    "Switch": "SWITCH",
    "Tea": "TEA",
    "Temple": "TEMPLE",
    "Thank you": "THANK_YOU",
    "Tiger": "TIGER",
    "Turtle": "TURTLE",
    "Umbrella": "UMBRELLA",
    "Uncle": "UNCLE",
    "Vegetables": "VEGETABLES",
    "Volcano": "VOLCANO",
    "What is your Name": "WHAT_IS_YOUR_NAME",
    "Wife": "WIFE",
    "Writer": "WRITER",
    "Wrong": "WRONG",
}


def extract_windows_from_video(
    video_path: Path,
    tracker: HandTracker,
    target_fps: int = TARGET_FPS,
) -> list[np.ndarray]:
    """Extract (WINDOW_SIZE, FEATURE_DIM) windows from a video file.

    The video is processed frame by frame through MediaPipe. Frames are
    sampled at target_fps. Each complete window of WINDOW_SIZE frames is
    returned as a numpy array.

    Parameters
    ----------
    video_path : Path
        Path to the MP4 file.
    tracker : HandTracker
        Initialised MediaPipe hand tracker.
    target_fps : int
        Target frame rate for sampling.

    Returns
    -------
    list of np.ndarray
        Each array has shape (WINDOW_SIZE, FEATURE_DIM).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame skip to achieve target_fps from the video's native fps
    frame_skip = max(1, round(video_fps / target_fps))

    frames_buffer: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Sample at target_fps
        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            features, _ = tracker.extract_landmarks(frame_rgb)
            frames_buffer.append(features)

            # When we have a full window, save it
            if len(frames_buffer) == WINDOW_SIZE:
                window = np.array(frames_buffer, dtype=np.float32)
                windows.append(window)
                # Slide by half the window for overlapping extraction
                frames_buffer = frames_buffer[WINDOW_SIZE // 2:]

        frame_idx += 1

    cap.release()

    # If we have remaining frames that form at least 70% of a window,
    # pad with zeros to make a full window
    if len(frames_buffer) >= int(WINDOW_SIZE * 0.7):
        while len(frames_buffer) < WINDOW_SIZE:
            frames_buffer.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        window = np.array(frames_buffer, dtype=np.float32)
        windows.append(window)

    return windows


def convert_dataset(
    input_dir: Path,
    output_dir: Path = DATA_RAW_DIR,
    max_videos_per_class: int = 0,
):
    """Convert the entire video dataset to landmark .npy files.

    Parameters
    ----------
    input_dir : Path
        Root directory of the extracted Kaggle dataset.
    output_dir : Path
        Target output directory (default: data/raw/).
    max_videos_per_class : int
        Limit videos per class (0 = no limit). Useful for quick testing.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Discover class folders
    class_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not class_dirs:
        # Maybe videos are one level deeper
        for sub in input_dir.iterdir():
            if sub.is_dir():
                deeper = sorted(d for d in sub.iterdir() if d.is_dir())
                if deeper:
                    class_dirs = deeper
                    break

    print(f"Found {len(class_dirs)} class folders in {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    tracker = HandTracker()

    total_samples = 0
    skipped_classes = []
    start_time = time.time()

    for class_dir in class_dirs:
        folder_name = class_dir.name

        # Map folder name to our label
        label = LABEL_MAP.get(folder_name)
        if label is None:
            # Try case-insensitive match
            for key, val in LABEL_MAP.items():
                if key.lower() == folder_name.lower():
                    label = val
                    break

        if label is None:
            skipped_classes.append(folder_name)
            continue

        # Create output directory
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Find existing samples to continue numbering
        existing = list(label_dir.glob("*.npy"))
        sample_count = len(existing)

        # Find video files
        videos = sorted(
            f for f in class_dir.iterdir()
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
        )

        if max_videos_per_class > 0:
            videos = videos[:max_videos_per_class]

        print(f"  {folder_name:25s} -> {label:25s} | "
              f"{len(videos)} videos, {sample_count} existing samples")

        for video_path in videos:
            windows = extract_windows_from_video(video_path, tracker)

            for window in windows:
                sample_count += 1
                filename = f"{label}_kaggle_{sample_count:04d}.npy"
                np.save(label_dir / filename, window)
                total_samples += 1

        print(f"{'':27s} -> {sample_count} total samples now")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  Conversion complete!")
    print(f"  New samples created : {total_samples}")
    print(f"  Time elapsed        : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output directory    : {output_dir}")
    if skipped_classes:
        print(f"  Skipped folders     : {skipped_classes}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaggle ISL video dataset to landmark .npy files"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the extracted Kaggle video dataset root folder"
    )
    parser.add_argument(
        "--output", type=str, default=str(DATA_RAW_DIR),
        help=f"Output directory (default: {DATA_RAW_DIR})"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Max videos per class (0 = all). Use 5 for a quick test."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SignScribe - Video Dataset Converter")
    print("=" * 60)
    print()

    convert_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        max_videos_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
