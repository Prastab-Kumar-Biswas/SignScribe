"""
CSV landmarks converter for the Kaggle ISL hand landmarks dataset.

Converts the pre-extracted MediaPipe landmarks CSV (A-Z alphabet classes)
into (30, 126) .npy files for our pipeline.

Since the CSV contains STATIC single-frame landmarks (not temporal
sequences), each row is repeated 30 times to create a "held sign"
window.  This is appropriate for ISL alphabet signs where you hold
a static hand shape.

Dataset: kaggle.com/datasets/eraakash/indian-sign-language-hand-landmarks-dataset
Expected CSV structure:
    - Each row is one sample
    - Columns include hand landmark coordinates (x, y, z per landmark)
    - A label/class column identifying A-Z
    - Possibly a 'uses_two_hands' engineered feature column

Usage:
    python scripts/convert_csv_landmarks.py --input <path_to_csv_file>

    Example:
    python scripts/convert_csv_landmarks.py --input C:/Users/ratna/Downloads/isl_landmarks.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    DATA_RAW_DIR,
    FEATURE_DIM,
    WINDOW_SIZE,
    NUM_LANDMARKS,
    NUM_COORDS,
    NUM_HANDS,
)


def detect_csv_format(df: pd.DataFrame) -> dict:
    """Auto-detect the CSV format and column mapping.

    Returns
    -------
    dict with keys:
        label_col: str - name of the label column
        feature_cols: list[str] - ordered list of feature columns
        num_features: int - total feature columns found
    """
    print(f"  CSV shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns[:10])}...")

    # Find the label column (typically 'label', 'class', 'sign', or 'gesture')
    label_col = None
    label_candidates = ["label", "class", "sign", "gesture", "target", "category"]
    for col in df.columns:
        if col.lower().strip() in label_candidates:
            label_col = col
            break

    # If not found by name, look for the column with A-Z string values
    if label_col is None:
        for col in df.columns:
            unique_vals = df[col].unique()
            if len(unique_vals) <= 26 and all(
                isinstance(v, str) and len(v) <= 3 for v in unique_vals
            ):
                label_col = col
                break

    if label_col is None:
        # Last resort: first or last column
        if df.dtypes.iloc[0] == object:
            label_col = df.columns[0]
        elif df.dtypes.iloc[-1] == object:
            label_col = df.columns[-1]
        else:
            raise ValueError(
                "Cannot auto-detect label column. "
                f"Columns: {list(df.columns)}"
            )

    print(f"  Label column: '{label_col}'")
    print(f"  Unique labels: {sorted(df[label_col].unique())}")

    # Feature columns = all numeric columns except the label
    # Also exclude 'uses_two_hands' and similar engineered features
    skip_cols = {label_col, "uses_two_hands", "index", "id", "Unnamed: 0"}
    feature_cols = [
        col for col in df.columns
        if col not in skip_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"  Feature columns: {len(feature_cols)}")

    return {
        "label_col": label_col,
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
    }


def convert_csv(
    csv_path: Path,
    output_dir: Path = DATA_RAW_DIR,
    samples_per_class: int = 0,
):
    """Convert CSV landmarks to .npy window files.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    output_dir : Path
        Output directory (default: data/raw/).
    samples_per_class : int
        Max samples per class (0 = all).
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    fmt = detect_csv_format(df)
    label_col = fmt["label_col"]
    feature_cols = fmt["feature_cols"]
    num_features = fmt["num_features"]

    # Determine how to map features to our (126,) vector.
    # The CSV might have:
    #   - 63 features (1 hand × 21 landmarks × 3 coords)
    #   - 126 features (2 hands × 21 landmarks × 3 coords)
    #   - 42 features (1 hand × 21 landmarks × 2 coords, no z)
    #   - Other count
    print(f"\n  Mapping {num_features} CSV features -> {FEATURE_DIM} pipeline features")

    if num_features == FEATURE_DIM:
        # Perfect match: 126 features
        print("  Direct 1:1 mapping (126 features)")
    elif num_features == FEATURE_DIM // 2:
        # 63 features = 1 hand only; pad second hand with zeros
        print("  Single-hand data (63 features); padding second hand with zeros")
    elif num_features == NUM_LANDMARKS * 2:
        # 42 features = 1 hand, x+y only; pad z and second hand
        print("  Single-hand x,y only (42 features); padding z and second hand")
    else:
        print(f"  Non-standard feature count ({num_features}); "
              f"will pad/truncate to {FEATURE_DIM}")

    total_samples = 0
    labels = sorted(df[label_col].unique())

    for label_value in labels:
        # Map to our uppercase label format
        label_str = str(label_value).strip().upper()

        label_dir = output_dir / label_str
        label_dir.mkdir(parents=True, exist_ok=True)

        # Count existing samples
        existing = list(label_dir.glob("*.npy"))
        sample_num = len(existing)

        # Get rows for this class
        class_df = df[df[label_col] == label_value]
        if samples_per_class > 0:
            class_df = class_df.head(samples_per_class)

        for _, row in class_df.iterrows():
            # Extract feature vector
            features = row[feature_cols].values.astype(np.float32)

            # Pad or truncate to FEATURE_DIM
            if len(features) < FEATURE_DIM:
                padded = np.zeros(FEATURE_DIM, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            elif len(features) > FEATURE_DIM:
                features = features[:FEATURE_DIM]

            # Create a (WINDOW_SIZE, FEATURE_DIM) window by repeating
            # the static frame (appropriate for held alphabet signs)
            window = np.tile(features, (WINDOW_SIZE, 1))  # (30, 126)

            # Add tiny random noise per frame to avoid the BiLSTM seeing
            # identical frames (which would be unrealistic)
            noise = np.random.normal(0, 0.002, window.shape).astype(np.float32)
            mask = window != 0
            window[mask] += noise[mask]

            sample_num += 1
            filename = f"{label_str}_csv_{sample_num:04d}.npy"
            np.save(label_dir / filename, window)
            total_samples += 1

        print(f"  {label_str:5s}: {len(class_df)} rows -> "
              f"{sample_num} total samples in {label_dir.name}/")

    print(f"\n  Conversion complete! {total_samples} new samples created.")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaggle ISL landmarks CSV to .npy window files"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the CSV file"
    )
    parser.add_argument(
        "--output", type=str, default=str(DATA_RAW_DIR),
        help=f"Output directory (default: {DATA_RAW_DIR})"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Max samples per class (0 = all). Use 100 for a quick test."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SignScribe - CSV Landmarks Converter")
    print("=" * 60)
    print()

    convert_csv(
        csv_path=Path(args.input),
        output_dir=Path(args.output),
        samples_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
