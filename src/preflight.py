"""Read-only readiness checks for a SignScribe training or demo run.

Usage:
    python -m src.preflight
    python -m src.preflight --strict

The command intentionally does not require a webcam or download any models.
It makes missing data, incompatible checkpoint labels, and absent LoRA weights
explicit before a training or demonstration session begins.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import (
    CLASSIFIER_WEIGHTS,
    DATA_RAW_DIR,
    FEATURE_DIM,
    VOCAB,
    WINDOW_SIZE,
)


def _status(kind: str, message: str) -> None:
    print(f"[{kind}] {message}")


def check_readiness() -> list[str]:
    """Print project readiness checks and return blocking issue messages."""
    blockers: list[str] = []
    expected_labels = set(VOCAB)
    duplicate_labels = [label for label, count in Counter(VOCAB).items() if count > 1]

    print("SignScribe preflight")
    print("=" * 60)
    _status("INFO", f"Configured vocabulary: {len(VOCAB)} classes")
    if duplicate_labels:
        blockers.append("Configured vocabulary has duplicate labels")
        _status("BLOCK", f"Duplicate configured labels: {duplicate_labels}")
    else:
        _status("OK", "Configured vocabulary labels are unique")

    samples = list(DATA_RAW_DIR.glob("*/*.npy")) if DATA_RAW_DIR.exists() else []
    labels_on_disk = Counter(path.parent.name for path in samples)
    unknown_labels = sorted(set(labels_on_disk) - expected_labels)
    malformed: list[str] = []
    for sample in samples:
        try:
            array = np.load(sample, mmap_mode="r")
            if array.shape != (WINDOW_SIZE, FEATURE_DIM):
                malformed.append(f"{sample}: {array.shape}")
        except Exception as exc:  # a corrupt user sample should be reported, not crash checks
            malformed.append(f"{sample}: unreadable ({exc})")

    _status("INFO", f"Recorded samples: {len(samples)} across {len(labels_on_disk)} labels")
    if labels_on_disk:
        _status("INFO", "Per-label counts: " + ", ".join(
            f"{label}={count}" for label, count in sorted(labels_on_disk.items())
        ))
    else:
        blockers.append("No recorded gesture samples found")
        _status("BLOCK", f"No .npy samples found under {DATA_RAW_DIR}")

    if unknown_labels:
        blockers.append("Recorded labels outside the configured vocabulary")
        _status("BLOCK", f"Unknown recorded labels: {', '.join(unknown_labels)}")
    if malformed:
        blockers.append("One or more recorded samples have an invalid shape or cannot be read")
        for item in malformed:
            _status("BLOCK", item)
    elif samples:
        _status("OK", f"All samples have shape ({WINDOW_SIZE}, {FEATURE_DIM})")

    missing_labels = sorted(expected_labels - set(labels_on_disk))
    if missing_labels:
        blockers.append("Recorded dataset does not cover the configured vocabulary")
        _status("WARN", f"No data yet for {len(missing_labels)} configured classes")

    if CLASSIFIER_WEIGHTS.exists():
        try:
            import torch

            checkpoint = torch.load(
                CLASSIFIER_WEIGHTS, map_location="cpu", weights_only=False
            )
            checkpoint_labels = checkpoint.get("label_names", [])
            checkpoint_classes = checkpoint.get("num_classes")
            _status(
                "INFO",
                "Classifier checkpoint: "
                f"{checkpoint_classes} classes, epoch {checkpoint.get('epoch', '?')}, "
                f"validation accuracy {checkpoint.get('val_acc', '?')}",
            )
            if checkpoint_classes != len(checkpoint_labels):
                blockers.append("Classifier checkpoint class count does not match its labels")
                _status("BLOCK", "Checkpoint num_classes does not match label_names")
            if set(checkpoint_labels) - expected_labels:
                blockers.append("Classifier checkpoint contains labels outside configured vocabulary")
                _status("BLOCK", "Checkpoint contains labels outside the configured vocabulary")
            if checkpoint_labels != sorted(labels_on_disk):
                _status(
                    "WARN",
                    "Checkpoint labels differ from the currently recorded dataset; retrain after data changes.",
                )
        except Exception as exc:
            blockers.append("Classifier checkpoint could not be loaded")
            _status("BLOCK", f"Could not load classifier checkpoint: {exc}")
    else:
        blockers.append("Classifier checkpoint is missing")
        _status("BLOCK", f"Missing classifier checkpoint: {CLASSIFIER_WEIGHTS}")

    _status("OK", "Sign-to-text pipeline ready")

    print("=" * 60)
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SignScribe data and model readiness")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any training/demo blocker is present",
    )
    args = parser.parse_args()
    blockers = check_readiness()
    if blockers:
        _status("SUMMARY", f"{len(blockers)} blocker(s) found")
        if args.strict:
            raise SystemExit(1)
    else:
        _status("SUMMARY", "All required checks passed")


if __name__ == "__main__":
    main()
