"""
M4 — Training script for the BiLSTM gesture classifier.

Runs LOCALLY (CPU).  Trains the model on data from data/raw/, saves
the best checkpoint to models/classifier.pt, and prints a final
evaluation on the test set.

Usage:
    python -m src.classifier.train [--epochs 50] [--batch-size 32] [--lr 0.001]

Benchmarks real training time so you can decide if CPU training is
tolerable at full dataset scale.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CLASSIFIER_WEIGHTS, MODELS_DIR
from src.classifier.model import build_model
from src.classifier.dataset import create_dataloaders


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate model. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / max(len(all_preds), 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(
        description="Train the BiLSTM gesture classifier"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience (default: 10)"
    )
    args = parser.parse_args()

    # ── Load data ───────────────────────────────────────────────
    print("=" * 60)
    print("  SignScribe - Classifier Training (Phase 3)")
    print("=" * 60)

    try:
        train_loader, val_loader, test_loader, label_names = (
            create_dataloaders(batch_size=args.batch_size)
        )
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        print("  Record some data first with:")
        print("    python -m src.buffering.data_recorder")
        return

    num_classes = len(label_names)
    print(f"\n  Classes ({num_classes}): {label_names}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches  : {len(val_loader)}")
    print(f"  Test batches : {len(test_loader)}")

    # ── Build model ─────────────────────────────────────────────
    print()
    model = build_model(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── Training loop ───────────────────────────────────────────
    print(f"\n  Training for up to {args.epochs} epochs "
          f"(patience={args.patience})...")
    print(f"  {'Epoch':>5s}  {'Train Loss':>10s}  {'Train Acc':>9s}  "
          f"{'Val Loss':>8s}  {'Val Acc':>7s}  {'Time':>6s}")
    print("  " + "-" * 55)

    best_val_loss = float("inf")
    patience_counter = 0
    training_start = time.perf_counter()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"  {epoch:5d}  {train_loss:10.4f}  {train_acc:9.4f}  "
            f"{val_loss:8.4f}  {val_acc:7.4f}  {epoch_time:5.1f}s"
        )

        scheduler.step(val_loss)

        # Early stopping + checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": label_names,
                    "num_classes": num_classes,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                CLASSIFIER_WEIGHTS,
            )
            print(f"         ^ Best model saved -> {CLASSIFIER_WEIGHTS.name}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    total_time = time.perf_counter() - training_start
    print(f"\n  Total training time: {total_time:.1f}s "
          f"({total_time/60:.1f} min)")

    # ── Benchmark: estimate full-scale training time ────────────
    # Your synopsis target: ~57,600 augmented samples
    samples_trained = len(train_loader.dataset) * min(epoch, args.epochs)
    time_per_sample = total_time / max(samples_trained, 1)
    full_scale_samples = 57_600 * args.epochs
    estimated_full_time = time_per_sample * full_scale_samples
    print(f"\n  [CPU Training Benchmark]")
    print(f"    Time per sample   : {time_per_sample*1000:.2f} ms")
    print(f"    Full-scale estimate ({57_600} samples x {args.epochs} epochs):")
    print(f"      ~{estimated_full_time/60:.0f} min "
          f"({estimated_full_time/3600:.1f} hours)")
    if estimated_full_time > 7200:  # > 2 hours
        print(f"    WARNING: That's over 2 hours -- consider moving "
              f"training to Colab if it's too painful.")
    else:
        print(f"    OK -- That's feasible on CPU.")

    # ── Final test evaluation ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  Test Set Evaluation")
    print("=" * 60)

    # Load best checkpoint
    checkpoint = torch.load(CLASSIFIER_WEIGHTS, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion
    )

    print(f"\n  Test Loss    : {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    if len(set(test_labels)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average="weighted", zero_division=0
        )
        print(f"  Precision    : {precision:.4f}")
        print(f"  Recall       : {recall:.4f}")
        print(f"  F1 Score     : {f1:.4f}")

        print(f"\n  Classification Report:")
        print(classification_report(
            test_labels,
            test_preds,
            target_names=[
                label_names[i]
                for i in sorted(set(test_labels + test_preds))
            ],
            zero_division=0,
        ))
    else:
        print("  (Too few classes in test set for detailed metrics)")

    print("=" * 60)


if __name__ == "__main__":
    main()
