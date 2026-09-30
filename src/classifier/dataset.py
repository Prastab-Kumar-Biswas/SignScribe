"""
M4 — Dataset loader with augmentation for gesture sequences.

Runs LOCALLY.  Loads .npy files from data/raw/<LABEL>/, applies
augmentation, and provides PyTorch DataLoaders for train/val/test.

Augmentations (from synopsis):
    1. Mirroring   — flip x-coordinates (simulates left↔right hand swap)
    2. Scaling     — random scale jitter on coordinates
    3. Temporal    — random frame shifting within the window
    4. Noise       — small Gaussian noise on landmarks

These produce the ~4× augmentation factor from the synopsis.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.config import (
    DATA_RAW_DIR,
    FEATURE_DIM,
    WINDOW_SIZE,
    NUM_LANDMARKS,
    NUM_COORDS,
    NUM_HANDS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


class GestureDataset(Dataset):
    """PyTorch dataset for recorded gesture sequences.

    Loads all .npy files from `data_dir/<LABEL>/`, assigns integer
    labels based on sorted unique label names, and optionally applies
    data augmentation.

    Each sample is a (WINDOW_SIZE, FEATURE_DIM) float32 tensor with
    an integer class label.
    """

    def __init__(
        self,
        data_dir: Path = DATA_RAW_DIR,
        augment: bool = False,
        augmentation_factor: int = 4,
    ):
        """
        Parameters
        ----------
        data_dir : Path
            Root directory containing per-label subdirectories.
        augment : bool
            If True, each original sample produces `augmentation_factor`
            augmented copies (plus the original).
        augmentation_factor : int
            How many augmented copies per original sample.
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.augmentation_factor = augmentation_factor

        # Discover labels and files
        self.samples: list[tuple[Path, int]] = []  # (filepath, label_idx)
        self.label_names: list[str] = []
        self._label_to_idx: dict[str, int] = {}

        self._discover_data()

    def _discover_data(self):
        """Scan data_dir for .npy files organised in label subdirectories."""
        if not self.data_dir.exists():
            print(f"WARNING: Data directory {self.data_dir} does not exist.")
            return

        # Get sorted list of label directories
        label_dirs = sorted(
            d for d in self.data_dir.iterdir() if d.is_dir()
        )

        self.label_names = [d.name for d in label_dirs]
        self._label_to_idx = {
            name: idx for idx, name in enumerate(self.label_names)
        }

        for label_dir in label_dirs:
            label_idx = self._label_to_idx[label_dir.name]
            npy_files = sorted(label_dir.glob("*.npy"))
            for f in npy_files:
                self.samples.append((f, label_idx))

        if self.samples:
            print(f"GestureDataset: found {len(self.samples)} raw samples "
                  f"across {len(self.label_names)} classes")
            if self.augment:
                effective = len(self.samples) * (1 + self.augmentation_factor)
                print(f"  With augmentation: ~{effective} effective samples")

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def __len__(self) -> int:
        if self.augment:
            return len(self.samples) * (1 + self.augmentation_factor)
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (window_tensor, label_idx).

        If augmentation is on, indices beyond len(samples) map to
        augmented versions of the original samples.
        """
        n_original = len(self.samples)

        if self.augment:
            # Map idx to (original_idx, augment_variant)
            original_idx = idx % n_original
            variant = idx // n_original  # 0 = original, 1+ = augmented
        else:
            original_idx = idx
            variant = 0

        filepath, label_idx = self.samples[original_idx]
        window = np.load(filepath).astype(np.float32)

        # Verify shape
        assert window.shape == (WINDOW_SIZE, FEATURE_DIM), (
            f"Expected ({WINDOW_SIZE}, {FEATURE_DIM}), "
            f"got {window.shape} from {filepath}"
        )

        # Apply augmentation if this is a non-original variant
        if variant > 0:
            window = self._augment(window, variant)

        return torch.from_numpy(window), label_idx

    def _augment(self, window: np.ndarray, variant: int) -> np.ndarray:
        """Apply one or more augmentations based on the variant number.

        Variant assignments:
            1 — mirror (flip x-coords)
            2 — scale jitter
            3 — temporal shift
            4 — Gaussian noise
        Additional variants cycle through combinations.
        """
        aug_type = (variant - 1) % 4

        if aug_type == 0:
            return self._augment_mirror(window)
        elif aug_type == 1:
            return self._augment_scale(window)
        elif aug_type == 2:
            return self._augment_temporal_shift(window)
        else:
            return self._augment_noise(window)

    @staticmethod
    def _augment_mirror(window: np.ndarray) -> np.ndarray:
        """Flip x-coordinates to simulate left↔right hand mirror.

        MediaPipe normalises x to [0, 1], so mirroring is x → (1 - x).
        Only the x-coordinate (every 3rd value starting from index 0)
        within each hand's 63-value block is flipped.
        """
        w = window.copy()
        for hand in range(NUM_HANDS):
            offset = hand * NUM_LANDMARKS * NUM_COORDS
            for lm in range(NUM_LANDMARKS):
                x_idx = offset + lm * NUM_COORDS  # x coordinate
                w[:, x_idx] = 1.0 - w[:, x_idx]
        return w

    @staticmethod
    def _augment_scale(
        window: np.ndarray,
        scale_range: tuple[float, float] = (0.85, 1.15),
    ) -> np.ndarray:
        """Apply random uniform scaling to all coordinates."""
        w = window.copy()
        scale = random.uniform(*scale_range)
        # Scale only non-zero values (don't scale padding)
        mask = w != 0
        w[mask] *= scale
        return w

    @staticmethod
    def _augment_temporal_shift(
        window: np.ndarray, max_shift: int = 3
    ) -> np.ndarray:
        """Shift frames forward/backward by a random amount, zero-padding."""
        w = window.copy()
        shift = random.randint(-max_shift, max_shift)
        if shift > 0:
            w[shift:] = w[:-shift]
            w[:shift] = 0
        elif shift < 0:
            w[:shift] = w[-shift:]
            w[shift:] = 0
        return w

    @staticmethod
    def _augment_noise(
        window: np.ndarray, noise_std: float = 0.01
    ) -> np.ndarray:
        """Add small Gaussian noise to all coordinates."""
        w = window.copy()
        mask = w != 0  # Don't add noise to zero-padding
        noise = np.random.normal(0, noise_std, w.shape).astype(np.float32)
        w[mask] += noise[mask]
        return w

    def idx_to_label(self, idx: int) -> str:
        """Convert integer label index back to string label."""
        return self.label_names[idx]


def create_dataloaders(
    data_dir: Path = DATA_RAW_DIR,
    batch_size: int = 32,
    augment_train: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create train/val/test DataLoaders from raw data.

    Parameters
    ----------
    data_dir : Path
        Root data directory.
    batch_size : int
        Batch size for all loaders.
    augment_train : bool
        Whether to apply augmentation to the training set.
    num_workers : int
        DataLoader workers (0 = main process, safest on Windows).

    Returns
    -------
    train_loader, val_loader, test_loader, label_names
    """
    # First load without augmentation to do the split on original data
    full_dataset = GestureDataset(data_dir, augment=False)

    if len(full_dataset) == 0:
        raise ValueError(
            f"No data found in {data_dir}. "
            "Record samples first with: python -m src.buffering.data_recorder"
        )

    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    # Ensure at least 1 sample per split
    if n_val == 0 and n_total >= 3:
        n_val = 1
        n_train -= 1
    if n_test == 0 and n_total >= 3:
        n_test = 1
        n_train -= 1

    print(f"Dataset split: {n_train} train / {n_val} val / {n_test} test")

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # For training: create a new augmented dataset from the train indices
    if augment_train and n_train > 0:
        train_files = [full_dataset.samples[i] for i in train_set.indices]
        train_dataset = _AugmentedSubset(
            train_files, full_dataset.label_names, augmentation_factor=4
        )
    else:
        train_dataset = train_set

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, full_dataset.label_names


class _AugmentedSubset(Dataset):
    """A subset of samples with on-the-fly augmentation.

    Used internally to apply augmentation only to training samples
    after the train/val/test split.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        label_names: list[str],
        augmentation_factor: int = 4,
    ):
        self.samples = samples
        self.label_names = label_names
        self.augmentation_factor = augmentation_factor

        # Preload augmentation functions from GestureDataset
        self._mirror = GestureDataset._augment_mirror
        self._scale = GestureDataset._augment_scale
        self._temporal = GestureDataset._augment_temporal_shift
        self._noise = GestureDataset._augment_noise

    def __len__(self) -> int:
        return len(self.samples) * (1 + self.augmentation_factor)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        n = len(self.samples)
        original_idx = idx % n
        variant = idx // n

        filepath, label_idx = self.samples[original_idx]
        window = np.load(filepath).astype(np.float32)

        if variant == 1:
            window = self._mirror(window)
        elif variant == 2:
            window = self._scale(window)
        elif variant == 3:
            window = self._temporal(window)
        elif variant == 4:
            window = self._noise(window)

        return torch.from_numpy(window), label_idx
