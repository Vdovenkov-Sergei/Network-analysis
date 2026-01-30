"""Utilities for loading and splitting preprocessed datasets.

This module provides a DataLoader class for loading preprocessed NumPy data,
splitting datasets into train/validation/test sets and a BatchIterator class
for iterating over the data in batches.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from typing_extensions import Self


@dataclass
class DataSplit:
    """Container for train/validation/test data splits.

    Attributes:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        feature_names: Names of features.
        target_names: Names of targets.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: np.ndarray
    target_names: np.ndarray


class DataLoader:
    """Class for loading and splitting preprocessed data.

    Handles loading NumPy files containing preprocessed features
    and targets, and splitting them into train/test sets.

    Attributes:
        data_dir: Directory containing preprocessed data files.
        prefix: Optional filename prefix.
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        prefix: str = "",
        random_seed: int = 42,
    ) -> None:
        """Initialize the DataLoader.

        Args:
            data_dir: Directory containing preprocessed data files.
            prefix: Optional filename prefix (e.g., "train_").
            random_seed: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.prefix = prefix
        self.random_seed = random_seed

        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._feature_names: Optional[np.ndarray] = None
        self._target_names: Optional[np.ndarray] = None

    def load(self) -> Self:
        """Load preprocessed data from NumPy files.

        Returns:
            self: The DataLoader instance with loaded data.

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        self._X = np.load(self.data_dir / f"{self.prefix}X_data.npy")
        self._y = np.load(self.data_dir / f"{self.prefix}y_data.npy")
        self._feature_names = np.load(
            self.data_dir / f"{self.prefix}feature_names.npy", allow_pickle=True
        )
        self._target_names = np.load(
            self.data_dir / f"{self.prefix}target_names.npy", allow_pickle=True
        )
        return self

    @property
    def X(self) -> np.ndarray:
        """Get feature matrix."""
        if self._X is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Get target vector."""
        if self._y is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._y

    @property
    def feature_names(self) -> np.ndarray:
        """Get feature names."""
        if self._feature_names is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._feature_names

    @property
    def target_names(self) -> np.ndarray:
        """Get target names."""
        if self._target_names is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._target_names

    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return self.X.shape[1]

    def split(
        self,
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
        shuffle: bool = True,
    ) -> DataSplit:
        """Split data into train and test sets.

        Args:
            train_ratio: Fraction of data for training (default: 0.7).
            test_ratio: Fraction of data for testing (default: 0.15).
            shuffle: Whether to shuffle data before splitting (default: True).

        Returns:
            DataSplit object containing all splits.

        Raises:
            ValueError: If ratios don't sum to 1.0 or are invalid.
            RuntimeError: If data is not loaded.
        """
        # --- Validate ratios ---
        total_ratio = train_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.2f}")

        if any(r <= 0 for r in [train_ratio, test_ratio]):
            raise ValueError("All ratios must be positive")

        n_samples = self.n_samples
        indices = np.arange(n_samples)

        # --- Shuffle if requested ---
        if shuffle:
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(indices)

        # --- Calculate split points ---
        train_end = int(n_samples * train_ratio)

        # --- Split indices ---
        train_idx = indices[:train_end]
        test_idx = indices[train_end:]

        return DataSplit(
            X_train=self.X[train_idx],
            y_train=self.y[train_idx],
            X_test=self.X[test_idx],
            y_test=self.y[test_idx],
            feature_names=self.feature_names,
            target_names=self.target_names,
        )
