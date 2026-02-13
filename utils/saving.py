"""Utilities for saving predictions, metrics, datasets, and class labels."""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from training.metrics import BaseMetrics


def save_predictions(y_pred: np.ndarray, save_path: Union[str, Path], model_name: str) -> None:
    """Save predictions to a NumPy file.

    Args:
        y_pred: Predicted values array.
        save_path: Directory to save the predictions.
        model_name: Name of the model (used in filename).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_predictions.npy"
    np.save(save_path / filename, y_pred)


def save_metrics(
    metrics: BaseMetrics, save_path: Union[str, Path], model_name: str, suffix: str = "metrics"
) -> None:
    """Save metrics to a JSON file.

    Args:
        metrics: Metrics object with to_dict() method.
        save_path: Directory to save the metrics.
        model_name: Name of the model (used in filename).
        suffix: Filename suffix.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_{suffix}.json"
    with open(save_path / filename, "w", encoding="utf-8") as json_file:
        json.dump(metrics.to_dict(), json_file, indent=4)


def save_dataset(X: pd.DataFrame, y: pd.Series, path: Union[str, Path], prefix: str = "") -> None:
    """Save feature matrix X and target y to .npy files.

    Creates: {prefix}X_data.npy, {prefix}y_data.npy, {prefix}feature_columns.npy,
    {prefix}target_column.npy.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        path: Directory to save files.
        prefix: Optional filename prefix.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / f"{prefix}X_data.npy", X.to_numpy())
    np.save(path / f"{prefix}feature_columns.npy", X.columns.to_numpy())

    np.save(path / f"{prefix}y_data.npy", y.to_numpy())
    np.save(path / f"{prefix}target_column.npy", np.array([y.name]))


def save_labels(labels: list[str], path: Union[str, Path], prefix: str = "") -> None:
    """Save class labels for classification tasks.

    Args:
        labels: Ordered list of class labels.
        path: Directory to save the file.
        prefix: Optional filename prefix.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"{prefix}class_labels.npy", np.array(labels))
