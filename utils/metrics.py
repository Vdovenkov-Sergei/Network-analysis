"""Metrics and evaluation utilities using sklearn.

This module provides functions for calculating regression metrics
and saving evaluation results.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


@dataclass
class RegressionMetrics:
    """
    Container for regression evaluation metrics.

    Attributes:
        r2: R-squared (coefficient of determination).
        mse: Mean squared error.
        rmse: Root mean squared error.
        mae: Mean absolute error.
    """

    r2: float
    mse: float
    rmse: float
    mae: float

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics to dictionary.

        Returns:
            Dictionary with metric names and values.
        """
        return asdict(self)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """
    Calculate all regression metrics for predictions.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.

    Returns:
        RegressionMetrics object with all calculated metrics.
    """
    return RegressionMetrics(
        r2=r2_score(y_true, y_pred),
        mse=mean_squared_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
    )


def save_predictions(y_pred: np.ndarray, save_path: Union[str, Path], model_name: str) -> None:
    """
    Save predictions to a NumPy file.

    Args:
        y_pred: Predicted values.
        save_path: Directory to save the predictions.
        model_name: Name of the model (used in filename).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_predictions.npy"
    np.save(save_path / filename, y_pred)


def save_metrics(metrics: RegressionMetrics, save_path: Union[str, Path], model_name: str) -> None:
    """
    Save metrics to a JSON file.

    Args:
        metrics: RegressionMetrics object.
        save_path: Directory to save the metrics.
        model_name: Name of the model (used in filename).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{model_name}_metrics.json"
    with open(save_path / filename, "w", encoding="utf-8") as json_file:
        json.dump(metrics.to_dict(), json_file, indent=4)
