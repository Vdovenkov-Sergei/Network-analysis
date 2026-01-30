"""Configuration for hyperparameter tuning.

This module contains the configuration dictionary that defines the hyperparameter grids
for each model.
"""

from typing import Any

RIDGE_PARAM_GRID: dict[str, list[Any]] = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

RANDOM_FOREST_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [1, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

GRADIENT_BOOSTING_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "min_samples_leaf": [10, 20, 50],
}
