"""Configuration for hyperparameter tuning."""

from typing import Any

REGRESSION_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [1, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "min_samples_leaf": [10, 20, 50],
    },
}

CLASSIFICATION_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [500, 1000],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10, 20],
    },
}
