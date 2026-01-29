"""Utilities for training models.

This module provides utilities for training models and tuning hyperparameters.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from training.regressors.base import BaseRegressor
from utils.metrics import RegressionMetrics


def train_model(
    model_cls: type[BaseRegressor],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_seed: int = 42,
    tune_hyperparams: bool = True,
) -> tuple[BaseRegressor, dict[str, RegressionMetrics]]:
    """Train a single model and evaluate on all splits.

    Args:
        model_cls: Model class to instantiate.
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        random_seed: Random seed for the model.
        tune_hyperparams: Whether to tune hyperparameters.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    model = model_cls(random_seed=random_seed, tune_hyperparams=tune_hyperparams)  # type: ignore[call-arg]

    # --- Fit model ---
    model.fit(X_train, y_train)

    # --- Evaluate on all splits ---
    metrics = {
        "train": model.evaluate(X_train, y_train),
        "test": model.evaluate(X_test, y_test),
    }

    return model, metrics


def tune_hyperparameters(
    estimator: BaseEstimator,
    param_grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "r2",
    n_jobs: int = -1,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Tune hyperparameters using GridSearchCV.

    Args:
        estimator: Scikit-learn estimator to tune.
        param_grid: Dictionary with parameter names and lists of values.
        X: Training features.
        y: Training targets.
        cv: Number of cross-validation folds.
        scoring: Scoring metric (default: "r2").
        n_jobs: Number of parallel jobs (-1 for all cores).

    Returns:
        Tuple of (best_estimator, best_params, best_score).
    """
    # --- Run GridSearchCV ---
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=True,
    )
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_
