"""Random Forest regression model.

This module provides a Random Forest regressor for salary prediction.
"""

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from typing_extensions import Self

from config.tuning import RANDOM_FOREST_PARAM_GRID
from training.regressors.base import BaseRegressor
from utils.training import tune_hyperparameters


class RandomForestRegressor(BaseRegressor):
    """Random Forest regressor for salary prediction.

    Uses an ensemble of decision trees with bagging.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees.
        min_samples_split: Minimum samples to split an internal node.
        min_samples_leaf: Minimum samples at a leaf node.
        n_jobs: Number of parallel jobs (-1 uses all cores).
        random_seed: Random seed for reproducibility.
        tune_hyperparams: Whether to tune hyperparameters.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        n_jobs: int = -1,
        random_seed: int = 42,
        tune_hyperparams: bool = True,
    ) -> None:
        """Initialize the Random Forest regressor.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees.
            min_samples_split: Minimum samples to split an internal node.
            min_samples_leaf: Minimum samples at a leaf node.
            n_jobs: Number of parallel jobs (-1 uses all cores).
            random_seed: Random seed for reproducibility.
            tune_hyperparams: Whether to tune hyperparameters.
        """
        super().__init__(random_seed=random_seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.tune_hyperparams = tune_hyperparams
        self.best_params_: Optional[dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the Random Forest model.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            self: The fitted regressor.
        """
        self.logger.info(f"Fitting '{self.__class__.__name__}'...")

        if self.tune_hyperparams:
            self.logger.info("Tuning hyperparameters...")
            base_estimator = SklearnRF(n_jobs=self.n_jobs, random_state=self.random_seed, verbose=0)
            self.model, self.best_params_ = tune_hyperparameters(
                estimator=base_estimator, param_grid=RANDOM_FOREST_PARAM_GRID, X=X, y=y
            )
            self.n_estimators = self.best_params_.get("n_estimators", self.n_estimators)
            self.max_depth = self.best_params_.get("max_depth", self.max_depth)
            self.logger.info(f"Best 'N Estimators': {self.n_estimators}")
            self.logger.info(f"Best 'Max Depth': {self.max_depth}")
        else:
            self.model = SklearnRF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=self.n_jobs,
                random_state=self.random_seed,
                verbose=0,
            )
            self.model.fit(X, y)
            self.logger.info(f"'N Estimators': {self.n_estimators}")
            self.logger.info(f"'Max Depth': {self.max_depth}")

        self._is_fitted = True
        return self
