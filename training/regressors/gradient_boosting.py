"""Gradient Boosting regression model.

This module provides a Gradient Boosting regressor using
scikit-learn's GradientBoostingRegressor.
"""

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBRegressor
from typing_extensions import Self

from config.tuning import GRADIENT_BOOSTING_PARAM_GRID
from training.regressors.base import BaseRegressor
from utils.training import tune_hyperparameters


class GradientBoostingRegressor(BaseRegressor):
    """Classic Gradient Boosting regressor for salary prediction.

    Uses tree-based gradient boosting with stage-wise optimization.

    Attributes:
        learning_rate: Boosting learning rate.
        max_depth: Maximum depth of trees.
        n_estimators: Number of boosting stages.
        min_samples_leaf: Minimum samples at leaf nodes.
        random_seed: Random seed for reproducibility.
        tune_hyperparams: Whether to tune hyperparameters.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 100,
        min_samples_leaf: int = 20,
        random_seed: int = 42,
        tune_hyperparams: bool = True,
    ) -> None:
        """Initialize the Gradient Boosting regressor.

        Args:
            learning_rate: Shrinkage parameter for boosting.
            max_depth: Maximum depth of individual trees.
            n_estimators: Number of boosting stages.
            min_samples_leaf: Minimum samples at leaf nodes.
            random_seed: Random seed for reproducibility.
            tune_hyperparams: Whether to tune hyperparameters.
        """
        super().__init__(random_seed=random_seed)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.tune_hyperparams = tune_hyperparams
        self.best_params_: Optional[dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the Gradient Boosting model.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            self: The fitted regressor.
        """
        self.logger.info(f"Fitting '{self.__class__.__name__}'...")

        if self.tune_hyperparams:
            self.logger.info("Tuning hyperparameters...")
            base_estimator = GBRegressor(random_state=self.random_seed, verbose=0)

            self.model, self.best_params_ = tune_hyperparameters(
                estimator=base_estimator, param_grid=GRADIENT_BOOSTING_PARAM_GRID, X=X, y=y
            )

            self.learning_rate = self.best_params_.get("learning_rate", self.learning_rate)
            self.max_depth = self.best_params_.get("max_depth", self.max_depth)
            self.n_estimators = self.best_params_.get("n_estimators", self.n_estimators)

            self.logger.info(f"Best 'Learning Rate': {self.learning_rate}")
            self.logger.info(f"Best 'Max Depth': {self.max_depth}")
            self.logger.info(f"Best 'N Estimators': {self.n_estimators}")

        else:
            self.model = GBRegressor(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_seed,
                verbose=0,
            )
            self.model.fit(X, y)

            self.logger.info(f"'Learning Rate': {self.learning_rate}")
            self.logger.info(f"'Max Depth': {self.max_depth}")
            self.logger.info(f"'N Estimators': {self.n_estimators}")

        self._is_fitted = True
        return self
