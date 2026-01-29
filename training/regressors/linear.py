"""Linear regression model with Ridge regularization.

This module provides a linear regression model with Ridge (L2)
regularization for salary prediction.
"""

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import Ridge
from typing_extensions import Self

from config.tuning import RIDGE_PARAM_GRID
from training.regressors.base import BaseRegressor
from utils.training import tune_hyperparameters


class LinearRegressor(BaseRegressor):
    """Ridge regression model for salary prediction.

    Uses L2 regularization to prevent overfitting.

    Attributes:
        alpha: Regularization strength.
        random_seed: Random seed for reproducibility.
        tune_hyperparams: Whether to tune hyperparameters.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        random_seed: int = 42,
        tune_hyperparams: bool = True,
    ) -> None:
        """Initialize the Ridge regressor.

        Args:
            alpha: Regularization strength.
            random_seed: Random seed for reproducibility.
            tune_hyperparams: Whether to tune hyperparameters using GridSearchCV.
        """
        super().__init__(random_seed=random_seed)
        self.alpha = alpha
        self.tune_hyperparams = tune_hyperparams
        self.best_params_: Optional[dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the Ridge regression model.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            self: The fitted regressor.
        """
        self.logger.info(f"Fitting '{self.__class__.__name__}'...")

        if self.tune_hyperparams:
            self.logger.info("Tuning hyperparameters...")
            base_estimator = Ridge(random_state=self.random_seed)
            self.model, self.best_params_ = tune_hyperparameters(
                estimator=base_estimator, param_grid=RIDGE_PARAM_GRID, X=X, y=y
            )
            self.alpha = self.best_params_["alpha"]
            self.logger.info(f"Best 'Alpha': {self.alpha}")
        else:
            self.model = Ridge(alpha=self.alpha, random_state=self.random_seed)
            self.model.fit(X, y)
            self.logger.info(f"'Alpha': {self.alpha}")

        self._is_fitted = True
        return self
