"""Gradient Boosting regression."""

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR

from training.regressors.base import BaseRegressor


class GradientBoostingRegressor(BaseRegressor):
    """Gradient Boosting for regression."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Gradient Boosting regressor."""
        super().__init__(estimator=SklearnGBR, **kwargs)
