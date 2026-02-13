"""Random Forest regression."""

from typing import Any

from sklearn.ensemble import RandomForestRegressor as SklearnRFR

from training.regressors.base import BaseRegressor


class RandomForestRegressor(BaseRegressor):
    """Random Forest for regression."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Random Forest regressor."""
        super().__init__(estimator=SklearnRFR, **kwargs)
