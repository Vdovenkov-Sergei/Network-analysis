"""Ridge regression (L2-regularized linear regression)."""

from typing import Any

from sklearn.linear_model import Ridge

from training.regressors.base import BaseRegressor


class RidgeRegressor(BaseRegressor):
    """Ridge model (L2) for regression."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Ridge regressor."""
        super().__init__(estimator=Ridge, **kwargs)
