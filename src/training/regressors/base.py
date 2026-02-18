import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from training.base import BaseModel
from training.metrics import RegressionMetrics


class BaseRegressor(BaseModel):
    """Base regressor with standardized evaluation metrics.

    Extends BaseModel to provide regression-specific evaluation using
    common metrics: R², MSE, RMSE, and MAE.
    """

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> RegressionMetrics:
        """Evaluate model performance on given data.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            Container with r2, mse, rmse, mae values.
        """
        y_pred = self.predict(X)
        return RegressionMetrics(
            r2=r2_score(y, y_pred),
            mse=mean_squared_error(y, y_pred),
            rmse=root_mean_squared_error(y, y_pred),
            mae=mean_absolute_error(y, y_pred),
        )
