"""Base regressor class for all regression models.

This module provides an abstract base class that defines the common
interface for all regression models in the training package.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from typing_extensions import Self

from utils.logging import setup_logger
from utils.metrics import RegressionMetrics, evaluate_predictions


class BaseRegressor(ABC):
    """Abstract base class for regression models.

    This class defines the common interface for training, evaluating,
    and saving regression models.

    Attributes:
        model: The underlying model object.
        logger: Logger instance for the model.
    """

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the regressor.

        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        self.model: Optional[BaseEstimator] = None
        self.logger = setup_logger(logger_name=self.__class__.__name__)
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model to training data.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            self: The fitted regressor.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")

        result: np.ndarray = self.model.predict(X)
        return result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> RegressionMetrics:
        """Evaluate the model on a dataset.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            RegressionMetrics object with evaluation results.
        """
        y_pred = self.predict(X)
        return evaluate_predictions(y, y_pred)

    def save_model(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model (should end with .pkl).

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before saving.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as fout:
            pickle.dump(self.model, fout)

        self.logger.info(f"Model saved to '{path}'")
