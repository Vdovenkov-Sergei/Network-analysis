"""Base model class for all trained models."""

import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from typing_extensions import Self

from utils.logging import setup_logger


class BaseModel:
    """Base class for training models with hyperparameter tuning support.

    This class provides a unified interface for training scikit-learn compatible
    estimators with optional GridSearchCV hyperparameter optimization.
    """

    def __init__(
        self,
        estimator: type[BaseEstimator],
        param_grid: dict[str, list[Any]],
        scoring: str,
        random_seed: int = 42,
        tune_hyperparams: bool = True,
        cv: int = 5,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            estimator: Estimator class.
            param_grid: Parameter grid for hyperparameter tuning.
            scoring: Scoring metric for hyperparameter tuning.
            random_seed: Random seed for reproducibility.
            tune_hyperparams: Whether to tune hyperparameters.
            cv: Cross-validation folds.
            n_jobs: Number of jobs to run in parallel.
            **kwargs: Additional keyword arguments for the estimator.
        """
        self.random_seed = random_seed
        self.estimator_class = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.tune_hyperparams = tune_hyperparams
        self.cv = cv
        self.n_jobs = n_jobs
        self.init_kwargs = kwargs
        self.model: Optional[BaseEstimator] = None
        self.logger = setup_logger(logger_name=self.__class__.__name__)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model to training data.

        Performs hyperparameter tuning or fits the base estimator with
        provided parameters.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Returns:
            The fitted model instance.
        """
        self.logger.info(f"Fitting '{self.__class__.__name__}'...")

        base = self.estimator_class(random_state=self.random_seed, **self.init_kwargs)

        if self.tune_hyperparams and self.param_grid:
            self.logger.info("Tuning hyperparameters...")
            grid_search = GridSearchCV(
                estimator=base,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=0,
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            for param, value in sorted(grid_search.best_params_.items()):
                self.logger.info(f"Best '{param}': {value}")
            for key, val in grid_search.best_params_.items():
                self.init_kwargs[key] = val
        else:
            self.model = base
            self.model.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values or labels.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before making predictions.")

        result: np.ndarray = self.model.predict(X)
        return result

    def save_model(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model (e.g. .pkl).

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

    def load_model(self, path: Union[str, Path]) -> None:
        """Load the trained model from a pickle file.

        Args:
            path: File path to the saved pickle.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{path}'")

        with open(path, "rb") as fin:
            self.model = pickle.load(fin)

        self._is_fitted = True
        self.logger.info(f"Model loaded from '{path}'")
