"""Gradient Boosting classifier."""

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC

from training.classifiers.base import BaseClassifier


class GradientBoostingClassifier(BaseClassifier):
    """Gradient Boosting for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Gradient Boosting classifier."""
        super().__init__(estimator=SklearnGBC, **kwargs)
