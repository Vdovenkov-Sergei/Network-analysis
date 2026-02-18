"""Random Forest classifier."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier as SklearnRFC

from training.classifiers.base import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """Random Forest for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Random Forest classifier."""
        super().__init__(estimator=SklearnRFC, **kwargs)
