"""Logistic regression classifier."""

from typing import Any

from sklearn.linear_model import LogisticRegression as SklearnLR

from training.classifiers.base import BaseClassifier


class LogisticClassifier(BaseClassifier):
    """Logistic regression for classification (binary or multinomial)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Logistic Regression classifier."""
        super().__init__(estimator=SklearnLR, **kwargs)
