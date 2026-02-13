"""Metric types for regression and classification evaluation."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BaseMetrics:
    """Base dataclass for evaluation metrics."""

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RegressionMetrics(BaseMetrics):
    """Regression evaluation metrics.

    Attributes:
        r2: R-squared (coefficient of determination).
        mse: Mean squared error.
        rmse: Root mean squared error.
        mae: Mean absolute error.
    """

    r2: float
    mse: float
    rmse: float
    mae: float


@dataclass
class ClassificationMetrics(BaseMetrics):
    """Classification evaluation metrics.

    Attributes:
        report: Full output from sklearn 'classification_report'.
        accuracy: Overall accuracy.
    """

    report: dict[str, Any]
    accuracy: float
