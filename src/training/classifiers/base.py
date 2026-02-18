from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from training.base import BaseModel
from training.metrics import ClassificationMetrics


class BaseClassifier(BaseModel):
    """Base classifier with standardized evaluation metrics.

    Extends BaseModel to provide classification-specific evaluation using
    common metrics: precision, recall, F1-score and accuracy.
    """

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_labels: Optional[list[str]] = None,
    ) -> ClassificationMetrics:
        """Evaluate model performance on given data.

        Args:
            X: Feature matrix.
            y: True class labels.
            class_labels: Optional list of class labels for the report.

        Returns:
            Container with report dict and accuracy.
        """
        y_pred = self.predict(X)
        report = classification_report(
            y,
            y_pred,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        accuracy = accuracy_score(y, y_pred)
        report.pop("accuracy", None)
        return ClassificationMetrics(report=report, accuracy=accuracy)
