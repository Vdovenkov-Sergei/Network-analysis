"""
Multi-label transformers.
"""

from typing import Any, Optional

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from parsers.base import BaseSingleColumnTransformer, BaseTextListNormalizer


class MultiLabelTransformer(BaseSingleColumnTransformer):
    """
    Transformer that applies `MultiLabelBinarizer` to a single column of lists of labels.

    Converts each list of labels into multiple one-hot encoded columns.

    Notes:
        - Input must be a pandas Series or a single-column DataFrame.
        - The resulting columns are named as "{input_column_name}_{class}".
    """

    def __init__(
        self,
        output_column: Optional[str] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the transformer.

        Args:
            output_column: Name of the output column. If None, the input column name is used.
            classes: Optional list of all possible classes.
        """
        super().__init__(output_column=output_column)
        self.classes = classes
        self.mlb: Optional[MultiLabelBinarizer] = None

    def fit(
        self, X: Any, y: Optional[pd.Series] = None
    ) -> "MultiLabelTransformer":
        """
        Fit the `MultiLabelBinarizer` on the input data.

        Args:
            X: Series or single-column DataFrame of lists of labels.
            y: Target values (ignored).

        Returns:
            self: Fitted transformer.

        Raises:
            TypeError: If input is not a Series or single-column DataFrame.
        """
        series = self._to_series(X)
        name = "feature" if series.name is None else str(series.name)
        self.feature_names_in_: list[str] = [name]

        self.mlb = MultiLabelBinarizer(classes=self.classes)
        self.mlb.fit(series.to_list())

        prefix = (
            self.output_column
            if self.output_column
            else self.feature_names_in_[0]
        )
        self.feature_names_out_: list[str] = [
            f"{prefix}_{cls}" for cls in self.mlb.classes_
        ]

        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """
        Transform the input column into one-hot encoded columns for multiple labels.

        If a row contains an empty list (or None/NaN normalized to []), all output columns
        for that row are set to None instead of 0 or 1.

        Args:
            X: Series or single-column DataFrame of lists of labels.

        Returns:
            One-hot encoded DataFrame with a column per class.

        Raises:
            RuntimeError: If the transformer has not been fitted.
        """
        if self.mlb is None:
            raise RuntimeError(
                "Transformer must be fitted before calling 'transform'"
            )

        series = self._to_series(X)
        rows_encoded = []

        for row in series.to_list():
            if not row:
                rows_encoded.append([None] * len(self.mlb.classes_))
            else:
                rows_encoded.append(self.mlb.transform([row])[0])

        return pd.DataFrame(
            rows_encoded, columns=self.feature_names_out_, index=series.index
        )


class EmploymentTypeNormalizer(BaseTextListNormalizer):
    """
    Transformer to normalize employment type descriptions in a column.

    This transformer processes each entry that may contain one or more employment
    types, applies a mapping to standardize them into consistent labels and returns
    a list of unique normalized types.
    """

    NORMALIZE_MAP: dict[str, str] = {
        "полная занятость": "full_time",
        "частичная занятость": "part_time",
        "проектная работа": "project_work",
        "стажировка": "work_placement",
        "волонтерство": "volunteering",
    }


class WorkScheduleNormalizer(BaseTextListNormalizer):
    """
    Transformer to normalize work schedule descriptions in a column.

    This transformer processes each entry that may contain one or more work schedule
    types, applies a mapping to standardize them into consistent labels and returns
    a list of unique normalized schedules.
    """

    NORMALIZE_MAP: dict[str, str] = {
        "полный день": "full_day",
        "гибкий график": "flexible_schedule",
        "сменный график": "shift_schedule",
        "удаленная работа": "remote_working",
        "вахтовый метод": "rotation_based_work",
    }
