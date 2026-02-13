"""Multi-label transformers."""

from typing import Optional, Union

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing_extensions import Self

from parsers.base.core import BaseSingleColumnTransformer, BaseTextListNormalizer


class MultiLabelTransformer(BaseSingleColumnTransformer):
    """Transformer that applies `MultiLabelBinarizer` to a single column of lists of labels.

    Converts each list into multiple one-hot columns; names "{prefix}_{class}".
    """

    def __init__(
        self,
        output_column: Optional[str] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        """Initialize the transformer.

        Args:
            output_column: Output column name; if None, input name used.
            classes: Optional list of all possible classes.
        """
        super().__init__(output_column=output_column)
        self.classes = classes
        self.mlb: Optional[MultiLabelBinarizer] = None

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[pd.Series] = None) -> Self:
        """Fit the `MultiLabelBinarizer` on the input data.

        Args:
            X: Series or single-column DataFrame of lists of labels.
            y: Target values (ignored).

        Returns:
            Fitted transformer.
        """
        series: pd.Series = self.validate(X)
        self.feature_names_in_ = [series.name if series.name is not None else "feature"]

        self.mlb = MultiLabelBinarizer(classes=self.classes)
        self.mlb.fit(series.to_list())

        prefix = self.output_column if self.output_column else self.feature_names_in_[0]
        self.feature_names_out_ = [f"{prefix}_{cls}" for cls in self.mlb.classes_]

        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Transform the column into one-hot encoded columns.

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
            raise RuntimeError("Transformer must be fitted before calling 'transform'")

        series = self.validate(X)
        rows_encoded = []

        for row in series.to_list():
            if not row:
                rows_encoded.append([None] * len(self.mlb.classes_))
            else:
                rows_encoded.append(self.mlb.transform([row]).flatten())

        return pd.DataFrame(rows_encoded, columns=self.feature_names_out_, index=series.index)


class EmploymentTypeNormalizer(BaseTextListNormalizer):
    """Transformer to normalize employment type descriptions in a column.

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
    """Transformer to normalize work schedule descriptions in a column.

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
