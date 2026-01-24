"""
Base classes for transformers.

This module contains abstract base classes that provide the foundation
for all column transformers used in the preprocessing pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Self


class BaseSingleColumnTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base transformer for operations on a single column.

    This class provides a foundation for transformers that operate on a single
    pandas Series or a single-column DataFrame. It tracks input and output
    feature names for compatibility with scikit-learn pipelines.

    Notes:
        - Subclasses must implement the `transform` method.
        - The output column name can be explicitly set via `output_column`.
        - If `output_column` is None, the original input column name is preserved.
    """

    def __init__(self, output_column: Optional[str] = None) -> None:
        """
        Initialize the transformer.

        Args:
            output_column: Name of the output column. If None, the input column name is used.
        """
        self.output_column = output_column

    def fit(
        self, X: Union[pd.Series, pd.DataFrame], y: Optional[pd.Series] = None
    ) -> Self:
        """
        Fit the transformer and record the column name.

        Args:
            X: Input data (Series or single-column DataFrame).
            y: Target values (ignored).

        Returns:
            self: Fitted transformer.
        """
        series: pd.Series = self._to_series(X)
        name = "feature" if series.name is None else str(series.name)

        self.feature_names_in_: list[str] = [name]
        self.feature_names_out_: list[str] = (
            [self.output_column]
            if self.output_column
            else self.feature_names_in_.copy()
        )

        return self

    @abstractmethod
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform the input data.

        Subclasses must implement this method to perform the actual transformation.

        Args:
            X: Input data (Series or single-column DataFrame).

        Returns:
            Transformed data as a pandas DataFrame with a single column.
        """
        pass

    def get_feature_names_out(
        self, input_features: Optional[list[str]] = None
    ) -> list[str]:
        """
        Get output feature name(s) for the transformer.

        Args:
            input_features: Ignored, included for scikit-learn compatibility.

        Returns:
            List of output column names.

        Raises:
            RuntimeError: If called before the transformer is fitted.
        """
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError(
                "Transformer must be fitted before calling 'get_feature_names_out'"
            )

        return self.feature_names_out_

    @staticmethod
    def _to_series(X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Validate input and return it as a pandas Series.

        Args:
            X: Input data (Series or single-column DataFrame).

        Returns:
            Pandas Series representing the column.

        Raises:
            ValueError: If a DataFrame has more than one column.
            TypeError: If input is neither a Series nor a DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    f"Input DataFrame must contain exactly one column, got {X.shape[1]}"
                )
            return X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            return X.copy()
        else:
            raise TypeError(
                "Input must be a pandas Series or a single-column DataFrame"
            )


class BaseRowWiseTransformer(BaseSingleColumnTransformer):
    """
    Abstract base class for single-column transformers that process
    individual values row-wise.

    Subclasses must implement the `process` method, which is applied
    to each element of the column.
    """

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply `process` to each element in the column.

        Args:
            X: Input Series or single-column DataFrame.

        Returns:
            DataFrame with a single transformed column.

        Raises:
            RuntimeError: If transformer is not fitted.
        """
        if not hasattr(self, "feature_names_in_") or not hasattr(
            self, "feature_names_out_"
        ):
            raise RuntimeError(
                "Transformer must be fitted before calling 'transform'"
            )

        series = self._to_series(X)
        transformed = series.apply(self.process)
        return pd.DataFrame(
            {self.feature_names_out_[0]: transformed}, index=series.index
        )

    @abstractmethod
    def process(self, value: Any) -> Any:
        """
        Transform a single value.

        Must be implemented in subclasses.

        Args:
            value: Input value.

        Returns:
            Transformed value.
        """
        pass


class BaseCategoricalTextExtractor(BaseRowWiseTransformer):
    """
    Base transformer for extracting a single categorical feature from a text column
    using regex patterns.

    Notes:
        - Subclasses should define the `PATTERNS` dictionary mapping labels to compiled regex.
        - If no pattern matches, `default_label` is returned.
    """

    PATTERNS: dict[Any, Any] = {}

    def __init__(
        self,
        output_column: Optional[str] = None,
        default_label: Optional[Any] = None,
    ) -> None:
        """
        Initialize the transformer.

        Args:
            output_column: Name of the output column. If None, the input column name is used.
            default_label: Label for unmatched titles.
        """
        super().__init__(output_column=output_column)
        self.default_label = default_label

    def process(self, text: Any) -> Optional[Any]:
        """
        Extract a categorical label from a text string using pre-defined patterns.

        Args:
            text: Input value to extract the label from. Can be string or any type.

        Returns:
            Matched label, `default_label` if no match,
            or None if input is invalid (non-strings).
        """
        if not isinstance(text, str):
            return None

        text = text.strip()

        for label, pattern in self.PATTERNS.items():
            if pattern.search(text):
                return label

        return self.default_label


class BaseTextListNormalizer(BaseRowWiseTransformer):
    """
    Base transformer for normalizing text columns with multiple categories.

    Splits strings by a delimiter, applies a mapping, converts normalized
    text to lowercase, replaces spaces with underscores, and returns
    a list of unique values.

    Notes:
        - Subclasses should define `NORMALIZE_MAP` for mapping specific values.
        - The delimiter can be customized via `DELIMITER`.
        - Returns an empty list for non-string inputs.
    """

    NORMALIZE_MAP: dict[str, Any] = {}
    DELIMITER: str = ","

    def process(self, text: Any) -> list[Any]:
        """
        Normalize a single row of text.

        Args:
            text: Input value containing one or more categories.

        Returns:
            List of normalized, unique categories.
            Non-string inputs return empty list.
        """
        if not isinstance(text, str):
            return []

        items = [t.strip() for t in text.split(self.DELIMITER)]
        seen, result = set(), []

        for item in items:
            normalized = self.NORMALIZE_MAP.get(item, item)
            normalized = normalized.replace(" ", "_")
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)

        return result
