"""Base classes for transformers.

This module contains abstract base classes that provide the foundation
for all column transformers used in the preprocessing pipeline.

Hierarchy:
    BaseColumnTransformer (__init__, get_feature_names_out)
    ├── BaseSingleColumnTransformer (fit, validate, _to_series)
    │   └── BaseRowWiseTransformerSingle (transform via process)
    │       ├── BaseCategoricalTextExtractor
    │       └── BaseTextListNormalizer
    └── BaseMultiColumnTransformer (fit, validate, _to_dataframe)
        └── BaseRowWiseTransformerMulti (transform via process)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Self


class BaseColumnTransformer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base transformer for column operations.

    Transformers receive data directly and extract column information from it.
    Subclasses must implement `fit`, `validate`, and `transform`.

    Attributes:
        output_column: Name of the output column. If None, derived from input.
        feature_names_in_: Column names from input data (set during fit).
        feature_names_out_: Output column names (set during fit).
    """

    def __init__(self, output_column: Optional[str] = None) -> None:
        """Initialize the transformer.

        Args:
            output_column: Name of the output column. If None, derived from input columns.
        """
        self.output_column = output_column

    @abstractmethod
    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[pd.Series] = None) -> Self:
        """Fit the transformer and record feature names.

        Args:
            X: Input data (one or more columns).
            y: Target values. Ignored.

        Returns:
            Fitted transformer instance.
        """
        pass

    @abstractmethod
    def validate(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Validate input data.

        Args:
            X: Input data to validate.

        Returns:
            Validated input (Series or DataFrame).
        """
        pass

    @abstractmethod
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Transform the input data.

        Args:
            X: Input data with one or more columns.

        Returns:
            Transformed data with one or more columns.
        """
        pass

    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        """Return output feature name(s) for the transformer.

        Args:
            input_features: Ignored, included for scikit-learn compatibility.

        Returns:
            List of output column names.

        Raises:
            RuntimeError: If called before the transformer is fitted.
        """
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Transformer must be fitted before calling 'get_feature_names_out'")

        return self.feature_names_out_  # type: ignore


class BaseSingleColumnTransformer(BaseColumnTransformer):
    """Abstract base transformer for a single input column.

    Extracts column name from input Series or DataFrame during fit.
    Subclasses must implement `transform`.
    """

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[pd.Series] = None) -> Self:
        """Fit the transformer and record feature names from input.

        Args:
            X: Input data (Series or single-column DataFrame).
            y: Target values. Ignored.

        Returns:
            Fitted transformer instance.
        """
        series = self.validate(X)
        self.feature_names_in_ = [series.name if series.name is not None else "feature"]
        self.feature_names_out_ = (
            [self.output_column] if self.output_column else self.feature_names_in_
        )
        return self

    def validate(self, X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Validate input data type and shape.

        Args:
            X: Input data (Series or single-column DataFrame).

        Returns:
            Input column as Series.

        Raises:
            ValueError: If DataFrame has more than one column.
            TypeError: If input is neither Series nor DataFrame.
        """
        return self._to_series(X)

    @staticmethod
    def _to_series(X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Convert input to a pandas Series (single column).

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
            raise TypeError("Input must be a pandas Series or a single-column DataFrame")


class BaseMultiColumnTransformer(BaseColumnTransformer):
    """Abstract base transformer for multiple input columns.

    Extracts column names from input DataFrame during fit and stores them in
    `feature_names_in_`. Column order is preserved from input.

    Important:
        Subclasses must document the expected column order in their docstring.
    """

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[pd.Series] = None) -> Self:
        """Fit the transformer and record feature names from input.

        Args:
            X: Input data (DataFrame with one or more columns).
            y: Target values. Ignored.

        Returns:
            Fitted transformer instance.
        """
        df = self.validate(X)
        self.feature_names_in_ = df.columns.tolist()
        self.feature_names_out_ = (
            [self.output_column] if self.output_column else [",".join(self.feature_names_in_)]
        )
        return self

    def validate(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Validate input data type and shape.

        Args:
            X: Input data (DataFrame with one or more columns).

        Returns:
            Input as DataFrame.

        Raises:
            ValueError: If input has no columns.
            TypeError: If input is neither Series nor DataFrame.
        """
        return self._to_dataframe(X)

    @staticmethod
    def _to_dataframe(X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Convert input to a pandas DataFrame (at least one column).

        Args:
            X: Input data (Series or DataFrame).

        Returns:
            DataFrame with at least one column.

        Raises:
            ValueError: If input is empty or has no columns.
            TypeError: If input is neither a Series nor a DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            if X.shape[1] < 1:
                raise ValueError("Input DataFrame must have at least one column")
            return X.copy()
        elif isinstance(X, pd.Series):
            return pd.DataFrame(X)
        else:
            raise TypeError("Input must be a pandas Series or DataFrame")


class BaseRowWiseTransformerSingle(BaseSingleColumnTransformer):
    """Abstract base for single-column transformers that process values row-wise.

    Subclasses must implement the `process` method, which is applied
    to each element of the column.
    """

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply `process` to each element in the column.

        Args:
            X: Input Series or single-column DataFrame.

        Returns:
            DataFrame with a single transformed column.

        Raises:
            RuntimeError: If transformer is not fitted.
        """
        if not hasattr(self, "feature_names_in_") or not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Transformer must be fitted before calling 'transform'")

        series = self.validate(X)
        transformed = series.apply(self.process)
        return pd.DataFrame({self.feature_names_out_[0]: transformed}, index=series.index)

    @abstractmethod
    def process(self, value: Any) -> Any:
        """Transform a single value.

        Must be implemented in subclasses.

        Args:
            value: Input value.

        Returns:
            Transformed value.
        """
        pass


class BaseRowWiseTransformerMulti(BaseMultiColumnTransformer):
    """Abstract base for multi-column transformers that process rows row-wise.

    Subclasses must implement the `process` method, which is applied
    to each row of the DataFrame.
    """

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply `process` to each row of the DataFrame.

        Args:
            X: Input data with one or more columns.

        Returns:
            DataFrame with a single transformed column.

        Raises:
            RuntimeError: If transformer is not fitted.
        """
        if not hasattr(self, "feature_names_in_") or not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Transformer must be fitted before calling 'transform'")

        df = self.validate(X)
        transformed = df.apply(self.process, axis=1)
        return pd.DataFrame({self.feature_names_out_[0]: transformed}, index=df.index)

    @abstractmethod
    def process(self, row: pd.Series) -> Any:
        """Transform a single row (values from multiple columns).

        Must be implemented in subclasses.

        Args:
            row: One row of the input data.

        Returns:
            Transformed value.
        """
        pass


class BaseCategoricalTextExtractor(BaseRowWiseTransformerSingle):
    """Base transformer for extracting a single categorical feature from a
    text column using regex patterns.

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
        """Initialize the transformer.

        Args:
            output_column: Name of the output column. If None, the input column name is used.
            default_label: Label for unmatched entries.
        """
        super().__init__(output_column=output_column)
        self.default_label = default_label

    def process(self, text: Any) -> Optional[Any]:
        """Extract a categorical label from a text string using pre-defined patterns.

        Args:
            text: Input value to extract the label from. Can be string or any type.

        Returns:
            Matched label, `default_label` if no match, or None if input is invalid (non-strings).
        """
        if not isinstance(text, str):
            return None

        text = text.strip()

        for label, pattern in self.PATTERNS.items():
            if pattern.search(text):
                return label

        return self.default_label


class BaseTextListNormalizer(BaseRowWiseTransformerSingle):
    """Base transformer for normalizing text columns with multiple categories.

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

    def process(self, text: Any) -> tuple[Any, ...]:
        """Normalize a single row of text to a list of categories.

        Args:
            text: Input value containing one or more categories.

        Returns:
            Tuple of normalized, unique categories. Non-string inputs return empty tuple.
        """
        if not isinstance(text, str):
            return ()

        items = [part.strip() for part in text.split(self.DELIMITER)]
        seen, result = set(), []

        for item in items:
            normalized = self.NORMALIZE_MAP.get(item, item)
            normalized = normalized.replace(" ", "_")
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)

        return tuple(result)
