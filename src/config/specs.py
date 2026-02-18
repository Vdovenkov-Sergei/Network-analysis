"""Shared specifications for data transformation and encoding pipelines."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from parsers.base.core import BaseColumnTransformer


@dataclass(frozen=True)
class TransformerSpec:
    """Specification for one transformation step.

    Attributes:
        transformer: Applied to the selected columns.
        input_columns: Ordered list of column names to extract and pass to the transformer.
        output_column: Name of the generated column. If None, derived from input.
        params: Optional keyword arguments for the transformer constructor.
    """

    transformer: type[BaseColumnTransformer]
    input_columns: list[str]
    output_column: Optional[str] = None
    params: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class EncodingSpec:
    """Specification for encoding feature columns to numeric.

    Columns not specified in any category are passed through unchanged.

    Attributes:
        one_hot: Categorical columns for one-hot encoding.
        iqr_masker: Numeric columns for IQR-based outlier masking.
        scale: Numeric columns for MinMax scaling.
        label_encode: Categorical columns for ordinal/label encoding.
        multi_label: Columns with list values for multi-label binarization.
        drop: Columns to exclude from modeling entirely.
    """

    one_hot: list[str]
    iqr_masker: list[str]
    scale: list[str]
    label_encode: list[str]
    multi_label: list[str]
    drop: list[str]


@dataclass(frozen=True)
class TargetSpec:
    """Specification for target column encoding.

    Attributes:
        column: Name of the target column.
        task: Task type - 'regression' or 'classification'.
        apply_iqr_masker: Whether to apply IQR-based outlier masking (regression only).
        return_labels: Whether to return class labels for saving (classification only).
    """

    column: str
    task: Literal["regression", "classification"]
    apply_iqr_masker: bool = False
    return_labels: bool = False
