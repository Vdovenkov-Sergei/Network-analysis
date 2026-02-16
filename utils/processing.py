"""Data processing utilities.

This module provides functions for transforming raw data, applying engineering steps,
encoding features and target to numeric arrays with proper cleaning between stages.
"""

from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder

from config.specs import EncodingSpec, TargetSpec, TransformerSpec
from parsers.transformers.multilabel import MultiLabelTransformer
from parsers.transformers.numerical import IQRMasker
from utils.cleaning import clean_dataframe, clean_raw_dataframe


def _build_column_transformer(config: list[TransformerSpec]) -> ColumnTransformer:
    """Build ColumnTransformer from list of TransformerSpec.

    Args:
        config: List of TransformerSpec with transformer class and column info.

    Returns:
        ColumnTransformer applying each spec to its input columns.
    """
    transformers_list = [
        (
            f"transform_{i}",
            spec.transformer(
                output_column=spec.output_column,
                **(spec.params or {}),
            ),
            spec.input_columns,
        )
        for i, spec in enumerate(config)
    ]
    return ColumnTransformer(transformers=transformers_list, verbose_feature_names_out=False)


def _build_encoding_pipeline(column: str, encoding_config: EncodingSpec) -> Pipeline:
    """Build a single-column encoding Pipeline from EncodingSpec.

    Args:
        column: Column name to encode.
        encoding_config: Encoding specification with column categories.

    Returns:
        Pipeline with encoding steps for this column.

    Raises:
        ValueError: If column is not in any encoding category.
    """
    steps = []
    if column in encoding_config.one_hot:
        steps.append(("onehot", OneHotEncoder(sparse_output=False)))
    if column in encoding_config.label_encode:
        steps.append(("ordinal", OrdinalEncoder()))
    if column in encoding_config.multi_label:
        steps.append(("multilabel", MultiLabelTransformer(output_column=column)))
    if column in encoding_config.iqr_masker:
        steps.append(("iqr", IQRMasker(output_column=column)))
    if column in encoding_config.scale:
        steps.append(("scale", MinMaxScaler()))
    if not steps:
        raise ValueError(f"Column '{column}' is not in any encoding category.")
    return Pipeline(steps, memory=None)


def _build_encoding_column_transformer(encoding_config: EncodingSpec) -> ColumnTransformer:
    """Build ColumnTransformer for all columns defined in EncodingSpec.

    Args:
        encoding_config: EncodingSpec with column categories.

    Returns:
        ColumnTransformer with one pipeline per column.
    """
    encode_cols = set(
        list(encoding_config.one_hot)
        + list(encoding_config.label_encode)
        + list(encoding_config.multi_label)
        + list(encoding_config.iqr_masker)
        + list(encoding_config.scale)
    )

    transformers_list = [
        (f"encoding_{i}", _build_encoding_pipeline(column, encoding_config), [column])
        for i, column in enumerate(sorted(encode_cols))
    ]
    return ColumnTransformer(
        transformers=transformers_list,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def apply_transforming(df: pd.DataFrame, transform_config: list[TransformerSpec]) -> pd.DataFrame:
    """Apply raw-to-structured transformation via ColumnTransformer.

    Args:
        df: Input DataFrame with raw features.
        transform_config: List of TransformerSpec defining transformations.

    Returns:
        DataFrame with structured columns only.

    Raises:
        KeyError: If required input columns are missing.
    """
    ct = _build_column_transformer(transform_config)
    required = [col for _, _, cols in ct.transformers for col in cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")

    transformed = ct.fit_transform(df)
    feature_names = [str(name) for name in ct.get_feature_names_out()]
    return pd.DataFrame(transformed, columns=feature_names, index=df.index)


def apply_engineering(df: pd.DataFrame, engineering_config: list[TransformerSpec]) -> pd.DataFrame:
    """Add derived columns via ColumnTransformer.

    Args:
        df: DataFrame with structured columns.
        engineering_config: List of TransformerSpec for derived features.

    Returns:
        DataFrame with additional engineered columns.

    Raises:
        KeyError: If required input columns are missing.
    """
    ct = _build_column_transformer(engineering_config)
    required = [col for _, _, cols in ct.transformers for col in cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")

    out = df.copy()
    out_arr = ct.fit_transform(df)
    feature_names = [str(name) for name in ct.get_feature_names_out()]

    for j, col in enumerate(feature_names):
        out[col] = out_arr[:, j] if out_arr.ndim > 1 else out_arr
    return out


def apply_encoding_features(df: pd.DataFrame, encoding_config: EncodingSpec) -> pd.DataFrame:
    """Encode feature columns to numeric.

    Columns not specified in `encoding_config` are passed through unchanged.
    Columns in `encoding_config.drop` are excluded from the result.

    Args:
        df: DataFrame with cleaned features.
        encoding_config: Spec for feature encoding.

    Returns:
        DataFrame with encoded features.

    Raises:
        KeyError: If columns specified in encoding_config are missing.
        ValueError: If no feature columns remain after excluding drop columns.
    """
    ct = _build_encoding_column_transformer(encoding_config)
    required = [col for _, _, cols in ct.transformers for col in cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")

    missing_drop = [col for col in encoding_config.drop if col not in df.columns]
    if missing_drop:
        raise ValueError(f"Drop columns {missing_drop} not found in DataFrame")

    encoded = ct.fit_transform(df)
    feature_names = [str(name) for name in ct.get_feature_names_out()]
    df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
    return df.drop(columns=encoding_config.drop)


def apply_encoding_target(
    df: pd.DataFrame, target_spec: TargetSpec
) -> tuple[pd.DataFrame, Optional[list[str]]]:
    """Encode target column for regression or classification.

    Args:
        df: DataFrame containing the target column.
        target_spec: TargetSpec with task type and encoding options.

    Returns:
        Tuple of (DataFrame with encoded target, class labels or None).
        For regression: (DataFrame with IQR applied if specified, None).
        For classification: (DataFrame with label encoding, sorted labels).

    Raises:
        ValueError: If target column is missing or task type is invalid.
    """
    if target_spec.column not in df.columns:
        raise ValueError(f"Target column '{target_spec.column}' not found in DataFrame")

    df_out = df.copy()
    y = df[target_spec.column]

    if target_spec.task == "regression":
        # Apply IQR masker if specified
        if target_spec.apply_iqr_masker:
            masker = IQRMasker(output_column=target_spec.column)
            y_masked = masker.fit_transform(y)
            df_out[target_spec.column] = y_masked.iloc[:, 0]
        return df_out, None

    elif target_spec.task == "classification":
        # Label encoding for classification
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(y)
        df_out[target_spec.column] = encoded

        if target_spec.return_labels:
            return df_out, list(label_encoder.classes_)
        return df_out, None

    else:
        raise ValueError(f"Invalid task type: {target_spec.task}")


def split_x_y(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features X and target y.

    Args:
        df: Transformed and cleaned DataFrame.
        target_column: Column name to use as target.

    Returns:
        Tuple of (feature DataFrame, target Series).

    Raises:
        ValueError: If target column is missing from the DataFrame.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def parse_data(
    filepath: str,
    transform_config: list[TransformerSpec],
    engineering_config: list[TransformerSpec],
    encoding_config: EncodingSpec,
    target_spec: TargetSpec,
) -> tuple[pd.DataFrame, Optional[list[str]]]:
    """Load, preprocess, and encode a dataset from CSV.

    Pipeline:
        1. Load raw CSV.
        2. Clean raw data.
        3. Apply transforming (raw -> structured columns).
        4. Apply engineering (add derived columns).
        5. Clean transformed features.
        6. Encode features.
        7. Clean after feature encoding.
        8. Encode target.
        9. Clean after target encoding.
        10. Convert to float64.

    Args:
        filepath: Path to CSV file.
        transform_config: List of TransformerSpec for column transformations.
        engineering_config: List of TransformerSpec for derived features.
        encoding_config: EncodingSpec for numeric encoding of features.
        target_spec: TargetSpec for target column encoding.

    Returns:
        Tuple of (processed DataFrame, class labels list or None).
    """
    # 1. Load raw CSV
    df = pd.read_csv(filepath, encoding="utf-8")

    # 2. Clean raw data
    df = clean_raw_dataframe(df)

    # 3. Apply transforming (raw -> structured columns)
    df = apply_transforming(df, transform_config)

    # 4. Apply engineering (add derived columns)
    df = apply_engineering(df, engineering_config)

    # 5. Clean transformed features
    df = clean_dataframe(df)

    # 6. Encode features
    df = apply_encoding_features(df, encoding_config)

    # 7. Clean after feature encoding
    df = clean_dataframe(df)

    # 8. Encode target
    df, labels = apply_encoding_target(df, target_spec)

    # 9. Clean after target encoding
    df = clean_dataframe(df)

    # 10. Convert to float64
    df = df.astype(float)

    return df, labels
