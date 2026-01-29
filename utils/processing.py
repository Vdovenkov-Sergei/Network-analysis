"""Data processing utilities.

This module provides functions for creating a ColumnTransformer from a configuration dictionary,
preprocessing a DataFrame using the ColumnTransformer, splitting a DataFrame into X and y,
and saving the processed data to .npy files.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from utils.cleaning import clean_dataframe, clean_raw_dataframe
from utils.pipeline_factory import PipelineFactory


def create_column_transformer(
    config: dict[str, list[dict[str, Any]]],
    verbose_feature_names_out: bool = False,
) -> ColumnTransformer:
    """
    Create ColumnTransformer from configuration dictionary.

    Args:
        config: Configuration dictionary with column specifications.
               Each key is an input column name, value is a list of dicts with:
               - 'type': Pipeline type ('numerical', 'categorical', 'multi_label', 'simple')
               - 'transformer': Transformer class
               - 'output_column': Name for the output column
               - 'params': Optional dict of pipeline parameters
               - 'transformer_kwargs': Optional dict of transformer arguments
        verbose_feature_names_out: Whether to include transformer names in feature names

    Returns:
        Configured ColumnTransformer.

    Raises:
        ValueError: If pipeline type is not one of supported types.
    """
    transformers = []

    for input_col, handlers in config.items():
        for handler_cfg in handlers:
            pipeline_type = handler_cfg.get("type", "simple")
            transformer_cls = handler_cfg["transformer"]
            column_name = handler_cfg["output_column"]
            transformer_kwargs = handler_cfg.get("transformer_kwargs", {})
            params_kwargs = handler_cfg.get("params", {})

            if pipeline_type == "numerical":
                pipeline = PipelineFactory.make_numerical_pipeline(
                    extractor_cls=transformer_cls,
                    column_name=column_name,
                    **params_kwargs,
                    **transformer_kwargs,
                )
            elif pipeline_type == "categorical":
                pipeline = PipelineFactory.make_categorical_pipeline(
                    extractor_cls=transformer_cls,
                    column_name=column_name,
                    **params_kwargs,
                    **transformer_kwargs,
                )
            elif pipeline_type == "multi_label":
                pipeline = PipelineFactory.make_multi_label_pipeline(
                    normalizer_cls=transformer_cls,
                    column_name=column_name,
                    **params_kwargs,
                    **transformer_kwargs,
                )
            elif pipeline_type == "simple":
                pipeline = PipelineFactory.make_simple_pipeline(
                    transformer_cls=transformer_cls,
                    column_name=column_name,
                    **params_kwargs,
                    **transformer_kwargs,
                )
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            transformers.append((f"{input_col}_{column_name}_pipe", pipeline, [input_col]))

    return ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=verbose_feature_names_out,
    )


def preprocess_dataframe(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Fit and transform a DataFrame using a ColumnTransformer, returning transformed output.

    This function automatically fits the preprocessor on the input DataFrame
    and returns the transformed result as DataFrame.

    Args:
        df: Input DataFrame with raw features.
        preprocessor: ColumnTransformer with pipelines for preprocessing.

    Returns:
        Transformed DataFrame with processed features.

    Raises:
        KeyError: If required input columns are missing from `df`.
    """
    # --- Verify that all required columns exist ---
    required_columns = [col for _, _, cols in preprocessor.transformers for col in cols]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Input DataFrame is missing required columns: {missing_columns}")

    # --- Fit and transform ---
    transformed_array = preprocessor.fit_transform(df)

    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed_array, columns=feature_names, index=df.index)


def split_x_y(
    df: pd.DataFrame, config: dict[str, list[dict[str, Any]]]
) -> tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    """
    Split a fully preprocessed DataFrame into X and y.

    The split is based on the `role` field in the transformer config:
    handlers with role="y" define target columns, all others are treated
    as features.

    Args:
        df: Transformed and cleaned DataFrame.
        config: Column transformer configuration with `role` and `output_column`.

    Returns:
        X: Feature DataFrame.
        y: Target variable (Series for single target, DataFrame for multi-target).

    Raises:
        ValueError: If no target columns are defined or expected targets
                    are missing from the DataFrame.
    """

    # --- Collect target column names from config ---
    y_columns = [
        handler["output_column"]
        for handlers in config.values()
        for handler in handlers
        if handler.get("role") == "y"
    ]

    if not y_columns:
        raise ValueError("No target columns defined (role='y').")

    # --- Validate presence of target columns ---
    missing = set(y_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Target columns missing in DataFrame: {sorted(missing)}")

    # --- Split ---
    X = df.drop(columns=y_columns)
    y = df[y_columns]

    return (X, y) if y.shape[1] > 1 else (X, y.iloc[:, 0])


def parse_data(filepath: str, config: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """
    Load, preprocess, and clean a dataset from CSV.

    Pipeline:
        1. Load raw CSV.
        2. Drop unnamed columns (e.g., index artifacts).
        3. Clean raw data (drop duplicates, empty rows).
        4. Apply feature preprocessing via ColumnTransformer.
        5. Clean transformed features (drop NaNs, unknowns, constant columns, duplicates).

    Args:
        filepath: Path to CSV file.
        config: Configuration dictionary describing feature pipelines.

    Returns:
        Fully cleaned, transformed DataFrame ready for modeling.
    """
    # --- Load dataset ---
    df = pd.read_csv(filepath, encoding="utf-8")

    # --- Drop unnamed columns ---
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    # --- Pre-clean: raw data ---
    df = clean_raw_dataframe(df)

    # --- Create ColumnTransformer with pipelines ---
    preprocessor = create_column_transformer(config, verbose_feature_names_out=False)

    # --- Preprocess DataFrame ---
    transformed_df = preprocess_dataframe(df, preprocessor)

    # --- Post-clean: features ---
    final_df = clean_dataframe(transformed_df)

    return final_df


def save_x_y(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    path: Union[str, Path],
    prefix: str = "",
) -> None:
    """
    Save feature matrix X and target y to .npy files.

    Files created:
        {prefix}X_data.npy
        {prefix}y_data.npy
        {prefix}feature_names.npy
        {prefix}target_names.npy

    Args:
        X: Feature DataFrame.
        y: Target variable (Series or DataFrame).
        path: Directory to save files.
        prefix: Optional filename prefix.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / f"{prefix}X_data.npy", X.to_numpy())
    np.save(path / f"{prefix}feature_names.npy", X.columns.to_numpy())

    np.save(path / f"{prefix}y_data.npy", y.to_numpy())
    if isinstance(y, pd.Series):
        np.save(path / f"{prefix}target_names.npy", np.array([y.name]))
    else:
        np.save(path / f"{prefix}target_names.npy", y.columns.to_numpy())
