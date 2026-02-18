"""Data cleaning utilities.

This module contains functions for cleaning raw and processed DataFrames.
"""

import pandas as pd


def clean_raw_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_empty_rows: bool = True,
    strip_strings: bool = True,
) -> pd.DataFrame:
    """
    Clean raw (preprocessed) data before feature engineering.

    This function performs lightweight, schema-agnostic cleaning
    that should be applied BEFORE any sklearn transformers.

    Cleaning steps:
        - Optionally drop fully duplicated rows.
        - Optionally drop rows that are completely empty.
        - Optionally strip whitespace from string columns.

    Args:
        df: Raw input.
        drop_duplicates: Whether to drop fully duplicated rows.
        drop_empty_rows: Whether to drop rows where all values are NaN.
        strip_strings: Whether to strip leading/trailing whitespace
                       from object (string) columns.

    Returns:
        Cleaned raw data.
    """
    df_clean = df.copy()

    # --- Drop unnamed columns ---
    df_clean = df_clean.loc[:, ~df_clean.columns.str.startswith("Unnamed:")]

    # --- Drop duplicated rows ---
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()

    # --- Drop rows where all values are NaN ---
    if drop_empty_rows:
        df_clean = df_clean.dropna(how="all")

    # --- Strip leading/trailing whitespace from string columns ---
    if strip_strings:
        for col in df_clean.select_dtypes(include="object"):
            df_clean[col] = df_clean[col].str.strip()

    return df_clean.reset_index(drop=True)


def clean_dataframe(
    df: pd.DataFrame,
    remove_all_na_columns: bool = True,
    remove_constant_columns: bool = True,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Clean transformed (feature-engineered) data.

    Cleaning steps:
        - Drop rows containing any NaN values.
        - Optionally drop duplicate rows.
        - Optionally drop columns that are entirely NaN.
        - Optionally drop columns with constant values.

    Args:
        df: Transformed data with engineered features.
        remove_all_na_columns: Whether to drop columns fully filled with NaN.
        remove_constant_columns: Whether to drop columns with a single unique value.
        drop_duplicates: Whether to drop duplicated rows.

    Returns:
        Cleaned data.
    """
    df_clean = df.copy()

    # --- Remove rows with NaN/None ---
    df_clean = df_clean.dropna(axis=0)

    # --- Remove duplicate rows ---
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()

    # --- Remove all-NaN columns ---
    if remove_all_na_columns:
        df_clean = df_clean.dropna(axis=1, how="all")

    # --- Remove constant columns ---
    if remove_constant_columns:
        constant_cols = [
            col for col in df_clean.columns if df_clean[col].nunique(dropna=False) <= 1
        ]
        df_clean = df_clean.drop(columns=constant_cols)

    return df_clean.reset_index(drop=True)
