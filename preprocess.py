"""Data preprocessing script for Head Hunter dataset.

This script provides a command-line interface for loading raw CSV data,
applying feature transformations, and saving preprocessed data as NumPy arrays.

Usage:
    python preprocess.py data/hh.csv -o data --task regression
    python preprocess.py data/hh.csv -o data --task classification

Output files:
    - X_data.npy: Feature matrix
    - y_data.npy: Target variable
    - feature_columns.npy: Feature column names
    - target_column.npy: Target column name
    - class_labels.npy: Class labels (classification only)
"""

import sys
from pathlib import Path

from config.cli import create_preprocess_parser
from config.encoding import (
    CLASSIFICATION_ENCODING_CONFIG,
    CLASSIFICATION_TARGET,
    REGRESSION_ENCODING_CONFIG,
    REGRESSION_TARGET,
)
from config.engineering import CLASSIFICATION_ENGINEERING_CONFIG, REGRESSION_ENGINEERING_CONFIG
from config.transform import COLUMN_TRANSFORM_CONFIG
from utils.logging import setup_logger
from utils.processing import parse_data, split_x_y
from utils.saving import save_dataset, save_labels


def main() -> None:
    """Main entry point for the preprocessing script.

    Loads raw CSV data, applies transformations defined in configs,
    and saves the resulting feature matrix (X) and target vector (y) as NumPy arrays.

    Exit codes:
        0: Success
        -1: Unexpected error
        -2: Validation error (file not found, wrong format)
    """
    parser = create_preprocess_parser()
    args = parser.parse_args()

    logger = setup_logger(logger_name="Preprocess", log_level=args.log_level)

    try:
        # --- Validate input file ---
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: '{input_path}'")
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Input file must be a CSV file, got: '{input_path.suffix}'")

        # --- Select configs based on task ---
        if args.task == "regression":
            engineering_config = REGRESSION_ENGINEERING_CONFIG
            encoding_config = REGRESSION_ENCODING_CONFIG
            target_spec = REGRESSION_TARGET
        elif args.task == "classification":
            engineering_config = CLASSIFICATION_ENGINEERING_CONFIG
            encoding_config = CLASSIFICATION_ENCODING_CONFIG
            target_spec = CLASSIFICATION_TARGET
        else:
            raise ValueError(f"Invalid task: '{args.task}'")

        logger.info(f"Task: '{args.task}'.")
        logger.info(f"Loading and processing data from '{input_path}'...")

        # --- Parse and encode data ---
        processed_data, labels = parse_data(
            str(input_path),
            transform_config=COLUMN_TRANSFORM_CONFIG,
            engineering_config=engineering_config,
            encoding_config=encoding_config,
            target_spec=target_spec,
        )
        logger.info(f"Processed data shape: {processed_data.shape}.")
        logger.info(f"Processed data columns: {len(processed_data.columns)}.")
        logger.info(f"Processed data 'NaN' count: {processed_data.isna().sum().sum()}.")
        logger.info(f"Labels shape: {len(labels) if labels is not None else 'None'}.")
        logger.info(f"Labels: {labels if labels is not None else 'None'}.")

        # --- Split into X and y ---
        logger.info("Splitting into features ('X') and target ('y')...")
        X, y = split_x_y(processed_data, target_column=target_spec.column)
        logger.info(f"Features ('X') shape: {X.shape}.")
        logger.info(f"Target ('y') shape: {y.shape}.")

        # --- Save to numpy files ---
        output_dir = Path(args.output_dir) / "processed" / args.task
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving processed data to: '{output_dir}'.")
        save_dataset(X=X, y=y, path=output_dir, prefix=args.prefix)
        if labels is not None:
            save_labels(labels=labels, path=output_dir, prefix=args.prefix)
        logger.info("Processing completed successfully.")

    except (FileNotFoundError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        logger.error(f"Error: {exc}.")
        sys.exit(-2)
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}.")
        sys.exit(-1)


if __name__ == "__main__":
    main()
