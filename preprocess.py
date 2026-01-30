"""Data preprocessing script for Head Hunter dataset.

This script provides a command-line interface for loading raw CSV data,
applying feature transformations, and saving preprocessed data as NumPy arrays.

Usage:
    python preprocess.py data/hh.csv -o data

Output files:
    - X_data.npy: Feature matrix
    - y_data.npy: Target variable (salary in RUB)
    - feature_names.npy: Feature column names
    - target_names.npy: Target column names
"""

import sys
from pathlib import Path

from config.cli import create_preprocess_parser
from config.parser import COLUMN_TRANSFORMER_CONFIG
from utils.logging import setup_logger
from utils.processing import parse_data, save_x_y, split_x_y


def main() -> None:
    """Main entry point for the preprocessing script.

    Loads raw CSV data, applies transformations defined in COLUMN_TRANSFORMER_CONFIG,
    and saves the resulting feature matrix (X) and target vector (y) as NumPy arrays.

    Exit codes:
        0: Success
        -1: Unexpected error
        -2: Validation error (file not found, wrong format)
    """
    try:
        parser = create_preprocess_parser()
        args = parser.parse_args()

        logger = setup_logger(logger_name="Preprocess", log_level=args.log_level)

        # --- Validate input file ---
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: '{input_path}'")
        if not input_path.suffix.lower() == ".csv":
            raise ValueError(f"Input file must be a CSV file, got: '{input_path.suffix}'")

        # --- Load and preprocess data ---
        logger.info(f"Loading and processing data from '{input_path}'...")
        processed_data = parse_data(str(input_path), config=COLUMN_TRANSFORMER_CONFIG)
        logger.info(f"Processed data shape: {processed_data.shape}.")
        logger.info(f"Processed data columns: {len(processed_data.columns)}.")

        # --- Split into X and y ---
        logger.info("Splitting into features (X) and target (y)...")
        X, y = split_x_y(processed_data, config=COLUMN_TRANSFORMER_CONFIG)
        logger.info(f"Features (X) shape: {X.shape}.")
        logger.info(f"Target (y) shape: {y.shape}.")

        # --- Save to numpy files ---
        output_dir = Path(args.output_dir) / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving processed data to: '{output_dir}'.")
        save_x_y(X=X, y=y, path=output_dir, prefix=args.prefix)

        logger.info("Processing completed successfully.")
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Error: {exc}.")
        sys.exit(-2)
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}.")
        sys.exit(-1)


if __name__ == "__main__":
    main()
