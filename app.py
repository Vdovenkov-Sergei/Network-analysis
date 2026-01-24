"""
Main application for processing Head Hunter dataset.

This script provides a command-line interface for loading, preprocessing,
and saving the Head Hunter job dataset.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import COLUMN_TRANSFORMER_CONFIG
from utils import parse_data, save_x_y, split_x_y


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Process Head Hunter dataset and save preprocessed data."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file containing Head Hunter data",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data (default: data)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for output files (default: empty)",
    )
    return parser


def main() -> None:
    """Main entry point for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        parser = create_arg_parser()
        args = parser.parse_args()

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
        output_dir = Path(args.output_dir)
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
