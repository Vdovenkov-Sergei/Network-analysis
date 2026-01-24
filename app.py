"""
Main application for processing Head Hunter dataset.

This script provides a command-line interface for loading, preprocessing,
and saving the Head Hunter job dataset.
"""

import argparse
import sys
from pathlib import Path

from config import COLUMN_TRANSFORMER_CONFIG
from logger import get_logger
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
    logger = get_logger(__name__)

    try:
        parser = create_arg_parser()
        args = parser.parse_args()

        # --- Validate input file ---
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not input_path.suffix.lower() == ".csv":
            raise ValueError(
                f"Input file must be a CSV file, got: {input_path.suffix}"
            )

        logger.info("Loading data from: %s", input_path)
        processed_data = parse_data(
            str(input_path), config=COLUMN_TRANSFORMER_CONFIG
        )

        logger.info("Processed data shape: %s", processed_data.shape)
        logger.info("Processed data columns: %s", len(processed_data.columns))

        # --- Split into X and y ---
        logger.info("Splitting into features (X) and target (y)...")
        X, y = split_x_y(processed_data, config=COLUMN_TRANSFORMER_CONFIG)

        logger.info("Features (X) shape: %s", X.shape)
        logger.info(
            "Target (y) shape: %s",
            y.shape if hasattr(y, "shape") else "Series",
        )

        # --- Save to numpy files ---
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving processed data to: %s", output_dir)
        save_x_y(X=X, y=y, path=output_dir, prefix=args.prefix)

        logger.info("Processing completed successfully!")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Error: %s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.exception("Unexpected error while processing data: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
