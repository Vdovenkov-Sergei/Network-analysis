"""Command-line interface argument parsers.

This module provides argument parsers for the preprocessing
and training scripts.
"""

import argparse


def create_preprocess_parser() -> argparse.ArgumentParser:
    """Create argument parser for the preprocessing script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess Head Hunter CSV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Root output directory; artifacts saved under {output_dir}/processed/{task} (default: data)",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Preprocessing task type (default: regression)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for output files (default: empty)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser


def create_train_parser() -> argparse.ArgumentParser:
    """Create argument parser for the training script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Train models on Head Hunter dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "data_dir",
        type=str,
        help="Root directory with preprocessed data",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Training task type (default: regression)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for input files (default: empty)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data",
        help="Root output directory; artifacts saved under {output_dir}/training/{task} (default: data)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio (default: 0.8)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test data ratio (default: 0.2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter tuning",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser
