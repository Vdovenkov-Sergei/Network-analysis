"""Command-line interface argument parsers.

This module provides argument parsers for the preprocessing
and training scripts.
"""

import argparse


def create_preprocess_parser() -> argparse.ArgumentParser:
    """Create argument parser for the preprocessing script.

    Returns:
        Configured ArgumentParser instance for preprocess.py.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess Head Hunter CSV dataset and save as NumPy arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python preprocess.py data/hh.csv -o data
    python preprocess.py data/hh.csv --prefix processed_
        """,
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

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser


def create_train_parser(model_choices: list[str]) -> argparse.ArgumentParser:
    """Create argument parser for the training script.

    Args:
        model_choices: List of available model names.

    Returns:
        Configured ArgumentParser instance for train.py.
    """
    parser = argparse.ArgumentParser(
        description="Train regression models on Head Hunter dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py data/processed -o resources
    python train.py data/processed --models ridge gradient_boosting
    python train.py data/processed --models all --no-tune
        """,
    )

    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing preprocessed data (X_data.npy, y_data.npy, etc.)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for output files (default: empty)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="resources",
        help="Output directory for results and predictions (default: resources)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=model_choices + ["all"],
        help="Models to train (default: all)",
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter tuning (use default params)",
    )

    return parser
