"""Unified training script for regression and classification models.

This script trains models on preprocessed Head Hunter dataset and saves results.

Usage:
    python train.py data/processed -o resources --task regression
    python train.py data/processed -o resources --task classification

Regression models:
    - ridge: Ridge Regression (L2)
    - random_forest: Random Forest
    - gradient_boosting: Gradient Boosting

Classification models:
    - logistic: Logistic Regression
    - random_forest: Random Forest
    - gradient_boosting: Gradient Boosting
"""

import sys
import warnings
from pathlib import Path

from config.cli import create_train_parser
from config.tuning import CLASSIFICATION_PARAM_GRIDS, REGRESSION_PARAM_GRIDS
from training.classifiers import (
    BaseClassifier,
    GradientBoostingClassifier,
    LogisticClassifier,
    RandomForestClassifier,
)
from training.regressors import (
    BaseRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RidgeRegressor,
)
from utils.loader import DataLoader
from utils.logging import setup_logger
from utils.training import run_classification_training, run_regression_training
from utils.visualization import plot_class_balance

# --- Model registries ---
REGRESSION_MODELS: dict[str, type[BaseRegressor]] = {
    "ridge": RidgeRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

CLASSIFICATION_MODELS: dict[str, type[BaseClassifier]] = {
    "logistic": LogisticClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def create_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create output directories for models, predictions, metrics, and plots.

    Args:
        output_dir: Base output directory.

    Returns:
        Dict with 'models', 'predictions', 'metrics', 'plots' paths.
    """
    dirs = {
        "models": output_dir / "models",
        "predictions": output_dir / "predictions",
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    """Main training entry point for regression and classification models.

    Trains regression or classification models based on --task argument.

    Exit codes:
        0: Success
        -1: Unexpected error
        -2: Validation error (file not found, wrong format)
    """
    warnings.filterwarnings("ignore")

    parser = create_train_parser()
    args = parser.parse_args()

    logger = setup_logger(logger_name="Train", log_level=args.log_level)

    try:
        # --- Load data ---
        logger.info(f"Task: '{args.task}'.")
        logger.info(f"Loading data from '{args.data_dir}'...")
        input_dir = Path(args.data_dir) / args.task
        loader = DataLoader(input_dir, prefix=args.prefix, random_seed=args.seed)
        loader.load()
        logger.info(f"Loaded ({loader.n_samples}) samples, ({loader.n_features}) features.")
        if loader.class_labels:
            logger.info(f"Class labels: {loader.class_labels}.")

        # --- Split data ---
        logger.info("Splitting data into train/test...")
        data = loader.split(train_ratio=args.train_ratio, test_ratio=args.test_ratio)
        logger.info(f"Train shape: {data.x_train.shape}.")
        logger.info(f"Test shape: {data.x_test.shape}.")

        # --- Create output directories ---
        output_dir = Path(args.output_dir) / args.task
        dirs = create_output_dirs(output_dir)

        # --- Train models ---
        if args.task == "regression":
            run_regression_training(
                data=data,
                models_registry=REGRESSION_MODELS,
                param_grids=REGRESSION_PARAM_GRIDS,
                dirs=dirs,
                random_seed=args.seed,
                tune_hyperparams=not args.no_tune,
                logger=logger,
            )
        else:
            # --- Plot class balance for classification ---
            if loader.class_labels:
                plot_class_balance(
                    data.y_train,
                    loader.class_labels,
                    title="Training Set Class Distribution",
                    save_path=dirs["plots"] / "class_balance.png",
                )
                logger.info("Saved class balance plot.")

            run_classification_training(
                data=data,
                models_registry=CLASSIFICATION_MODELS,
                param_grids=CLASSIFICATION_PARAM_GRIDS,
                dirs=dirs,
                random_seed=args.seed,
                tune_hyperparams=not args.no_tune,
                class_labels=loader.class_labels,
                logger=logger,
            )

        logger.info("Training completed successfully.")

    except (FileNotFoundError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        logger.error(f"Error: {exc}.")
        sys.exit(-2)
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}.")
        sys.exit(-1)


if __name__ == "__main__":
    main()
