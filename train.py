"""Main training script for regression models.

This script trains multiple regression models on the preprocessed
Head Hunter dataset and saves the best performing model.

Usage:
    python train.py data/processed -o resources --models all

Models:
    - ridge: Ridge Regression (L2)
    - random_forest: Random Forest
    - gradient_boosting: Gradient Boosting
"""

import sys
import warnings
from pathlib import Path

from config.cli import create_train_parser
from training.regressors.base import BaseRegressor
from training.regressors.gradient_boosting import GradientBoostingRegressor
from training.regressors.linear import LinearRegressor
from training.regressors.random_forest import RandomForestRegressor
from utils.loader import DataLoader
from utils.logging import setup_logger
from utils.metrics import RegressionMetrics, save_metrics, save_predictions
from utils.training import train_model
from utils.visualization import plot_predictions_vs_actual

# --- Available models registry ---
MODELS: dict[str, type[BaseRegressor]] = {
    "ridge": LinearRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def main() -> None:
    """
    Main training entry point.

    This script trains multiple regression models on the preprocessed
    Head Hunter dataset and saves the best performing model.

    Error codes:
        0: Success
        -1: Unexpected error
        -2: Validation error (file not found, wrong format)
    """
    warnings.filterwarnings("ignore")

    parser = create_train_parser(list(MODELS.keys()))
    args = parser.parse_args()

    logger = setup_logger(logger_name="Train", log_level=args.log_level)
    try:
        # --- Load data ---
        logger.info(f"Loading data from '{args.data_dir}'...")
        data_loader = DataLoader(args.data_dir, prefix=args.prefix, random_seed=args.seed)
        data_loader.load()
        logger.info(
            f"Loaded {data_loader.n_samples} samples with {data_loader.n_features} features."
        )

        # --- Split data ---
        logger.info("Splitting data into train/test...")
        data = data_loader.split(train_ratio=args.train_ratio, test_ratio=args.test_ratio)
        logger.info(f"Train shape: {data.X_train.shape}")
        logger.info(f"Test shape: {data.X_test.shape}")

        # --- Determine which models to train ---
        models_to_train = list(MODELS.keys()) if "all" in args.models else args.models

        # --- Create output directories ---
        output_dir = Path(args.output_dir)
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- Train models ---
        results: dict[str, tuple[BaseRegressor, dict[str, RegressionMetrics]]] = {}
        for model_name in models_to_train:
            logger.info(f"Training '{model_name}'...")

            model_cls = MODELS[model_name]
            model, metrics = train_model(
                model_cls=model_cls,
                X_train=data.X_train,
                y_train=data.y_train,
                X_test=data.X_test,
                y_test=data.y_test,
                random_seed=args.seed,
                tune_hyperparams=not args.no_tune,
            )
            results[model_name] = (model, metrics)

            # --- Log metrics ---
            logger.info(f"Model '{model_name}' results:")
            logger.info(f"---> Train 'R^2': {metrics['train'].r2:.4f}")
            logger.info(f"---> Test 'R^2': {metrics['test'].r2:.4f}")

            # --- Save predictions, metrics, model and plot ---
            y_pred_test = model.predict(data.X_test)
            save_predictions(y_pred_test, predictions_dir, model_name)
            save_metrics(metrics["test"], metrics_dir, model_name)
            model.save_model(models_dir / f"{model_name}.pkl")
            plot_predictions_vs_actual(
                y_true=data.y_test,
                y_pred=y_pred_test,
                title=f"{model_name}: Predictions vs Actual",
                save_path=plots_dir / f"{model_name}_predictions.png",
            )

        logger.info("Training completed successfully!")

    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Error: {exc}.")
        sys.exit(-2)
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}.")
        sys.exit(-1)


if __name__ == "__main__":
    main()
