"""Training utilities for regression and classification models."""

import logging
from pathlib import Path
from typing import Optional

from training.classifiers.base import BaseClassifier
from training.metrics import ClassificationMetrics, RegressionMetrics
from training.regressors.base import BaseRegressor
from utils.loader import DataSplit
from utils.saving import save_metrics, save_predictions
from utils.visualization import plot_classification_report, plot_predictions_vs_actual


def train_regressor(
    model_cls: type[BaseRegressor],
    data: DataSplit,
    param_grid: dict,
    random_seed: int = 42,
    tune_hyperparams: bool = True,
) -> tuple[BaseRegressor, dict[str, RegressionMetrics]]:
    """Train a regression model and evaluate on train/test sets.

    Args:
        model_cls: Regressor class to instantiate.
        data: DataSplit with train and test data.
        param_grid: Hyperparameter grid for tuning.
        random_seed: Random seed for reproducibility.
        tune_hyperparams: Whether to perform hyperparameter tuning.

    Returns:
        Tuple of (trained model, dict with 'train' and 'test' RegressionMetrics).
    """
    model = model_cls(
        param_grid=param_grid,
        scoring="r2",
        random_seed=random_seed,
        tune_hyperparams=tune_hyperparams and bool(param_grid),
    )  # type: ignore
    model.fit(data.x_train, data.y_train)

    metrics = {
        "train": model.evaluate(data.x_train, data.y_train),
        "test": model.evaluate(data.x_test, data.y_test),
    }

    return model, metrics


def train_classifier(
    model_cls: type[BaseClassifier],
    data: DataSplit,
    param_grid: dict,
    random_seed: int = 42,
    tune_hyperparams: bool = True,
    class_labels: Optional[list[str]] = None,
) -> tuple[BaseClassifier, dict[str, ClassificationMetrics]]:
    """Train a classification model and evaluate on train/test sets.

    Args:
        model_cls: Classifier class to instantiate.
        data: DataSplit with train and test data.
        param_grid: Hyperparameter grid for tuning.
        random_seed: Random seed for reproducibility.
        tune_hyperparams: Whether to perform hyperparameter tuning.
        class_labels: Class labels for classification report.

    Returns:
        Tuple of (trained model, dict with 'train' and 'test' ClassificationMetrics).
    """
    model = model_cls(
        param_grid=param_grid,
        scoring="f1_macro",
        random_seed=random_seed,
        tune_hyperparams=tune_hyperparams and bool(param_grid),
    )  # type: ignore
    model.fit(data.x_train, data.y_train)

    metrics = {
        "train": model.evaluate(data.x_train, data.y_train, class_labels=class_labels),
        "test": model.evaluate(data.x_test, data.y_test, class_labels=class_labels),
    }

    return model, metrics


def run_regression_training(
    data: DataSplit,
    models_registry: dict[str, type[BaseRegressor]],
    param_grids: dict[str, dict],
    dirs: dict[str, Path],
    random_seed: int,
    tune_hyperparams: bool,
    logger: logging.Logger,
) -> None:
    """Train and evaluate multiple regression models.

    Args:
        data: DataSplit with train and test data.
        models_registry: Dict mapping model names to regressor classes.
        param_grids: Dict mapping model names to hyperparameter grids.
        dirs: Output directories dict with 'models', 'predictions', 'metrics', 'plots'.
        random_seed: Random seed for reproducibility.
        tune_hyperparams: Whether to tune hyperparameters.
        logger: Logger instance.
    """
    best_r2 = -float("inf")
    best_model_name = ""

    for model_name in models_registry.keys():
        logger.info(f"Training '{model_name}'...")
        model_cls = models_registry[model_name]
        param_grid = param_grids.get(model_name, {}) if tune_hyperparams else {}

        # Train model
        model, metrics = train_regressor(
            model_cls=model_cls,
            data=data,
            param_grid=param_grid,
            random_seed=random_seed,
            tune_hyperparams=tune_hyperparams,
        )

        # Log metrics
        train_r2 = metrics["train"].r2
        test_r2 = metrics["test"].r2
        logger.info(f"Model '{model_name}' results:")
        logger.info(f"---> Train 'R^2': {train_r2:.4f}")
        logger.info(f"---> Test 'R^2': {test_r2:.4f}")

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_name = model_name

        # Save artifacts
        y_pred = model.predict(data.x_test)
        save_predictions(y_pred, dirs["predictions"], model_name)
        save_metrics(metrics["test"], dirs["metrics"], model_name)
        model.save_model(dirs["models"] / f"{model_name}.pkl")
        plot_predictions_vs_actual(
            y_true=data.y_test,
            y_pred=y_pred,
            title=f"{model_name}: Predictions vs Actual",
            save_path=dirs["plots"] / f"{model_name}_predictions.png",
        )

    logger.info(f"Best model: '{best_model_name}' ('R^2' = {best_r2:.4f})")


def run_classification_training(
    data: DataSplit,
    models_registry: dict[str, type[BaseClassifier]],
    param_grids: dict[str, dict],
    dirs: dict[str, Path],
    random_seed: int,
    tune_hyperparams: bool,
    class_labels: Optional[list[str]],
    logger: logging.Logger,
) -> None:
    """Train and evaluate multiple classification models.

    Args:
        data: DataSplit with train and test data.
        models_registry: Dict mapping model names to classifier classes.
        param_grids: Dict mapping model names to hyperparameter grids.
        dirs: Output directories dict with 'models', 'predictions', 'metrics', 'plots'.
        random_seed: Random seed for reproducibility.
        tune_hyperparams: Whether to tune hyperparameters.
        class_labels: List of class labels for reporting.
        logger: Logger instance.
    """
    best_f1 = -float("inf")
    best_model_name = ""

    for model_name in models_registry.keys():
        logger.info(f"Training '{model_name}'...")
        model_cls = models_registry[model_name]
        param_grid = param_grids.get(model_name, {}) if tune_hyperparams else {}

        # Train model
        model, metrics = train_classifier(
            model_cls=model_cls,
            data=data,
            param_grid=param_grid,
            random_seed=random_seed,
            tune_hyperparams=tune_hyperparams,
            class_labels=class_labels,
        )

        # Log metrics
        test_metrics = metrics["test"]
        accuracy = test_metrics.accuracy
        f1_macro = test_metrics.report.get("macro avg", {}).get("f1-score", 0.0)
        logger.info(f"Model '{model_name}' results:")
        logger.info(f"---> Test 'accuracy': {accuracy:.4f}")
        logger.info(f"---> Test 'macro F1': {f1_macro:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = model_name

        # Save artifacts
        y_pred = model.predict(data.x_test)
        save_predictions(y_pred, dirs["predictions"], f"{model_name}_clf")
        save_metrics(test_metrics, dirs["metrics"], f"{model_name}_clf")
        model.save_model(dirs["models"] / f"{model_name}_clf.pkl")

        if class_labels:
            plot_classification_report(
                test_metrics.report,
                class_labels,
                title=f"{model_name}: Classification Report",
                save_path=dirs["plots"] / f"{model_name}_clf_report.png",
            )

    logger.info(f"Best model: '{best_model_name}' ('macro F1' = {best_f1:.4f})")
