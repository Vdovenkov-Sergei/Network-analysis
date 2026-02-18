"""Visualization utilities for models.

This module provides functions for creating publication-quality plots
using matplotlib and seaborn.
"""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _setup_style() -> None:
    """Configure global plot styling.

    This function sets the default style for matplotlib.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def _finalize_figure(
    fig: plt.Figure,
    save_path: Optional[Union[str, Path]],
    show: bool,
) -> None:
    """Save figure to disk and/or display it.

    Args:
        fig: Matplotlib figure to finalize.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot. If False, closes the figure.
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """Create a scatter plot of predictions vs actual values.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
        title: Plot title.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", s=20)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal")

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()

    _finalize_figure(fig, save_path, show)


def plot_class_balance(
    y: np.ndarray,
    class_labels: list[str],
    title: str = "Class Distribution",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """Plot class balance (number of samples per class).

    Args:
        y: Integer-encoded class labels.
        class_labels: Ordered list of class labels.
        title: Plot title.
        save_path: Path to save the figure.
        show: Whether to display the plot.
    """
    _setup_style()
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_labels[int(idx)] for idx in unique]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)

    for bar, count in zip(bars, counts):
        ax.annotate(
            str(count),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    _finalize_figure(fig, save_path, show)


def plot_classification_report(
    report_dict: dict[str, Any],
    class_labels: list[str],
    title: str = "Classification Report",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """Plot classification report as a heatmap (precision, recall, f1-score).

    Args:
        report_dict: Dict with class metrics.
        class_labels: Class labels for display.
        title: Plot title.
        save_path: Path to save the figure.
        show: Whether to display the plot.
    """
    _setup_style()
    metrics = ["precision", "recall", "f1-score"]
    data = []
    for class_label in class_labels:
        if class_label in report_dict:
            row = [report_dict[class_label].get(metric, 0) for metric in metrics]
        else:
            row = [0.0, 0.0, 0.0]
        data.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        np.array(data).T,
        annot=True,
        fmt=".2f",
        xticklabels=class_labels,
        yticklabels=metrics,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)

    _finalize_figure(fig, save_path, show)
