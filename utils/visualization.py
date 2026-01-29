"""Visualization utilities for model training and evaluation.

This module provides functions for creating publication-quality plots
using matplotlib.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def setup_style() -> None:
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
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Scatter plot ---
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none", s=20)

    # --- Diagonal line (perfect predictions) ---
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal")

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
