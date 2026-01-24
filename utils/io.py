"""
I/O utilities for saving processed data.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def save_x_y(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
    path: Union[str, Path],
    prefix: str = "",
) -> None:
    """
    Save feature matrix X and target y to .npy files.

    Files created:
        {prefix}X_data.npy
        {prefix}y_data.npy
        {prefix}feature_names.npy
        {prefix}target_names.npy

    Args:
        X: Feature DataFrame.
        y: Target variable (Series or DataFrame).
        path: Directory to save files.
        prefix: Optional filename prefix.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / f"{prefix}X_data.npy", X.to_numpy())
    np.save(path / f"{prefix}feature_names.npy", X.columns.to_numpy())

    np.save(path / f"{prefix}y_data.npy", y.to_numpy())
    if isinstance(y, pd.Series):
        np.save(path / f"{prefix}target_names.npy", np.array([y.name]))
    else:
        np.save(path / f"{prefix}target_names.npy", y.columns.to_numpy())
