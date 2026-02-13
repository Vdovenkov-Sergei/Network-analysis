"""Categorical feature engineering.

This module provides transformers for extracting derived categorical features
from already-parsed columns.
"""

from typing import Optional

import pandas as pd

from parsers.base.core import BaseRowWiseTransformerMulti


class ITDeveloperLevelExtractor(BaseRowWiseTransformerMulti):
    """Extracts developer level (junior/middle/senior) from position and experience.

    Expected column order (unpacked from `row.index` in process()):
        0: position column (categorical: programmer, qa, etc.)
        1: experience column (numeric: months of experience)
    """

    JUNIOR_MAX_MONTHS = 24  # 0-2 years
    MIDDLE_MAX_MONTHS = 60  # 2-5 years
    IT_DEVELOPER_POSITIONS = {"programmer", "qa"}

    def process(self, row: pd.Series) -> Optional[str]:
        """Extract developer level from a single row.

        Args:
            row: One row with position and experience data.

        Returns:
            'junior', 'middle', or 'senior' for IT developers; None otherwise.
        """
        position_col, experience_col = row.index

        position = self._to_label(row[position_col])
        if position not in self.IT_DEVELOPER_POSITIONS:
            return None

        experience = self._to_int(row[experience_col])
        if experience is None:
            return None
        if experience <= self.JUNIOR_MAX_MONTHS:
            return "junior"
        if experience <= self.MIDDLE_MAX_MONTHS:
            return "middle"
        return "senior"

    @staticmethod
    def _to_label(val: object) -> str:
        """Convert value to lowercase label string.

        Args:
            val Any value.

        Returns:
            Stripped lowercase string, or "" if None/NaN.
        """
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return str(val).strip().lower()

    @staticmethod
    def _to_int(val: object) -> Optional[int]:
        """Convert value to integer.

        Args:
            val: Any value.

        Returns:
            Integer value, or None if invalid.
        """
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return int(float(str(val).strip().replace(",", ".")))
        except (ValueError, TypeError):
            return None
