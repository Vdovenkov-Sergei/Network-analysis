"""
Numeric transformers.
"""

import re
import warnings
from datetime import datetime
from typing import Any, Optional
from xml.etree import ElementTree

import pandas as pd
import requests
from typing_extensions import Self

from parsers.base.core import BaseRowWiseTransformer


class IQRMasker(BaseRowWiseTransformer):
    """
    Transformer that detects outliers using the IQR rule and replaces them with None.

    Processes one numeric column at a time. Non-numeric values are ignored.
    Outliers are replaced with None according to [Q1 - k * IQR, Q3 + k * IQR].
    """

    def __init__(self, output_column: Optional[str] = None, k: float = 1.5) -> None:
        """
        Initialize the transformer.

        Args:
            output_column: Name of the output column. If None, the input column name is used.
            k: IQR multiplier used to determine outlier thresholds.
        """
        super().__init__(output_column=output_column)
        self.k = k
        self.lower_: Optional[float] = None
        self.upper_: Optional[float] = None

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> Self:
        """
        Compute IQR-based bounds for a single numeric column.

        Args:
            X: Single numeric column (Series or single-column DataFrame)
            y: Target values (ignored)

        Returns:
            self: Fitted transformer.

        Raises:
            ValueError: If input column is not numeric.
        """
        series = self._to_series(X)

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Transformer requires a numeric column")

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr

        super().fit(X=X, y=y)
        return self

    def process(self, value: Any) -> Any:
        """
        Replace a value with None if it is outside the IQR bounds.

        Args:
            value: Numeric value

        Returns:
            Original value or None if it is an outlier
        """
        if value is None or pd.isna(value):
            return None
        if value < self.lower_ or value > self.upper_:
            return None
        return value


class AgeExtractor(BaseRowWiseTransformer):
    """
    Transformer that extracts the first age value from a column.

    The transformer looks for patterns like "25 лет", "30 года", "20 years", etc.,
    and returns the first integer found. Non-string inputs or texts without a match
    return None.
    """

    AGE_PATTERN: re.Pattern = re.compile(r"(\d+)\s*(лет|года|год|years?|yrs?)", re.IGNORECASE)

    def process(self, text: Any) -> Optional[int]:
        """
        Extract age value from a single text entry.

        Args:
            text: Input string containing age information.

        Returns:
            Extracted age as integer, or None if not found or input is invalid.
        """
        if not isinstance(text, str):
            return None

        text = text.strip()
        match_obj = self.AGE_PATTERN.search(text)

        if match_obj:
            try:
                return int(match_obj.group(1))
            except ValueError:
                return None

        return None


class ExperienceInMonthsExtractor(BaseRowWiseTransformer):
    """
    Transformer to extract work experience from a column into total months.

    Only parses text after the first occurrence of 'Опыт работы'.
    Parses strings containing years and months in Russian or English and returns
    the total experience in months.
    """

    EXPERIENCE_PATTERN: re.Pattern = re.compile(
        r"Опыт работы.*?\b(\d+)\s*(?:years?|год(?:а|ов)?|лет)"
        r"(?:\s*(\d+)\s*(?:months?|месяц(?:а|ев)?))?",
        re.IGNORECASE | re.DOTALL,
    )

    def process(self, text: Any) -> Optional[int]:
        """
        Extract total months of experience from a single text entry.

        Args:
            text: Input string containing experience information.

        Returns:
            - Total experience in months as an integer if a match is found.
            - 0 if no experience pattern is found.
            - None if the input is not a string.
        """
        if not isinstance(text, str):
            return None

        match_obj = self.EXPERIENCE_PATTERN.search(text)
        if not match_obj:
            return 0

        years_str, months_str = match_obj.groups()
        total_months = int(years_str) * 12
        if months_str:
            total_months += int(months_str)

        return total_months


class CurrencyToRUBTransformer(BaseRowWiseTransformer):
    """
    Transformer that converts currency amounts to RUB (Russian Rubles).

    This transformer handles strings containing an amount and a currency code,
    converts the amount to RUB using the Central Bank of Russia (CBR) exchange
    rates for a specified year.

    Notes:
        - Supports currency codes like 'USD', 'EUR', 'RUB', and common textual aliases.
        - Non-numeric or unrecognized currency inputs return None.
    """

    CBR_URL_TEMPLATE: str = "https://www.cbr.ru/scripts/XML_daily.asp?date_req={date}"
    DEFAULT_ALIASES: dict[str, str] = {
        "руб.": "RUB",
        "руб": "RUB",
        "грн.": "UAH",
        "грн": "UAH",
        "бел. руб.": "BYN",
        "бел руб": "BYN",
    }

    def __init__(self, output_column: Optional[str] = None, year: Optional[int] = None) -> None:
        """
        Args:
            output_column: Name of the output column. If None, the input column name is used.
            year: Year for which to fetch exchange rates from CBR. If None, current year is used.
        """
        super().__init__(output_column=output_column)
        self.year = year or datetime.now().year
        self.exchange_rates: dict[str, float] = {}

    def fit(self, X: Any, y: Optional[pd.Series] = None) -> Self:
        """
        Fetch exchange rates for the specified year from CBR.

        Args:
            X: Input data (Series or single-column DataFrame).
            y: Target values (ignored).

        Returns:
            self: Fitted transformer.
        """
        self.exchange_rates = self._fetch_annual_cbr_rates(self.year)
        self.exchange_rates["RUB"] = 1.0
        super().fit(X=X, y=y)
        return self

    def process(self, value: Any) -> Optional[float]:
        """
        Convert a single currency value to RUB (Russian Rubles).

        Args:
            value: Input value containing an amount and optional currency code.

        Returns:
            Amount converted to RUB, or None if input is invalid or cannot be parsed.
        """
        if not isinstance(value, str):
            return None

        code = self._extract_currency_code(value)
        number = self._extract_numeric(value)
        if number is None:
            return None

        rate = self.exchange_rates.get(code, 1.0)
        return number * rate

    def _extract_currency_code(self, value: str) -> str:
        """
        Extract the currency code from a string.

        Args:
            value: String containing the currency

        Returns:
            str: Canonical 3-letter currency code
        """
        # --- Try 3-letter code at the end ---
        match_obj = re.search(r"([A-Za-z]{3})$", value.strip(), flags=re.IGNORECASE)
        code = match_obj.group(1).upper() if match_obj else None

        # --- Try aliases if not found ---
        if not code:
            for alias, canon in self.DEFAULT_ALIASES.items():
                if alias in value.lower():
                    code = canon
                    break

        return code or "RUB"

    def _extract_numeric(self, value: str) -> Optional[float]:
        """
        Extract the numeric portion from a string and convert it to float.

        Args:
            value: String containing a number

        Returns:
            The numeric value as a float, or None if conversion fails.
        """
        num_str = re.sub(r"[^0-9,.\-]", "", value).replace(",", ".")
        try:
            return float(num_str)
        except ValueError:
            return None

    def _fetch_annual_cbr_rates(self, year: int) -> dict[str, float]:
        """
        Fetch average exchange rates from the Central Bank of Russia (CBR) for a given year.

        Args:
            year: Year for which to fetch exchange rates (e.g., 2019).

        Returns:
            Dictionary mapping 3-letter currency codes (e.g., 'USD', 'EUR')
            to their average annual exchange rate relative to RUB.

        Raises:
            RuntimeError: If failed to fetch rates for all months.
        """
        rates: dict[str, list[float]] = {}

        for month in range(1, 12 + 1):
            date = self._get_last_valid_day(year, month)
            url = self.CBR_URL_TEMPLATE.format(date=date.strftime("%d/%m/%Y"))
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                root = ElementTree.fromstring(resp.content)
            except (requests.RequestException, ElementTree.ParseError) as exc:
                warnings.warn(f"Failed to fetch CBR rates for {date:%d/%m/%Y}: {exc}")
                continue

            for valute in root.findall("Valute"):
                code_node = valute.find("CharCode")
                nominal_node = valute.find("Nominal")
                value_node = valute.find("Value")

                if code_node is None or nominal_node is None or value_node is None:
                    continue

                code_text = code_node.text
                nominal_text = nominal_node.text
                value_text = value_node.text

                if not code_text or not nominal_text or not value_text:
                    continue

                nominal = int(nominal_text)
                value = float(value_text.replace(",", "."))
                per_unit = value / nominal
                rates.setdefault(code_text, []).append(per_unit)

        if not rates:
            raise RuntimeError("Failed to fetch CBR rates for all months.")

        # --- Compute average rates ---
        avg_rates = {code: sum(values) / len(values) for code, values in rates.items()}
        return avg_rates

    @staticmethod
    def _get_last_valid_day(year: int, month: int) -> datetime:
        """
        Return the last valid day of a month for a given year.

        Args:
            year: Year.
            month: Month (1-12).

        Returns:
            datetime object representing the last valid day of the month.
        """
        day = 31
        while True:
            try:
                return datetime(year, month, day)
            except ValueError:
                day -= 1
