"""
Parsers package for network analysis.

This package contains transformers and parsers for processing Head Hunter dataset.
All classes can be imported directly from the parsers package.
"""

from parsers.base import (
    BaseCategoricalTextExtractor,
    BaseRowWiseTransformer,
    BaseSingleColumnTransformer,
    BaseTextListNormalizer,
)
from parsers.transformers import (
    AgeExtractor,
    BusinessTripReadinessExtractor,
    CarOwnershipExtractor,
    CityCategorizer,
    CurrencyToRUBTransformer,
    EducationLevelExtractor,
    EmploymentTypeNormalizer,
    ExperienceInMonthsExtractor,
    GenderExtractor,
    IQRMasker,
    MultiLabelTransformer,
    RelocationReadinessExtractor,
    WorkPositionCategorizer,
    WorkScheduleNormalizer,
)

__all__ = [
    # --- Base classes ---
    "BaseSingleColumnTransformer",
    "BaseRowWiseTransformer",
    "BaseCategoricalTextExtractor",
    "BaseTextListNormalizer",
    # --- Transformers ---
    "AgeExtractor",
    "BusinessTripReadinessExtractor",
    "CarOwnershipExtractor",
    "CityCategorizer",
    "CurrencyToRUBTransformer",
    "EducationLevelExtractor",
    "EmploymentTypeNormalizer",
    "ExperienceInMonthsExtractor",
    "GenderExtractor",
    "IQRMasker",
    "MultiLabelTransformer",
    "RelocationReadinessExtractor",
    "WorkPositionCategorizer",
    "WorkScheduleNormalizer",
]
