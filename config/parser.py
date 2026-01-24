"""
Configuration for data parsing pipeline.

This module contains the configuration dictionary that defines how each column
in the Head Hunter dataset should be processed and transformed.
"""

from parsers import (
    AgeExtractor,
    BusinessTripReadinessExtractor,
    CarOwnershipExtractor,
    CityCategorizer,
    CurrencyToRUBTransformer,
    EducationLevelExtractor,
    EmploymentTypeNormalizer,
    ExperienceInMonthsExtractor,
    GenderExtractor,
    RelocationReadinessExtractor,
    WorkPositionCategorizer,
    WorkScheduleNormalizer,
)

# --- Configuration dictionary for column transformers ---
# --- Each key is an input column name, value is a list of transformation handlers ---
COLUMN_TRANSFORMER_CONFIG = {
    "ЗП": [
        {
            "role": "y",
            "type": "simple",
            "transformer": CurrencyToRUBTransformer,
            "output_column": "salary",
            "transformer_kwargs": {"year": 2019},
        }
    ],
    "Опыт (двойное нажатие для полной версии)": [
        {
            "role": "X",
            "type": "numerical",
            "transformer": ExperienceInMonthsExtractor,
            "output_column": "experience",
            "params": {"iqr_k": 3.5},
        }
    ],
    "Пол, возраст": [
        {
            "role": "X",
            "type": "categorical",
            "transformer": GenderExtractor,
            "output_column": "is",
            "params": {"drop_first": True},
        },
        {
            "role": "X",
            "type": "numerical",
            "transformer": AgeExtractor,
            "output_column": "age",
            "params": {"iqr_k": 3.5},
        },
    ],
    "Ищет работу на должность:": [
        {
            "role": "X",
            "type": "categorical",
            "transformer": WorkPositionCategorizer,
            "output_column": "desired_position",
            "transformer_kwargs": {"default_label": "other"},
        }
    ],
    "Последеняя/нынешняя должность": [
        {
            "role": "X",
            "type": "categorical",
            "transformer": WorkPositionCategorizer,
            "output_column": "last_position",
            "transformer_kwargs": {"default_label": "other"},
        }
    ],
    "Город": [
        {
            "role": "X",
            "type": "categorical",
            "transformer": CityCategorizer,
            "output_column": "city",
            "transformer_kwargs": {"default_label": "small"},
        },
        {
            "role": "X",
            "type": "categorical",
            "transformer": RelocationReadinessExtractor,
            "output_column": "relocation_readiness",
        },
        {
            "role": "X",
            "type": "categorical",
            "transformer": BusinessTripReadinessExtractor,
            "output_column": "business_trip_readiness",
        },
    ],
    "Авто": [
        {
            "role": "X",
            "type": "categorical",
            "transformer": CarOwnershipExtractor,
            "output_column": "car",
            "transformer_kwargs": {"default_label": "not_own"},
            "params": {"drop_first": True},
        }
    ],
    "Образование и ВУЗ": [
        {
            "role": "X",
            "type": "categorical",
            "transformer": EducationLevelExtractor,
            "output_column": "education",
        }
    ],
    "Занятость": [
        {
            "role": "X",
            "type": "multi_label",
            "transformer": EmploymentTypeNormalizer,
            "output_column": "employment",
        }
    ],
    "График": [
        {
            "role": "X",
            "type": "multi_label",
            "transformer": WorkScheduleNormalizer,
            "output_column": "schedule",
        }
    ],
}
