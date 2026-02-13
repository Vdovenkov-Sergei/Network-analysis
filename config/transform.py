"""Column transformation pipeline configuration."""

from config.specs import TransformerSpec
from parsers.transformers.categorical import (
    BusinessTripReadinessExtractor,
    CarOwnershipExtractor,
    CityCategorizer,
    EducationLevelExtractor,
    GenderExtractor,
    RelocationReadinessExtractor,
    WorkPositionCategorizer,
)
from parsers.transformers.multilabel import EmploymentTypeNormalizer, WorkScheduleNormalizer
from parsers.transformers.numerical import (
    AgeExtractor,
    CurrencyToRUBTransformer,
    ExperienceInMonthsExtractor,
)

COLUMN_TRANSFORM_CONFIG: list[TransformerSpec] = [
    TransformerSpec(
        transformer=CurrencyToRUBTransformer,
        input_columns=["ЗП"],
        output_column="salary",
        params={"year": 2019},
    ),
    TransformerSpec(
        transformer=ExperienceInMonthsExtractor,
        input_columns=["Опыт (двойное нажатие для полной версии)"],
        output_column="experience",
    ),
    TransformerSpec(
        transformer=GenderExtractor,
        input_columns=["Пол, возраст"],
        output_column="sex",
    ),
    TransformerSpec(
        transformer=AgeExtractor,
        input_columns=["Пол, возраст"],
        output_column="age",
    ),
    TransformerSpec(
        transformer=WorkPositionCategorizer,
        input_columns=["Ищет работу на должность:"],
        output_column="desired_position",
        params={"default_label": "other"},
    ),
    TransformerSpec(
        transformer=WorkPositionCategorizer,
        input_columns=["Последеняя/нынешняя должность"],
        output_column="last_position",
        params={"default_label": "other"},
    ),
    TransformerSpec(
        transformer=CityCategorizer,
        input_columns=["Город"],
        output_column="city",
        params={"default_label": "small"},
    ),
    TransformerSpec(
        transformer=RelocationReadinessExtractor,
        input_columns=["Город"],
        output_column="relocation_readiness",
    ),
    TransformerSpec(
        transformer=BusinessTripReadinessExtractor,
        input_columns=["Город"],
        output_column="business_trip_readiness",
    ),
    TransformerSpec(
        transformer=CarOwnershipExtractor,
        input_columns=["Авто"],
        output_column="has_car",
        params={"default_label": "no"},
    ),
    TransformerSpec(
        transformer=EducationLevelExtractor,
        input_columns=["Образование и ВУЗ"],
        output_column="education",
    ),
    TransformerSpec(
        transformer=EmploymentTypeNormalizer,
        input_columns=["Занятость"],
        output_column="employment",
    ),
    TransformerSpec(
        transformer=WorkScheduleNormalizer,
        input_columns=["График"],
        output_column="schedule",
    ),
]
