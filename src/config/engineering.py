"""Feature engineering configurations for regression and classification pipelines."""

from config.specs import TransformerSpec
from parsers.engineering.categorical import ITDeveloperLevelExtractor

REGRESSION_ENGINEERING_CONFIG: list[TransformerSpec] = []

CLASSIFICATION_ENGINEERING_CONFIG: list[TransformerSpec] = [
    TransformerSpec(
        transformer=ITDeveloperLevelExtractor,
        input_columns=["last_position", "experience"],
        output_column="developer_level",
    ),
]
