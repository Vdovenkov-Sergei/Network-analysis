"""Encoding configurations for regression and classification pipelines."""

from config.specs import EncodingSpec, TargetSpec

# --- Target specifications ---

REGRESSION_TARGET = TargetSpec(
    column="salary",
    task="regression",
    apply_iqr_masker=True,
)

CLASSIFICATION_TARGET = TargetSpec(
    column="developer_level",
    task="classification",
    return_labels=True,
)


# --- Feature encoding specifications ---

REGRESSION_ENCODING_CONFIG = EncodingSpec(
    one_hot=[
        "desired_position",
        "last_position",
        "city",
        "relocation_readiness",
        "business_trip_readiness",
        "education",
    ],
    iqr_masker=["age", "experience"],
    scale=["age", "experience"],
    label_encode=["sex", "has_car"],
    multi_label=["employment", "schedule"],
    drop=[],
)

CLASSIFICATION_ENCODING_CONFIG = EncodingSpec(
    one_hot=[
        "desired_position",
        "city",
        "relocation_readiness",
        "business_trip_readiness",
        "education",
    ],
    iqr_masker=["salary", "age"],
    scale=["salary", "age"],
    label_encode=["sex", "has_car"],
    multi_label=["employment", "schedule"],
    drop=["experience", "last_position"],
)
