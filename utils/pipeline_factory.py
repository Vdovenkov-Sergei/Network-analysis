"""
Pipeline creation utilities.

This module contains functions for creating preprocessing pipelines.
"""

from typing import Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from parsers.base import (
    BaseCategoricalTextExtractor,
    BaseRowWiseTransformer,
    BaseSingleColumnTransformer,
    BaseTextListNormalizer,
)
from parsers.transformers import IQRMasker, MultiLabelTransformer


class PipelineFactory:
    """
    Factory class for creating standardized preprocessing pipelines.

    This class provides methods to create different types of preprocessing
    pipelines (numerical, categorical, multi-label).
    """

    @staticmethod
    def make_numerical_pipeline(
        extractor_cls: type[BaseRowWiseTransformer],
        column_name: str,
        use_iqr: bool = True,
        iqr_k: float = 1.5,
        use_scaler: bool = True,
        scaler_range: tuple[float, float] = (0, 1),
        **extractor_kwargs: dict[str, Any],
    ) -> Pipeline:
        """
        Create a pipeline for numerical columns.

        Args:
            extractor_cls: Numerical extractor class
            column_name: Output column name
            use_iqr: Whether to apply IQR outlier masking
            iqr_k: Multiplier for IQR outlier detection
            use_scaler: Whether to apply MinMax scaling
            scaler_range: Desired range for MinMaxScaler
            **extractor_kwargs: Additional arguments for the extractor

        Returns:
            Configured numerical preprocessing pipeline.
        """
        steps = [
            (
                "extractor",
                extractor_cls(output_column=column_name, **extractor_kwargs),
            )
        ]

        if use_iqr:
            steps.append(("iqr_masker", IQRMasker(k=iqr_k)))

        if use_scaler:
            steps.append(
                ("min_max_scaler", MinMaxScaler(feature_range=scaler_range))
            )

        return Pipeline(steps)

    @staticmethod
    def make_categorical_pipeline(
        extractor_cls: type[BaseCategoricalTextExtractor],
        column_name: str,
        default_label: str = "unknown",
        drop_first: bool = False,
        **extractor_kwargs: dict[str, Any],
    ) -> Pipeline:
        """
        Create a pipeline for categorical text columns.

        Args:
            extractor_cls: Categorical extractor class
            column_name: Output column name
            default_label: Default label for unmatched entries
            drop_first: Whether to drop first category in one-hot encoding
            **extractor_kwargs: Additional arguments for the extractor

        Returns:
            Configured categorical preprocessing pipeline.
        """
        return Pipeline(
            [
                (
                    "extractor",
                    extractor_cls(
                        output_column=column_name,
                        default_label=default_label,
                        **extractor_kwargs,
                    ),
                ),
                (
                    "one_hot_encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first" if drop_first else None,
                    ),
                ),
            ]
        )

    @staticmethod
    def make_multi_label_pipeline(
        normalizer_cls: type[BaseTextListNormalizer],
        column_name: str,
        classes: Optional[list[str]] = None,
        **normalizer_kwargs: dict[str, Any],
    ) -> Pipeline:
        """
        Create a pipeline for multi-label columns.

        Args:
            normalizer_cls: Multi-label normalizer class
            column_name: Output column name
            classes: Optional list of all possible classes
            **normalizer_kwargs: Additional arguments for the normalizer

        Returns:
            Configured multi-label pipeline.
        """
        return Pipeline(
            [
                (
                    "normalizer",
                    normalizer_cls(
                        output_column=column_name, **normalizer_kwargs
                    ),
                ),
                (
                    "multi_label_binarizer",
                    MultiLabelTransformer(classes=classes),
                ),
            ]
        )

    @staticmethod
    def make_simple_pipeline(
        transformer_cls: type[BaseSingleColumnTransformer],
        column_name: str,
        **transformer_kwargs: dict[str, Any],
    ) -> Pipeline:
        """
        Create a simple single-step pipeline.

        Useful for transformers that don't need additional processing.

        Args:
            transformer_cls: Transformer class
            column_name: Output column name
            **transformer_kwargs: Additional arguments for the transformer

        Returns:
            Simple single-step pipeline.
        """
        return Pipeline(
            [
                (
                    "transformer",
                    transformer_cls(
                        output_column=column_name, **transformer_kwargs
                    ),
                )
            ]
        )
