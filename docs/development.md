# Development

## Code Quality

```bash
poetry run black src
poetry run isort src
poetry run ruff check src
poetry run mypy src
```

## Adding Parsers (Transformers)

Parsers live under `src/parsers/`: base classes in `base/core.py`, concrete implementations in `transformers/` (main pipeline) or `engineering/` (derived features).

### Base classes

- **Single column, row-wise**: `BaseRowWiseTransformerSingle` — implement `process(value)` returning the transformed value.
- **Categorical from text**: `BaseCategoricalTextExtractor` — define class attribute `PATTERNS: dict[label, re.Pattern]`; optional `default_label` via `params` in config.
- **Multi column, row-wise**: `BaseRowWiseTransformerMulti` — implement `process(row)`; unpack column names from `row.index` and use `row[col]`; document expected column order in the docstring.

### Where to add

| Kind | File |
|------|------|
| Categorical, Binary (regex-based) | `parsers/transformers/categorical.py` |
| Numerical | `parsers/transformers/numerical.py` |
| Multi-label (lists) | `parsers/transformers/multilabel.py` |
| Engineering (categorical) | `parsers/engineering/categorical.py` |
| Engineering (numerical) | `parsers/engineering/numerical.py` |
| Engineering (multi-label) | `parsers/engineering/multilabel.py` |

### Registering

**Main pipeline** (raw → structured columns): add a `TransformerSpec` to `COLUMN_TRANSFORM_CONFIG` in `src/config/transform.py`:

```python
TransformerSpec(
    transformer=YourTransformer,
    input_columns=["Raw column name"],
    output_column="output_name",
    params={},  # optional, e.g. default_label="other"
),
```

**Feature engineering** (derived columns): add a `TransformerSpec` to `REGRESSION_ENGINEERING_CONFIG` or `CLASSIFICATION_ENGINEERING_CONFIG` in `src/config/engineering.py`. Use already-transformed column names in `input_columns` (e.g. `["last_position", "experience"]`).

### Encoding

To use a new column in encoding, add its name to the right list in `src/config/encoding.py` in `REGRESSION_ENCODING_CONFIG` and/or `CLASSIFICATION_ENCODING_CONFIG` (`EncodingSpec`): `one_hot`, `label_encode`, `multi_label`, `iqr_masker`, `scale`, or `drop`.

## Adding New Models

**Regressor**
1. Add a module under `src/training/regressors/`
2. Inherit from `BaseRegressor`
3. Implement `__init__()` with model setup
4. Register in `REGRESSION_MODELS` in `src/train.py`
5. Add a hyperparameter grid in `REGRESSION_PARAM_GRIDS` in `src/config/tuning.py`

**Classifier**
1. Add a module under `src/training/classifiers/`
2. Inherit from `BaseClassifier`
3. Implement `__init__()` with model setup
4. Register in `CLASSIFICATION_MODELS` in `src/train.py`
5. Add a hyperparameter grid in `CLASSIFICATION_PARAM_GRIDS` in `src/config/tuning.py`
