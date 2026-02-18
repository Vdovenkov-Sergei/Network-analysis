# Training

The training script trains regression or classification models on preprocessed data and saves models, predictions, metrics, and plots.

## Usage

```bash
poetry run python src/train.py <data_dir> -o data [--task regression|classification] [options]
```

- **data_dir**: Root directory containing preprocessed data. The script loads from `{data_dir}/{task}` (e.g. `data/processed/regression` when `data_dir=data/processed` and `--task regression`).
- **-o, --output-dir**: Root output directory (default: `data`). Artifacts are saved under `{output_dir}/training/{task}`.
- **--task**: `regression` or `classification`. Default: `regression`.
- **--prefix**: Optional prefix for input files.
- **--train-ratio**: Training set ratio (default: `0.8`).
- **--test-ratio**: Test set ratio (default: `0.2`).
- **--seed**: Random seed (default: `42`).
- **--no-tune**: Disable hyperparameter tuning.
- **--log-level**: Logging level (default: `INFO`).

## Data Splitting

Default ratios (overridable via CLI):
- **Training**: 80%
- **Testing**: 20%

## Hyperparameter Tuning

When enabled (default), `GridSearchCV` with 5-fold cross-validation is used. Disable with `--no-tune` for faster runs.

## Evaluation Metrics

**Regression**
- **R²** (coefficient of determination) — primary
- MSE, RMSE, MAE

**Classification**
- **Macro F1** — primary (handles class imbalance)
- Accuracy, per-class precision, recall, F1-score
