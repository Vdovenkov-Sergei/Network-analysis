# Head Hunter Salary Prediction

A machine learning project for salary prediction based on Head Hunter resume dataset. Includes a complete pipeline from data preprocessing to training multiple regression models.

## Features

- **Data Preprocessing**: Modular transformer classes for feature extraction and processing
- **Multiple Models**: Ridge Regression, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: Automatic tuning via GridSearchCV with cross-validation
- **Visualization**: Prediction plots using matplotlib
- **Evaluation**: R², MSE, RMSE, MAE metrics

## Project Structure

```
.
├── parsers/                     # Transformer classes for feature extraction
│   ├── base/                    # Abstract base classes
│   │   └── core.py
│   └── transformers/            # Concrete transformer implementations
│       ├── categorical.py
│       ├── multilabel.py
│       └── numerical.py
├── training/                    # Model training module
│   └── regressors/              # Regressor implementations
│       ├── base.py              # Base regressor class
│       ├── linear.py            # Ridge regression
│       ├── random_forest.py     # Random Forest
│       └── gradient_boosting.py # Gradient Boosting
├── utils/                       # Utility functions
│   ├── cleaning.py              # Data cleaning
│   ├── loader.py                # Data loading and splitting
│   ├── logging.py               # Logging configuration
│   ├── metrics.py               # Evaluation metrics
│   ├── pipeline_factory.py      # Pipeline creation
│   ├── processing.py            # Data processing
│   ├── training.py              # Training utilities
│   └── visualization.py         # Plotting functions
├── config/                      # Configuration files
│   ├── cli.py                   # CLI argument parsers
│   ├── parser.py                # Transformer config
│   └── tuning.py                # Hyperparameter grids
├── data/                        # Data directory (git-ignored)
│   ├── hh.csv                   # Raw dataset
│   └── processed/               # Preprocessed data
├── resources/                   # Training results (git-ignored)
│   ├── models/                  # Saved model files
│   ├── predictions/             # Model predictions
│   ├── metrics/                 # Evaluation metrics
│   └── plots/                   # Visualization plots
├── preprocess.py                # Preprocessing script
└── train.py                     # Training script
```

## Installation

This project uses Poetry for dependency management:

```bash
git clone https://github.com/yourusername/network-analysis.git
cd network-analysis
poetry install
```

## Quick Start

### 1. Preprocess Data

Convert CSV to NumPy arrays:

```bash
python preprocess.py data/hh.csv -o data
```

Output files:
- `X_data.npy` — Feature matrix
- `y_data.npy` — Target variable (salary in RUB)
- `feature_names.npy` — Feature names
- `target_names.npy` — Target names

### 2. Train Models

Train all models:

```bash
python train.py data/processed -o resources
```

Train specific models:

```bash
python train.py data/processed --models ridge gradient_boosting
```

Skip hyperparameter tuning (use default parameters):

```bash
python train.py data/processed --no-tune
```

Available models:
- `ridge` — Ridge Regression (L2 regularization)
- `random_forest` — Random Forest
- `gradient_boosting` — Gradient Boosting

### 3. Results

After training, results are saved to `resources/` (or specified output directory):
- `models/*.pkl` — Trained model files
- `predictions/*_predictions.npy` — Model predictions on test set
- `metrics/*_metrics.json` — Evaluation metrics (R², MSE, RMSE, MAE)
- `plots/*_predictions.png` — Predictions vs Actual scatter plots

## Training Configuration

### Data Splitting

Default ratios (can be changed via CLI):
- **Training**: 80%
- **Testing**: 20%

```bash
python train.py data/processed --train-ratio 0.7 --test-ratio 0.3
```

### Hyperparameter Tuning

When enabled (default), `GridSearchCV` with 5-fold cross-validation is used on the training set to find optimal hyperparameters. Disable with `--no-tune` flag for faster training.

### Evaluation Metric

Primary metric is **R² (coefficient of determination)**. Additional metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

## Features

The preprocessing pipeline extracts the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `experience` | Numerical | Work experience in months |
| `age` | Numerical | Candidate age |
| `is_*` | Categorical | Gender (one-hot encoded) |
| `desired_position_*` | Categorical | Desired job position |
| `last_position_*` | Categorical | Previous job position |
| `city_*` | Categorical | City category |
| `relocation_readiness_*` | Categorical | Relocation readiness |
| `business_trip_readiness_*` | Categorical | Business trip readiness |
| `car_*` | Categorical | Car ownership |
| `education_*` | Categorical | Education level |
| `employment_*` | Multi-label | Employment type |
| `schedule_*` | Multi-label | Work schedule |

**Target variable**: `salary` (in Russian Rubles)

## Development

### Code Quality

```bash
poetry run black .       # formatting
poetry run isort .       # import sorting
poetry run ruff check .  # linting
poetry run mypy .        # type checking
```

### Adding New Regressor Models

1. Create a file in `training/regressors/`
2. Inherit from `BaseRegressor`
3. Implement the `fit()` method
4. Add to `MODELS` dict in `train.py`
5. Optionally add hyperparameter grid to `config/tuning.py`
