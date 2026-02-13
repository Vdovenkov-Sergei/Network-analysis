# Head Hunter Resume Analysis

A machine learning project for Head Hunter resume dataset: salary prediction (`regression`) and developer level classification (`junior/middle/senior`). Includes a complete pipeline from data preprocessing to training multiple models.

## Features

- **Data Preprocessing**: Modular transformer classes for feature extraction and processing
- **Regression Models**: Ridge Regression, Random Forest, Gradient Boosting (`salary prediction`)
- **Classification Models**: Logistic Regression, Random Forest, Gradient Boosting (`developer level`)
- **Hyperparameter Tuning**: Automatic tuning via GridSearchCV with cross-validation
- **Visualization**: Prediction plots, class balance, classification report heatmaps
- **Evaluation**: R², MSE, RMSE, MAE for regression; precision, recall, F1 for classification

## Project Structure

```
.
├── parsers/                     # Transformer classes for feature extraction
│   ├── base/                    # Abstract base classes
│   │   └── core.py              # BaseColumnTransformer, Single/Multi variants
│   ├── transformers/            # Concrete transformer implementations
│   │   ├── categorical.py       # Gender, position, city, etc.
│   │   ├── multilabel.py        # Employment type, work schedule
│   │   └── numerical.py         # Age, experience, salary, IQR masker
│   └── engineering/             # Feature engineering transformers
│       └── categorical.py       # IT developer level extraction
├── training/                    # Model training module
│   ├── base.py                  # BaseModel with fit/predict/evaluate
│   ├── metrics.py               # RegressionMetrics, ClassificationMetrics
│   ├── regressors/              # Regressor implementations
│   │   ├── base.py
│   │   ├── ridge.py
│   │   ├── random_forest.py
│   │   └── gradient_boosting.py
│   └── classifiers/             # Classifier implementations
│       ├── base.py
│       ├── logistic.py
│       ├── random_forest.py
│       └── gradient_boosting.py
├── utils/                       # Utility functions
│   ├── cleaning.py              # Data cleaning functions
│   ├── loader.py                # DataLoader for preprocessed data
│   ├── logging.py               # Logger setup
│   ├── processing.py            # Data transformation and encoding pipeline
│   ├── saving.py                # Save metrics, predictions, datasets
│   ├── training.py              # Training pipeline (regression + classification)
│   └── visualization.py         # Plots for evaluation
├── config/
│   ├── cli.py                   # Command-line argument parsers
│   ├── specs.py                 # TransformerSpec, EncodingSpec, TargetSpec
│   ├── transform.py             # Column transformation config
│   ├── engineering.py           # Feature engineering config
│   ├── encoding.py              # Encoding config (regression + classification)
│   └── tuning.py                # Hyperparameter grids
├── preprocess.py                # Data preprocessing script
└── train.py                     # Unified training script (regression + classification)
```

## Installation

This project uses Poetry for dependency management:

```bash
git clone https://github.com/Vdovenkov-Sergei/Network-analysis
cd Network-analysis
poetry install
```

## Quick Start

### 1. Preprocess Data

Convert CSV to NumPy arrays:

```bash
# For regression
python preprocess.py data/hh.csv -o data/processed --task regression

# For classification
python preprocess.py data/hh.csv -o data/processed --task classification
```

Output:
- `data/processed/regression/` — Regression data (X, y, feature columns, target column)
- `data/processed/classification/` — Classification data (X, y, feature columns, target column, class labels)

### 2. Train Models

Train all models for a task:

```bash
# Regression
python train.py data/processed -o resources --task regression

# Classification
python train.py data/processed -o resources --task classification
```

### 3. Results

Output structure:
```
resources/
├── regression/
│   ├── models/          # *.pkl files
│   ├── predictions/     # *_predictions.npy
│   ├── metrics/         # *_metrics.json (R², MSE, RMSE, MAE)
│   └── plots/           # predictions vs actual
└── classification/
    ├── models/          # *_clf.pkl files
    ├── predictions/     # *_clf_predictions.npy
    ├── metrics/         # *_clf_metrics.json (accuracy, precision, recall, F1)
    └── plots/           # class_balance.png, *_clf_report.png
```

## Training Configuration

### Data Splitting

Default ratios (can be changed via CLI):
- **Training**: 80%
- **Testing**: 20%

```bash
python train.py data/processed --task regression --train-ratio 0.7 --test-ratio 0.3
```

### Hyperparameter Tuning

When enabled (default), `GridSearchCV` with 5-fold cross-validation is used on the training set to find optimal hyperparameters. Disable with `--no-tune` flag for faster training.

```bash
python train.py data/processed --task regression --no-tune
```

### Evaluation Metrics

**Regression**:
- **R²** (coefficient of determination) — primary metric
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Classification**:
- **Macro F1** — primary metric
- Accuracy
- Per-class precision, recall, F1-score

## Features

The preprocessing pipeline extracts the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `experience` | Numerical | Work experience in months |
| `age` | Numerical | Candidate age |
| `sex` | Binary | Gender |
| `desired_position_*` | Categorical | Desired job position |
| `last_position_*` | Categorical | Previous job position |
| `city_*` | Categorical | City category |
| `relocation_readiness_*` | Categorical | Relocation readiness |
| `business_trip_readiness_*` | Categorical | Business trip readiness |
| `has_car` | Binary | Car ownership |
| `education_*` | Categorical | Education level |
| `employment_*` | Multi-label | Employment type |
| `schedule_*` | Multi-label | Work schedule |

**Regression target**: `salary` (in Russian Rubles)

**Classification target**: `developer_level` — junior / middle / senior (IT developers only, inferred from job title keywords and experience)

## Model Performance & Results

### Regression (Salary Prediction)

| Model | R² | RMSE (₽) | MAE (₽) |
|-------|-----|----------|---------|
| **Random Forest** | **0.499** | **36,802** | **25,321** |
| Gradient Boosting | 0.485 | 37,330 | 26,420 |
| Ridge Regression | 0.432 | 39,183 | 28,614 |

**Key Findings**:
- **Random Forest achieved the best performance** with R² = 0.499, explaining ~50% of salary variance
- RMSE of ~37K rubles indicates average prediction error of approximately 37,000 rubles
- Linear model (Ridge) performed worst, suggesting **non-linear relationships** in the data
- Moderate R² (~0.5) is attributed to:
  - High salary variability even for similar resumes
  - Missing key features: skills, specific technologies, company size, exact location

---

### Classification (Developer Level)

| Model | Accuracy | Macro F1 | Junior F1 | Middle F1 | Senior F1 |
|-------|----------|----------|-----------|-----------|-----------|
| **Gradient Boosting** | **0.747** | **0.627** | **0.519** | **0.491** | **0.871** |
| Random Forest | 0.738 | 0.617 | 0.545 | 0.444 | 0.862 |
| Logistic Regression | 0.708 | 0.537 | 0.373 | 0.376 | 0.864 |

**Key Findings**:
- **Gradient Boosting is the best model** with macro F1 = 0.627 and accuracy = 74.7%
- **Strong class imbalance**: Senior class dominates, resulting in:
  - Excellent performance on Senior (F1 = 0.871)
  - Poor performance on Junior/Middle (F1 = 0.519 / 0.491)
- **Logistic Regression significantly underperforms** (macro F1 = 0.537), indicating **non-linear class boundaries**
- Main challenges:
  - Fuzzy class boundaries: 2-5 years of experience can be junior, middle, or senior depending on skills
  - Limited features: no information about skills, projects, or education quality

## Development

### Code Quality

```bash
poetry run black .       # formatting
poetry run isort .       # import sorting
poetry run ruff check .  # linting
poetry run mypy .        # type checking
```

### Adding New Models

**Regressor**:
1. Create a file in `training/regressors/`
2. Inherit from `BaseRegressor`
3. Implement `__init__()` with model initialization
4. Add to `REGRESSION_MODELS` in `train.py`
5. Add hyperparameter grid to `REGRESSION_PARAM_GRIDS` in `config/tuning.py`

**Classifier**:
1. Create a file in `training/classifiers/`
2. Inherit from `BaseClassifier`
3. Implement `__init__()` with model initialization
4. Add to `CLASSIFICATION_MODELS` in `train.py`
5. Add hyperparameter grid to `CLASSIFICATION_PARAM_GRIDS` in `config/tuning.py`
