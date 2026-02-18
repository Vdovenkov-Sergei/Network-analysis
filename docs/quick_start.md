# Quick Start

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
git clone https://github.com/Vdovenkov-Sergei/Network-analysis
cd Network-analysis
poetry config virtualenvs.in-project true
poetry install
```

## Virtual environment

The virtual environment is created in the project root as `.venv`. Activate it so you can run `python` without the `poetry run` prefix.

**Activate** (from project root):

| OS | Command |
|----|---------|
| Linux / macOS | `source .venv/bin/activate` |
| Windows (Cmd) | `.venv\Scripts\activate` |

**Deactivate**:

```bash
deactivate
```

## 1. Preprocess Data

Convert CSV to NumPy arrays. Root output directory defaults to `data`; preprocessed files are written under `{output_dir}/processed/{task}`.

```bash
# Regression (salary prediction)
poetry run python src/preprocess.py data/hh.csv -o data --task regression

# Classification (developer level)
poetry run python src/preprocess.py data/hh.csv -o data --task classification
```

Output:
- `data/processed/regression/` — Regression data (_X_, _y_, _feature columns_, _target column_)
- `data/processed/classification/` — Classification data (_X_, _y_, _feature columns_, _target column_, _class labels_)

## 2. Train Models

Root output directory defaults to `data`. Artifacts are saved under `{output_dir}/training/{task}`.

```bash
# Regression
poetry run python src/train.py data/processed -o data --task regression

# Classification
poetry run python src/train.py data/processed -o data --task classification
```

Available models:
- **Regression**: `ridge`, `random_forest`, `gradient_boosting`
- **Classification**: `logistic`, `random_forest`, `gradient_boosting`

## 3. Results

Output structure:

```
data/
├── processed/
│   ├── regression/       # Preprocessed regression data
│   └── classification/   # Preprocessed classification data
└── training/
    ├── regression/
    │   ├── models/       # *.pkl
    │   ├── predictions/  # *_predictions.npy
    │   ├── metrics/      # *_metrics.json (R², MSE, RMSE, MAE)
    │   └── plots/        # Predictions vs actual
    └── classification/
        ├── models/       # *_clf.pkl
        ├── predictions/  # *_clf_predictions.npy
        ├── metrics/      # *_clf_metrics.json (accuracy, precision, recall, F1)
        └── plots/        # class_balance.png, *_clf_report.png
```
