# Project Structure

```
.
├── src/
│   ├── parsers/                     # Transformer classes for feature extraction
│   │   ├── base/                    # Abstract base classes
│   │   │   └── core.py              # BaseColumnTransformer, Single/Multi variants
│   │   ├── transformers/            # Concrete transformer implementations
│   │   │   ├── categorical.py       # Gender, position, city, etc.
│   │   │   ├── multilabel.py        # Employment type, work schedule
│   │   │   └── numerical.py         # Age, experience, salary, IQR masker
│   │   └── engineering/             # Feature engineering transformers
│   │       └── categorical.py       # IT developer level extraction
│   ├── training/                    # Model training module
│   │   ├── base.py                  # BaseModel with fit/predict/evaluate
│   │   ├── metrics.py               # RegressionMetrics, ClassificationMetrics
│   │   ├── regressors/              # Regressor implementations
│   │   │   ├── base.py
│   │   │   ├── ridge.py
│   │   │   ├── random_forest.py
│   │   │   └── gradient_boosting.py
│   │   └── classifiers/             # Classifier implementations
│   │       ├── base.py
│   │       ├── logistic.py
│   │       ├── random_forest.py
│   │       └── gradient_boosting.py
│   ├── utils/                       # Utility functions
│   │   ├── cleaning.py              # Data cleaning functions
│   │   ├── loader.py                # DataLoader for preprocessed data
│   │   ├── logging.py               # Logger setup
│   │   ├── processing.py            # Data transformation and encoding pipeline
│   │   ├── saving.py                # Save metrics, predictions, datasets
│   │   ├── training.py              # Training pipeline (regression + classification)
│   │   └── visualization.py         # Plots for evaluation
│   ├── config/
│   │   ├── cli.py                   # Command-line argument parsers
│   │   ├── specs.py                 # TransformerSpec, EncodingSpec, TargetSpec
│   │   ├── transform.py             # Column transformation config
│   │   ├── engineering.py           # Feature engineering config
│   │   ├── encoding.py              # Encoding config (regression + classification)
│   │   └── tuning.py                # Hyperparameter grids
│   ├── preprocess.py                # Data preprocessing script
│   └── train.py                     # Unified training script (regression + classification)
├── data/                            # Data directory (created by preprocessing/training)
│   ├── processed/                   # Preprocessed data per task
│   │   ├── regression/
│   │   └── classification/
│   └── training/                    # Training artifacts per task
│       ├── regression/
│       └── classification/
├── docs/                            # Documentation
├── README.md
└── pyproject.toml
```
