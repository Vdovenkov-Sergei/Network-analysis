# Head Hunter Resume Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Poetry](https://img.shields.io/badge/Poetry-60A5FA?style=for-the-badge&logo=poetry&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-EC407A?style=for-the-badge&logo=seaborn&logoColor=white)

A machine learning project for Head Hunter resume dataset: salary prediction (`regression`) and developer level classification (`junior` / `middle` / `senior`). Includes a complete pipeline from data preprocessing to training multiple models.

## Requirements

- **Python**: 3.10 or higher
- **Poetry**: for dependency management and virtual environment
- **Disk**: ~100 MB for the project and dependencies; more if storing large datasets under `data/`
- **RAM**: 2 GB minimum; 4 GB+ recommended for training with hyperparameter tuning

## Features

- **Data Preprocessing**: Modular transformer classes for feature extraction and processing
- **Regression Models**: _Ridge Regression_, _Random Forest_, _Gradient Boosting_ (`salary prediction`)
- **Classification Models**: _Logistic Regression_, _Random Forest_, _Gradient Boosting_ (`developer level`)
- **Hyperparameter Tuning**: Automatic tuning via **GridSearchCV** with cross-validation
- **Visualization**: Prediction plots, class balance, classification report heatmaps
- **Evaluation**: _R²_, _MSE_, _RMSE_, _MAE_ for `regression` and  _precision_, _recall_, _f1-score_ for `classification`

## Documentation

- [Project Structure](docs/project_structure.md)
- [Quick Start](docs/quick_start.md)
- [Preprocessing](docs/preprocessing.md)
- [Training](docs/training.md)
- [Model Performance & Results](docs/results.md)
- [Development](docs/development.md)
