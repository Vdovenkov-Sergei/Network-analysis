# Network Analysis

A project for processing and analyzing `Head Hunter` job dataset. This project provides a structured pipeline for preprocessing job resume data, extracting features, and preparing data for machine learning models.

## Features

- **Structured Parsers**: Modular transformer classes for extracting and processing features from raw data
- **Pipeline Factory**: Easy creation of preprocessing pipelines for `numerical`, `categorical`, and `multi-label` features
- **Data Cleaning**: Utilities for cleaning raw and processed DataFrames
- **Command-Line Interface**: Simple CLI for processing CSV files and saving results

## Project Structure

```
.
├── parsers/              # Transformer classes
│   ├── __init__.py       # Package exports
│   ├── base/             # Base transformer classes
│   │   ├── __init__.py
│   │   └── core.py
│   └── transformers/     # Concrete transformer implementations
│       ├── __init__.py
│       ├── categorical.py
│       ├── multilabel.py
│       └── numerical.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── cleaning.py       # Data cleaning functions
│   ├── io.py             # Saving helpers
│   ├── pipeline_factory.py
│   └── processing.py
├── config/               # Configuration files
│   ├── __init__.py
│   └── parser.py
├── app.py                # Main CLI application
└── data/                 # Output directory (ignored by git)
```

## Installation

This project uses Poetry for dependency management. To install:

```bash
poetry install
```

## Usage

### Command-Line Interface

Process a CSV file and save the preprocessed data:

```bash
python app.py input_file.csv
```

Or with custom output directory:

```bash
python app.py input_file.csv -o output_dir
```

With a prefix for output files:

```bash
python app.py input_file.csv --prefix "processed_"
```

## Parsers

The `parsers` package contains transformer classes organized by functionality:

- **Base Classes**: Abstract base classes for creating custom transformers
- **Extractors**: Classes for extracting specific information (age, gender, experience, etc.)
- **Normalizers**: Classes for normalizing text data (employment types, schedules, etc.)
- **Categorizers**: Classes for categorizing text into predefined categories

All classes can be imported directly from the `parsers` package.

## Configuration

The `config/parser.py` file contains the configuration dictionary that defines how each column in the dataset should be processed. You can modify this file to adjust the preprocessing pipeline.

Notes:

- Columns named `Unnamed:*` are dropped automatically (CSV index artifacts).
- Extra columns are ignored by the preprocessing pipeline.

## Output

The processed data is saved as NumPy arrays in the output directory:

- `X_data.npy`: Feature matrix
- `y_data.npy`: Target variable
- `feature_names.npy`: Feature column names
- `target_names.npy`: Target column names

## Development

This project uses:

- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking
- **isort** for import sorting

Run formatting and linting:

```bash
poetry run black .
poetry run ruff check .
poetry run mypy .
poetry run isort .
```
