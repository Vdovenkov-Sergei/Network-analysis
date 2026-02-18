# Preprocessing

The preprocessing script loads raw Head Hunter CSV data, applies transformations and encoding, and saves feature matrix (`X`) and target (`y`) as NumPy arrays.

## Usage

```bash
poetry run python src/preprocess.py <input_file> -o data [--task regression|classification] [--prefix PREFIX] [--log-level LEVEL]
```

- **input_file**: Path to the input CSV (e.g. `data/hh.csv`)
- **-o, --output-dir**: Root output directory (default: `data`). Output is written to `{output_dir}/processed/{task}`.
- **--task**: `regression` (salary) or `classification` (developer level). Default: `regression`.
- **--prefix**: Optional prefix for output filenames.
- **--log-level**: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`.

## Output Files

Under `{output_dir}/processed/{task}/`:

| File | Description |
|------|-------------|
| `X_data.npy` | Feature matrix |
| `y_data.npy` | Target variable |
| `feature_columns.npy` | Feature column names |
| `target_column.npy` | Target column name |
| `class_labels.npy` | Class labels (classification only) |

## Features

The pipeline extracts the following features:

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

**Regression target**: `salary` (Russian Rubles)

**Classification target**: `developer_level` — junior / middle / senior (IT developers only, inferred from job title keywords and experience)
