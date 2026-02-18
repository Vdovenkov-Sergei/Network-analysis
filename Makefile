# Run in Git Bash

SRC_CODE_DIR = src
POETRY_CMD = poetry run

.PHONY: format lint style clean

format:
	$(POETRY_CMD) black $(SRC_CODE_DIR)
	$(POETRY_CMD) isort $(SRC_CODE_DIR)

lint:
	$(POETRY_CMD) ruff check $(SRC_CODE_DIR)
	$(POETRY_CMD) mypy $(SRC_CODE_DIR)

style: format lint

clean:
	find . -type d -name "__pycache__" ! -path "./.venv/*" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) ! -path "./.venv/*" -delete
	rm -rf .mypy_cache .ruff_cache