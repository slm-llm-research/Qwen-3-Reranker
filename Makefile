.PHONY: help install install-dev test lint format clean download analyze preprocess docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests with pytest
	pytest tests/ -v --cov=src/ranking_qwen --cov-report=html --cov-report=term

test-fast:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linting checks
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

clean:  ## Clean build artifacts and cache files
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

download:  ## Download the dataset
	python -m ranking_qwen.cli.download

analyze:  ## Analyze the dataset
	python -m ranking_qwen.cli.analyze --detailed

preprocess:  ## Preprocess the dataset
	python -m ranking_qwen.cli.preprocess --output data/preprocessed.parquet

docs:  ## Build documentation
	cd docs && make html

serve-docs:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server

check: lint test  ## Run all checks (lint + test)
