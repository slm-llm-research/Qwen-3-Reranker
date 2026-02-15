#!/bin/bash
# Run comprehensive test suite

set -e

echo "======================================"
echo "Running Test Suite"
echo "======================================"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run linting
echo ""
echo "Running code quality checks..."
echo "--------------------------------------"
flake8 src/ tests/ || echo "⚠ Flake8 found issues"
black --check src/ tests/ || echo "⚠ Black formatting needed"
isort --check-only src/ tests/ || echo "⚠ Isort formatting needed"

# Run type checking
echo ""
echo "Running type checking..."
echo "--------------------------------------"
mypy src/ || echo "⚠ Mypy found type issues"

# Run tests with coverage
echo ""
echo "Running unit tests..."
echo "--------------------------------------"
pytest tests/ -v --cov=src/ranking_qwen --cov-report=html --cov-report=term

echo ""
echo "======================================"
echo "✓ Test suite completed!"
echo "======================================"
echo ""
echo "Coverage report available at: htmlcov/index.html"
