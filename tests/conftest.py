"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "id": [1, 2, 3, 4, 5],
        "entity_id": [100, 101, 102, 100, 103],
        "name": [
            "Product A",
            "Product B",
            "Product C",
            "Product A",
            "Product D",
        ],
        "query": [
            "test query",
            "test query",
            "another query",
            "another query",
            "test query",
        ],
        "relevance": [3.0, 2.5, 2.0, 1.5, 1.0],
        "description": [
            "Description for product A",
            "Description for product B",
            "Description for product C",
            "Description for product A again",
            "Description for product D",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return {
        "data": {
            "data_dir": "data",
            "dataset_name": "bstds/home_depot",
            "default_format": "parquet",
        },
        "preprocessing": {
            "clean_text": True,
            "create_combined_text": True,
        },
        "split": {
            "test_size": 0.2,
            "random_state": 42,
        },
    }
