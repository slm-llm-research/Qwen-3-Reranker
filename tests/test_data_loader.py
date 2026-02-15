"""Tests for data loader module."""

import pytest
import pandas as pd
from pathlib import Path

from ranking_qwen.data.dataset_loader import HomeDepotDataset


class TestHomeDepotDataset:
    """Test cases for HomeDepotDataset class."""
    
    def test_initialization(self, temp_data_dir):
        """Test dataset initialization."""
        dataset = HomeDepotDataset(
            data_path=temp_data_dir,
            format="parquet",
        )
        assert dataset.data_path == temp_data_dir
        assert dataset.format == "parquet"
        assert dataset.df is None
    
    def test_load_from_dataframe(self, sample_dataset, temp_data_dir):
        """Test loading dataset from saved file."""
        # Save sample dataset
        file_path = temp_data_dir / "test.parquet"
        sample_dataset.to_parquet(file_path, index=False)
        
        # Load it back
        dataset = HomeDepotDataset(data_path=file_path, format="parquet")
        df = dataset.load()
        
        assert len(df) == len(sample_dataset)
        assert list(df.columns) == list(sample_dataset.columns)
    
    def test_get_statistics(self, sample_dataset, temp_data_dir):
        """Test getting dataset statistics."""
        file_path = temp_data_dir / "test.parquet"
        sample_dataset.to_parquet(file_path, index=False)
        
        dataset = HomeDepotDataset(data_path=file_path, format="parquet")
        dataset.load()
        
        stats = dataset.get_statistics()
        
        assert stats["total_records"] == len(sample_dataset)
        assert stats["unique_queries"] == sample_dataset["query"].nunique()
        assert stats["unique_products"] == sample_dataset["entity_id"].nunique()
    
    def test_filter_by_relevance(self, sample_dataset, temp_data_dir):
        """Test filtering by relevance score."""
        file_path = temp_data_dir / "test.parquet"
        sample_dataset.to_parquet(file_path, index=False)
        
        dataset = HomeDepotDataset(data_path=file_path, format="parquet")
        dataset.load()
        
        # Filter high relevance
        high_rel = dataset.filter_by_relevance(min_score=2.5)
        assert len(high_rel) == 2  # scores 3.0 and 2.5
        
        # Filter low relevance
        low_rel = dataset.filter_by_relevance(max_score=2.0)
        assert len(low_rel) == 3  # scores 2.0, 1.5, 1.0
    
    def test_filter_by_query(self, sample_dataset, temp_data_dir):
        """Test filtering by query."""
        file_path = temp_data_dir / "test.parquet"
        sample_dataset.to_parquet(file_path, index=False)
        
        dataset = HomeDepotDataset(data_path=file_path, format="parquet")
        dataset.load()
        
        # Filter by single query
        filtered = dataset.filter_by_query("test query")
        assert len(filtered) == 3
        
        # Filter by multiple queries
        filtered = dataset.filter_by_query(["test query", "another query"])
        assert len(filtered) == 5
    
    def test_split_train_test(self, sample_dataset, temp_data_dir):
        """Test train/test split."""
        file_path = temp_data_dir / "test.parquet"
        sample_dataset.to_parquet(file_path, index=False)
        
        dataset = HomeDepotDataset(data_path=file_path, format="parquet")
        dataset.load()
        
        train_df, test_df = dataset.split_train_test(
            test_size=0.4,
            random_state=42,
            group_by_query=False,
        )
        
        assert len(train_df) + len(test_df) == len(sample_dataset)
        assert len(test_df) == 2  # 40% of 5
    
    def test_get_top_queries(self, sample_dataset, temp_data_dir):
        """Test getting top queries."""
        file_path = temp_data_dir / "test.parquet"
        sample_dataset.to_parquet(file_path, index=False)
        
        dataset = HomeDepotDataset(data_path=file_path, format="parquet")
        dataset.load()
        
        top_queries = dataset.get_top_queries(n=2)
        
        assert len(top_queries) == 2
        assert top_queries.iloc[0] == 3  # "test query" appears 3 times
