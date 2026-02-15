"""Tests for data preprocessor module."""

import pytest
import pandas as pd

from ranking_qwen.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
    
    def test_clean_text(self, sample_dataset):
        """Test text cleaning."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_text(sample_dataset)
        
        # Check that all text columns are strings
        assert df_clean["query"].dtype == object
        assert df_clean["name"].dtype == object
        assert df_clean["description"].dtype == object
        
        # Check no null values
        assert not df_clean["query"].isnull().any()
        assert not df_clean["name"].isnull().any()
        assert not df_clean["description"].isnull().any()
    
    def test_create_combined_text(self, sample_dataset):
        """Test combined text creation."""
        preprocessor = DataPreprocessor()
        df_combined = preprocessor.create_combined_text(sample_dataset)
        
        assert "combined_text" in df_combined.columns
        assert df_combined["combined_text"].str.contains("[SEP]").all()
    
    def test_create_relevance_labels_binary(self, sample_dataset):
        """Test binary relevance label creation."""
        preprocessor = DataPreprocessor()
        df_labels = preprocessor.create_relevance_labels(
            sample_dataset,
            num_classes=2,
        )
        
        assert "relevance_label" in df_labels.columns
        assert "relevance_label_name" in df_labels.columns
        assert set(df_labels["relevance_label"].unique()).issubset({0, 1})
    
    def test_create_relevance_labels_multiclass(self, sample_dataset):
        """Test multiclass relevance label creation."""
        preprocessor = DataPreprocessor()
        df_labels = preprocessor.create_relevance_labels(
            sample_dataset,
            num_classes=3,
        )
        
        assert "relevance_label" in df_labels.columns
        assert set(df_labels["relevance_label"].unique()).issubset({0, 1, 2})
    
    def test_create_features(self, sample_dataset):
        """Test feature creation."""
        preprocessor = DataPreprocessor()
        df_features = preprocessor.create_features(sample_dataset)
        
        expected_features = [
            "query_length",
            "name_length",
            "description_length",
            "query_word_count",
            "name_word_count",
            "query_in_name",
        ]
        
        for feature in expected_features:
            assert feature in df_features.columns
    
    def test_preprocess_full_pipeline(self, sample_dataset):
        """Test full preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess(
            sample_dataset,
            clean_text=True,
            create_combined=True,
            create_labels=True,
            num_classes=3,
            create_features=True,
        )
        
        # Check that all processing steps were applied
        assert "combined_text" in df_processed.columns
        assert "relevance_label" in df_processed.columns
        assert "query_length" in df_processed.columns
        assert len(df_processed) == len(sample_dataset)
