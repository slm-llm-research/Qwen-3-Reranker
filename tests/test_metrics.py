"""Tests for evaluation metrics module."""

import pytest
import numpy as np
import pandas as pd

from ranking_qwen.evaluation.metrics import RankingMetrics


class TestRankingMetrics:
    """Test cases for RankingMetrics class."""
    
    def test_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([3.0, 2.5, 2.0, 1.5, 1.0])
        y_pred = np.array([2.8, 2.4, 2.1, 1.6, 1.2])
        
        rmse = RankingMetrics.rmse(y_true, y_pred)
        assert rmse > 0
        assert rmse < 1  # Should be relatively small for these values
    
    def test_mae(self):
        """Test MAE calculation."""
        y_true = np.array([3.0, 2.5, 2.0, 1.5, 1.0])
        y_pred = np.array([2.8, 2.4, 2.1, 1.6, 1.2])
        
        mae = RankingMetrics.mae(y_true, y_pred)
        assert mae > 0
        assert mae < 1
    
    def test_dcg_at_k(self):
        """Test DCG@k calculation."""
        relevances = [3.0, 2.0, 1.0, 0.0]
        
        # DCG@2
        dcg = RankingMetrics.dcg_at_k(relevances, k=2)
        assert dcg > 0
        
        # DCG with all items
        dcg_all = RankingMetrics.dcg_at_k(relevances, k=None)
        assert dcg_all >= dcg
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        y_true = [3.0, 2.0, 1.0, 0.0]
        y_pred = [2.8, 1.9, 0.9, 0.1]  # Similar order
        
        ndcg = RankingMetrics.ndcg_at_k(y_true, y_pred, k=3)
        
        # NDCG should be between 0 and 1
        assert 0 <= ndcg <= 1
        
        # Perfect ranking should give NDCG = 1
        y_pred_perfect = [3.0, 2.0, 1.0, 0.0]
        ndcg_perfect = RankingMetrics.ndcg_at_k(y_true, y_pred_perfect, k=3)
        assert abs(ndcg_perfect - 1.0) < 0.01
    
    def test_evaluate_predictions(self, sample_dataset):
        """Test full evaluation pipeline."""
        # Add predicted relevance scores
        df = sample_dataset.copy()
        df["predicted_relevance"] = df["relevance"] + np.random.normal(0, 0.1, len(df))
        
        metrics = RankingMetrics.evaluate_predictions(
            df,
            y_true_col="relevance",
            y_pred_col="predicted_relevance",
            group_by_query=True,
            k_values=[1, 3],
        )
        
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "ndcg@1" in metrics
        assert "ndcg@3" in metrics
