"""Ranking evaluation metrics."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class RankingMetrics:
    """
    Calculate evaluation metrics for ranking models.
    
    This class provides implementations of common ranking metrics including
    RMSE, MAE, NDCG, and others.
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: True relevance scores
            y_pred: Predicted relevance scores
        
        Returns:
            RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True relevance scores
            y_pred: Predicted relevance scores
        
        Returns:
            MAE value
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def dcg_at_k(relevances: List[float], k: Optional[int] = None) -> float:
        """
        Calculate Discounted Cumulative Gain at k.
        
        Args:
            relevances: List of relevance scores in ranked order
            k: Rank cutoff (if None, use all)
        
        Returns:
            DCG@k value
        """
        if k is not None:
            relevances = relevances[:k]
        
        if len(relevances) == 0:
            return 0.0
        
        # DCG = sum(rel_i / log2(i+1)) for i in 1..k
        dcg = relevances[0]  # First item has no discount
        for i, rel in enumerate(relevances[1:], start=2):
            dcg += rel / np.log2(i + 1)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(
        y_true: List[float],
        y_pred: List[float],
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            y_true: True relevance scores
            y_pred: Predicted relevance scores (used for ranking)
            k: Rank cutoff (if None, use all)
        
        Returns:
            NDCG@k value (between 0 and 1)
        """
        # Sort by predicted scores
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_true = [y_true[i] for i in sorted_indices]
        
        # Calculate DCG
        dcg = RankingMetrics.dcg_at_k(sorted_true, k)
        
        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_sorted = sorted(y_true, reverse=True)
        idcg = RankingMetrics.dcg_at_k(ideal_sorted, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def evaluate_predictions(
        df: pd.DataFrame,
        y_true_col: str = "relevance",
        y_pred_col: str = "predicted_relevance",
        group_by_query: bool = True,
        k_values: Optional[List[int]] = None,
    ) -> dict:
        """
        Evaluate predictions with multiple metrics.
        
        Args:
            df: DataFrame containing true and predicted values
            y_true_col: Column name for true relevance scores
            y_pred_col: Column name for predicted scores
            group_by_query: Whether to calculate query-level metrics
            k_values: List of k values for NDCG@k (e.g., [1, 3, 5, 10])
        
        Returns:
            Dictionary of metric values
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        results = {}
        
        # Overall metrics
        y_true = df[y_true_col].values
        y_pred = df[y_pred_col].values
        
        results["rmse"] = RankingMetrics.rmse(y_true, y_pred)
        results["mae"] = RankingMetrics.mae(y_true, y_pred)
        
        logger.info(f"Overall RMSE: {results['rmse']:.4f}")
        logger.info(f"Overall MAE: {results['mae']:.4f}")
        
        # Query-level metrics
        if group_by_query and "query" in df.columns:
            ndcg_scores = {}
            
            for k in k_values:
                ndcg_list = []
                
                for query, group in df.groupby("query"):
                    if len(group) > 0:
                        ndcg = RankingMetrics.ndcg_at_k(
                            group[y_true_col].tolist(),
                            group[y_pred_col].tolist(),
                            k=k,
                        )
                        ndcg_list.append(ndcg)
                
                avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
                ndcg_scores[f"ndcg@{k}"] = avg_ndcg
                logger.info(f"Average NDCG@{k}: {avg_ndcg:.4f}")
            
            results.update(ndcg_scores)
        
        return results
    
    @staticmethod
    def print_metrics(metrics: dict) -> None:
        """
        Pretty print evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        logger.info("=" * 60)
        logger.info("Evaluation Metrics")
        logger.info("=" * 60)
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name.upper():20s}: {value:.4f}")
        
        logger.info("=" * 60)
