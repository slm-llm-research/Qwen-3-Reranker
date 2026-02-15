"""Reranker-specific evaluation metrics for ranking tasks."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict


def compute_dcg_at_k(relevances: List[float], k: Optional[int] = None) -> float:
    """
    Compute Discounted Cumulative Gain at k.
    
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
    
    # DCG = sum((2^rel - 1) / log2(i+2)) for i in 0..k-1
    dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    
    return dcg


def compute_ndcg_at_k(
    true_relevances: List[float],
    predicted_scores: List[float],
    k: Optional[int] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.
    
    Args:
        true_relevances: True relevance scores
        predicted_scores: Predicted scores (used for ranking)
        k: Rank cutoff (if None, use all)
    
    Returns:
        NDCG@k value (between 0 and 1)
    """
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_relevances = [true_relevances[i] for i in sorted_indices]
    
    # Calculate DCG
    dcg = compute_dcg_at_k(sorted_relevances, k)
    
    # Calculate IDCG (ideal DCG with perfect ranking)
    ideal_relevances = sorted(true_relevances, reverse=True)
    idcg = compute_dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_map(
    true_relevances: List[float],
    predicted_scores: List[float],
    relevance_threshold: float = 2.33,
) -> float:
    """
    Compute Mean Average Precision.
    
    Args:
        true_relevances: True relevance scores
        predicted_scores: Predicted scores (used for ranking)
        relevance_threshold: Threshold for binary relevance
    
    Returns:
        MAP value
    """
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_relevances = [true_relevances[i] for i in sorted_indices]
    
    # Binary relevance
    binary_relevances = [1 if r >= relevance_threshold else 0 for r in sorted_relevances]
    
    if sum(binary_relevances) == 0:
        return 0.0
    
    # Compute precisions at relevant positions
    precisions = []
    num_relevant = 0
    
    for i, rel in enumerate(binary_relevances):
        if rel == 1:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    return np.mean(precisions) if precisions else 0.0


def compute_mrr(
    true_relevances: List[float],
    predicted_scores: List[float],
    relevance_threshold: float = 2.33,
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        true_relevances: True relevance scores
        predicted_scores: Predicted scores (used for ranking)
        relevance_threshold: Threshold for binary relevance
    
    Returns:
        MRR value
    """
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_relevances = [true_relevances[i] for i in sorted_indices]
    
    # Find first relevant position
    for i, rel in enumerate(sorted_relevances):
        if rel >= relevance_threshold:
            return 1.0 / (i + 1)
    
    return 0.0


def compute_precision_at_k(
    true_relevances: List[float],
    predicted_scores: List[float],
    k: int,
    relevance_threshold: float = 2.33,
) -> float:
    """
    Compute Precision@K.
    
    Args:
        true_relevances: True relevance scores
        predicted_scores: Predicted scores (used for ranking)
        k: Rank cutoff
        relevance_threshold: Threshold for binary relevance
    
    Returns:
        Precision@K value
    """
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(predicted_scores)[::-1][:k]
    top_k_relevances = [true_relevances[i] for i in sorted_indices]
    
    # Count relevant documents
    num_relevant = sum(1 for r in top_k_relevances if r >= relevance_threshold)
    
    return num_relevant / k if k > 0 else 0.0


def compute_recall_at_k(
    true_relevances: List[float],
    predicted_scores: List[float],
    k: int,
    relevance_threshold: float = 2.33,
) -> float:
    """
    Compute Recall@K.
    
    Args:
        true_relevances: True relevance scores
        predicted_scores: Predicted scores (used for ranking)
        k: Rank cutoff
        relevance_threshold: Threshold for binary relevance
    
    Returns:
        Recall@K value
    """
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(predicted_scores)[::-1][:k]
    top_k_relevances = [true_relevances[i] for i in sorted_indices]
    
    # Count relevant documents in top-k
    num_relevant_retrieved = sum(1 for r in top_k_relevances if r >= relevance_threshold)
    
    # Count total relevant documents
    total_relevant = sum(1 for r in true_relevances if r >= relevance_threshold)
    
    return num_relevant_retrieved / total_relevant if total_relevant > 0 else 0.0


def evaluate_ranking_metrics(
    df: pd.DataFrame,
    true_col: str = 'relevance',
    pred_col: str = 'predicted_score',
    query_col: str = 'query',
    k_values: Optional[List[int]] = None,
    relevance_threshold: float = 2.33,
) -> Dict[str, float]:
    """
    Compute all ranking metrics for a dataset.
    
    Args:
        df: DataFrame with predictions
        true_col: Column name for true relevance scores
        pred_col: Column name for predicted scores
        query_col: Column name for query identifiers
        k_values: List of K values for metrics
        relevance_threshold: Threshold for binary relevance
    
    Returns:
        Dictionary of average metrics across queries
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    metrics = defaultdict(list)
    
    # Group by query and compute metrics
    for query, group in df.groupby(query_col):
        true_rels = group[true_col].tolist()
        pred_scores = group[pred_col].tolist()
        
        # Skip if no data
        if len(true_rels) == 0:
            continue
        
        # Compute metrics for each K
        for k in k_values:
            if len(true_rels) >= k:
                ndcg = compute_ndcg_at_k(true_rels, pred_scores, k)
                precision = compute_precision_at_k(true_rels, pred_scores, k, relevance_threshold)
                recall = compute_recall_at_k(true_rels, pred_scores, k, relevance_threshold)
                
                metrics[f'ndcg@{k}'].append(ndcg)
                metrics[f'precision@{k}'].append(precision)
                metrics[f'recall@{k}'].append(recall)
        
        # MAP and MRR (no K cutoff)
        map_score = compute_map(true_rels, pred_scores, relevance_threshold)
        mrr_score = compute_mrr(true_rels, pred_scores, relevance_threshold)
        
        metrics['map'].append(map_score)
        metrics['mrr'].append(mrr_score)
    
    # Average across queries
    avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    
    return avg_metrics


def print_ranking_metrics(metrics: Dict[str, float], title: str = "Ranking Metrics"):
    """
    Pretty print ranking metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    
    # Group metrics by type
    ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg')}
    precision_metrics = {k: v for k, v in metrics.items() if k.startswith('precision')}
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith('recall')}
    other_metrics = {k: v for k, v in metrics.items() if k in ['map', 'mrr']}
    
    if ndcg_metrics:
        print("\nNDCG Metrics:")
        for k, v in sorted(ndcg_metrics.items()):
            print(f"  {k.upper():15s}: {v:.4f}")
    
    if precision_metrics:
        print("\nPrecision Metrics:")
        for k, v in sorted(precision_metrics.items()):
            print(f"  {k.capitalize():15s}: {v:.4f}")
    
    if recall_metrics:
        print("\nRecall Metrics:")
        for k, v in sorted(recall_metrics.items()):
            print(f"  {k.capitalize():15s}: {v:.4f}")
    
    if other_metrics:
        print("\nOther Metrics:")
        for k, v in other_metrics.items():
            print(f"  {k.upper():15s}: {v:.4f}")
    
    print("=" * 60)


def analyze_error_distribution(
    df: pd.DataFrame,
    true_col: str = 'relevance',
    pred_col: str = 'predicted_score',
) -> pd.DataFrame:
    """
    Analyze error distribution across relevance levels.
    
    Args:
        df: DataFrame with predictions
        true_col: Column name for true relevance scores
        pred_col: Column name for predicted scores
    
    Returns:
        DataFrame with error analysis by relevance level
    """
    # Compute absolute error
    df_analysis = df.copy()
    df_analysis['abs_error'] = np.abs(df_analysis[true_col] - df_analysis[pred_col])
    
    # Group by relevance level
    analysis = df_analysis.groupby(true_col).agg({
        pred_col: ['count', 'mean', 'std', 'min', 'max'],
        'abs_error': ['mean', 'std'],
    }).round(4)
    
    analysis.columns = [
        'count', 'pred_mean', 'pred_std', 'pred_min', 'pred_max',
        'mae', 'mae_std'
    ]
    
    return analysis.reset_index()
