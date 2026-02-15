#!/usr/bin/env python3
"""
Evaluation script for Qwen3-Reranker on Home Depot test set.

This script evaluates a fine-tuned reranker model:
- Computes relevance scores for test query-document pairs
- Calculates ranking metrics (NDCG@K, MAP, MRR)
- Performs error analysis by relevance level
- Saves predictions and detailed results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking_qwen.models.qwen_reranker import QwenReranker, create_data_collator
from src.ranking_qwen.data.reranker_dataset import RerankerDatasetPreparator
from src.ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-Reranker on Home Depot dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Base model name (for reference)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/home_depot.json",
        help="Path to Home Depot dataset"
    )
    parser.add_argument(
        "--test_split_path",
        type=str,
        default=None,
        help="Path to saved test split (optional)"
    )
    parser.add_argument(
        "--relevance_threshold",
        type=float,
        default=2.33,
        help="Threshold for binary relevance labels"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed (for reproducibility)"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help="K values for NDCG@K, Precision@K"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to CSV"
    )
    
    return parser.parse_args()


def compute_ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Compute NDCG@K for a single query.
    
    Args:
        relevances: True relevance scores in ranked order
        k: Cutoff position
    
    Returns:
        NDCG@K score
    """
    relevances = relevances[:k]
    
    if len(relevances) == 0:
        return 0.0
    
    # DCG
    dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    
    # IDCG
    ideal_rels = sorted(relevances, reverse=True)
    idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_map(relevances: List[float], threshold: float = 0.5) -> float:
    """
    Compute Mean Average Precision.
    
    Args:
        relevances: True relevance scores in ranked order
        threshold: Threshold for considering a document relevant
    
    Returns:
        MAP score
    """
    relevant = [1 if r >= threshold else 0 for r in relevances]
    
    if sum(relevant) == 0:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, rel in enumerate(relevant):
        if rel == 1:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    return np.mean(precisions) if precisions else 0.0


def compute_mrr(relevances: List[float], threshold: float = 0.5) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        relevances: True relevance scores in ranked order
        threshold: Threshold for considering a document relevant
    
    Returns:
        MRR score
    """
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(relevances: List[float], k: int, threshold: float = 0.5) -> float:
    """
    Compute Precision@K.
    
    Args:
        relevances: True relevance scores in ranked order
        k: Cutoff position
        threshold: Threshold for considering a document relevant
    
    Returns:
        Precision@K score
    """
    relevances = relevances[:k]
    relevant = sum(1 for r in relevances if r >= threshold)
    return relevant / k if k > 0 else 0.0


def evaluate_rankings(
    predictions_df: pd.DataFrame,
    k_values: List[int],
) -> Dict[str, float]:
    """
    Compute ranking metrics grouped by query.
    
    Args:
        predictions_df: DataFrame with columns: query, relevance, predicted_score
        k_values: List of K values for metrics
    
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing ranking metrics...")
    
    metrics = defaultdict(list)
    
    # Group by query
    for query, group in predictions_df.groupby('query'):
        # Sort by predicted score (descending)
        group_sorted = group.sort_values('predicted_score', ascending=False)
        true_relevances = group_sorted['relevance'].tolist()
        
        # Compute metrics for each K
        for k in k_values:
            ndcg = compute_ndcg_at_k(true_relevances, k)
            precision = compute_precision_at_k(true_relevances, k, threshold=2.33)
            
            metrics[f'ndcg@{k}'].append(ndcg)
            metrics[f'precision@{k}'].append(precision)
        
        # MAP and MRR (no K cutoff)
        map_score = compute_map(true_relevances, threshold=2.33)
        mrr_score = compute_mrr(true_relevances, threshold=2.33)
        
        metrics['map'].append(map_score)
        metrics['mrr'].append(mrr_score)
    
    # Average across queries
    avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    
    return avg_metrics


def evaluate_classification(
    predictions_df: pd.DataFrame,
    threshold: float = 2.33,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        predictions_df: DataFrame with relevance and predicted_score
        threshold: Relevance threshold for binary labels
    
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing classification metrics...")
    
    # Binary labels
    y_true = (predictions_df['relevance'] >= threshold).astype(int)
    y_pred = (predictions_df['predicted_score'] >= 0.5).astype(int)
    y_score = predictions_df['predicted_score']
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_true, y_score)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


def analyze_by_relevance_level(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by relevance level.
    
    Args:
        predictions_df: DataFrame with predictions
    
    Returns:
        DataFrame with analysis by relevance level
    """
    logger.info("Analyzing by relevance level...")
    
    # Group by relevance score
    analysis = predictions_df.groupby('relevance').agg({
        'predicted_score': ['count', 'mean', 'std'],
    }).round(4)
    
    analysis.columns = ['count', 'mean_score', 'std_score']
    analysis = analysis.reset_index()
    
    return analysis


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Qwen3-Reranker Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    model = QwenReranker(model_name=args.base_model_name)
    model.load_checkpoint(args.model_path)
    model.eval()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    data = load_dataset('json', data_files=args.data_path, split='train')
    df = data.to_pandas()
    
    # Prepare test dataset
    preparator = RerankerDatasetPreparator(
        tokenizer=model.get_tokenizer(),
        relevance_threshold=args.relevance_threshold,
    )
    
    # If test split is not provided, create it
    if args.test_split_path:
        logger.info(f"Loading test split from {args.test_split_path}")
        test_df = pd.read_csv(args.test_split_path)
    else:
        logger.info("Creating test split (15%)...")
        _, _, test_df = preparator.stratified_split_by_query(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=args.random_seed,
        )
    
    logger.info(f"Test set: {len(test_df):,} samples")
    
    # Prepare test dataset
    test_dataset = preparator.prepare_dataset(test_df, split_type='test')
    
    # Create dataloader
    collate_fn = create_data_collator(model.get_tokenizer())
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Run inference
    logger.info("=" * 60)
    logger.info("Running inference on test set...")
    logger.info("=" * 60)
    
    all_scores = []
    all_labels = []
    all_queries = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Forward pass
            outputs = model.get_model()(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            
            # Compute scores
            true_logits = logits[:, model.token_true_id]
            false_logits = logits[:, model.token_false_id]
            scores = torch.sigmoid(true_logits - false_logits)
            
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {(i+1) * args.batch_size} samples...")
    
    logger.info(f"Inference completed: {len(all_scores)} predictions")
    
    # Create predictions DataFrame
    predictions_df = test_df.copy()
    predictions_df['predicted_score'] = all_scores[:len(predictions_df)]
    
    # Evaluate rankings
    ranking_metrics = evaluate_rankings(predictions_df, args.k_values)
    
    # Evaluate classification
    classification_metrics = evaluate_classification(predictions_df, args.relevance_threshold)
    
    # Combine all metrics
    all_metrics = {**ranking_metrics, **classification_metrics}
    
    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    
    logger.info("\nRanking Metrics:")
    for k in args.k_values:
        logger.info(f"  NDCG@{k:2d}:      {ranking_metrics[f'ndcg@{k}']:.4f}")
        logger.info(f"  Precision@{k:2d}: {ranking_metrics[f'precision@{k}']:.4f}")
    logger.info(f"  MAP:          {ranking_metrics['map']:.4f}")
    logger.info(f"  MRR:          {ranking_metrics['mrr']:.4f}")
    
    logger.info("\nClassification Metrics:")
    logger.info(f"  Accuracy:  {classification_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {classification_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {classification_metrics['recall']:.4f}")
    logger.info(f"  F1:        {classification_metrics['f1']:.4f}")
    logger.info(f"  AUC:       {classification_metrics['auc']:.4f}")
    
    # Relevance level analysis
    relevance_analysis = analyze_by_relevance_level(predictions_df)
    logger.info("\nPerformance by Relevance Level:")
    logger.info("\n" + relevance_analysis.to_string(index=False))
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nSaved metrics to {results_path}")
    
    # Save relevance analysis
    analysis_path = os.path.join(args.output_dir, "relevance_analysis.csv")
    relevance_analysis.to_csv(analysis_path, index=False)
    logger.info(f"Saved relevance analysis to {analysis_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")
    
    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
