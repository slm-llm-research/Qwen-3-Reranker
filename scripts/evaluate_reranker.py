#!/usr/bin/env python3
"""
Evaluation script for Qwen3-Reranker using HuggingFace models.

Evaluates both base (pretrained) and fine-tuned models, then compares metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ranking_qwen.data import RerankerDatasetPreparator
from src.ranking_qwen.models import create_data_collator
from src.ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-Reranker models")
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Base model name from HuggingFace"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/home_depot.json",
        help="Path to dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash attention"
    )
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    token_true_id: int,
    token_false_id: int,
    batch_size: int,
    max_length: int,
    model_name: str
):
    """Evaluate a single model and return predictions."""
    logger.info(f"\nEvaluating: {model_name}")
    logger.info("=" * 60)
    
    model.eval()
    
    # Create dataloader
    data_collator = create_data_collator(tokenizer, max_length=max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    
    all_scores = []
    all_labels = []
    
    logger.info(f"Running inference on {len(test_dataset)} samples...")
    
    for i, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        
        # Extract yes/no logits and compute scores
        true_logits = logits[:, token_true_id]
        false_logits = logits[:, token_false_id]
        scores = torch.sigmoid(true_logits - false_logits)
        
        # Convert to FP32 before numpy (BF16 not supported in numpy)
        all_scores.extend(scores.float().cpu().numpy().tolist())
        all_labels.extend(batch['labels'].float().cpu().numpy().tolist())
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {(i + 1) * batch_size}/{len(test_dataset)} samples")
    
    logger.info(f"✓ Completed: {len(all_scores)} predictions")
    
    return all_scores, all_labels


def compute_ranking_metrics(predictions_df, k_values=[1, 3, 5, 10]):
    """Compute ranking metrics grouped by query."""
    metrics = defaultdict(list)
    
    for query, group in predictions_df.groupby('query'):
        # Sort by predicted score (descending)
        group_sorted = group.sort_values('predicted_score', ascending=False)
        true_relevances = group_sorted['relevance'].tolist()
        
        # NDCG@K
        for k in k_values:
            if len(true_relevances) >= k:
                rels = true_relevances[:k]
                dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(rels))
                ideal_rels = sorted(true_relevances, reverse=True)[:k]
                idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                metrics[f'ndcg@{k}'].append(ndcg)
        
        # MAP
        binary_rels = [1 if r >= 2.33 else 0 for r in true_relevances]
        if sum(binary_rels) > 0:
            precisions = []
            num_rel = 0
            for i, rel in enumerate(binary_rels):
                if rel == 1:
                    num_rel += 1
                    precisions.append(num_rel / (i + 1))
            map_score = np.mean(precisions)
            metrics['map'].append(map_score)
        
        # MRR
        for i, rel in enumerate(true_relevances):
            if rel >= 2.33:
                metrics['mrr'].append(1.0 / (i + 1))
                break
    
    return {k: np.mean(v) for k, v in metrics.items()}


def compute_classification_metrics(predictions_df, threshold=2.33):
    """Compute binary classification metrics."""
    y_true = (predictions_df['relevance'] >= threshold).astype(int)
    y_pred = (predictions_df['predicted_score'] >= 0.5).astype(int)
    y_score = predictions_df['predicted_score']
    
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


def print_metrics(metrics, model_name):
    """Pretty print metrics."""
    print(f"\n{'=' * 60}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'=' * 60}")
    
    print("\n📊 Ranking Metrics:")
    for k in [1, 3, 5, 10]:
        if f'ndcg@{k}' in metrics:
            print(f"  NDCG@{k:2d}:  {metrics[f'ndcg@{k}']:.4f}")
    if 'map' in metrics:
        print(f"  MAP:      {metrics['map']:.4f}")
    if 'mrr' in metrics:
        print(f"  MRR:      {metrics['mrr']:.4f}")
    
    print("\n🎯 Classification Metrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1:        {metrics.get('f1', 0):.4f}")
    print(f"  AUC:       {metrics.get('auc', 0):.4f}")
    print(f"{'=' * 60}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Qwen3-Reranker Evaluation: Base vs Fine-Tuned")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Fine-tuned model: {args.finetuned_model}")
    logger.info(f"Data: {args.data_path}")
    logger.info("=" * 60)
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load dataset and prepare test split
    logger.info("\n1. Loading and preparing dataset...")
    data = load_dataset('json', data_files=args.data_path, split='train')
    df = data.to_pandas()
    
    # Load base tokenizer for data preparation
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side='left',
        trust_remote_code=True
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    preparator = RerankerDatasetPreparator(
        tokenizer=base_tokenizer,
        relevance_threshold=2.33,
        max_description_tokens=350,
    )
    
    # Create test split (same split as training used)
    _, _, test_dataset = preparator.prepare_full_pipeline(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    
    # Get test dataframe for metrics
    test_df = df[df['query'].isin(test_dataset['query'])]
    logger.info(f"Test set: {len(test_dataset)} samples, {test_df['query'].nunique()} queries")
    
    # Get token IDs (same for base and fine-tuned)
    token_true_id = base_tokenizer.convert_tokens_to_ids('yes')
    token_false_id = base_tokenizer.convert_tokens_to_ids('no')
    
    logger.info(f"Token 'yes' ID: {token_true_id}")
    logger.info(f"Token 'no' ID: {token_false_id}")
    
    # 2. Evaluate BASE model
    logger.info("\n2. Evaluating BASE model (pretrained)...")
    
    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'device_map': 'auto',
    }
    if args.use_flash_attn:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    
    base_scores, base_labels = evaluate_model(
        base_model,
        base_tokenizer,
        test_dataset,
        token_true_id,
        token_false_id,
        args.batch_size,
        args.max_length,
        "BASE (pretrained)"
    )
    
    # Clear base model from memory
    del base_model
    torch.cuda.empty_cache()
    
    # 3. Evaluate FINE-TUNED model
    logger.info("\n3. Evaluating FINE-TUNED model...")
    
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model,
        **model_kwargs
    )
    finetuned_tokenizer = AutoTokenizer.from_pretrained(
        args.finetuned_model,
        padding_side='left',
        trust_remote_code=True
    )
    
    finetuned_scores, finetuned_labels = evaluate_model(
        finetuned_model,
        finetuned_tokenizer,
        test_dataset,
        token_true_id,
        token_false_id,
        args.batch_size,
        args.max_length,
        "FINE-TUNED"
    )
    
    # 4. Compute metrics for both models
    logger.info("\n4. Computing metrics...")
    
    # Convert test_dataset to dataframe (it has all the original columns)
    test_results_df = test_dataset.to_pandas()
    
    # Add predictions
    base_predictions = test_results_df[['query', 'relevance']].copy()
    base_predictions['predicted_score'] = base_scores
    
    finetuned_predictions = test_results_df[['query', 'relevance']].copy()
    finetuned_predictions['predicted_score'] = finetuned_scores
    
    # Ranking metrics
    base_ranking = compute_ranking_metrics(base_predictions)
    finetuned_ranking = compute_ranking_metrics(finetuned_predictions)
    
    # Classification metrics
    base_classification = compute_classification_metrics(base_predictions)
    finetuned_classification = compute_classification_metrics(finetuned_predictions)
    
    # Combine metrics
    base_metrics = {**base_ranking, **base_classification}
    finetuned_metrics = {**finetuned_ranking, **finetuned_classification}
    
    # 5. Print results
    print_metrics(base_metrics, "BASE MODEL (Pretrained)")
    print_metrics(finetuned_metrics, "FINE-TUNED MODEL")
    
    # 6. Print improvement comparison
    print(f"\n{'=' * 60}")
    print("📈 IMPROVEMENT (Fine-tuned vs Base)")
    print(f"{'=' * 60}")
    
    improvements = {}
    for key in base_metrics:
        if key in finetuned_metrics:
            base_val = base_metrics[key]
            finetuned_val = finetuned_metrics[key]
            improvement = ((finetuned_val - base_val) / base_val * 100) if base_val > 0 else 0
            improvements[key] = improvement
            
            symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"  {symbol} {key.upper():15s}: {improvement:+.2f}% "
                  f"({base_val:.4f} → {finetuned_val:.4f})")
    
    print(f"{'=' * 60}")
    
    # 7. Save results
    results = {
        'base_model': {
            'name': args.base_model,
            'metrics': base_metrics
        },
        'finetuned_model': {
            'name': args.finetuned_model,
            'metrics': finetuned_metrics
        },
        'improvements': improvements
    }
    
    results_path = f"{args.output_dir}/comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Results saved to: {results_path}")
    
    # Save predictions
    base_predictions.to_csv(f"{args.output_dir}/base_predictions.csv", index=False)
    finetuned_predictions.to_csv(f"{args.output_dir}/finetuned_predictions.csv", index=False)
    logger.info(f"💾 Predictions saved to: {args.output_dir}/")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
