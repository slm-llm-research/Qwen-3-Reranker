#!/usr/bin/env python3
"""
Training script for Qwen3-Reranker fine-tuning on Home Depot dataset.

This script implements the training pipeline described in instruction_plan.md:
- Query-stratified dataset splitting
- Binary classification training with yes/no tokens
- Gradient accumulation and mixed precision
- Checkpoint saving and validation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking_qwen.models.qwen_reranker import QwenReranker, create_data_collator
from src.ranking_qwen.data.reranker_dataset import RerankerDatasetPreparator
from src.ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-Reranker on Home Depot dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        choices=["Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"],
        help="Qwen reranker model to use"
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash attention 2 (requires flash-attn package)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/home_depot.json",
        help="Path to Home Depot dataset"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Proportion for training set"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion for validation set"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion for test set"
    )
    parser.add_argument(
        "--relevance_threshold",
        type=float,
        default=2.33,
        help="Threshold for binary relevance labels"
    )
    parser.add_argument(
        "--max_description_tokens",
        type=int,
        default=350,
        help="Maximum tokens for product description"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps"],
        help="When to save checkpoints"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (if save_strategy=steps)"
    )
    
    return parser.parse_args()


def load_and_prepare_data(args) -> tuple:
    """
    Load Home Depot dataset and prepare for training.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    logger.info("=" * 60)
    logger.info("Loading and preparing dataset")
    logger.info("=" * 60)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    data = load_dataset('json', data_files=args.data_path, split='train')
    df = data.to_pandas()
    logger.info(f"Loaded {len(df):,} records")
    
    # Initialize reranker to get tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side='left',
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Initialize dataset preparator
    preparator = RerankerDatasetPreparator(
        tokenizer=tokenizer,
        relevance_threshold=args.relevance_threshold,
        max_description_tokens=args.max_description_tokens,
    )
    
    # Prepare full pipeline
    train_dataset, val_dataset, test_dataset = preparator.prepare_full_pipeline(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
    )
    
    return train_dataset, val_dataset, test_dataset, tokenizer


def create_dataloaders(
    train_dataset,
    val_dataset,
    tokenizer,
    batch_size: int,
) -> tuple:
    """
    Create DataLoaders for training and validation.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating DataLoaders...")
    
    collate_fn = create_data_collator(tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def train_epoch(
    model: QwenReranker,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for step, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)
        
        # Forward pass
        loss = model.compute_loss(input_ids, attention_mask, labels)
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def validate(
    model: QwenReranker,
    val_loader: DataLoader,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_probs = []
    all_labels = []
    
    for batch in val_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)
        
        # Forward pass
        loss = model.compute_loss(input_ids, attention_mask, labels)
        total_loss += loss.item()
        num_batches += 1
        
        # Get predictions for metrics
        outputs = model.get_model()(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        
        true_logits = logits[:, model.token_true_id]
        false_logits = logits[:, model.token_false_id]
        probs = torch.sigmoid(true_logits - false_logits)
        
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute accuracy
    import numpy as np
    predictions = (np.array(all_probs) >= 0.5).astype(int)
    accuracy = (predictions == np.array(all_labels)).mean()
    
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    logger.info("=" * 60)
    logger.info("Qwen3-Reranker Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save training config
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved training config to {config_path}")
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset, tokenizer = load_and_prepare_data(args)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        tokenizer,
        args.batch_size,
    )
    
    # Initialize model
    logger.info("=" * 60)
    logger.info("Initializing model")
    logger.info("=" * 60)
    
    model = QwenReranker(
        model_name=args.model_name,
        use_flash_attn=args.use_flash_attn,
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'=' * 60}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            epoch,
            args.log_interval,
        )
        
        logger.info(f"Epoch {epoch + 1} | Average train loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader)
        logger.info(
            f"Epoch {epoch + 1} | Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )
        
        # Save checkpoint
        if args.save_strategy == "epoch":
            checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            model.save_model(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_dir = os.path.join(args.output_dir, "best_model")
            model.save_model(best_model_dir)
            logger.info(f"Saved best model to {best_model_dir}")
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved in: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
