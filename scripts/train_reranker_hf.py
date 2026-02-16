#!/usr/bin/env python3
"""
Training script for Qwen3-Reranker using HuggingFace Trainer.

This is MUCH simpler than raw PyTorch and handles:
- Mixed precision automatically
- Gradient accumulation
- Checkpointing
- Memory optimization
- Distributed training
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ranking_qwen.data import RerankerDatasetPreparator
from src.ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class QwenRerankerModel(torch.nn.Module):
    """Wrapper model for Qwen3-Reranker with HuggingFace Trainer."""
    
    def __init__(self, model_name: str):
        super().__init__()
        # Load in FP32 - Trainer will handle FP16/BF16 conversion automatically
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get yes/no token IDs
        self.token_true_id = self.tokenizer.convert_tokens_to_ids('yes')
        self.token_false_id = self.tokenizer.convert_tokens_to_ids('no')
        
        logger.info(f"Token 'yes' ID: {self.token_true_id}")
        logger.info(f"Token 'no' ID: {self.token_false_id}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with HuggingFace Trainer.
        
        Returns dict with 'loss' key.
        """
        # Forward through model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Extract yes/no logits
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        
        # Compute logit difference
        logit_diff = true_logits - false_logits
        
        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            logit_diff,
            labels.to(logit_diff.dtype)
        )
        
        return {"loss": loss}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Qwen3-Reranker using HuggingFace Trainer"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        choices=["Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"],
        help="Model to use"
    )
    
    # Logging arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen3-reranker-finetuning",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name (default: auto-generated)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/home_depot.json",
        help="Path to dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples for quick testing (None = use all)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints_hf",
        help="Output directory"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup WandB if requested
    if args.use_wandb:
        import wandb
        import os
        
        # Set WandB API key from environment or default
        wandb_token = os.environ.get("WANDB_API_KEY")
        if wandb_token:
            wandb.login(key=wandb_token)
            logger.info("✓ WandB authentication successful")
        
        # Initialize WandB run
        run_name = args.wandb_run_name or f"{args.model_name.split('/')[-1]}-lr{args.learning_rate}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        logger.info(f"✓ WandB run initialized: {run_name}")
    
    logger.info("=" * 60)
    logger.info("Qwen3-Reranker Training (HuggingFace Trainer)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
    
    # 1. Load dataset
    logger.info("\n1. Loading dataset...")
    data = load_dataset('json', data_files=args.data_path, split='train')
    df = data.to_pandas()
    
    if args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
        logger.info(f"   Using {len(df)} samples for testing")
    
    # 2. Initialize model and tokenizer
    logger.info("\n2. Loading model...")
    model = QwenRerankerModel(args.model_name)
    tokenizer = model.tokenizer
    
    # 3. Prepare datasets
    logger.info("\n3. Preparing datasets...")
    preparator = RerankerDatasetPreparator(
        tokenizer=tokenizer,
        relevance_threshold=2.33,
        max_description_tokens=350,
    )
    
    train_dataset, val_dataset, test_dataset = preparator.prepare_full_pipeline(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    
    # 4. Setup Trainer
    logger.info("\n4. Setting up Trainer...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Mixed precision (automatic!) - BF16 is more stable on A100
        bf16=True,  # Use bf16 on A100 (more stable than fp16)
        
        # Logging
        logging_steps=50,
        logging_dir=f"{args.output_dir}/logs",
        
        # Evaluation
        eval_strategy="epoch",
        save_strategy="no",  # Don't auto-save during training (we'll save at the end)
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="loss"
        
        # Memory optimization
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        
        # Other
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name if args.use_wandb else None
    )
    
    # Data collator
    from src.ranking_qwen.models import create_data_collator
    data_collator = create_data_collator(tokenizer)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 5. Train
    logger.info("\n5. Starting training...")
    logger.info(f"   Total train samples: {len(train_dataset)}")
    logger.info(f"   Total val samples: {len(val_dataset)}")
    logger.info(f"   Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info("")
    
    trainer.train()
    
    # 6. Save final model
    logger.info("\n6. Saving final model...")
    import os
    output_path = f"{args.output_dir}/final_model"
    os.makedirs(output_path, exist_ok=True)
    
    # Save using model's save_pretrained (handles tied weights correctly)
    model.model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Model saved to: {args.output_dir}/final_model")
    logger.info("=" * 60)
    
    # Finish WandB run
    if args.use_wandb:
        import wandb
        wandb.finish()
        logger.info("✓ WandB run finished")


if __name__ == "__main__":
    main()
