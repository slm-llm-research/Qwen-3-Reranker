#!/usr/bin/env python3
"""
Training script for Qwen3-Reranker using pure HuggingFace Trainer.

Clean, well-organized code using only HuggingFace components.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ranking_qwen.data import RerankerDatasetPreparator
from src.ranking_qwen.models import create_data_collator
from src.ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class RerankerTrainer(Trainer):
    """Custom Trainer for Qwen3-Reranker with binary classification loss."""
    
    def __init__(self, token_true_id: int, token_false_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_true_id = token_true_id
        self.token_false_id = token_false_id
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute binary cross-entropy loss on yes/no tokens.
        
        This is where the reranker-specific logic lives.
        """
        # Extract inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Extract yes/no logits
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        
        # Compute logit difference
        logit_diff = true_logits - false_logits
        
        # Binary cross-entropy with logits (numerically stable)
        loss = F.binary_cross_entropy_with_logits(
            logit_diff,
            labels.to(logit_diff.dtype)
        )
        
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
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
        help="Max samples for testing (None = use all)"
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
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (reduces memory)"
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile for speedup (requires PyTorch 2.0+)"
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use flash attention 2 (requires flash-attn package)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048, model supports up to 32768)"
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
    
    return parser.parse_args()


def setup_wandb(args: argparse.Namespace):
    """Setup Weights & Biases logging."""
    if not args.use_wandb:
        return
    
    import wandb
    
    # Login with API key from environment
    wandb_token = os.environ.get("WANDB_API_KEY")
    if wandb_token:
        wandb.login(key=wandb_token)
        logger.info("✓ WandB authentication successful")
    
    # Initialize run
    run_name = args.wandb_run_name or f"{args.model_name.split('/')[-1]}-lr{args.learning_rate}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )
    logger.info(f"✓ WandB run initialized: {run_name}")


def load_model_and_tokenizer(
    model_name: str,
    use_flash_attn: bool = False,
    use_compile: bool = False
) -> tuple:
    """Load model and tokenizer with optional optimizations."""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left',
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model in BF16 for A100
    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
    }
    
    # Add flash attention if requested
    if use_flash_attn:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
        logger.info("  Using Flash Attention 2")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Compile model for speedup (PyTorch 2.0+)
    if use_compile:
        logger.info("  Compiling model with torch.compile...")
        model = torch.compile(model)
        logger.info("  ✓ Model compiled")
    
    # Get yes/no token IDs
    token_true_id = tokenizer.convert_tokens_to_ids('yes')
    token_false_id = tokenizer.convert_tokens_to_ids('no')
    
    logger.info(f"  Token 'yes' ID: {token_true_id}")
    logger.info(f"  Token 'no' ID: {token_false_id}")
    
    return model, tokenizer, token_true_id, token_false_id


def prepare_datasets(args: argparse.Namespace, tokenizer) -> tuple:
    """Load and prepare datasets."""
    logger.info("Loading dataset...")
    data = load_dataset('json', data_files=args.data_path, split='train')
    df = data.to_pandas()
    
    if args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
        logger.info(f"Using {len(df)} samples for testing")
    
    # Prepare datasets
    logger.info("Preparing datasets...")
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
    
    return train_dataset, val_dataset, test_dataset


def create_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Create TrainingArguments configuration."""
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Mixed precision
        bf16=True,  # BF16 for A100
        
        # Logging
        logging_steps=50,
        logging_dir=f"{args.output_dir}/logs",
        
        # Evaluation and checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Memory and speed optimization
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster CPU→GPU transfer
        
        # Other
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name if args.use_wandb else None,
    )


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup WandB
    setup_wandb(args)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Qwen3-Reranker Training (HuggingFace Trainer)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size} × {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")
    logger.info(f"WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load model and tokenizer
    logger.info("\n1. Loading model and tokenizer...")
    model, tokenizer, token_true_id, token_false_id = load_model_and_tokenizer(
        args.model_name,
        use_flash_attn=args.use_flash_attn,
        use_compile=args.use_compile
    )
    
    # 2. Prepare datasets
    logger.info("\n2. Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(args, tokenizer)
    
    # 3. Setup Trainer
    logger.info("\n3. Setting up Trainer...")
    training_args = create_training_args(args)
    data_collator = create_data_collator(tokenizer, max_length=args.max_length)
    logger.info(f"  Max sequence length: {args.max_length}")
    
    trainer = RerankerTrainer(
        token_true_id=token_true_id,
        token_false_id=token_false_id,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 4. Train
    logger.info("\n4. Starting training...")
    logger.info(f"   Total train samples: {len(train_dataset):,}")
    logger.info(f"   Total val samples: {len(val_dataset):,}")
    logger.info(f"   Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info("")
    
    trainer.train()
    
    # 5. Save best model
    logger.info("\n5. Saving best model...")
    output_path = f"{args.output_dir}/best_model"
    os.makedirs(output_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    logger.info(f"Best model saved to: {output_path}")
    
    # Finish WandB
    if args.use_wandb:
        import wandb
        wandb.finish()
        logger.info("✓ WandB run finished")
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Checkpoints saved in: {args.output_dir}")
    logger.info(f"Best model: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
