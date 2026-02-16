#!/usr/bin/env python3
"""
Example: Quick start for training Qwen3-Reranker using HuggingFace Trainer.

This script demonstrates minimal setup for training a reranker model.
Uses HuggingFace Trainer for simplicity and robustness.
"""

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking_qwen.data import RerankerDatasetPreparator
from src.ranking_qwen.models import create_data_collator


class QwenRerankerModel(torch.nn.Module):
    """Wrapper model for Qwen3-Reranker with HuggingFace Trainer."""
    
    def __init__(self, model_name: str):
        super().__init__()
        # Load in FP32 - Trainer will handle FP16 conversion automatically
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
        
        print(f"  Token 'yes' ID: {self.token_true_id}")
        print(f"  Token 'no' ID: {self.token_false_id}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - returns dict with 'loss' key."""
        # Forward through model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
        
        return {"loss": loss}


def main():
    """Train a small reranker model using HuggingFace Trainer."""
    
    print("=" * 60)
    print("Qwen3-Reranker Training Example (HuggingFace Trainer)")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    data = load_dataset('json', data_files='data/home_depot.json', split='train')
    df = data.to_pandas()
    
    # Take a small subset for quick training
    df_small = df.sample(n=1000, random_state=42)
    print(f"   Using {len(df_small)} samples for quick demo")
    
    # 2. Initialize model
    print("\n2. Loading Qwen3-Reranker-0.6B...")
    model = QwenRerankerModel("Qwen/Qwen3-Reranker-0.6B")
    tokenizer = model.tokenizer
    
    # 3. Prepare data
    print("\n3. Preparing dataset...")
    preparator = RerankerDatasetPreparator(
        tokenizer=tokenizer,
        relevance_threshold=2.33,
        max_description_tokens=350,
    )
    
    train_dataset, val_dataset, test_dataset = preparator.prepare_full_pipeline(
        df_small,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    # 4. Setup Trainer
    print("\n4. Setting up HuggingFace Trainer...")
    
    training_args = TrainingArguments(
        output_dir="models/example_checkpoint",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Mixed precision (automatic!) - BF16 is more stable on A100
        bf16=True,  # Use bf16 on A100 (more stable than fp16)
        
        # Logging
        logging_steps=10,
        
        # Evaluation
        eval_strategy="epoch",
        save_strategy="no",  # Don't auto-save during training (we'll save manually)
        save_total_limit=2,
        load_best_model_at_end=False
        
        # Other
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Data collator
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
    print("\n5. Training...")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Effective batch size: {2 * 4}")
    print()
    
    trainer.train()
    
    # 6. Save model (manually to avoid tied weights issue)
    print("\n6. Saving model...")
    output_path = "models/example_checkpoint/final_model"
    
    # Save model weights only (not using safetensors)
    import os
    os.makedirs(output_path, exist_ok=True)
    torch.save(model.model.state_dict(), f"{output_path}/pytorch_model.bin")
    tokenizer.save_pretrained(output_path)
    model.model.config.save_pretrained(output_path)
    print(f"   Saved to {output_path}")
    
    # 7. Test inference
    print("\n7. Testing inference...")
    
    # Load the trained model for inference
    from src.ranking_qwen.models import QwenReranker
    
    reranker = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
    reranker.load_checkpoint(output_path)
    reranker.eval()
    
    query = "drill bits"
    documents = [
        "DEWALT 14-Piece Titanium Drill Bit Set for drilling metal and wood",
        "Milwaukee Cobalt Red Helix Drill Bit Set with storage case",
        "Black+Decker Screwdriver Set - 20 pieces",
    ]
    
    scores = reranker.compute_scores(
        queries=[query] * len(documents),
        documents=documents,
    )
    
    print(f"\n   Query: '{query}'")
    print("   Ranked results:")
    
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    for i, (doc, score) in enumerate(ranked, 1):
        print(f"   {i}. [{score:.4f}] {doc[:60]}...")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run full training: python scripts/train_reranker_hf.py")
    print("  - Evaluate model: python scripts/evaluate_reranker.py --help")
    print("  - Read guide: TRAINING_GUIDE.md")


if __name__ == "__main__":
    main()
