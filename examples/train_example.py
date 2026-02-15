#!/usr/bin/env python3
"""
Example: Quick start for training Qwen3-Reranker.

This script demonstrates minimal setup for training a reranker model.
For full options, see scripts/train_reranker.py or RERANKER_TRAINING_GUIDE.md
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking_qwen.models import QwenReranker
from src.ranking_qwen.data import RerankerDatasetPreparator
from datasets import load_dataset
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def main():
    """Train a small reranker model."""
    
    print("=" * 60)
    print("Qwen3-Reranker Training Example")
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
    model = QwenReranker(
        model_name="Qwen/Qwen3-Reranker-0.6B",
        use_flash_attn=False,  # Set True if you have flash-attn installed
    )
    
    # 3. Prepare data
    print("\n3. Preparing dataset...")
    preparator = RerankerDatasetPreparator(
        tokenizer=model.get_tokenizer(),
        relevance_threshold=2.33,
        max_description_tokens=350,
    )
    
    train_dataset, val_dataset, test_dataset = preparator.prepare_full_pipeline(
        df_small,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    # 4. Create dataloaders
    print("\n4. Creating dataloaders...")
    from ranking_qwen.models import create_data_collator
    from torch.utils.data import DataLoader
    
    collate_fn = create_data_collator(model.get_tokenizer())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # 5. Setup optimizer
    print("\n5. Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    
    num_epochs = 2
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    # 6. Training loop
    print("\n6. Training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Steps per epoch: {len(train_loader)}")
    print(f"   Total steps: {total_steps}")
    print()
    
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Forward pass
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            loss = model.compute_loss(input_ids, attention_mask, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"  Step {step + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                
                loss = model.compute_loss(input_ids, attention_mask, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"\nValidation loss: {avg_val_loss:.4f}")
        
        model.train()
    
    # 7. Save model
    print("\n7. Saving model...")
    output_path = "models/example_checkpoint"
    model.save_model(output_path)
    print(f"   Saved to {output_path}")
    
    # 8. Test inference
    print("\n8. Testing inference...")
    model.eval()
    
    query = "drill bits"
    documents = [
        "DEWALT 14-Piece Titanium Drill Bit Set for drilling metal and wood",
        "Milwaukee Cobalt Red Helix Drill Bit Set with storage case",
        "Black+Decker Screwdriver Set - 20 pieces",
    ]
    
    scores = model.compute_scores(
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
    print("  - Run full training: python scripts/train_reranker.py --help")
    print("  - Evaluate model: python scripts/evaluate_reranker.py --help")
    print("  - Read guide: RERANKER_TRAINING_GUIDE.md")


if __name__ == "__main__":
    main()
