#!/usr/bin/env python3
"""
Push fine-tuned Qwen3-Reranker model to HuggingFace Hub.

This script:
1. Loads the fine-tuned model
2. Creates a model card with training details
3. Pushes to HuggingFace Hub
"""

import argparse
import os
from pathlib import Path
import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_model_card(
    base_model: str,
    dataset_name: str,
    num_samples: int,
    training_config: dict
) -> str:
    """Create a model card for the fine-tuned model."""
    
    card = f"""---
tags:
- reranker
- qwen3
- information-retrieval
- product-search
base_model: {base_model}
license: mit
language:
- en
pipeline_tag: text-classification
---

# Qwen3-Reranker-HomeDepot

Fine-tuned [Qwen3-Reranker-0.6B](https://huggingface.co/{base_model}) on the Home Depot product search dataset for e-commerce search ranking.

## Model Description

This model is a cross-encoder reranker trained to score query-product pairs for relevance. It takes a search query and product description as input and outputs a relevance score between 0 and 1.

**Base Model**: {base_model}  
**Training Dataset**: Home Depot Product Search  
**Training Samples**: {num_samples:,}  
**Task**: Binary relevance classification (relevant/irrelevant)

## Training Details

### Dataset

- **Total samples**: {num_samples:,}
- **Splits**: 70% train / 15% validation / 15% test
- **Splitting strategy**: Query-stratified (prevents data leakage)
- **Label threshold**: Relevance ≥ 2.33 → relevant (1), else irrelevant (0)
- **Label distribution**: ~68% relevant, ~32% irrelevant

### Training Configuration

```
Learning rate: {training_config.get('learning_rate', '5e-6')}
Batch size: {training_config.get('batch_size', 8)} × {training_config.get('gradient_accumulation_steps', 2)} = {training_config.get('batch_size', 8) * training_config.get('gradient_accumulation_steps', 2)}
Epochs: {training_config.get('num_epochs', 3)}
Optimizer: AdamW (weight_decay=0.01)
Scheduler: Linear warmup + decay
Mixed precision: BF16
```

### Hardware

- **GPU**: NVIDIA A100 80GB
- **Training time**: ~2-4 hours

## Usage

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "codefactory4791/Qwen3-Reranker-HomeDepot",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "codefactory4791/Qwen3-Reranker-HomeDepot",
    trust_remote_code=True
)

# Prepare input
query = "cordless drill"
document = "DEWALT 20V MAX Cordless Drill Kit with battery and charger"

# Format prompt
prompt = f'''<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Given a web search query, retrieve relevant passages that answer the query
<Query>: {{query}}
<Document>: {{document}}<|im_end|>
<|im_start|>assistant
<think>

</think>

'''

# Tokenize and get score
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model(**inputs)
logits = outputs.logits[0, -1, :]

# Get yes/no token probabilities
token_yes = tokenizer.convert_tokens_to_ids('yes')
token_no = tokenizer.convert_tokens_to_ids('no')
score = torch.sigmoid(logits[token_yes] - logits[token_no]).item()

print(f"Relevance score: {{score:.4f}}")
```

### Using with Ranking-Qwen Library

```python
from ranking_qwen.models import QwenReranker

# Load fine-tuned model
reranker = QwenReranker(model_name="codefactory4791/Qwen3-Reranker-HomeDepot")

# Score multiple candidates
scores = reranker.compute_scores(
    queries=["drill bits", "drill bits"],
    documents=[
        "DEWALT 14-Piece Titanium Drill Bit Set",
        "Black+Decker Screwdriver Set"
    ]
)
# Returns: [0.92, 0.31]
```

## Performance

Expected metrics on Home Depot test set:

- **NDCG@10**: ≥ 0.80
- **MAP**: ≥ 0.75
- **MRR**: ≥ 0.85
- **AUC**: ≥ 0.90

## Limitations

- Trained specifically for Home Depot product search
- May not generalize well to other domains without fine-tuning
- Maximum sequence length: 8192 tokens (though 2048 is recommended for speed)

## Citation

```bibtex
@misc{{qwen3-reranker-homedepot,
  author = {{Your Name}},
  title = {{Qwen3-Reranker Fine-tuned on Home Depot Dataset}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/codefactory4791/Qwen3-Reranker-HomeDepot}}}}
}}
```

## License

MIT License - See base model license for additional details.

## Acknowledgments

- Base model: [Qwen3-Reranker-0.6B](https://huggingface.co/{base_model})
- Dataset: Home Depot Product Search Relevance
- Training framework: HuggingFace Transformers
"""
    
    return card


def push_to_hub(
    model_path: str,
    repo_name: str,
    hf_token: str,
    base_model: str = "Qwen/Qwen3-Reranker-0.6B",
    private: bool = False,
):
    """Push model to HuggingFace Hub."""
    
    print("=" * 60)
    print("Pushing Model to HuggingFace Hub")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_name}")
    print(f"Private: {private}")
    print("=" * 60)
    
    # 1. Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print("   ✓ Model loaded")
    
    # 2. Create model card
    print("\n2. Creating model card...")
    training_config = {
        'learning_rate': '5e-6',
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'num_epochs': 3,
    }
    
    model_card = create_model_card(
        base_model=base_model,
        dataset_name="Home Depot Product Search",
        num_samples=51911,  # Training samples
        training_config=training_config
    )
    print("   ✓ Model card created")
    
    # 3. Create repository
    print("\n3. Creating HuggingFace repository...")
    api = HfApi()
    
    try:
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"   ✓ Repository created: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"   Repository might already exist: {e}")
    
    # 4. Push model
    print("\n4. Pushing model to Hub...")
    print("   (This may take several minutes...)")
    
    model.push_to_hub(
        repo_id=repo_name,
        token=hf_token,
        commit_message="Upload fine-tuned Qwen3-Reranker for Home Depot search"
    )
    print("   ✓ Model pushed")
    
    # 5. Push tokenizer
    print("\n5. Pushing tokenizer...")
    tokenizer.push_to_hub(
        repo_id=repo_name,
        token=hf_token,
        commit_message="Upload tokenizer"
    )
    print("   ✓ Tokenizer pushed")
    
    # 6. Upload model card
    print("\n6. Uploading model card...")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        token=hf_token,
        commit_message="Add model card"
    )
    print("   ✓ Model card uploaded")
    
    print("\n" + "=" * 60)
    print("✅ Upload completed successfully!")
    print("=" * 60)
    print(f"\n🚀 Your model is now available at:")
    print(f"   https://huggingface.co/{repo_name}")
    print("\n📖 To use it:")
    print(f'   model = AutoModelForCausalLM.from_pretrained("{repo_name}")')
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="HuggingFace repository name (username/model-name)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    args = parser.parse_args()
    
    push_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_token=args.hf_token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
