# Qwen3-Reranker Fine-Tuning Guide

This guide explains how to fine-tune **Qwen3-Reranker** models (0.6B or 4B) on the Home Depot product search dataset using the newly created training pipeline.

## Overview

The training pipeline implements the methodology described in `instruction_plan.md`:

- ✅ **Query-stratified dataset splitting** (prevents data leakage)
- ✅ **Binary classification** with yes/no token logits
- ✅ **Generative reranker** architecture (cross-encoder)
- ✅ **Gradient accumulation** for large effective batch sizes
- ✅ **Mixed precision training** (FP16/BF16)
- ✅ **Comprehensive evaluation metrics** (NDCG@K, MAP, MRR, Precision@K)

## Quick Start

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install -r requirements.txt

# Optional: Install flash-attention for 2-4x speedup (requires CUDA)
pip install flash-attn --no-build-isolation

# Install the package in development mode
pip install -e .
```

### 2. Train a Model

**Option A: Using the Python script directly**

```bash
python scripts/train_reranker.py \
    --model_name Qwen/Qwen3-Reranker-0.6B \
    --data_path data/home_depot.json \
    --output_dir models/checkpoints \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --use_flash_attn
```

**Option B: Using the CLI command** (after `pip install -e .`)

```bash
ranking-train \
    --model_name Qwen/Qwen3-Reranker-0.6B \
    --data_path data/home_depot.json \
    --output_dir models/checkpoints \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8
```

### 3. Evaluate a Model

```bash
python scripts/evaluate_reranker.py \
    --model_path models/checkpoints/best_model \
    --base_model_name Qwen/Qwen3-Reranker-0.6B \
    --data_path data/home_depot.json \
    --output_dir evaluation_results \
    --batch_size 4 \
    --save_predictions
```

Or using the CLI:

```bash
ranking-evaluate \
    --model_path models/checkpoints/best_model \
    --data_path data/home_depot.json \
    --output_dir evaluation_results
```

## Architecture Components

### 1. Data Preparation (`src/ranking_qwen/data/reranker_dataset.py`)

**`RerankerDatasetPreparator`** handles:

- **Text normalization**: Lowercase, whitespace cleanup
- **Document construction**: Combines product name + truncated description
- **Binary labeling**: Converts relevance scores to binary (threshold=2.33)
- **Query-stratified splitting**: 70% train / 15% val / 15% test
- **Message formatting**: Builds prompts for generative reranker

Example usage:

```python
from ranking_qwen.data import RerankerDatasetPreparator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
preparator = RerankerDatasetPreparator(
    tokenizer=tokenizer,
    relevance_threshold=2.33,
    max_description_tokens=350,
)

train_dataset, val_dataset, test_dataset = preparator.prepare_full_pipeline(df)
```

### 2. Model Wrapper (`src/ranking_qwen/models/qwen_reranker.py`)

**`QwenReranker`** provides:

- Model loading with flash attention support
- Prompt formatting with system instructions
- Score computation from yes/no token logits
- Binary cross-entropy loss for training
- Checkpoint saving/loading

Example usage:

```python
from ranking_qwen.models import QwenReranker

model = QwenReranker(
    model_name="Qwen/Qwen3-Reranker-0.6B",
    use_flash_attn=True,
)

# Inference
score = model.compute_scores(
    queries="angle bracket",
    documents="Simpson Strong-Tie 12-Gauge Angle Bracket...",
)

# Training
loss = model.compute_loss(input_ids, attention_mask, labels)
```

### 3. Training Script (`scripts/train_reranker.py`)

Implements:

- Gradient accumulation (simulate larger batches)
- Learning rate warmup and linear decay
- Gradient clipping (max_norm=1.0)
- Validation after each epoch
- Automatic checkpoint saving

### 4. Evaluation Script (`scripts/evaluate_reranker.py`)

Computes:

- **Ranking metrics**: NDCG@K, MAP, MRR, Precision@K
- **Classification metrics**: Accuracy, Precision, Recall, F1, AUC
- **Error analysis**: Performance by relevance level

### 5. Metrics Module (`src/ranking_qwen/evaluation/reranker_metrics.py`)

Provides ranking-specific metrics:

```python
from ranking_qwen.evaluation import evaluate_ranking_metrics

metrics = evaluate_ranking_metrics(
    df=predictions_df,
    true_col='relevance',
    pred_col='predicted_score',
    query_col='query',
    k_values=[1, 3, 5, 10],
)
```

## Training Configuration

### Recommended Hyperparameters

#### For Qwen3-Reranker-0.6B (~600M parameters)

```bash
--model_name Qwen/Qwen3-Reranker-0.6B
--batch_size 2
--gradient_accumulation_steps 8         # Effective batch = 16
--learning_rate 5e-6
--num_epochs 3
--warmup_ratio 0.1
--max_grad_norm 1.0
```

**Hardware**: Requires ~16-20 GB GPU memory (24GB recommended)

#### For Qwen3-Reranker-4B (~4B parameters)

```bash
--model_name Qwen/Qwen3-Reranker-4B
--batch_size 1
--gradient_accumulation_steps 16        # Effective batch = 16
--learning_rate 3e-6
--num_epochs 3
--warmup_ratio 0.1
```

**Hardware**: Requires ~40-48 GB GPU memory (A100 recommended)

### Dataset Configuration

```bash
--train_ratio 0.70                      # 70% for training
--val_ratio 0.15                        # 15% for validation
--test_ratio 0.15                       # 15% for testing
--relevance_threshold 2.33              # Binary label threshold
--max_description_tokens 350            # Truncate long descriptions
--random_seed 42                        # For reproducibility
```

### Training Tips

1. **Flash Attention**: Use `--use_flash_attn` for 2-4x speedup (requires flash-attn package)

2. **Gradient Accumulation**: Increase if you run out of memory:
   ```bash
   --batch_size 1 --gradient_accumulation_steps 16
   ```

3. **Learning Rate**: Start with 5e-6; increase to 1e-5 if training is slow

4. **Early Stopping**: Monitor validation loss; stop if it plateaus

5. **Checkpointing**: Models are saved after each epoch + best model saved separately

## Evaluation Metrics

### Ranking Metrics (Query-Level)

- **NDCG@K**: Normalized Discounted Cumulative Gain at positions 1, 3, 5, 10
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **Precision@K**: Fraction of relevant items in top K

### Classification Metrics (Sample-Level)

- **Accuracy**: Overall correctness
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives
- **F1**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Target Performance

Based on dataset analysis, target metrics for a well-tuned model:

- **NDCG@10**: ≥ 0.80
- **MAP**: ≥ 0.75
- **MRR**: ≥ 0.85
- **AUC**: ≥ 0.90

## File Structure

```
Ranking-Qwen/
├── data/
│   └── home_depot.json                      # Dataset
├── scripts/
│   ├── train_reranker.py                    # Training script
│   └── evaluate_reranker.py                 # Evaluation script
├── src/ranking_qwen/
│   ├── data/
│   │   └── reranker_dataset.py              # Dataset preparation
│   ├── models/
│   │   └── qwen_reranker.py                 # Model wrapper
│   ├── evaluation/
│   │   └── reranker_metrics.py              # Ranking metrics
│   └── cli/
│       ├── train.py                         # CLI: ranking-train
│       └── evaluate.py                      # CLI: ranking-evaluate
├── models/
│   └── checkpoints/                         # Saved models
│       ├── epoch_1/
│       ├── epoch_2/
│       ├── epoch_3/
│       └── best_model/                      # Best validation loss
└── evaluation_results/
    ├── evaluation_metrics.json              # Metrics
    ├── relevance_analysis.csv               # By-level analysis
    └── predictions.csv                      # Predictions (optional)
```

## Advanced Usage

### Custom Instruction

Modify the task instruction for domain-specific reranking:

```python
preparator = RerankerDatasetPreparator(
    tokenizer=tokenizer,
    instruction="Given a shopper's query, determine if the product matches their intent",
)
```

### Continuous Relevance (MSE Loss)

Instead of binary classification, map scores to [0, 1] and use MSE:

```python
# In reranker_dataset.py
df['label'] = (df['relevance'] - 1.0) / 2.0  # Scale to [0, 1]

# In qwen_reranker.py (modify compute_loss)
loss = F.mse_loss(probs, labels.float())
```

### LoRA Fine-Tuning

Use PEFT for parameter-efficient training:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model.get_model(), lora_config)
```

### Inference Example

```python
from ranking_qwen.models import QwenReranker

# Load fine-tuned model
reranker = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
reranker.load_checkpoint("models/checkpoints/best_model")

# Rank candidates for a query
query = "drill bits"
candidates = [
    "DEWALT 14-Piece Titanium Drill Bit Set",
    "Milwaukee Cobalt Red Helix Drill Bit Set",
    "Black+Decker Screwdriver Set",
]

scores = reranker.compute_scores(
    queries=[query] * len(candidates),
    documents=candidates,
)

# Sort by score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
for doc, score in ranked:
    print(f"{score:.4f}: {doc}")
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 32`
3. Reduce max_length in tokenizer (default 8192)
4. Use smaller model: Qwen3-Reranker-0.6B instead of 4B

### Slow Training

1. Enable flash attention: `--use_flash_attn`
2. Use mixed precision (automatic with FP16)
3. Reduce logging frequency: `--log_interval 200`

### Poor Performance

1. Increase training epochs: `--num_epochs 5`
2. Adjust learning rate: try 1e-5 or 2e-5
3. Use different relevance threshold: `--relevance_threshold 2.0`
4. Mine hard negatives (see instruction_plan.md section 2.3)

## References

- [Qwen3-Reranker Model Card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Instruction Plan](instruction_plan.md) - Detailed methodology
- [Dataset Research Report](DATASET_RESEARCH_REPORT.md) - Dataset analysis

## License

MIT License - See LICENSE file for details.
