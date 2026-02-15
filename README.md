# Ranking-Qwen: Fine-Tuning Qwen3-Reranker for Product Search

A complete training pipeline for fine-tuning **Qwen3-Reranker** models (0.6B and 4B) on the Home Depot product search dataset.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Train a Model

```bash
python scripts/train_reranker.py \
    --model_name Qwen/Qwen3-Reranker-0.6B \
    --data_path data/home_depot.json \
    --output_dir models/checkpoints \
    --num_epochs 3 \
    --use_flash_attn
```

### 3. Evaluate the Model

```bash
python scripts/evaluate_reranker.py \
    --model_path models/checkpoints/best_model \
    --base_model_name Qwen/Qwen3-Reranker-0.6B \
    --data_path data/home_depot.json \
    --output_dir evaluation_results
```

## 📚 Complete Documentation

**➡️ See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the complete guide covering:**
- Dataset preparation
- Model training configuration
- Evaluation metrics
- Troubleshooting
- Advanced usage

## 🎯 What This Pipeline Does

1. **Dataset Preparation**: Query-stratified splitting, binary labeling, message formatting
2. **Training**: Gradient accumulation, mixed precision, flash attention, checkpoint management
3. **Evaluation**: NDCG@K, MAP, MRR, Precision@K, and classification metrics

## 🏗️ Project Structure

```
Ranking-Qwen/
├── data/
│   └── home_depot.json              # Dataset (74,067 samples)
├── scripts/
│   ├── train_reranker.py            # Training script
│   └── evaluate_reranker.py         # Evaluation script
├── examples/
│   ├── train_example.py             # Quick training demo
│   └── inference_example.py         # Inference examples
├── src/ranking_qwen/
│   ├── data/reranker_dataset.py     # Dataset preparation
│   ├── models/qwen_reranker.py      # Model wrapper
│   └── evaluation/reranker_metrics.py  # Metrics
└── TRAINING_GUIDE.md                # ← COMPLETE GUIDE (START HERE)
```

## 🛠️ CLI Commands

```bash
# Train model
ranking-train --model_name Qwen/Qwen3-Reranker-0.6B --data_path data/home_depot.json

# Evaluate model
ranking-evaluate --model_path models/checkpoints/best_model --data_path data/home_depot.json

# Download dataset (if needed)
ranking-download
```

## 💡 Key Features

- ✅ Query-stratified dataset splitting (prevents data leakage)
- ✅ Binary classification with yes/no token logits
- ✅ Flash Attention 2 support (2-4x speedup)
- ✅ Gradient accumulation for memory efficiency
- ✅ Automatic best model checkpoint saving
- ✅ Comprehensive evaluation metrics

## 📊 Expected Performance

| Metric | Target |
|--------|--------|
| NDCG@10 | ≥ 0.80 |
| MAP | ≥ 0.75 |
| MRR | ≥ 0.85 |
| AUC | ≥ 0.90 |

## 🔧 Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.51.0
- CUDA-capable GPU (16GB+ for 0.6B, 48GB+ for 4B)

Optional:
- `flash-attn` for 2-4x speedup
- `tensorboard` for training monitoring

## 📖 Documentation Files

- **TRAINING_GUIDE.md** ← **Main guide (read this!)**
- **instruction_plan.md** - Detailed methodology and theory
- **DATASET_RESEARCH_REPORT.md** - Dataset analysis and statistics

## 🎓 Examples

### Quick Training Demo

```bash
python examples/train_example.py
```

### Inference Example

```python
from ranking_qwen.models import QwenReranker

# Load model
reranker = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
reranker.load_checkpoint("models/checkpoints/best_model")

# Score query-document pairs
scores = reranker.compute_scores(
    queries=["cordless drill", "cordless drill"],
    documents=[
        "DEWALT 20V MAX Cordless Drill Kit",
        "Black+Decker Screwdriver Set"
    ],
)
# Output: [0.92, 0.31]
```

## 🆘 Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 16`

**Slow training?**
- Enable flash attention: `--use_flash_attn`
- Reduce logging: `--log_interval 200`

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for more troubleshooting tips.

## 📝 License

MIT License - See LICENSE file for details.

## 🔗 References

- [Qwen3-Reranker Model Card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Training Methodology](instruction_plan.md)
- [Dataset Analysis](DATASET_RESEARCH_REPORT.md)

---

**Ready to start?** Run `python examples/train_example.py` or read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
