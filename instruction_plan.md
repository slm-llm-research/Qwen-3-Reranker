# Home Depot Reranker Fine‑Tuning Plan

## Introduction

This document describes how to fine‑tune the **Qwen/Qwen3‑Reranker‑0.6B** model on the Home Depot product search dataset.  The goal is to learn a model that accepts a user query and a candidate product description and outputs a relevance score.  The Qwen reranker is a **generative, cross‑encoder reranking model** built on the Qwen3 foundation.  Cross‑encoders process both the query and the document simultaneously using self‑attention and are therefore able to capture fine‑grained interactions between tokens【493181220393779†L95-L103】.  Qwen rerankers are implemented as **causal language models**; they judge relevance by comparing the logits of special tokens (“yes” vs. “no”) at the final position【967128664425397†L118-L125】.  Training these models requires converting each query–document pair into a prompt that ends with a fixed instruction; the model should learn to generate **“yes”** when the product is relevant and **“no”** otherwise.

The instructions below are suitable for executing in **CURSOR IDE**.  They cover dataset preparation, building a custom training script, evaluating the model before and after fine‑tuning, and provide guidance on hyper‑parameter choices and best practices.  Example code is written in Python using **PyTorch**, **Transformers (≥ 4.51)** and **Datasets**.  Advanced users can optionally leverage the **ModelScope SWIFT** library for listwise training, but this plan focuses on a pointwise binary‑classification approach.

## 1. Environment Setup

1. **Python Environment** – Create a new virtual environment and install dependencies.  Use Python 3.10 or later.

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   # core libraries
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers==4.51.0 datasets evaluate scikit‑learn
   # optional for logging & monitoring
   pip install tensorboard wandb
   ```

2. **GPU and Mixed Precision** – The Qwen3‑Reranker‑0.6B has ~600 million parameters and supports 32 k token contexts.  To train efficiently you should have at least one 24 GB or larger GPU.  Enable **flash‑attention** in Transformers by passing `attn_implementation="flash_attention_2"` when loading the model【90629648899456†L172-L180】.  Mixed precision (FP16 or BF16) can reduce memory usage and should be turned on if supported.

3. **Dataset Files** – Ensure that the Home Depot dataset (e.g., `home_depot.json`) and this training plan are available in your project.  The dataset contains 74 067 query–product pairs with fields: `id`, `entity_id`, `name`, `query`, `relevance`, and `description`.  Each product appears in multiple query contexts and relevance scores range from 1.0 to 3.0 (13 distinct levels).

4. **Directory Structure** – Organise your project as follows:

   ```text
   project_root/
   ├── data/
   │   └── home_depot.json            # raw dataset
   ├── scripts/
   │   ├── train_reranker.py         # training script
   │   └── evaluate_reranker.py      # evaluation script
   ├── models/
   │   └── checkpoints/              # fine‑tuned models saved here
   └── logs/                         # tensorboard logs
   ```

## 2. Dataset Preparation

### 2.1 Loading and Splitting

1. **Load the JSON dataset** using the 🤗 Datasets library:

   ```python
   from datasets import load_dataset, Dataset

   data = load_dataset('json', data_files='data/home_depot.json', split='train')
   # Inspect fields
   print(data.features)
   ```

2. **Group by query for splitting.**  To avoid leaking information across splits, keep all products belonging to the same query in the same partition.  Stratify by average relevance to preserve score distribution.  A common split is 70 % train, 15 % validation, 15 % test.  Here is an example splitting function:

   ```python
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   # Convert to pandas for grouping
   df = data.to_pandas()
   # Compute average relevance per query
   query_groups = df.groupby('query')['relevance'].mean().reset_index(name='avg_rel')
   # Stratify by binned average relevance
   bins = [0, 1.67, 2.0, 2.33, 3.0]
   query_groups['bin'] = pd.cut(query_groups['avg_rel'], bins=bins, labels=False)
   train_queries, temp_queries = train_test_split(query_groups, test_size=0.3, stratify=query_groups['bin'], random_state=42)
   val_queries, test_queries = train_test_split(temp_queries, test_size=0.5, stratify=temp_queries['bin'], random_state=42)

   # Filter original dataframe
   train_df = df[df['query'].isin(train_queries['query'])]
   val_df   = df[df['query'].isin(val_queries['query'])]
   test_df  = df[df['query'].isin(test_queries['query'])]

   # Convert back to datasets
   train_data = Dataset.from_pandas(train_df)
   val_data   = Dataset.from_pandas(val_df)
   test_data  = Dataset.from_pandas(test_df)
   ```

3. **Sanity check** – Verify that no query is shared across splits and that relevance score distributions in each set mirror the global distribution (see dataset report).  Use histograms or counts by bin.

### 2.2 Pre‑processing Text

The Qwen generative reranker expects a **messages**‑based input.  Each sample must be converted into a system/user prompt template ending with a `yes`/`no` answer.  The default template described in ModelScope’s documentation is【967128664425397†L250-L260】:

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {Instruction}
<Query>: {Query}
<Document>: {Document}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

For this fine‑tuning task we keep the default instruction:

```text
Given a web search query, retrieve relevant passages that answer the query
```

1. **Construct document text** by concatenating the product name and a truncated description.  Descriptions vary from 153 to 5 516 characters (median ~885) and often exceed the model’s context limit.  Extract the first 256–384 tokens of the description while preserving entire bullet points.  Example:

   ```python
   import re
   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B')

   def truncate_description(text, max_tokens=350):
       tokens = tokenizer.encode(text, add_special_tokens=False)
       if len(tokens) > max_tokens:
           tokens = tokens[:max_tokens]
       return tokenizer.decode(tokens, skip_special_tokens=True)

   def build_document(row):
       desc = truncate_description(row['description'])
       return f"{row['name']}. {desc}"

   for split_name, split in [('train', train_data), ('validation', val_data), ('test', test_data)]:
       split = split.add_column('document', [build_document(row) for row in split])
   ```

2. **Lowercase normalization** – Convert queries and documents to lowercase (97 % of queries are already lowercase) to reduce vocabulary size.  Do **not** remove numbers or special characters because they convey important product specifications.

3. **Label engineering** – Convert the 13‑level relevance scores into binary labels suitable for a generative classification reranker.  A simple mapping is:
   * **relevant (1)** if `relevance ≥ 2.33` (good, very good, perfect match) – roughly top 46 % of samples.
   * **irrelevant (0)** if `relevance < 2.33`.

   This threshold balances positive and negative classes while still leaving enough negative examples.  Save the binary label in a new column `label`.  For more nuanced models you can map scores to continuous probabilities: `p = (relevance − 1.0) / 2.0`, then use mean‑squared error instead of cross‑entropy.

4. **Generate message dictionaries** – For each sample create a dictionary with the query and document formatted for the generative reranker:

   ```python
   def build_message(example, instruction="Given a web search query, retrieve relevant passages that answer the query"):
       user_content = f"<Instruct>: {instruction}\n<Query>: {example['query']}\n<Document>: {example['document']}"
       # positive_messages contains a single answer; the model is expected to output "yes" for relevant and "no" otherwise
       return {
           'messages': [{'role': 'user', 'content': user_content}],
           'positive_messages': [[{'role': 'assistant', 'content': 'yes'}]] if example['label'] == 1 else [],
           'negative_messages': [[{'role': 'assistant', 'content': 'no'}]] if example['label'] == 0 else []
       }

   train_msgs = train_data.map(build_message)
   val_msgs   = val_data.map(build_message)
   test_msgs  = test_data.map(build_message)
   ```

   The format above mirrors the `messages`, `positive_messages` and `negative_messages` fields used by ModelScope SWIFT【967128664425397†L182-L228】.  Only one positive or negative message is provided per sample; SWIFT will group multiple negatives automatically.  The binary label is implied by the presence of the positive or negative message.

### 2.3 Negative Sampling (optional)

To strengthen the model, include **hard negatives**, i.e., non‑relevant products that are similar to the query.  Sentence Transformers provides a `mine_hard_negatives` utility for this purpose【938936884683624†L281-L334】.  You can use a lightweight embedding model (e.g., `sentence-transformers/static-retrieval-mrl-en-v1`) to find hard negatives within each query group and add them to `negative_messages`.  This step improves the model’s ability to discriminate between very similar products but is optional if computational resources are limited.

## 3. Model Loading

The Qwen reranker is a generative language model.  We load it via `AutoModelForCausalLM` and `AutoTokenizer`.  The model expects the query and document to be prefaced by a **system message** and uses `yes` and `no` tokens to compute relevance probability【90629648899456†L172-L190】.  The following code shows how to load the model and set up special tokens:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen3-Reranker-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation='flash_attention_2'
).cuda()

token_true_id  = tokenizer.convert_tokens_to_ids('yes')
token_false_id = tokenizer.convert_tokens_to_ids('no')
max_length = 8192  # maximum sequence length supported by the model

# Prepare prefix and suffix tokens for the template
prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
```

The Qwen reranker uses left padding; ensure that `tokenizer.padding_side` is set accordingly.  Enabling flash attention accelerates attention computations, as suggested in the model card【90629648899456†L172-L180】.

## 4. Building the Training Script (train_reranker.py)

### 4.1 Data Collator

Define a collator that converts the message dictionaries into token IDs with the proper prefix and suffix.  Each input sequence should respect the model’s context length (8192 tokens).  Unused tokens should be padded on the left so that the final tokens correspond to the answer.  For each sample, we return input IDs and the **target label** (1 for relevant → “yes”, 0 for irrelevant → “no”).

```python
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq

class RerankerDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn(batch):
    messages = []
    labels   = []
    for example in batch:
        # Build instruction prompt
        user_content = example['messages'][0]['content']
        full_text   = prefix + user_content + suffix
        # Tokenize and truncate from the left if necessary
        input_ids = tokenizer.encode(full_text, add_special_tokens=False, truncation=True, max_length=max_length - 1)
        # Append the answer token placeholder; we will teach the model to generate 'yes' or 'no'
        input_ids = prefix_tokens + input_ids + suffix_tokens
        input_ids = input_ids[-max_length:]
        # Pad on the left
        padding_length = max_length - len(input_ids)
        input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
        messages.append(torch.tensor(input_ids, dtype=torch.long))
        # Determine label: 1 if positive_messages present, else 0
        label = 1 if example['positive_messages'] else 0
        labels.append(label)
    batch_input_ids = torch.stack(messages)
    batch_labels    = torch.tensor(labels, dtype=torch.float32)
    return {'input_ids': batch_input_ids.cuda(), 'labels': batch_labels.cuda()}

train_dataset = RerankerDataset(train_msgs)
val_dataset   = RerankerDataset(val_msgs)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
```

Notes:

* The batch size of 2 is chosen for demonstration.  In practice you should use the largest batch size that fits into GPU memory.  Because Qwen uses 32k tokens, each sequence consumes significant memory.  If memory becomes a bottleneck, use **gradient accumulation** to achieve a larger effective batch size.

* Each input ends with the `<think>` tag; the model will output the next token (either “yes” or “no”) at the last position.  We ignore the “positive_messages” or “negative_messages” text because the generative reranker uses only the tokens “yes” and “no” at inference time【967128664425397†L118-L125】.

### 4.2 Loss Function

For pointwise training we treat the task as **binary classification**: for each query–document pair the model should generate “yes” if the product is relevant and “no” otherwise.  This matches the pointwise loss definition in the SWIFT documentation【967128664425397†L131-L141】.  We compute the binary cross‑entropy between the model’s predicted probability for the positive token and the target label.

```python
from torch.nn import functional as F

def compute_loss(logits, labels):
    # logits: [batch_size, vocab_size] at the last position
    # Extract logits for "yes" and "no"
    true_logits  = logits[:, token_true_id]
    false_logits = logits[:, token_false_id]
    # Compute probability that the model chooses "yes"
    probs = torch.sigmoid(true_logits - false_logits)
    # Binary cross‑entropy loss
    loss = F.binary_cross_entropy(probs, labels)
    return loss
```

### 4.3 Training Loop

The following skeleton illustrates a simple training loop with gradient accumulation, learning rate scheduling, and mixed precision.  Save this script as `scripts/train_reranker.py`.

```python
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

num_epochs = 3
accum_steps = 8  # accumulate gradients to simulate larger batch
learning_rate = 5e-6
warmup_ratio  = 0.1

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
total_steps = len(train_loader) // accum_steps * num_epochs
warmup_steps = int(total_steps * warmup_ratio)
scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        labels    = batch['labels']
        # Forward pass with labels masked so the model only predicts at final position
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        loss   = compute_loss(logits, labels) / accum_steps
        loss.backward()
        total_loss += loss.item()
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch+1} step {step+1}: loss={total_loss/(step+1):.4f}")
    # Validation after each epoch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['input_ids'])
            logits  = outputs.logits[:, -1, :]
            loss    = compute_loss(logits, batch['labels'])
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")
    model.train()
    # Save checkpoint
    model.save_pretrained(f"models/checkpoints/epoch_{epoch+1}")
```

**Hyper‑parameters** – You may adjust the learning rate (5e‑6 to 2e‑5) and number of epochs (3–5) based on validation loss.  Use early stopping if the validation loss stops improving.  Since the dataset is balanced across high and low relevance scores, you usually do not need label weighting, but you can weight the positive class if needed.  Warm‑up for 10 % of the total steps helps stabilise training.

### 4.4 Advanced: Listwise Training with SWIFT

ModelScope’s **SWIFT** framework supports **listwise** generative reranking, where each query is associated with one positive document and multiple negatives; the model learns to choose the positive among them【967128664425397†L149-L177】.  To use SWIFT for the Home Depot dataset:

1. Install SWIFT: `pip install ms-swift`.
2. Convert the dataset into the **LLM reranker format** described in the documentation【967128664425397†L182-L228】.  For each query, identify one positive (highest‑scoring) product and at most seven negatives (lowest‑scoring) products.  Example entry:

   ```json
   {
     "messages": [{"role": "user", "content": "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: angle bracket\n<Document>: Simpson Strong‑Tie 12‑Gauge Angle. Not only do angles make joints stronger..."}],
     "positive_messages": [[{"role": "assistant", "content": "yes"}]],
     "negative_messages": [[{"role": "assistant", "content": "no"}], [{"role": "assistant", "content": "no"}], ...]
   }
   ```

3. Run SWIFT’s training script.  For pointwise classification, use the `generative_reranker` loss; for listwise ranking, use `listwise_generative_reranker`.  Example command (adjust `model`, `output_dir`, and batch sizes):

   ```sh
   nproc_per_node=2
   swift sft \
       --model Qwen/Qwen3-Reranker-0.6B \
       --task_type generative_reranker \
       --loss_type generative_reranker \
       --train_type full \
       --dataset path/to/home_depot_reranker_dataset.json \
       --split_dataset_ratio 0.1 \
       --output_dir models/swift_checkpoint \
       --num_train_epochs 3 \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 16 \
       --learning_rate 6e-6 \
       --eval_strategy steps \
       --eval_steps 200 \
       --save_steps 1000 \
       --label_names labels \
       --dataloader_drop_last true
   ```

SWIFT handles prompt formatting internally and implements both pointwise and listwise losses.  Use `MAX_POSITIVE_SAMPLES` and `MAX_NEGATIVE_SAMPLES` environment variables to control the number of examples per query【967128664425397†L210-L229】.  Monitor GPU memory usage and adjust `gradient_accumulation_steps` accordingly.

## 5. Evaluation

### 5.1 Computing Relevance Scores

After fine‑tuning, evaluate the model’s performance on the held‑out test set.  To compute relevance scores for each query–document pair, follow the inference example from the model card【90629648899456†L168-L213】:

```python
@torch.no_grad()
def compute_scores(model, input_ids_batch):
    outputs = model(input_ids_batch)
    logits  = outputs.logits[:, -1, :]
    true_logits  = logits[:, token_true_id]
    false_logits = logits[:, token_false_id]
    probs = torch.sigmoid(true_logits - false_logits)
    return probs.cpu().numpy()

test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)
model.eval()
all_scores = []
all_labels = []
for batch in test_loader:
    scores = compute_scores(model, batch['input_ids'])
    all_scores.extend(scores)
    all_labels.extend(batch['labels'].cpu().numpy())
```

Scores range between 0 and 1 and represent the model’s confidence that the product matches the query.  To obtain the final ranking for a query, group candidates by query and sort by descending score.  Optionally, rescale scores back to the original 1–3 relevance range using `scaled = 1 + 2 * score`.

### 5.2 Ranking Metrics

Compute ranking metrics such as **NDCG@10**, **MAP**, **MRR**, and **Precision@K** as recommended in the dataset research.  The `evaluate` library or `scikit‑learn` can be used for this purpose.  For example:

```python
from collections import defaultdict
import numpy as np

def compute_metrics(scores, labels, queries):
    # group by query
    groups = defaultdict(list)
    for s, l, q in zip(scores, labels, queries):
        groups[q].append((s, l))
    ndcg_values = []
    map_values  = []
    mrr_values  = []
    for q, pairs in groups.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        rels = [l for _, l in pairs_sorted]
        # DCG@10
        dcg = sum((2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rels[:10]))
        # Ideal DCG
        ideal = sorted(rels, reverse=True)
        idcg = sum((2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(ideal[:10])) or 1
        ndcg_values.append(dcg / idcg)
        # MAP
        hits = 0
        precisions = []
        for i, rel in enumerate(rels):
            if rel > 0:
                hits += 1
                precisions.append(hits / (i + 1))
        map_values.append(np.mean(precisions) if precisions else 0)
        # MRR
        try:
            first_rel = rels.index(1)
            mrr_values.append(1 / (first_rel + 1))
        except ValueError:
            mrr_values.append(0)
    return {
        'NDCG@10': np.mean(ndcg_values),
        'MAP': np.mean(map_values),
        'MRR': np.mean(mrr_values)
    }

metrics = compute_metrics(all_scores, all_labels, list(test_df['query']))
print(metrics)
```

Compare metrics **before** fine‑tuning (using the base Qwen3 reranker) and **after** fine‑tuning.  A strong model should significantly improve NDCG@10, MAP and MRR over the baseline (e.g., aiming for NDCG@10 ≥ 0.80 as suggested in the dataset report).

### 5.3 Error Analysis

1. **Relevance Level Analysis** – Segment test cases by their human relevance scores (1.0–3.0) and examine whether the model struggles at the boundaries (e.g., distinguishing 2.0 vs. 2.33).  Compare predicted scores across these segments.
2. **Query Type Analysis** – Use the query characteristics from the dataset report (brand vs. specification vs. typo queries) to evaluate performance on different query types.  This can inform targeted augmentation or weighting strategies.
3. **Failure Modes** – Inspect high scoring false positives and low scoring false negatives to understand whether the model is misled by certain features (e.g., synonyms, synonyms in descriptions, or unusual brand names).  Use these insights to design further fine‑tuning or data cleaning.

## 6. Post‑Training Deployment

1. **Model Quantisation** – For production inference consider converting the fine‑tuned model to **INT8** or **4‑bit** weights using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) or `transformers.quantization`.  Quantisation reduces memory usage and speeds up inference with minimal accuracy loss.
2. **Serving** – Deploy the model behind a web service.  You can use **vLLM** for high‑throughput generation; the model card provides vLLM usage example【90629648899456†L215-L309】.  For simpler deployments, wrap inference in a FastAPI or Flask service with GPU support.
3. **Pipeline Integration** – In a two‑stage retrieval system, first use a fast dense retriever (e.g., Qwen3‑Embedding) to fetch top‑100 candidates, then apply the fine‑tuned reranker to refine the top‑k results.  This balances efficiency and precision.

## 7. Tips and Best Practices

1. **Instruction Tuning** – Qwen rerankers support user‑defined instructions.  The model card notes that using an instruction generally improves retrieval performance by 1 %–5 %【90629648899456†L314-L317】.  Experiment with domain‑specific instructions (e.g., “Given a shopper’s query, determine whether the product matches the shopper’s intent”) and include them in the prompt.
2. **Loss Variants** – If binary classification is too coarse, map relevance scores to continuous probabilities and train with mean squared error or Huber loss.  You can also discretize scores into more than two bins and use multi‑class cross‑entropy with tokens “low”, “medium”, “high”.
3. **Negative Sampling** – Mining hard negatives using an embedding model helps the reranker learn subtle distinctions【938936884683624†L281-L334】.  Balance easy and hard negatives to avoid overfitting.
4. **Cross‑Validation** – Consider 5‑fold query‑stratified cross‑validation to obtain robust performance estimates.  The average across folds gives a reliable picture of generalisation.
5. **Monitoring & Logging** – Use TensorBoard or Weights & Biases to monitor training loss, validation loss, and evaluation metrics.  This helps catch overfitting and compare runs.
6. **LoRA/PEFT** – To reduce fine‑tuning costs, apply Low‑Rank Adaptation (LoRA) or QLoRA to only train a small number of adapter parameters while keeping the base model frozen.  The Qwen3 reranker accepts LoRA adapters (via `peft` library) because it is built on the same underlying architecture【493181220393779†L95-L103】.
7. **Ethical Considerations** – Ensure that the model does not inadvertently encode bias or present discriminatory results.  Evaluate fairness across product categories and check for spurious correlations.

## 8. Conclusion

This plan outlines a complete workflow for fine‑tuning the **Qwen3‑Reranker‑0.6B** on the Home Depot product search dataset.  By preparing the data carefully, constructing appropriate prompt‑based messages, and using a pointwise binary classification loss, you can teach the model to recognise relevant products with high precision.  Optionally, SWIFT enables more advanced listwise training with relative ranking losses.  Robust evaluation and error analysis ensure that the fine‑tuned model meets performance targets and yields insight for further improvement.

**References**

* Qwen3 Embedding blog – highlights the dual/cross‑encoder architecture of reranking models【493181220393779†L95-L103】 and notes that reranker models are trained on high‑quality labelled data【493181220393779†L117-L124】.
* Qwen3‑Reranker model card – shows how to construct prompts, use “yes”/“no” tokens, and suggests enabling flash attention for better performance【90629648899456†L172-L190】.
* ModelScope SWIFT documentation – explains that generative rerankers compute the probability of “yes”/“no” tokens and use binary cross‑entropy or listwise losses for training【967128664425397†L118-L160】 and provides the dataset format for training【967128664425397†L182-L228】.
* Sentence Transformers training guide – emphasises mining hard negatives and using appropriate loss functions for cross‑encoder reranking【938936884683624†L281-L334】.
