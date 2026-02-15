# Home Depot Product Search Ranking Dataset - Comprehensive Research Report

**Report Date:** February 15, 2026  
**Dataset:** home_depot.json  
**Target Model:** Qwen/Qwen3-Reranker-0.6B  
**Task:** Fine-tuning for E-commerce Product Ranking

---

## Executive Summary

This report provides a comprehensive analysis of the Home Depot product search ranking dataset for fine-tuning the Qwen/Qwen3-Reranker-0.6B model. The dataset contains **74,067 query-product pairs** with human-annotated relevance scores, representing real-world e-commerce search scenarios across home improvement products. The dataset exhibits strong characteristics for training a reranker model, including diverse relevance distributions, realistic user queries, and rich product descriptions.

---

## 1. Dataset Overview

### 1.1 Basic Statistics

| Metric | Value |
|--------|-------|
| **Total Records** | 74,067 |
| **Unique Products (entity_id)** | 54,667 |
| **Unique Queries** | 11,795 |
| **Average Queries per Product** | 1.4 (range: 1-21) |
| **Average Products per Query** | 6.3 (range: 1-16) |
| **Data Completeness** | 100% (no missing fields) |
| **ID Uniqueness** | 100% (no duplicate IDs) |

### 1.2 Data Schema

Each record contains the following fields:

```json
{
  "id": 2,
  "entity_id": 100001,
  "name": "Simpson Strong-Tie 12-Gauge Angle",
  "query": "angle bracket",
  "relevance": 3.0,
  "description": "Not only do angles make joints stronger..."
}
```

**Field Descriptions:**
- `id`: Unique identifier for each query-product pair
- `entity_id`: Product identifier (multiple queries can reference same product)
- `name`: Product name/title
- `query`: User search query
- `relevance`: Human-annotated relevance score (1.0 to 3.0 scale)
- `description`: Detailed product description with features and specifications

---

## 2. Relevance Score Analysis

### 2.1 Score Distribution

The dataset uses a granular relevance scoring system with 13 distinct values ranging from 1.0 to 3.0:

| Score | Count | Percentage | Interpretation |
|-------|-------|------------|----------------|
| **1.0** | 2,105 | 2.84% | Not Relevant / Poor Match |
| **1.33** | 3,006 | 4.06% | Marginally Relevant |
| **1.67** | 6,780 | 9.15% | Somewhat Relevant |
| **2.0** | 11,730 | 15.84% | Moderately Relevant |
| **2.33** | 16,060 | 21.68% | Good Match |
| **2.67** | 15,202 | 20.52% | Very Good Match |
| **3.0** | 19,125 | 25.82% | Highly Relevant / Perfect Match |

**Minor scores** (1.25, 1.5, 1.75, 2.25, 2.5, 2.75): Combined < 0.1%

### 2.2 Statistical Properties

- **Mean Relevance:** 2.382
- **Median Relevance:** 2.330
- **Standard Deviation:** 0.534
- **Relevance Pattern:** Predominantly follows thirds pattern (X.00, X.33, X.67)

### 2.3 Class Balance Considerations

**Distribution Characteristics:**
- **High Relevance (2.5-3.0):** 46.37% - Good representation of positive examples
- **Medium Relevance (2.0-2.4):** 37.53% - Substantial middle ground for learning nuances
- **Low Relevance (1.0-1.9):** 16.07% - Sufficient negative examples

**Training Implications:**
- Well-balanced dataset with slight skew toward higher relevance scores
- This reflects real-world e-commerce where most presented results have some relevance
- The granular scoring enables learning fine-grained ranking distinctions
- May benefit from weighted loss functions to address slight class imbalance

---

## 3. Query Characteristics

### 3.1 Query Length Statistics

| Metric | Value |
|--------|-------|
| **Character Length** | Min: 2, Max: 60, Avg: 19.0 |
| **Word Count** | Min: 1, Max: 14, Avg: 3.2 |

### 3.2 Query Composition

- **Lowercase Queries:** 96.7% (71,645 queries)
- **Queries with Numbers:** 24.6% (18,256 queries)
- **Queries with Special Characters (/-&*#+):** 6.5% (4,785 queries)

### 3.3 Top 10 Most Frequent Queries

Each appears 15-16 times in the dataset:

1. metal sheet
2. 3 WAY TOGGLE SWITCH
3. everblit heavy duty canvas dropcloth
4. bed frames headboaed
5. 1/2 zip wall
6. contact paoer
7. anderson windows 400 seriesimpact resistant
8. moen chat oil bronze tub/shower faucet
9. burgundy red foot stools
10. closetmaid

### 3.4 Query Quality Observations

**Strengths:**
- Natural, user-generated queries reflecting real search behavior
- Diverse query types: brand names, product categories, specifications
- Includes long-tail queries (up to 14 words)

**Challenges:**
- Contains typos and misspellings (e.g., "sprkinler", "paoer", "headboaed")
- Inconsistent formatting and spacing
- Mixed specificity levels (generic to highly specific)

**Training Implications:**
- Model must handle noisy, real-world input
- Robustness to typos and variations is critical
- Lowercase normalization appears standard but not universal

---

## 4. Product/Document Characteristics

### 4.1 Product Name Analysis

| Metric | Value |
|--------|-------|
| **Character Length** | Min: 9, Max: 147, Avg: 69.3 |
| **Word Count** | Min: 1, Max: 35, Avg: 11.6 |

**Characteristics:**
- Product names are structured and informative
- Often include brand, model, size, color, and key features
- Standardized format typical of e-commerce listings

### 4.2 Product Description Analysis

| Metric | Value |
|--------|-------|
| **Character Length** | Min: 153, Max: 5,516, Avg: 885.7 |
| **Word Count** | Min: 21, Max: 857, Avg: 133.7 |

**Content Characteristics:**
- Rich, detailed descriptions with specifications
- Include product features, dimensions, materials, use cases
- Contain marketing language and technical details
- May include legal notices (Proposition 65, warranty information)
- Feature bullet-point style information (often concatenated)

### 4.3 Domain Coverage

**Sample Brand Distribution** (from 10,000 record sample):

| Brand | Frequency |
|-------|-----------|
| Hampton Bay | 333 |
| Whirlpool | 124 |
| Delta | 110 |
| HDX | 101 |
| Lithonia | 91 |
| Samsung | 80 |
| Simpson | 62 |
| Toro | 59 |
| BEHR | 54 |
| Quikrete | 24 |

**Product Categories Identified:**
- Home Improvement (lumber, hardware, tools)
- Appliances (washers, microwaves)
- Plumbing (faucets, fixtures)
- Electrical (lighting, switches)
- Building Materials (concrete, paint, fencing)
- Lawn & Garden (mowers, sprinklers)
- Kitchen & Bath
- Furniture & Decor

---

## 5. Query-Product Relationship Patterns

### 5.1 Multi-Product Queries

**Example Patterns:**

1. **Query: "l bracket"** → 3 products with scores: [3.0, 2.5, 2.33]
2. **Query: "deck over"** → 2 products with scores: [3.0, 1.67]
3. **Query: "rain shower head"** → 2 products with scores: [2.67, 2.33]
4. **Query: "emergency light"** → 2 products with scores: [3.0, 2.67]

**Observations:**
- Same query can match multiple products with varying relevance
- Relevance scores show meaningful distinctions between products
- Some queries have clear best matches (3.0) with lower-scored alternatives

### 5.2 Relevance Examples by Score Level

**Relevance 1.0 (Poor Match):**
- Query: "door guards" → Product: "MD Building Products 36 in. x 36 in. Cloverleaf Aluminum Sheet"
- Minimal semantic overlap, product tangentially related

**Relevance 2.0 (Moderate Match):**
- Query: "honda mower" → Product: "Toro Personal Pace Recycler 22 in. ... Briggs & Stratton Engine"
- Same category but wrong brand, relevant but not ideal match

**Relevance 3.0 (Perfect Match):**
- Query: "angle bracket" → Product: "Simpson Strong-Tie 12-Gauge Angle"
- Direct semantic match, product precisely addresses query intent

---

## 6. Data Quality Assessment

### 6.1 Strengths

✅ **Completeness:** 100% - No missing values in any field  
✅ **Uniqueness:** All IDs are unique, no duplicates  
✅ **Volume:** 74K+ samples provide substantial training data  
✅ **Diversity:** 11,795 unique queries across 54,667 products  
✅ **Granularity:** 13 relevance levels enable fine-grained learning  
✅ **Real-World:** Authentic user queries with typos and variations  
✅ **Rich Context:** Detailed product descriptions (avg 134 words)  
✅ **Domain Coverage:** Comprehensive home improvement product range  

### 6.2 Considerations

⚠️ **Query Noise:** Contains typos and inconsistent formatting  
⚠️ **Query Repetition:** Max 16 occurrences per query (but most unique)  
⚠️ **Class Imbalance:** Slight skew toward higher relevance scores  
⚠️ **Annotation Granularity:** Mix of 1/3 and 1/4 increments suggests multiple annotators or methods  
⚠️ **Description Length Variance:** Wide range (153-5,516 chars) may require truncation strategies  

---

## 7. Reranking Task Specifics

### 7.1 Task Definition

**Input:** A user query and a candidate product (name + description)  
**Output:** Relevance score predicting how well the product matches the query  

**Task Type:** Pointwise or listwise reranking with regression/ordinal classification

### 7.2 Evaluation Metrics Recommendations

Given the dataset structure with multiple products per query:

**Recommended Metrics:**
1. **NDCG (Normalized Discounted Cumulative Gain)** - Primary metric for ranking quality
2. **MAP (Mean Average Precision)** - Measures ranking of relevant items
3. **MRR (Mean Reciprocal Rank)** - First relevant result position
4. **Spearman/Kendall Correlation** - Ranking order correlation
5. **MSE/MAE** - For relevance score prediction accuracy

**Metric Considerations:**
- The granular relevance scores (13 levels) support graded relevance metrics (NDCG)
- Average of 6.3 products per query enables meaningful ranking evaluation
- Both regression (predicting exact score) and ranking (relative ordering) approaches viable

---

## 8. Model Architecture & Fine-tuning Considerations

### 8.1 Qwen3-Reranker-0.6B Model Overview

**Base Model:** Qwen3-Reranker-0.6B  
**Architecture:** Transformer-based reranking model  
**Size:** 600M parameters  
**Task:** Cross-encoder style reranking  

### 8.2 Input Format Considerations

**Query-Product Pair Construction:**
```
[CLS] query [SEP] product_name [SEP] product_description [/SEP]
```

**Text Length Challenges:**
- Average description: 134 words (~885 chars)
- Max description: 857 words (~5,516 chars)
- Most rerankers use 512 token limit
- **Recommendation:** Truncate descriptions to first 256-384 tokens after product name

### 8.3 Training Approach Options

**Option 1: Pointwise (Regression)**
- Predict relevance score directly (1.0 to 3.0)
- Loss: MSE, Huber, or smooth L1
- Pro: Simple, uses all score granularity
- Con: Doesn't explicitly optimize ranking

**Option 2: Pairwise (Ranking Loss)**
- Learn relative ordering between pairs
- Loss: Margin ranking loss, hinge loss
- Pro: Directly optimizes ranking objective
- Con: Requires pair construction (memory intensive)

**Option 3: Listwise (Group-wise)**
- Process all products for a query together
- Loss: ListNet, ListMLE, ApproxNDCG
- Pro: Best for ranking metrics
- Con: Variable-length groups, complex implementation

**Option 4: Ordinal Classification**
- Treat relevance as ordinal categories
- Map to discrete classes or bins
- Pro: Respects ordinal nature
- Con: Loses score granularity

**Recommendation:** Start with **Pointwise + Pairwise hybrid**
- Combine MSE loss for score prediction with margin loss for ranking
- Balance with weighted sum: `Loss = α * MSE + β * MarginRankingLoss`

### 8.4 Data Preprocessing Requirements

**Query Processing:**
- Lowercase normalization (optional, 96.7% already lowercase)
- Handle typos: Consider augmentation or leave as-is for robustness
- Special character handling: Preserve (important for model numbers)

**Product Processing:**
- Combine name and description
- Truncate long descriptions intelligently (preserve key features)
- Consider structured description parsing (bullets, specs)

**Relevance Score Processing:**
- **Option A:** Use raw scores (1.0-3.0 regression)
- **Option B:** Normalize to [0,1] range
- **Option C:** Bin into 3-5 classes (low/medium/high relevance)

---

## 9. Dataset Split Strategy

### 9.1 Splitting Challenges

**Considerations:**
- Each query appears with multiple products
- Each product may appear with multiple queries
- Need to avoid data leakage

### 9.2 Recommended Split Strategy

**Strategy: Query-based Stratified Split**

1. **Group by Query:** Ensure all products for a query stay together
2. **Stratify by Relevance Distribution:** Maintain score distribution across splits
3. **Consider Product Overlap:** Monitor entity_id overlap (acceptable if queries differ)

**Proposed Split Ratios:**
- **Training:** 70% (~8,257 queries, ~51,847 pairs)
- **Validation:** 15% (~1,769 queries, ~11,110 pairs)
- **Test:** 15% (~1,769 queries, ~11,110 pairs)

**Implementation Note:**
```python
# Pseudocode
unique_queries = dataset.groupby('query')
train_queries, temp_queries = stratified_split(unique_queries, 
                                                test_size=0.3, 
                                                stratify=avg_relevance_per_query)
val_queries, test_queries = stratified_split(temp_queries, 
                                               test_size=0.5)
```

### 9.3 Cross-Validation Consideration

For robust evaluation, consider **5-fold query-stratified cross-validation**:
- More reliable performance estimates
- Accounts for query diversity
- Better handles data variability

---

## 10. Training Configuration Recommendations

### 10.1 Hyperparameters

**Learning Rate:**
- Start: 1e-5 to 5e-5 (typical for fine-tuning)
- Schedule: Linear decay with warmup
- Warmup: 5-10% of total steps

**Batch Size:**
- Effective batch size: 32-64 pairs
- Gradient accumulation if memory constrained
- For pairwise: Consider in-batch negatives

**Epochs:**
- Typical: 3-5 epochs
- Monitor validation metrics closely
- Early stopping with patience=2-3

**Optimizer:**
- AdamW with weight decay (0.01)
- Gradient clipping (max_norm=1.0)

**Sequence Length:**
- Max tokens: 512
- Query: ~50 tokens, Name: ~50 tokens, Description: 300-400 tokens

### 10.2 Data Augmentation Strategies

**Query Augmentation:**
- Synonym replacement (limited, preserve intent)
- Back-translation (for robustness)
- ❌ Avoid heavy augmentation (typos are natural)

**Negative Sampling:**
- In-batch negatives (other products in batch)
- Hard negatives (low-scored products for same query)
- Random negatives (products from different queries)

**Description Augmentation:**
- Different truncation positions
- Shuffle bullet points (if parsed)

### 10.3 Regularization

- **Dropout:** 0.1 (typically in model)
- **Weight Decay:** 0.01
- **Label Smoothing:** 0.1 (if using classification)
- **Early Stopping:** Based on validation NDCG

---

## 11. Evaluation Framework

### 11.1 Offline Metrics

**Primary Metrics:**
1. **NDCG@10:** Ranking quality for top results
2. **MAP (Mean Average Precision):** Overall ranking performance
3. **MRR:** First relevant result

**Secondary Metrics:**
4. **Spearman Correlation:** Rank order correlation
5. **MSE/MAE:** Score prediction accuracy
6. **Precision@K** (K=1,3,5,10)

### 11.2 Error Analysis

**Recommended Analyses:**
1. **Performance by Relevance Level:** Identify weak score ranges
2. **Performance by Query Length:** Short vs. long queries
3. **Performance by Query Type:** Brand, category, specification queries
4. **Failure Mode Analysis:** Examine low-scoring predictions
5. **Confusion Matrix:** For binned relevance classes

### 11.3 Diagnostic Subsets

Create evaluation subsets for specific scenarios:
- **Brand Queries:** Queries containing brand names
- **Specification Queries:** Queries with numbers/measurements
- **Typo Queries:** Identified misspelled queries
- **Long-tail Queries:** Rare queries (frequency ≤ 2)
- **Ambiguous Queries:** Single-word or very generic queries

---

## 12. Dataset Characteristics Summary for GPT Research

### 12.1 Key Dataset Properties

| Property | Value | Training Implication |
|----------|-------|----------------------|
| **Domain** | E-commerce (Home Depot) | Domain-specific reranking |
| **Task** | Query-product relevance | Cross-encoder reranking |
| **Size** | 74,067 pairs | Medium-sized dataset |
| **Queries** | 11,795 unique | Good diversity |
| **Products** | 54,667 unique | Extensive product coverage |
| **Relevance Levels** | 13 distinct scores | Fine-grained ranking |
| **Avg Query Length** | 3.2 words | Short, keyword-based |
| **Avg Description** | 134 words | Rich context |
| **Query Noise** | Typos, variations | Robustness required |
| **Class Balance** | Slight positive skew | Consider weighting |

### 12.2 Success Criteria

**Minimum Viable Model:**
- NDCG@10 > 0.70
- MAP > 0.65
- MRR > 0.75

**Strong Performance:**
- NDCG@10 > 0.80
- MAP > 0.75
- MRR > 0.85

**Production-Ready:**
- NDCG@10 > 0.85
- MAP > 0.80
- MRR > 0.90
- Latency < 50ms per query (with batching)

---

## 13. Challenges & Mitigation Strategies

### 13.1 Identified Challenges

| Challenge | Impact | Mitigation Strategy |
|-----------|--------|---------------------|
| **Query Typos** | Model confusion | Keep typos for robustness training |
| **Description Length** | Memory/truncation | Smart truncation, preserve key features |
| **Class Imbalance** | Bias toward high scores | Weighted loss, stratified sampling |
| **Annotation Variance** | Inconsistent labels | Robust loss functions, smoothing |
| **Product Overlap** | Data leakage | Query-based splitting |
| **Domain Specificity** | Limited generalization | Accept, optimize for domain |

### 13.2 Training Stability

**Potential Issues:**
- Large batch requirements for pairwise training
- Memory constraints with long sequences
- Overfitting on frequent queries

**Solutions:**
- Gradient accumulation
- Mixed precision training (FP16/BF16)
- Query dropout (random query masking)
- Product diversity in batches

---

## 14. Baseline & Comparison Strategies

### 14.1 Baselines to Establish

**Traditional IR Baselines:**
1. **BM25:** Query-product matching
2. **TF-IDF + Cosine Similarity**
3. **Jaccard Similarity**

**Neural Baselines:**
1. **Sentence-BERT Cosine:** Bi-encoder approach
2. **Pre-trained Reranker (zero-shot):** Qwen3-Reranker before fine-tuning
3. **Generic Cross-Encoder:** e.g., cross-encoder/ms-marco-MiniLM

**Learning-based Baselines:**
1. **LambdaMART / XGBoost:** With hand-crafted features
2. **BERT-base Fine-tuned:** Standard BERT classification head

### 14.2 Ablation Studies

**Recommended Ablations:**
1. Name only vs. Name+Description
2. Different truncation strategies
3. Score regression vs. classification
4. Different loss functions
5. With/without data augmentation

---

## 15. Deployment Considerations

### 15.1 Inference Requirements

**Latency Target:** < 50ms per query (batch of 10-20 products)

**Optimization Strategies:**
- Model quantization (INT8)
- ONNX conversion for faster inference
- Batch processing of candidate products
- GPU inference for high throughput

### 15.2 Production Integration

**Typical Pipeline:**
1. **Candidate Retrieval:** BM25/Bi-encoder retrieves top 100-200 candidates
2. **Reranking:** Fine-tuned Qwen3-Reranker scores top candidates
3. **Final Ranking:** Top 10-20 results presented to user

**Model Serving:**
- Triton Inference Server, TorchServe, or FastAPI
- Caching for common queries
- A/B testing framework

---

## 16. Additional Dataset Insights

### 16.1 Linguistic Patterns

**Query Patterns:**
- Product-centric: "lawn mower", "shower head"
- Brand-specific: "samsung washer", "toro mower"  
- Specification-driven: "4x8 paneling", "3 way toggle switch"
- Use-case: "grill gazebo", "emergency light"

**Description Patterns:**
- Feature-heavy marketing language
- Technical specifications embedded
- Regulatory information (Prop 65, warranties)
- Installation and usage instructions

### 16.2 Domain-Specific Vocabulary

- Product dimensions (12 in., 72 ft., etc.)
- Material specifications (MDF, stainless steel, galvanized)
- Technical certifications (ENERGY STAR, NSF, UL)
- Brand-specific terminology

**Training Implication:** Model needs to learn domain lexicon and measurement matching

---

## 17. Future Enhancements & Extensions

### 17.1 Dataset Augmentation Possibilities

1. **User Behavior Signals:** Click-through rates, purchase data
2. **Product Images:** Multi-modal reranking
3. **User Context:** Geographic location, past purchases
4. **Seasonal Patterns:** Time-based relevance adjustments
5. **Product Relationships:** Cross-selling, substitutes

### 17.2 Model Enhancement Paths

1. **Multi-Task Learning:** Combine relevance + click prediction
2. **Personalization Layer:** User-specific reranking
3. **Explainability:** Attention visualization for relevance factors
4. **Active Learning:** Identify uncertain predictions for human review
5. **Continual Learning:** Adapt to new products and queries

---

## 18. Recommended Training Plan Outline for GPT Research

Based on this analysis, GPT Research should develop a training plan covering:

### Phase 1: Data Preparation (Week 1)
- [ ] Query-stratified train/val/test split (70/15/15)
- [ ] Text preprocessing pipeline (truncation, normalization)
- [ ] Baseline metric establishment (BM25, zero-shot model)
- [ ] Data loader implementation with batching strategy

### Phase 2: Model Setup (Week 1-2)
- [ ] Load Qwen3-Reranker-0.6B with appropriate tokenizer
- [ ] Design input format (query [SEP] name [SEP] description)
- [ ] Implement loss function (MSE + margin ranking hybrid)
- [ ] Configure optimizer (AdamW, learning rate schedule)

### Phase 3: Initial Training (Week 2-3)
- [ ] Pointwise training with relevance regression
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Monitor validation NDCG, MAP, MRR
- [ ] Error analysis on validation set

### Phase 4: Advanced Training (Week 3-4)
- [ ] Implement pairwise/listwise components
- [ ] Hard negative mining
- [ ] Cross-validation for robust estimates
- [ ] Model checkpointing and selection

### Phase 5: Evaluation & Analysis (Week 4-5)
- [ ] Comprehensive test set evaluation
- [ ] Performance breakdown by query types
- [ ] Ablation studies
- [ ] Comparison with baselines

### Phase 6: Optimization & Deployment (Week 5-6)
- [ ] Model quantization and optimization
- [ ] Latency benchmarking
- [ ] Integration testing
- [ ] Documentation and handoff

---

## 19. Conclusion

The Home Depot product search ranking dataset provides an excellent foundation for fine-tuning the Qwen/Qwen3-Reranker-0.6B model for e-commerce reranking tasks. Key strengths include:

✅ **High-Quality Annotations:** 13-level granular relevance scores  
✅ **Real-World Queries:** Natural language with typos and variations  
✅ **Rich Product Context:** Detailed descriptions averaging 134 words  
✅ **Sufficient Scale:** 74K pairs covering 11.8K queries and 54.7K products  
✅ **Clean Data:** 100% completeness, no duplicates  
✅ **Domain Coverage:** Comprehensive home improvement product range  

The dataset's characteristics support both pointwise and pairwise training approaches, with opportunities for sophisticated ranking optimization. The primary challenge lies in handling the description length variance and ensuring robust performance on noisy queries.

**Recommended Approach:** Hybrid pointwise-pairwise training with query-stratified splitting, smart description truncation, and comprehensive evaluation using NDCG, MAP, and MRR metrics.

This dataset should enable training a production-quality reranker that significantly improves upon traditional IR methods and generic pre-trained models for the home improvement e-commerce domain.

---

## 20. Appendix: Data Samples

### A.1 Sample High-Quality Matches (Relevance 3.0)

**Example 1:**
- **Query:** "angle bracket"
- **Product:** Simpson Strong-Tie 12-Gauge Angle
- **Description:** Features angle brackets for joints, 3x3x1.5 in., 12-gauge steel
- **Why 3.0:** Exact intent match, product directly addresses query

**Example 2:**
- **Query:** "deck over"
- **Product:** BEHR Premium Textured DeckOver 1-gal. Tugboat Wood and Concrete Coating
- **Description:** Deck coating product, revives wood/concrete decks
- **Why 3.0:** Product name contains query, purpose aligns perfectly

### A.2 Sample Moderate Matches (Relevance 2.0)

**Example:**
- **Query:** "honda mower"
- **Product:** Toro Personal Pace Recycler 22 in. Lawn Mower with Briggs & Stratton Engine
- **Description:** Gas lawn mower, similar category
- **Why 2.0:** Right category but wrong brand (Toro, not Honda)

### A.3 Sample Poor Matches (Relevance 1.0)

**Example:**
- **Query:** "door guards"
- **Product:** MD Building Products 36 in. x 36 in. Cloverleaf Aluminum Sheet
- **Description:** Decorative aluminum sheet for screens/cabinets
- **Why 1.0:** Tangentially related (metal sheet could theoretically protect), but not door guards

---

## Document Metadata

- **Report Version:** 1.0
- **Dataset Version:** home_depot.json (74,067 records)
- **Analysis Date:** February 15, 2026
- **Prepared For:** GPT Research - Qwen3-Reranker Fine-tuning Project
- **Next Steps:** Detailed training plan development by GPT Research

---

*End of Report*
