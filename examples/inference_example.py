#!/usr/bin/env python3
"""
Example: Using a fine-tuned reranker for inference.

This script shows how to load and use a trained reranker model
for ranking products given a search query.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking_qwen.models import QwenReranker


def main():
    """Demonstrate reranker inference."""
    
    print("=" * 60)
    print("Qwen3-Reranker Inference Example")
    print("=" * 60)
    
    # 1. Load model (use base model or fine-tuned checkpoint)
    print("\nLoading model...")
    
    # Option A: Use base pretrained model
    model = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
    
    # Option B: Load fine-tuned checkpoint
    # model = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
    # model.load_checkpoint("models/checkpoints/best_model")
    
    model.eval()
    
    # 2. Define queries and candidate products
    print("\nRanking products for queries...")
    
    queries_and_products = [
        {
            "query": "cordless drill",
            "products": [
                "DEWALT 20V MAX Cordless Drill/Driver Kit with battery and charger",
                "Milwaukee M18 FUEL Cordless Hammer Drill with case",
                "Black+Decker 8V MAX Cordless Screwdriver",
                "Makita 18V LXT Lithium-Ion Cordless Impact Driver Kit",
                "WORX 20V Switchdriver 2-in-1 Cordless Drill and Driver",
            ]
        },
        {
            "query": "led light bulbs",
            "products": [
                "Philips LED Dimmable Soft White Light Bulb 60W Equivalent - 4 Pack",
                "GE Relax LED Light Bulbs, 60 Watt, Soft White, 8 Pack",
                "Feit Electric LED String Lights 48ft for outdoor use",
                "Sylvania LED Daylight Light Bulbs A19, 100W = 15W, 12 Pack",
                "TCP LED Light Bulbs 60W Equivalent 2700K Warm White",
            ]
        },
        {
            "query": "garden hose",
            "products": [
                "Flexzilla Garden Hose 5/8 in. x 50 ft., Heavy Duty, Lightweight",
                "Gilmour Pro Commercial Hose 5/8 in. x 100 ft. with brass fittings",
                "Craftsman Premium Rubber Garden Hose 5/8 in. x 75 ft.",
                "Rain Bird Soaker Hose for garden watering",
                "Generic plastic watering can 2 gallon capacity",
            ]
        },
    ]
    
    # 3. Rank products for each query
    for i, item in enumerate(queries_and_products, 1):
        query = item["query"]
        products = item["products"]
        
        print(f"\n{'-' * 60}")
        print(f"Query {i}: '{query}'")
        print(f"{'-' * 60}")
        
        # Compute scores
        scores = model.compute_scores(
            queries=[query] * len(products),
            documents=products,
        )
        
        # Sort by score (descending)
        ranked = sorted(
            zip(products, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Print ranked results
        print("\nRanked Results:")
        for rank, (product, score) in enumerate(ranked, 1):
            relevance = "✓✓✓" if score >= 0.8 else "✓✓" if score >= 0.5 else "✓"
            print(f"{rank}. [{score:.4f}] {relevance} {product}")
    
    # 4. Batch inference for efficiency
    print(f"\n{'=' * 60}")
    print("Batch Inference Example")
    print(f"{'=' * 60}")
    
    # Prepare batch
    all_queries = []
    all_products = []
    
    for item in queries_and_products:
        query = item["query"]
        products = item["products"]
        all_queries.extend([query] * len(products))
        all_products.extend(products)
    
    print(f"\nRanking {len(all_products)} query-product pairs in batch...")
    
    # Batch scoring with batch_size parameter
    batch_scores = model.compute_scores(
        queries=all_queries,
        documents=all_products,
        batch_size=8,  # Process 8 pairs at a time
    )
    
    print(f"✓ Computed {len(batch_scores)} scores")
    print(f"  Score range: [{min(batch_scores):.4f}, {max(batch_scores):.4f}]")
    print(f"  Average score: {sum(batch_scores) / len(batch_scores):.4f}")
    
    # 5. Single pair example
    print(f"\n{'=' * 60}")
    print("Single Query-Product Pair")
    print(f"{'=' * 60}")
    
    query = "angle bracket"
    product = "Simpson Strong-Tie 12-Gauge Angle Bracket for wood construction"
    
    score = model.compute_scores(query, product)
    
    print(f"\nQuery:   '{query}'")
    print(f"Product: '{product}'")
    print(f"Score:   {score:.4f}")
    
    if score >= 0.8:
        print("Result:  Highly relevant ✓✓✓")
    elif score >= 0.5:
        print("Result:  Somewhat relevant ✓✓")
    else:
        print("Result:  Not very relevant ✓")
    
    print("\n" + "=" * 60)
    print("Inference examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
