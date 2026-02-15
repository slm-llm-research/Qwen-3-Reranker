#!/usr/bin/env python3
"""
Example script demonstrating how to use the downloaded Home Depot dataset.

This script shows various ways to load and analyze the dataset.
"""

import pandas as pd
from pathlib import Path


def load_dataset(format: str = "parquet") -> pd.DataFrame:
    """
    Load the Home Depot dataset in the specified format.
    
    Args:
        format: File format to load ("csv", "parquet", or "json")
    
    Returns:
        DataFrame containing the dataset
    """
    data_dir = Path("data")
    
    if format == "csv":
        file_path = data_dir / "home_depot.csv"
        df = pd.read_csv(file_path)
    elif format == "parquet":
        file_path = data_dir / "home_depot.parquet"
        df = pd.read_parquet(file_path)
    elif format == "json":
        file_path = data_dir / "home_depot.json"
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"✅ Loaded {len(df):,} records from {file_path}")
    return df


def analyze_dataset(df: pd.DataFrame):
    """
    Perform basic analysis on the dataset.
    
    Args:
        df: DataFrame containing the Home Depot dataset
    """
    print("\n" + "=" * 60)
    print("Dataset Analysis")
    print("=" * 60)
    
    # Basic statistics
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   - Rows: {df.shape[0]:,}")
    print(f"   - Columns: {df.shape[1]}")
    
    # Column information
    print(f"\n📋 Columns:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    # Relevance score distribution
    print(f"\n🎯 Relevance Score Statistics:")
    print(df['relevance'].describe())
    
    print(f"\n📈 Relevance Score Distribution:")
    relevance_counts = df['relevance'].value_counts().sort_index()
    for score, count in relevance_counts.items():
        pct = (count / len(df)) * 100
        print(f"   Score {score:.1f}: {count:,} ({pct:.1f}%)")
    
    # Top queries
    print(f"\n🔍 Top 10 Most Common Search Queries:")
    top_queries = df['query'].value_counts().head(10)
    for query, count in top_queries.items():
        print(f"   '{query}': {count} times")
    
    # Unique entities
    print(f"\n🏷️  Unique Products: {df['entity_id'].nunique():,}")
    print(f"🔎 Unique Search Queries: {df['query'].nunique():,}")
    
    # Sample high and low relevance items
    print(f"\n✨ Sample High Relevance (3.0) Records:")
    high_rel = df[df['relevance'] == 3.0].head(3)
    for idx, row in high_rel.iterrows():
        print(f"\n   Query: '{row['query']}'")
        print(f"   Product: {row['name']}")
        print(f"   Relevance: {row['relevance']}")
    
    print(f"\n⚠️  Sample Low Relevance (1.0) Records:")
    low_rel = df[df['relevance'] == 1.0].head(3)
    for idx, row in low_rel.iterrows():
        print(f"\n   Query: '{row['query']}'")
        print(f"   Product: {row['name']}")
        print(f"   Relevance: {row['relevance']}")


def filter_examples(df: pd.DataFrame):
    """
    Show examples of filtering the dataset.
    
    Args:
        df: DataFrame containing the Home Depot dataset
    """
    print("\n" + "=" * 60)
    print("Filtering Examples")
    print("=" * 60)
    
    # Filter by relevance
    high_relevance = df[df['relevance'] >= 2.5]
    print(f"\n✅ High relevance items (≥2.5): {len(high_relevance):,} ({len(high_relevance)/len(df)*100:.1f}%)")
    
    low_relevance = df[df['relevance'] < 2.0]
    print(f"❌ Low relevance items (<2.0): {len(low_relevance):,} ({len(low_relevance)/len(df)*100:.1f}%)")
    
    # Search for specific query
    query = "angle bracket"
    query_results = df[df['query'] == query]
    if len(query_results) > 0:
        print(f"\n🔍 Results for query '{query}': {len(query_results)} products")
        print(query_results[['name', 'relevance']].head())


def main():
    """Main function to demonstrate dataset usage."""
    # Load the dataset (using parquet for speed)
    df = load_dataset(format="parquet")
    
    # Analyze the dataset
    analyze_dataset(df)
    
    # Show filtering examples
    filter_examples(df)
    
    print("\n" + "=" * 60)
    print("✨ Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
