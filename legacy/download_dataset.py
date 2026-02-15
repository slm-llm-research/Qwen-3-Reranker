#!/usr/bin/env python3
"""
Download the Home Depot dataset from Hugging Face.

This script downloads the Home Depot product search relevance dataset
and saves it in multiple formats for easy access.

Dataset Info:
- Contains product search terms and relevance scores (1-3)
- Relevance: 1 (not relevant) to 3 (highly relevant)
- ~74k search term/product pairs with human-rated relevance scores
"""

import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def download_home_depot_dataset(data_dir: str = "data", save_formats: list = ["csv", "parquet", "json"]):
    """
    Download the Home Depot dataset from Hugging Face.
    
    Args:
        data_dir: Directory to save the dataset (default: "data")
        save_formats: List of formats to save the dataset in 
                     Options: ["csv", "parquet", "json", "datasets"]
                     (default: ["csv", "parquet", "json"])
    
    Returns:
        dataset: The loaded dataset object
    """
    print("=" * 60)
    print("Downloading Home Depot Dataset")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset from Hugging Face
    print("\n📥 Downloading dataset from Hugging Face...")
    print("Dataset: bstds/home_depot")
    
    try:
        dataset = load_dataset("bstds/home_depot")
        print("✅ Dataset downloaded successfully!")
        
        # Get the train split
        train_data = dataset["train"]
        print(f"\n📊 Dataset Statistics:")
        print(f"   - Number of rows: {len(train_data):,}")
        print(f"   - Number of columns: {len(train_data.column_names)}")
        print(f"   - Columns: {', '.join(train_data.column_names)}")
        
        # Convert to pandas for easier manipulation and saving
        df = train_data.to_pandas()
        
        # Display basic info
        print(f"\n📈 Data Preview:")
        print(df.head())
        print(f"\n📉 Relevance Score Distribution:")
        print(df['relevance'].describe())
        
        # Save in requested formats
        print(f"\n💾 Saving dataset to '{data_dir}' directory...")
        
        if "csv" in save_formats:
            csv_path = data_path / "home_depot.csv"
            df.to_csv(csv_path, index=False)
            print(f"   ✅ Saved as CSV: {csv_path}")
        
        if "parquet" in save_formats:
            parquet_path = data_path / "home_depot.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"   ✅ Saved as Parquet: {parquet_path}")
        
        if "json" in save_formats:
            json_path = data_path / "home_depot.json"
            df.to_json(json_path, orient="records", lines=True)
            print(f"   ✅ Saved as JSON Lines: {json_path}")
        
        if "datasets" in save_formats:
            datasets_path = data_path / "home_depot_hf"
            train_data.save_to_disk(str(datasets_path))
            print(f"   ✅ Saved as Hugging Face Dataset: {datasets_path}")
        
        print(f"\n✨ All done! Dataset ready for use.")
        print(f"\n📁 Files saved in: {data_path.absolute()}")
        
        return dataset
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        raise


def main():
    """Main function to run the download script."""
    # Download and save the dataset
    dataset = download_home_depot_dataset(
        data_dir="data",
        save_formats=["csv", "parquet", "json"]
    )
    
    print("\n" + "=" * 60)
    print("Dataset Information:")
    print("=" * 60)
    print("""
The Home Depot dataset contains:
- id: Unique identifier for each record
- entity_id: Product entity ID
- name: Product name
- query: Search query term
- relevance: Relevance score (1.0 - 3.0)
- description: Product description

Relevance Scores:
- 3.0: Highly relevant
- 2.0: Moderately relevant  
- 1.0: Not relevant

Use cases:
- Search relevance modeling
- Learning to rank
- Information retrieval
- Product recommendation systems
    """)


if __name__ == "__main__":
    main()
