"""CLI tool for analyzing the dataset."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from ranking_qwen.data.dataset_loader import HomeDepotDataset
from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function for the analyze CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze the Home Depot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset from default directory
  python -m ranking_qwen.cli.analyze
  
  # Analyze specific file
  python -m ranking_qwen.cli.analyze --data-path data/home_depot.parquet
  
  # Show detailed statistics
  python -m ranking_qwen.cli.analyze --detailed
        """,
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to dataset file or directory (default: data)",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["csv", "parquet", "json"],
        help="Dataset format (default: parquet)",
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statistics and analysis",
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top queries to show (default: 10)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.setLevel(args.log_level)
    
    try:
        logger.info("Loading dataset...")
        
        # Load dataset
        dataset = HomeDepotDataset(
            data_path=args.data_path,
            format=args.format,
        )
        df = dataset.load()
        
        # Get statistics
        stats = dataset.get_statistics()
        
        # Print basic statistics
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        print(f"Total Records:       {stats['total_records']:,}")
        print(f"Number of Columns:   {stats['num_columns']}")
        print(f"Unique Products:     {stats['unique_products']:,}")
        print(f"Unique Queries:      {stats['unique_queries']:,}")
        
        print("\nRelevance Score Statistics:")
        for key, value in stats['relevance_stats'].items():
            print(f"  {key:10s}: {value:.4f}")
        
        print("\nRelevance Distribution:")
        relevance_dist = pd.Series(stats['relevance_distribution']).sort_index()
        for score, count in relevance_dist.items():
            pct = (count / stats['total_records']) * 100
            print(f"  Score {score:.1f}: {count:6,} ({pct:5.1f}%)")
        
        # Top queries
        print(f"\nTop {args.top_n} Most Common Queries:")
        top_queries = dataset.get_top_queries(n=args.top_n)
        for i, (query, count) in enumerate(top_queries.items(), 1):
            print(f"  {i:2d}. '{query}' ({count} times)")
        
        # Detailed analysis
        if args.detailed:
            print("\n" + "=" * 70)
            print("DETAILED ANALYSIS")
            print("=" * 70)
            
            # Query-level statistics
            query_stats = dataset.analyze_relevance_by_query()
            
            print("\nTop 10 Queries by Average Relevance:")
            for i, row in query_stats.head(10).iterrows():
                print(
                    f"  '{row['query'][:50]}': "
                    f"avg={row['avg_relevance']:.2f}, "
                    f"count={row['num_products']}"
                )
            
            print("\nBottom 10 Queries by Average Relevance:")
            for i, row in query_stats.tail(10).iterrows():
                print(
                    f"  '{row['query'][:50]}': "
                    f"avg={row['avg_relevance']:.2f}, "
                    f"count={row['num_products']}"
                )
            
            # Text length statistics
            print("\nText Length Statistics:")
            print(f"  Query length (avg):       {df['query'].str.len().mean():.1f} chars")
            print(f"  Product name length (avg): {df['name'].str.len().mean():.1f} chars")
            print(f"  Description length (avg):  {df['description'].str.len().mean():.1f} chars")
        
        print("\n" + "=" * 70)
        print("✅ Analysis completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
