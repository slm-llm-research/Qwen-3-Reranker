"""CLI tool for preprocessing the dataset."""

import argparse
import sys
from pathlib import Path

from ranking_qwen.data.dataset_loader import HomeDepotDataset
from ranking_qwen.data.preprocessor import DataPreprocessor
from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function for the preprocess CLI."""
    parser = argparse.ArgumentParser(
        description="Preprocess the Home Depot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preprocessing pipeline
  python -m ranking_qwen.cli.preprocess --output data/preprocessed.parquet
  
  # Clean text only
  python -m ranking_qwen.cli.preprocess --clean-text-only --output data/clean.csv
  
  # Create labels for classification
  python -m ranking_qwen.cli.preprocess --create-labels --num-classes 3
        """,
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to input dataset (default: data)",
    )
    
    parser.add_argument(
        "--input-format",
        type=str,
        default="parquet",
        choices=["csv", "parquet", "json"],
        help="Input dataset format (default: parquet)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (format inferred from extension)",
    )
    
    parser.add_argument(
        "--clean-text",
        action="store_true",
        default=True,
        help="Clean text columns (default: True)",
    )
    
    parser.add_argument(
        "--no-clean-text",
        action="store_false",
        dest="clean_text",
        help="Skip text cleaning",
    )
    
    parser.add_argument(
        "--create-combined",
        action="store_true",
        default=True,
        help="Create combined text field (default: True)",
    )
    
    parser.add_argument(
        "--create-labels",
        action="store_true",
        help="Create categorical relevance labels",
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of relevance classes (default: 3)",
    )
    
    parser.add_argument(
        "--create-features",
        action="store_true",
        default=True,
        help="Create additional features (default: True)",
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
        logger.info("Starting preprocessing...")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = HomeDepotDataset(
            data_path=args.data_path,
            format=args.input_format,
        )
        df = dataset.load()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Run preprocessing
        df_processed = preprocessor.preprocess(
            df,
            clean_text=args.clean_text,
            create_combined=args.create_combined,
            create_labels=args.create_labels,
            num_classes=args.num_classes,
            create_features=args.create_features,
        )
        
        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving preprocessed dataset to {output_path}")
        
        if output_path.suffix == ".csv":
            df_processed.to_csv(output_path, index=False)
        elif output_path.suffix == ".parquet":
            df_processed.to_parquet(output_path, index=False)
        elif output_path.suffix == ".json":
            df_processed.to_json(output_path, orient="records", lines=True)
        else:
            logger.warning(f"Unknown format: {output_path.suffix}, saving as CSV")
            df_processed.to_csv(output_path, index=False)
        
        logger.info(f"Preprocessed {len(df_processed):,} records")
        logger.info(f"Number of columns: {len(df_processed.columns)}")
        logger.info(f"New columns: {', '.join(df_processed.columns)}")
        
        logger.info("\n✅ Preprocessing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
