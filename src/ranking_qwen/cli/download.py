"""CLI tool for downloading the dataset."""

import argparse
import sys
from pathlib import Path

from ranking_qwen.data.downloader import DatasetDownloader
from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function for the download CLI."""
    parser = argparse.ArgumentParser(
        description="Download the Home Depot dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with default settings (CSV, Parquet, JSON)
  python -m ranking_qwen.cli.download
  
  # Download only Parquet format
  python -m ranking_qwen.cli.download --formats parquet
  
  # Download to custom directory
  python -m ranking_qwen.cli.download --data-dir ./my_data
  
  # Force re-download even if files exist
  python -m ranking_qwen.cli.download --force
        """,
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save the dataset (default: data)",
    )
    
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["csv", "parquet", "json"],
        choices=["csv", "parquet", "json", "datasets"],
        help="Formats to save the dataset in (default: csv parquet json)",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files already exist",
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
        logger.info("Starting dataset download...")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Formats: {', '.join(args.formats)}")
        
        # Initialize downloader
        downloader = DatasetDownloader(data_dir=args.data_dir)
        
        # Download dataset
        df = downloader.download(
            save_formats=args.formats,
            force_download=args.force,
        )
        
        logger.info(f"Successfully downloaded {len(df):,} records")
        
        # Print dataset info
        info = downloader.get_dataset_info()
        logger.info("\nDataset files:")
        for format_name, file_info in info["files"].items():
            if file_info.get("exists"):
                logger.info(f"  {format_name}: {file_info['path']} ({file_info['size']})")
        
        logger.info("\n✅ Download completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
