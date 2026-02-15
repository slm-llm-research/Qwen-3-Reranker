"""Dataset downloader module."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset

from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetDownloader:
    """
    Download and save the Home Depot dataset from Hugging Face.
    
    This class handles downloading the dataset and saving it in multiple formats
    for different use cases (CSV for compatibility, Parquet for performance, etc.).
    """
    
    DATASET_NAME = "bstds/home_depot"
    SUPPORTED_FORMATS = ["csv", "parquet", "json", "datasets"]
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir: Directory to save the dataset (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DatasetDownloader with data_dir: {self.data_dir}")
    
    def download(
        self,
        save_formats: Optional[List[str]] = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Download the dataset from Hugging Face.
        
        Args:
            save_formats: List of formats to save the dataset in.
                         Options: ["csv", "parquet", "json", "datasets"]
                         Default: ["csv", "parquet", "json"]
            force_download: If True, download even if files exist
        
        Returns:
            DataFrame containing the dataset
        
        Raises:
            ValueError: If an unsupported format is specified
            Exception: If download fails
        """
        if save_formats is None:
            save_formats = ["csv", "parquet", "json"]
        
        # Validate formats
        invalid_formats = set(save_formats) - set(self.SUPPORTED_FORMATS)
        if invalid_formats:
            raise ValueError(
                f"Unsupported formats: {invalid_formats}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        
        # Check if already downloaded
        if not force_download and self._check_files_exist(save_formats):
            logger.info("Dataset files already exist. Use force_download=True to re-download.")
            return self._load_existing()
        
        logger.info(f"Downloading dataset from Hugging Face: {self.DATASET_NAME}")
        
        try:
            dataset = load_dataset(self.DATASET_NAME)
            train_data = dataset["train"]
            
            logger.info(f"Successfully downloaded {len(train_data):,} records")
            logger.info(f"Columns: {', '.join(train_data.column_names)}")
            
            # Convert to pandas
            df = train_data.to_pandas()
            
            # Log statistics
            self._log_statistics(df)
            
            # Save in requested formats
            self._save_dataset(df, train_data, save_formats)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def _check_files_exist(self, formats: List[str]) -> bool:
        """Check if dataset files already exist for given formats."""
        format_paths = {
            "csv": self.data_dir / "home_depot.csv",
            "parquet": self.data_dir / "home_depot.parquet",
            "json": self.data_dir / "home_depot.json",
            "datasets": self.data_dir / "home_depot_hf",
        }
        
        for fmt in formats:
            if fmt in format_paths:
                path = format_paths[fmt]
                if not path.exists():
                    return False
        
        return True
    
    def _load_existing(self) -> pd.DataFrame:
        """Load existing dataset from disk."""
        parquet_path = self.data_dir / "home_depot.parquet"
        if parquet_path.exists():
            logger.info(f"Loading existing dataset from {parquet_path}")
            return pd.read_parquet(parquet_path)
        
        csv_path = self.data_dir / "home_depot.csv"
        if csv_path.exists():
            logger.info(f"Loading existing dataset from {csv_path}")
            return pd.read_csv(csv_path)
        
        raise FileNotFoundError("No existing dataset files found")
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Log dataset statistics."""
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Relevance score statistics:\n{df['relevance'].describe()}")
        logger.info(f"Unique products: {df['entity_id'].nunique():,}")
        logger.info(f"Unique queries: {df['query'].nunique():,}")
    
    def _save_dataset(
        self,
        df: pd.DataFrame,
        hf_dataset,
        formats: List[str],
    ) -> None:
        """Save dataset in requested formats."""
        logger.info(f"Saving dataset to '{self.data_dir}' directory...")
        
        if "csv" in formats:
            csv_path = self.data_dir / "home_depot.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved as CSV: {csv_path} ({self._get_file_size(csv_path)})")
        
        if "parquet" in formats:
            parquet_path = self.data_dir / "home_depot.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved as Parquet: {parquet_path} ({self._get_file_size(parquet_path)})")
        
        if "json" in formats:
            json_path = self.data_dir / "home_depot.json"
            df.to_json(json_path, orient="records", lines=True)
            logger.info(f"Saved as JSON Lines: {json_path} ({self._get_file_size(json_path)})")
        
        if "datasets" in formats:
            datasets_path = self.data_dir / "home_depot_hf"
            hf_dataset.save_to_disk(str(datasets_path))
            logger.info(f"Saved as Hugging Face Dataset: {datasets_path}")
    
    @staticmethod
    def _get_file_size(path: Path) -> str:
        """Get human-readable file size."""
        size_bytes = path.stat().st_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get information about the downloaded dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "data_dir": str(self.data_dir),
            "dataset_name": self.DATASET_NAME,
            "files": {},
        }
        
        for format_name, filename in [
            ("csv", "home_depot.csv"),
            ("parquet", "home_depot.parquet"),
            ("json", "home_depot.json"),
        ]:
            file_path = self.data_dir / filename
            if file_path.exists():
                info["files"][format_name] = {
                    "path": str(file_path),
                    "size": self._get_file_size(file_path),
                    "exists": True,
                }
            else:
                info["files"][format_name] = {"exists": False}
        
        return info
