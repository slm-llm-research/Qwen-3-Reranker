"""Dataset loader with preprocessing and splitting capabilities."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class HomeDepotDataset:
    """
    Home Depot dataset loader with preprocessing and analysis capabilities.
    
    This class provides a high-level interface for loading and working with
    the Home Depot product search relevance dataset.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        format: str = "parquet",
    ):
        """
        Initialize the dataset loader.
        
        Args:
            data_path: Path to dataset file or directory
            format: Format of the dataset file ("csv", "parquet", "json")
        
        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If dataset file doesn't exist
        """
        self.data_path = Path(data_path)
        self.format = format.lower()
        self.df: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized HomeDepotDataset with path: {self.data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load the dataset from disk.
        
        Returns:
            DataFrame containing the dataset
        
        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If file doesn't exist
        """
        # If data_path is a directory, construct the file path
        if self.data_path.is_dir():
            file_map = {
                "csv": "home_depot.csv",
                "parquet": "home_depot.parquet",
                "json": "home_depot.json",
            }
            if self.format not in file_map:
                raise ValueError(f"Unsupported format: {self.format}")
            
            file_path = self.data_path / file_map[self.format]
        else:
            file_path = self.data_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading dataset from {file_path}")
        
        if self.format == "csv":
            self.df = pd.read_csv(file_path)
        elif self.format == "parquet":
            self.df = pd.read_parquet(file_path)
        elif self.format == "json":
            self.df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(
                f"Unsupported format: {self.format}. "
                "Supported: csv, parquet, json"
            )
        
        logger.info(f"Loaded {len(self.df):,} records")
        return self.df
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        
        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        stats = {
            "total_records": len(self.df),
            "num_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "unique_products": self.df["entity_id"].nunique(),
            "unique_queries": self.df["query"].nunique(),
            "relevance_stats": self.df["relevance"].describe().to_dict(),
            "relevance_distribution": self.df["relevance"].value_counts().to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
        }
        
        return stats
    
    def filter_by_relevance(
        self,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Filter dataset by relevance score.
        
        Args:
            min_score: Minimum relevance score (inclusive)
            max_score: Maximum relevance score (inclusive)
        
        Returns:
            Filtered DataFrame
        
        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        df_filtered = self.df.copy()
        
        if min_score is not None:
            df_filtered = df_filtered[df_filtered["relevance"] >= min_score]
            logger.info(f"Filtered by min_score >= {min_score}: {len(df_filtered):,} records")
        
        if max_score is not None:
            df_filtered = df_filtered[df_filtered["relevance"] <= max_score]
            logger.info(f"Filtered by max_score <= {max_score}: {len(df_filtered):,} records")
        
        return df_filtered
    
    def filter_by_query(self, queries: Union[str, List[str]]) -> pd.DataFrame:
        """
        Filter dataset by specific queries.
        
        Args:
            queries: Single query string or list of queries
        
        Returns:
            Filtered DataFrame
        
        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        if isinstance(queries, str):
            queries = [queries]
        
        df_filtered = self.df[self.df["query"].isin(queries)]
        logger.info(f"Filtered by {len(queries)} queries: {len(df_filtered):,} records")
        
        return df_filtered
    
    def split_train_test(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        group_by_query: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Proportion of the dataset to include in the test split
            random_state: Random seed for reproducibility
            group_by_query: If True, ensure queries don't appear in both splits
        
        Returns:
            Tuple of (train_df, test_df)
        
        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        if group_by_query:
            # Use GroupShuffleSplit to ensure queries don't leak between splits
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state,
            )
            
            train_idx, test_idx = next(
                splitter.split(self.df, groups=self.df["query"])
            )
            
            train_df = self.df.iloc[train_idx].reset_index(drop=True)
            test_df = self.df.iloc[test_idx].reset_index(drop=True)
            
            logger.info(
                f"Split by query groups: train={len(train_df):,}, test={len(test_df):,}"
            )
            logger.info(
                f"Unique queries - train: {train_df['query'].nunique():,}, "
                f"test: {test_df['query'].nunique():,}"
            )
        else:
            # Simple random split
            test_df = self.df.sample(frac=test_size, random_state=random_state)
            train_df = self.df.drop(test_df.index).reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            
            logger.info(
                f"Random split: train={len(train_df):,}, test={len(test_df):,}"
            )
        
        return train_df, test_df
    
    def get_top_queries(self, n: int = 10) -> pd.Series:
        """
        Get the most common queries.
        
        Args:
            n: Number of top queries to return
        
        Returns:
            Series with query counts
        
        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        return self.df["query"].value_counts().head(n)
    
    def analyze_relevance_by_query(self) -> pd.DataFrame:
        """
        Analyze relevance scores grouped by query.
        
        Returns:
            DataFrame with query-level statistics
        
        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        query_stats = self.df.groupby("query").agg({
            "relevance": ["mean", "std", "min", "max", "count"],
            "entity_id": "nunique",
        }).reset_index()
        
        query_stats.columns = [
            "query",
            "avg_relevance",
            "std_relevance",
            "min_relevance",
            "max_relevance",
            "num_products",
            "unique_products",
        ]
        
        return query_stats.sort_values("avg_relevance", ascending=False)
