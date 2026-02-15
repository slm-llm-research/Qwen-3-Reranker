"""Data preprocessing utilities."""

from typing import Optional

import pandas as pd
import numpy as np

from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocess the Home Depot dataset for model training.
    
    This class provides methods for cleaning, transforming, and preparing
    the dataset for machine learning models.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        logger.info("Initialized DataPreprocessor")
    
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text columns (query, name, description).
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with cleaned text
        """
        logger.info("Cleaning text columns...")
        df_clean = df.copy()
        
        text_columns = ["query", "name", "description"]
        
        for col in text_columns:
            if col in df_clean.columns:
                # Convert to string and handle missing values
                df_clean[col] = df_clean[col].fillna("").astype(str)
                
                # Remove extra whitespace
                df_clean[col] = df_clean[col].str.strip()
                df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
                
                # Convert to lowercase (optional, can be made configurable)
                # df_clean[col] = df_clean[col].str.lower()
        
        logger.info("Text cleaning completed")
        return df_clean
    
    def create_combined_text(
        self,
        df: pd.DataFrame,
        separator: str = " [SEP] ",
    ) -> pd.DataFrame:
        """
        Create combined text field for model input.
        
        Args:
            df: Input DataFrame
            separator: Separator between query and product text
        
        Returns:
            DataFrame with 'combined_text' column
        """
        logger.info("Creating combined text field...")
        df_combined = df.copy()
        
        # Combine query, name, and optionally description
        df_combined["combined_text"] = (
            df_combined["query"].fillna("") +
            separator +
            df_combined["name"].fillna("") +
            " " +
            df_combined["description"].fillna("").str[:200]  # Truncate description
        )
        
        logger.info("Combined text field created")
        return df_combined
    
    def create_relevance_labels(
        self,
        df: pd.DataFrame,
        num_classes: int = 3,
    ) -> pd.DataFrame:
        """
        Create categorical relevance labels from continuous scores.
        
        Args:
            df: Input DataFrame
            num_classes: Number of relevance classes (2 or 3)
        
        Returns:
            DataFrame with 'relevance_label' column
        
        Raises:
            ValueError: If num_classes is not 2 or 3
        """
        if num_classes not in [2, 3]:
            raise ValueError("num_classes must be 2 or 3")
        
        logger.info(f"Creating {num_classes}-class relevance labels...")
        df_labeled = df.copy()
        
        if num_classes == 2:
            # Binary: high (2.5+) vs low (<2.5)
            df_labeled["relevance_label"] = (
                df_labeled["relevance"] >= 2.5
            ).astype(int)
            label_map = {0: "not_relevant", 1: "relevant"}
        else:  # num_classes == 3
            # Three classes: low, medium, high
            df_labeled["relevance_label"] = pd.cut(
                df_labeled["relevance"],
                bins=[0, 1.8, 2.5, 3.0],
                labels=[0, 1, 2],
                include_lowest=True,
            ).astype(int)
            label_map = {0: "low", 1: "medium", 2: "high"}
        
        df_labeled["relevance_label_name"] = df_labeled["relevance_label"].map(label_map)
        
        logger.info(f"Relevance label distribution:\n{df_labeled['relevance_label'].value_counts()}")
        return df_labeled
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "fill",
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ("fill" or "drop")
        
        Returns:
            DataFrame with handled missing values
        
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy not in ["fill", "drop"]:
            raise ValueError("strategy must be 'fill' or 'drop'")
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        df_handled = df.copy()
        
        if strategy == "fill":
            # Fill text columns with empty string
            text_columns = ["query", "name", "description"]
            for col in text_columns:
                if col in df_handled.columns:
                    df_handled[col] = df_handled[col].fillna("")
            
            # Fill numeric columns with median
            numeric_columns = df_handled.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_handled[col].isnull().any():
                    median_val = df_handled[col].median()
                    df_handled[col] = df_handled[col].fillna(median_val)
        
        else:  # strategy == "drop"
            initial_len = len(df_handled)
            df_handled = df_handled.dropna()
            logger.info(f"Dropped {initial_len - len(df_handled)} rows with missing values")
        
        return df_handled
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for modeling.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating additional features...")
        df_features = df.copy()
        
        # Text length features
        df_features["query_length"] = df_features["query"].str.len()
        df_features["name_length"] = df_features["name"].str.len()
        df_features["description_length"] = df_features["description"].str.len()
        
        # Word count features
        df_features["query_word_count"] = df_features["query"].str.split().str.len()
        df_features["name_word_count"] = df_features["name"].str.split().str.len()
        
        # Query in name feature (simple text matching)
        df_features["query_in_name"] = df_features.apply(
            lambda row: int(
                str(row["query"]).lower() in str(row["name"]).lower()
            ),
            axis=1,
        )
        
        logger.info(f"Created {6} additional features")
        return df_features
    
    def preprocess(
        self,
        df: pd.DataFrame,
        clean_text: bool = True,
        create_combined: bool = True,
        create_labels: bool = False,
        num_classes: int = 3,
        create_features: bool = True,
    ) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            clean_text: Whether to clean text columns
            create_combined: Whether to create combined text field
            create_labels: Whether to create categorical labels
            num_classes: Number of classes for labels (if create_labels=True)
            create_features: Whether to create additional features
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")
        df_processed = df.copy()
        
        if clean_text:
            df_processed = self.clean_text(df_processed)
        
        if create_combined:
            df_processed = self.create_combined_text(df_processed)
        
        if create_labels:
            df_processed = self.create_relevance_labels(df_processed, num_classes)
        
        if create_features:
            df_processed = self.create_features(df_processed)
        
        logger.info("Preprocessing pipeline completed")
        return df_processed
