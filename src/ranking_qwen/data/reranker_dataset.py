"""Reranker dataset preparation for Qwen3-Reranker models."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class RerankerDatasetPreparator:
    """
    Prepare dataset for training Qwen3-Reranker models.
    
    This class handles:
    - Query-stratified splitting to prevent data leakage
    - Document text construction with truncation
    - Binary label creation from relevance scores
    - Message formatting for generative reranker training
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        relevance_threshold: float = 2.33,
        max_description_tokens: int = 350,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        """
        Initialize the dataset preparator.
        
        Args:
            tokenizer: Tokenizer for text processing
            relevance_threshold: Threshold for binary classification (>=threshold → relevant)
            max_description_tokens: Maximum tokens to keep from description
            instruction: Instruction text for the reranker prompt
        """
        self.tokenizer = tokenizer
        self.relevance_threshold = relevance_threshold
        self.max_description_tokens = max_description_tokens
        self.instruction = instruction
        
        logger.info(f"Initialized RerankerDatasetPreparator")
        logger.info(f"  Relevance threshold: {relevance_threshold}")
        logger.info(f"  Max description tokens: {max_description_tokens}")
    
    def truncate_description(self, text: str) -> str:
        """
        Truncate description to maximum token length.
        
        Args:
            text: Description text to truncate
        
        Returns:
            Truncated description
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_description_tokens:
            tokens = tokens[:self.max_description_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def build_document(self, row: Dict) -> str:
        """
        Build document text from product name and description.
        
        Args:
            row: Dictionary with 'name' and 'description' keys
        
        Returns:
            Combined document text
        """
        name = str(row.get('name', ''))
        description = str(row.get('description', ''))
        
        # Truncate description
        desc_truncated = self.truncate_description(description)
        
        # Combine name and description
        document = f"{name}. {desc_truncated}"
        return document
    
    def create_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary labels from relevance scores.
        
        Args:
            df: DataFrame with 'relevance' column
        
        Returns:
            DataFrame with added 'label' column
        """
        df = df.copy()
        df['label'] = (df['relevance'] >= self.relevance_threshold).astype(int)
        
        logger.info(f"Binary label distribution:")
        logger.info(f"  Relevant (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean() * 100:.1f}%)")
        logger.info(f"  Irrelevant (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean() * 100:.1f}%)")
        
        return df
    
    def normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize text fields (lowercase, strip whitespace).
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with normalized text
        """
        df = df.copy()
        
        text_columns = ['query', 'name', 'description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
                df[col] = df[col].str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                # Lowercase normalization (97% of queries already lowercase)
                df[col] = df[col].str.lower()
        
        logger.info("Text normalization completed")
        return df
    
    def stratified_split_by_query(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by query with stratification on average relevance.
        
        This ensures:
        - No query appears in multiple splits
        - Relevance distribution is preserved across splits
        
        Args:
            df: Input DataFrame with 'query' and 'relevance' columns
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        logger.info("Starting query-stratified split...")
        
        # Compute average relevance per query
        query_groups = df.groupby('query')['relevance'].mean().reset_index(name='avg_rel')
        
        # Create bins for stratification
        bins = [0, 1.67, 2.0, 2.33, 3.0]
        query_groups['bin'] = pd.cut(
            query_groups['avg_rel'],
            bins=bins,
            labels=False,
            include_lowest=True
        )
        
        # First split: train vs (val + test)
        train_queries, temp_queries = train_test_split(
            query_groups,
            test_size=(val_ratio + test_ratio),
            stratify=query_groups['bin'],
            random_state=random_state
        )
        
        # Second split: val vs test
        val_queries, test_queries = train_test_split(
            temp_queries,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_queries['bin'],
            random_state=random_state
        )
        
        # Filter original dataframe
        train_df = df[df['query'].isin(train_queries['query'])].reset_index(drop=True)
        val_df = df[df['query'].isin(val_queries['query'])].reset_index(drop=True)
        test_df = df[df['query'].isin(test_queries['query'])].reset_index(drop=True)
        
        logger.info(f"Split completed:")
        logger.info(f"  Train: {len(train_df):,} samples, {len(train_queries):,} queries")
        logger.info(f"  Val:   {len(val_df):,} samples, {len(val_queries):,} queries")
        logger.info(f"  Test:  {len(test_df):,} samples, {len(test_queries):,} queries")
        
        # Sanity check - no query overlap
        train_q = set(train_df['query'].unique())
        val_q = set(val_df['query'].unique())
        test_q = set(test_df['query'].unique())
        
        assert len(train_q & val_q) == 0, "Train-Val query overlap detected!"
        assert len(train_q & test_q) == 0, "Train-Test query overlap detected!"
        assert len(val_q & test_q) == 0, "Val-Test query overlap detected!"
        
        logger.info("✓ No query leakage across splits")
        
        return train_df, val_df, test_df
    
    def build_message(self, example: Dict) -> Dict:
        """
        Build message format for generative reranker training.
        
        Format follows Qwen3-Reranker specification:
        - messages: User prompt with instruction, query, and document
        - positive_messages: Assistant response "yes" if relevant
        - negative_messages: Assistant response "no" if irrelevant
        
        Args:
            example: Dictionary with 'query', 'document', and 'label' keys
        
        Returns:
            Dictionary with messages for training
        """
        user_content = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {example['query']}\n"
            f"<Document>: {example['document']}"
        )
        
        messages = [{'role': 'user', 'content': user_content}]
        
        # Label determines whether we have positive or negative message
        if example['label'] == 1:
            positive_messages = [[{'role': 'assistant', 'content': 'yes'}]]
            negative_messages = []
        else:
            positive_messages = []
            negative_messages = [[{'role': 'assistant', 'content': 'no'}]]
        
        return {
            'messages': messages,
            'positive_messages': positive_messages,
            'negative_messages': negative_messages,
            'label': example['label'],
            'query': example['query'],
        }
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        split_type: str = 'full',
        add_documents: bool = True,
    ) -> Dataset:
        """
        Prepare dataset for reranker training.
        
        Args:
            df: Input DataFrame
            split_type: Split type ('train', 'val', 'test', 'full')
            add_documents: Whether to build document text and messages
        
        Returns:
            HuggingFace Dataset ready for training
        """
        logger.info(f"Preparing {split_type} dataset...")
        
        df_prep = df.copy()
        
        # Normalize text
        df_prep = self.normalize_text(df_prep)
        
        # Create binary labels
        df_prep = self.create_binary_labels(df_prep)
        
        if add_documents:
            # Build document text
            logger.info("Building document text...")
            df_prep['document'] = df_prep.apply(self.build_document, axis=1)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df_prep)
        
        if add_documents:
            # Build messages for training
            logger.info("Building training messages...")
            dataset = dataset.map(self.build_message)
        
        logger.info(f"Dataset prepared: {len(dataset)} samples")
        return dataset
    
    def prepare_full_pipeline(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Complete pipeline: split and prepare all datasets.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("=" * 60)
        logger.info("Starting full dataset preparation pipeline")
        logger.info("=" * 60)
        
        # Split by query
        train_df, val_df, test_df = self.stratified_split_by_query(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )
        
        # Prepare each split
        train_dataset = self.prepare_dataset(train_df, split_type='train')
        val_dataset = self.prepare_dataset(val_df, split_type='val')
        test_dataset = self.prepare_dataset(test_df, split_type='test')
        
        logger.info("=" * 60)
        logger.info("Dataset preparation completed")
        logger.info("=" * 60)
        
        return train_dataset, val_dataset, test_dataset
