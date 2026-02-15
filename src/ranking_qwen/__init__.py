"""
Ranking-Qwen: A production-ready framework for search ranking and relevance modeling.

This package provides tools for downloading, processing, and modeling the Home Depot
product search relevance dataset.
"""

__version__ = "0.1.0"
__author__ = "ML Engineering Team"
__email__ = "ml-team@example.com"

from ranking_qwen.data.dataset_loader import HomeDepotDataset
from ranking_qwen.utils.logger import get_logger

__all__ = [
    "HomeDepotDataset",
    "get_logger",
    "__version__",
]
