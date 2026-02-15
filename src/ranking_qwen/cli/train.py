"""CLI command for training Qwen reranker."""

import sys
from pathlib import Path

# Import the training script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.train_reranker import main


def train_cli():
    """Entry point for ranking-train CLI command."""
    main()


if __name__ == "__main__":
    train_cli()
