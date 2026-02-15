"""CLI command for evaluating Qwen reranker."""

import sys
from pathlib import Path

# Import the evaluation script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.evaluate_reranker import main


def evaluate_cli():
    """Entry point for ranking-evaluate CLI command."""
    main()


if __name__ == "__main__":
    evaluate_cli()
