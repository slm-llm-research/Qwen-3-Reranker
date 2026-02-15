"""Command-line interface tools."""

from ranking_qwen.cli.train import train_cli
from ranking_qwen.cli.evaluate import evaluate_cli

__all__ = ["train_cli", "evaluate_cli"]
