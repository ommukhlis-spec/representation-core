"""
ARC-AGI benchmark integration.

Loads ARC tasks and converts them to Observations for representation analysis.

ARC is the first experimental arena — not the foundation.
The framework is designed so that other benchmarks (sudoku, maze,
planning, robotics) can be added without changing the core.
"""

from .loader import load_task, load_dataset

__all__ = ["load_task", "load_dataset"]
