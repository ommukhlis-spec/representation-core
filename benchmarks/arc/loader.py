"""
ARC-AGI task loader.

Converts ARC JSON task files into Observation objects consumable
by the representation framework.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.representation import Observation


def grid_to_observation(grid: List[List[int]]) -> Observation:
    """Convert an ARC grid (list of lists) to an Observation."""
    arr = np.array(grid, dtype=np.int32)
    return Observation(grid=arr)


def load_task(task_path: str) -> Dict[str, List[Observation]]:
    """
    Load a single ARC task from a JSON file.

    Returns:
        {
            "train": [(input_obs, output_obs), ...],
            "test":  [(input_obs, output_obs), ...],
        }
    """
    with open(task_path, "r") as f:
        data = json.load(f)

    result: Dict[str, List[Tuple[Observation, Observation]]] = {
        "train": [],
        "test": [],
    }

    for pair in data.get("train", []):
        inp = grid_to_observation(pair["input"])
        out = grid_to_observation(pair["output"])
        result["train"].append((inp, out))

    for pair in data.get("test", []):
        inp = grid_to_observation(pair["input"])
        # Test outputs may be absent or placeholder
        if "output" in pair and pair["output"]:
            out = grid_to_observation(pair["output"])
        else:
            out = None
        result["test"].append((inp, out))

    return result


def load_dataset(dataset_dir: str) -> List[Dict[str, List[Tuple[Observation, Observation]]]]:
    """
    Load all ARC tasks from a directory.

    Args:
        dataset_dir: Path to directory containing ARC JSON task files.

    Returns:
        List of task dicts (as returned by load_task).
    """
    tasks = []
    for fname in sorted(os.listdir(dataset_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(dataset_dir, fname)
            tasks.append(load_task(fpath))
    return tasks
