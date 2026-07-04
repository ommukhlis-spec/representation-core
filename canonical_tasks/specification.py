"""
Canonical Representation Task Specification.

A CanonicalTask is a synthetic puzzle designed to test a specific
property of representations. Unlike ARC tasks (which are black-box
and may require multiple representations to solve), each canonical
task has a KNOWN ground truth:

  - Which representation SHOULD be the best fit
  - Which representation SHOULD fail
  - WHY (the scientific justification)

This transforms representation evaluation from "does it solve ARC?"
to "does the framework correctly identify the right representation
for the right reason?"

Design principles:
  1. Minimal — the simplest grid that isolates one property
  2. Diagnostic — the best/worst representation is unambiguous
  3. Justified — a clear scientific reason, not just empirical ranking
  4. Reproducible — purely synthetic, no external data dependency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from core.representation import Observation


# ─────────────────────────────────────────────────────────────────────
# Task specification
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CanonicalTask:
    """
    A synthetic task that tests a specific representational capability.

    Each task declares:
      - The observation grid(s)
      - Which representation(s) SHOULD perform best
      - Which SHOULD perform worst
      - The scientific justification

    This is the "admission exam" for any new representation added to the zoo.
    """

    # Identity
    task_id: str          # unique identifier, e.g. "count_discrete_objects"
    name: str             # human-readable, e.g. "Count Discrete Objects"
    category: str         # "counting", "symmetry", "topology", "region", "constraint"

    # Description
    description: str      # what the task tests
    justification: str    # WHY the expected representation should win

    # Failure mode this task diagnoses
    failure_mode: str = ""  # e.g. "object_merge_ambiguity", "reflection_ambiguity"

    # Data
    train_inputs: List[Observation] = field(default_factory=list)
    train_outputs: List[Observation] = field(default_factory=list)
    test_input: Optional[Observation] = None

    # Ground truth — which representation(s) should be best/worst
    best_representation: str = ""        # e.g. "object"
    worst_representation: str = ""       # e.g. "pixel"
    also_acceptable: List[str] = field(default_factory=list)  # tied alternatives

    # Tags for classification
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.justification:
            raise ValueError(f"Task {self.task_id}: justification is required")
        if not self.best_representation:
            raise ValueError(f"Task {self.task_id}: best_representation is required")

    def __repr__(self) -> str:
        return f"CanonicalTask({self.task_id}, best={self.best_representation})"


# ─────────────────────────────────────────────────────────────────────
# Helper to construct Observations inline
# ─────────────────────────────────────────────────────────────────────

def grid(data: List[List[int]]) -> Observation:
    """Create an Observation from a list-of-lists."""
    return Observation(grid=np.array(data, dtype=np.int32))


def empty_grid(h: int, w: int) -> Observation:
    """Create an empty (all-zero) observation."""
    return Observation(grid=np.zeros((h, w), dtype=np.int32))
