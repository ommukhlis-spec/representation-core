"""
Canonical Representation Tasks.

Each module defines tasks where a specific representation is the
unambiguously correct choice. These tasks serve as the admission
exam for new representations added to the zoo.

A new representation must:
  1. Be the BEST on at least one canonical task where no existing
     representation wins
  2. Not regress: existing winning representations must continue to win
     on their canonical tasks
"""

from .counting import (
    COUNT_DISCRETE_OBJECTS,
    COUNT_BY_COLOR,
    UNIFORM_OBJECT_SIZE,
    MANY_SMALL_OBJECTS,
)
from .symmetry import (
    HORIZONTAL_REFLECTION,
    ROTATIONAL_SYMMETRY_4,
    TRANSLATIONAL_SYMMETRY,
    NO_SYMMETRY,
)
from .topology import (
    SINGLE_HOLE,
    MULTIPLE_HOLES,
    CONNECTIVITY_CHECK,
    NESTED_CONTAINMENT,
)
from .region import (
    OVERLAPPING_REGIONS,
    CONTINUOUS_GRADIENT,
    SPATIAL_ADJACENCY_ZONES,
    REGION_FILLING,
)
from .constraint import (
    COLOR_ORDERING_RULE,
    COLOR_CONSERVATION,
    SEQUENTIAL_PATTERN,
    XOR_PATTERN,
)

__all__ = [
    # Counting
    "COUNT_DISCRETE_OBJECTS",
    "COUNT_BY_COLOR",
    "UNIFORM_OBJECT_SIZE",
    "MANY_SMALL_OBJECTS",
    # Symmetry
    "HORIZONTAL_REFLECTION",
    "ROTATIONAL_SYMMETRY_4",
    "TRANSLATIONAL_SYMMETRY",
    "NO_SYMMETRY",
    # Topology
    "SINGLE_HOLE",
    "MULTIPLE_HOLES",
    "CONNECTIVITY_CHECK",
    "NESTED_CONTAINMENT",
    # Region
    "OVERLAPPING_REGIONS",
    "CONTINUOUS_GRADIENT",
    "SPATIAL_ADJACENCY_ZONES",
    "REGION_FILLING",
    # Constraint
    "COLOR_ORDERING_RULE",
    "COLOR_CONSERVATION",
    "SEQUENTIAL_PATTERN",
    "XOR_PATTERN",
]
