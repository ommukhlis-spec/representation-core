"""
Concrete representation implementations.

Each representation implements the abstract Representation contract
for a specific way of structuring ARC grid observations.

The zoo will grow to include:
  - pixel      : raw grid (identity encoding)
  - object     : object graph (connected components, properties)
  - region     : spatial regions (segmentation, voronoi-like)
  - symmetry   : symmetry groups (rotational, reflectional, translational)
  - topology   : topological structure (connectivity, holes, adjacency)
  - constraint  : constraint networks (relations, invariants between parts)
"""

from .pixel import PixelRepresentation
from .object import ObjectRepresentation
from .symmetry import SymmetryRepresentation

__all__ = [
    "PixelRepresentation",
    "ObjectRepresentation",
    "SymmetryRepresentation",
]
