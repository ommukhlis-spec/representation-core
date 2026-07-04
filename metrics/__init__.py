"""
Metrics for evaluating and comparing representations.

Primary abstraction: QualityVector — multidimensional quality assessment.
"""

from .metrics import (
    QualityVector,
    compute_quality,
    compare_representations,
    compression_ratio,
    dominance_matrix,
    find_pareto_front,
    invariance_density,
    rank_representations,
    representation_tension,
)

__all__ = [
    "QualityVector",
    "compute_quality",
    "compare_representations",
    "compression_ratio",
    "dominance_matrix",
    "find_pareto_front",
    "invariance_density",
    "rank_representations",
    "representation_tension",
]
