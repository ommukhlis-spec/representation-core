"""
Metrics for evaluating and comparing representations.

Representation quality is NOT a scalar — it is a vector:

    Q(R) = (C, H, I, E, X)

where:
  C = complexity (representation cost)
  H = hypothesis space (log-size)
  I = invariance count
  E = explainability
  X = execution cost (estimated reasoning cost)

Ranking is a PARTIAL ORDER: R_i dominates R_j only if R_i is strictly
better on at least one dimension and not worse on any other.

All metrics are:
  - Comparable across different representations
  - Computable from the representation alone (no oracle needed)
  - Grounded in measurable quantities (no hand-waving)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.representation import Observation, Representation


# ─────────────────────────────────────────────────────────────────────
# Quality Vector — the multidimensional representation of quality
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QualityVector:
    """
    A multidimensional quality assessment of a representation.

    Q(R | D) = (C, H, I, E, X)

    This is the PRIMARY abstraction for comparing representations.
    Scalar metrics (tension, compression ratio) are DERIVED from this
    vector for convenience, but the vector is the ground truth.
    """

    complexity: float        # C — representation cost in bits-equivalent
    hypothesis_log_size: float  # H — log2(|hypothesis space|)
    invariance_count: int    # I — number of invariants made explicit
    explainability: float    # E — structural interpretability score [0, 1]
    execution_cost: float    # X — estimated reasoning cost

    # Convenience accessors
    @property
    def C(self) -> float:
        return self.complexity

    @property
    def H(self) -> float:
        return self.hypothesis_log_size

    @property
    def I(self) -> int:
        return self.invariance_count

    @property
    def E(self) -> float:
        return self.explainability

    @property
    def X(self) -> float:
        return self.execution_cost

    def to_tuple(self) -> Tuple[float, float, int, float, float]:
        return (self.C, self.H, self.I, self.E, self.X)

    def to_dict(self) -> Dict[str, float]:
        return {
            "complexity": self.C,
            "hypothesis_log_size": self.H,
            "invariance_count": self.I,
            "explainability": self.E,
            "execution_cost": self.X,
        }

    # ── Partial order operations ────────────────────────────────────

    def dominates(self, other: QualityVector) -> bool:
        """
        Check if self strictly dominates other.

        Dominance: self is at least as good on ALL dimensions AND
        strictly better on at least one.

        For C, H, X: lower is better → negate for comparison.
        For I, E: higher is better.
        """
        better_dims = 0
        # Lower is better
        if self.C <= other.C:
            if self.C < other.C:
                better_dims += 1
        else:
            return False
        if self.H <= other.H:
            if self.H < other.H:
                better_dims += 1
        else:
            return False
        if self.X <= other.X:
            if self.X < other.X:
                better_dims += 1
        else:
            return False
        # Higher is better
        if self.I >= other.I:
            if self.I > other.I:
                better_dims += 1
        else:
            return False
        if self.E >= other.E:
            if self.E > other.E:
                better_dims += 1
        else:
            return False
        return better_dims > 0

    def is_dominated_by(self, other: QualityVector) -> bool:
        """Check if other strictly dominates self."""
        return other.dominates(self)

    def is_equivalent_to(self, other: QualityVector) -> bool:
        """Two quality vectors are equivalent if no dimension differs."""
        return (
            self.C == other.C
            and self.H == other.H
            and self.I == other.I
            and self.E == other.E
            and self.X == other.X
        )

    def is_comparable_with(self, other: QualityVector) -> bool:
        """Two vectors are comparable if one dominates the other."""
        return self.dominates(other) or self.is_dominated_by(other)

    def distance_to(self, other: QualityVector) -> float:
        """Euclidean distance in quality space (normalized)."""
        # Normalize each dimension to [0, 1] range for comparison
        # Using soft normalization based on typical ranges
        c_norm = abs(self.C - other.C) / max(abs(self.C), abs(other.C), 1.0)
        h_norm = abs(self.H - other.H) / max(abs(self.H), abs(other.H), 1.0)
        i_norm = abs(self.I - other.I) / max(abs(self.I), abs(other.I), 1.0)
        e_norm = abs(self.E - other.E)
        x_norm = abs(self.X - other.X) / max(abs(self.X), abs(other.X), 1.0)
        return float(np.sqrt(c_norm**2 + h_norm**2 + i_norm**2 + e_norm**2 + x_norm**2))

    def __repr__(self) -> str:
        return (
            f"Q(C={self.C:.1f}, H={self.H:.1f}, I={self.I}, "
            f"E={self.E:.2f}, X={self.X:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────
# Quality vector computation
# ─────────────────────────────────────────────────────────────────────

def compute_quality(
    rep: Representation, observation: Observation
) -> QualityVector:
    """
    Compute the full quality vector for a representation on an observation.

    This is the PRIMARY metric function. All scalar metrics are derived
    from this vector.
    """
    C = rep.complexity(observation)
    H = np.log2(float(max(rep.hypothesis_space_size(observation), 1)))
    I = len(rep.invariants(observation))
    X = rep.estimated_reasoning_cost(observation)

    # Explainability: proxy based on whether explain() produces
    # non-trivial output. Could be replaced with a more rigorous measure.
    explanation = rep.explain(observation)
    E = min(len(explanation) / 500.0, 1.0)  # crude heuristic — to be refined

    return QualityVector(
        complexity=C,
        hypothesis_log_size=H,
        invariance_count=I,
        explainability=E,
        execution_cost=X,
    )


# ─────────────────────────────────────────────────────────────────────
# Scalar metrics (derived from QualityVector for convenience)
# ─────────────────────────────────────────────────────────────────────

def compression_ratio(
    rep: Representation, observation: Observation
) -> float:
    """
    Ratio of raw size to encoded size.

    Higher = more compression. Derived from QualityVector.C.
    """
    raw_bits = observation.height * observation.width * np.log2(10)
    encoded_bits = rep.complexity(observation)
    if encoded_bits == 0:
        return float("inf")
    return raw_bits / encoded_bits


def invariance_density(
    rep: Representation, observation: Observation
) -> float:
    """
    Invariants per unit of representational complexity.

    Higher = more structural insight per unit cost.
    Derived from QualityVector.I / QualityVector.C.
    """
    q = compute_quality(rep, observation)
    if q.C == 0:
        return float("inf")
    return q.I / q.C


def representation_tension(
    rep: Representation, observation: Observation
) -> float:
    """
    Heuristic proxy for reasoning cost.

    Tension(R) = C + α · H

    Lower tension → simpler representation + smaller hypothesis space.

    IMPORTANT: This is a HEURISTIC, not a theorem. Its validity must be
    empirically tested by correlating tension with actual reasoning cost.

    Derived from QualityVector.C and QualityVector.H.
    """
    q = compute_quality(rep, observation)
    alpha = 0.5
    return float(q.C + alpha * q.H)


# ─────────────────────────────────────────────────────────────────────
# Comparative analysis (vector-aware)
# ─────────────────────────────────────────────────────────────────────

def compare_representations(
    representations: List[Representation],
    observation: Observation,
) -> Dict[str, QualityVector]:
    """
    Compute quality vectors for multiple representations.

    Returns:
        {rep_name: QualityVector}
    """
    return {rep.name: compute_quality(rep, observation) for rep in representations}


def find_pareto_front(
    representations: List[Representation],
    observation: Observation,
) -> List[Representation]:
    """
    Find representations on the Pareto front — those not dominated
    by any other representation in the set.

    These are the "non-dominated" representations. Any one of them
    could be the best choice depending on task requirements.
    """
    vectors = {rep: compute_quality(rep, observation) for rep in representations}
    front: List[Representation] = []

    for rep_i, q_i in vectors.items():
        dominated = False
        for rep_j, q_j in vectors.items():
            if rep_i is rep_j:
                continue
            if q_j.dominates(q_i):
                dominated = True
                break
        if not dominated:
            front.append(rep_i)

    return front


def rank_representations(
    representations: List[Representation],
    observation: Observation,
    by: str = "tension",
) -> List[Representation]:
    """
    Rank representations by a single scalar criterion.

    For multi-dimensional comparison, use find_pareto_front() instead.

    Default: rank by tension (lowest first).
    """
    if by == "tension":
        key = lambda r: representation_tension(r, observation)
        reverse = False
    elif by == "compression_ratio":
        key = lambda r: compression_ratio(r, observation)
        reverse = True
    elif by == "execution_cost":
        key = lambda r: r.estimated_reasoning_cost(observation)
        reverse = False
    elif by == "complexity":
        key = lambda r: r.complexity(observation)
        reverse = False
    elif by == "invariance_count":
        key = lambda r: len(r.invariants(observation))
        reverse = True  # more invariants = better
    else:
        raise ValueError(f"Unknown ranking criterion: {by}")

    return sorted(representations, key=key, reverse=reverse)


def dominance_matrix(
    representations: List[Representation],
    observation: Observation,
) -> Dict[str, Dict[str, str]]:
    """
    Build a dominance matrix: for each pair (R_i, R_j), report
    whether R_i dominates R_j, is dominated by R_j, they are
    equivalent, or incomparable.
    """
    vectors = {rep.name: compute_quality(rep, observation) for rep in representations}
    names = list(vectors.keys())
    matrix: Dict[str, Dict[str, str]] = {}

    for name_i in names:
        matrix[name_i] = {}
        for name_j in names:
            if name_i == name_j:
                matrix[name_i][name_j] = "="
            elif vectors[name_i].dominates(vectors[name_j]):
                matrix[name_i][name_j] = ">"
            elif vectors[name_i].is_dominated_by(vectors[name_j]):
                matrix[name_i][name_j] = "<"
            elif vectors[name_i].is_equivalent_to(vectors[name_j]):
                matrix[name_i][name_j] = "="
            else:
                matrix[name_i][name_j] = "∥"  # incomparable

    return matrix
