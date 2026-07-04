"""
Formal definition of Representation for Adaptive Reasoning.

A Representation is not merely an encoding. It is a structured mapping
from observation space to a symbolic space that:
  - Preserves certain invariants
  - Discards irrelevant information
  - Supports operations (reasoning) that are closed in the symbolic space
  - Can explain its own structural choices

This module defines the abstract contract that all representations must fulfill.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Observation type (ARC-specific initially, extensible later)
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    """
    An observation is a grid of discrete values (ARC-native format).

    For ARC: shape = (H, W), values in {0..9}.

    This is deliberately ARC-specific. The theory of representation
    should be born from pressure of a concrete domain. Generalization
    comes after the foundations are proven.
    """

    grid: np.ndarray  # shape (H, W), dtype int, values 0..9

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def __post_init__(self):
        if self.grid.ndim != 2:
            raise ValueError(f"Observation grid must be 2D, got shape {self.grid.shape}")
        if not np.issubdtype(self.grid.dtype, np.integer):
            raise ValueError(f"Observation values must be integers, got {self.grid.dtype}")


# ─────────────────────────────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Invariant:
    """
    A property of the observation that:
      1. Is preserved across input-output pairs (robust)
      2. Is independent of any particular representation
      3. Simplifies description when made explicit

    Examples:
      - "exactly 3 colors appear in every example"
      - "all objects have the same bounding box area"
      - "red pixels are always adjacent to blue pixels"
    """

    name: str
    description: str
    predicate: str  # executable expression or formal statement

    def __repr__(self) -> str:
        return f"Invariant({self.name})"


@dataclass(frozen=True)
class FailureSignature:
    """
    A structured description of WHY a representation fails on a task.

    Not just "it failed", but HOW it failed. The failure pattern
    determines which representation to try next.

    Failure modes:
      - overlap: objects overlap and cannot be segmented
      - non_discrete: no discrete objects exist
      - partial_symmetry: symmetry holds only approximately
      - ambiguous_topology: connectivity is unclear
      - sequential_only: pattern is not sequential
      - too_many_exceptions: too many ad-hoc rules needed
    """

    mode: str
    detail: str = ""
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"FailureSignature(mode={self.mode}, confidence={self.confidence:.2f})"


# ─────────────────────────────────────────────────────────────────────
# Transition operator
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Transition:
    """
    A transition from one representation to another.

    Not a random jump — a structured transformation guided by:
      - FailureSignature (push: why leave current R)
      - Emerging invariants (pull: what structure is visible)
    """

    source: str  # name of source representation
    target: str  # name of target representation
    trigger: str  # "failure" | "invariant" | "both"
    failure_signature: Optional[FailureSignature] = None
    emerging_invariant: Optional[Invariant] = None
    rationale: str = ""

    def __repr__(self) -> str:
        return f"Transition({self.source} -> {self.target}, trigger={self.trigger})"


# ─────────────────────────────────────────────────────────────────────
# The Representation contract
# ─────────────────────────────────────────────────────────────────────

class Representation(ABC):
    """
    Abstract contract for a representation.

    A Representation R is a triple:
      R = (E_R, OP_R, Inv_R)

    where:
      E_R   : Observation -> SymbolicSpace   (encoding)
      OP_R  : operations closed over SymbolicSpace (reasoning)
      Inv_R : invariants preserved by this representation

    Concrete implementations MUST override all @abstractmethod methods.

    Design principle:
      Every method must be implementable for a specific representation
      AND produce measurable, comparable outputs. No hand-waving.
    """

    # ── Identity ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this representation (e.g. 'object', 'symmetry')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version for reproducibility."""
        ...

    # ── Core encoding / decoding ──────────────────────────────────

    @abstractmethod
    def encode(self, observation: Observation) -> Any:
        """
        Map observation into this representation's symbolic space.

        Returns: A structure in the symbolic space S_R.
          - Object rep returns: ObjectGraph
          - Symmetry rep returns: SymmetryGroup
          - Pixel rep returns: np.ndarray (identity)
        """
        ...

    @abstractmethod
    def decode(self, encoded: Any) -> Observation:
        """
        Reconstruct observation from symbolic space.

        May be lossy — perfect reconstruction is not required.
        What matters is whether invariants are preserved.
        """
        ...

    # ── Structural properties ─────────────────────────────────────

    @abstractmethod
    def complexity(self, observation: Observation) -> float:
        """
        Measure the complexity of representing this observation.

        C_rep(R | D) in our framework notation.

        Lower is better. Must be comparable across representations.

        Measured as: description length of the encoded structure
        (number of primitives, edges, rules, etc.)
        """
        ...

    @abstractmethod
    def invariants(self, observation: Observation) -> List[Invariant]:
        """
        Extract invariants that this representation makes explicit.

        An invariant is "made explicit" if detecting it requires
        O(1) or O(n) operations in this representation, vs
        exponential search in raw pixels.
        """
        ...

    # ── Reasoning support ─────────────────────────────────────────

    @abstractmethod
    def hypothesis_space_size(self, observation: Observation) -> int:
        """
        Estimate the size of the hypothesis space under this representation.

        This is the key metric that connects representation to reasoning cost.
        A good representation makes |H| small.
        """
        ...

    @abstractmethod
    def estimated_reasoning_cost(
        self, observation: Observation
    ) -> float:
        """
        Estimate C_reason(T | R) — the cost of reasoning under this representation.

        This is a HEURISTIC, not an exact computation (which would be circular).
        Based on: |H|, number of invariants, structure depth, branching factor.
        """
        ...

    # ── Applicability ─────────────────────────────────────────────

    @abstractmethod
    def applicability(self, observation: Observation) -> float:
        """
        Estimate how well this representation fits the observation.

        α(R, D) ∈ [0, 1]

        1.0 = perfect fit (representation's structural assumptions match
              the observation's regularities)
        0.0 = total mismatch (representation is the wrong lens)

        Applicability is used for:
          - Initial representation selection (before reasoning begins)
          - Gating whether to even attempt this representation
          - Explaining WHY a representation was chosen

        Implementation guidance:
          - Object rep returns low α when no discrete objects exist
          - Symmetry rep returns low α when no repeating patterns exist
          - Pixel rep always returns 1.0 (it fits everything — poorly)

        This is the HEART of adaptive reasoning. The system does not
        blindly try all representations. It evaluates which ones are
        plausible given the structural signals in the observation.
        """
        ...

    # ── Transition dynamics ────────────────────────────────────────

    @abstractmethod
    def failure_detect(
        self, observation: Observation, reasoning_progress: Dict[str, Any]
    ) -> Optional[FailureSignature]:
        """
        Detect whether this representation is failing on the given observation.

        reasoning_progress contains signals like:
          - steps_taken: number of reasoning steps attempted
          - constraints_found: number of consistent constraints
          - contradictions: number of contradictory constraints
          - complexity_trend: is description growing or shrinking?

        Returns None if the representation is still making progress.
        Returns FailureSignature if stagnation or contradiction is detected.
        """
        ...

    @abstractmethod
    def transition_candidates(
        self, observation: Observation, failure: FailureSignature
    ) -> List[Transition]:
        """
        Propose candidate transitions from this representation.

        Based on:
          - The failure signature (push)
          - Emerging invariants visible even in the failing encoding (pull)

        Returns a list of plausible next representations, ordered by
        estimated suitability.
        """
        ...

    # ── Explainability ────────────────────────────────────────────

    @abstractmethod
    def explain(self, observation: Observation) -> str:
        """
        Explain WHY this representation structures the observation the way it does.

        Must answer questions like:
          "Why did you group these pixels as one object?"
          "Why did you detect rotational symmetry here?"
          "What principle guided this particular encoding?"

        This is essential for scientific comparison of representations.
        """
        ...

    # ── Dunder ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, v{self.version})"
