"""
Pixel Representation — the baseline.

This is the identity representation: encode(x) = x.
No structure is extracted. No compression. No invariants beyond the trivial.

It serves as the CONTROL condition for all experiments:
  - Maximum complexity (raw pixel count)
  - Maximum hypothesis space (all possible grid transformations)
  - Maximum reasoning cost

Every other representation should outperform pixel on at least one metric.
If a representation does NOT beat pixel, it is worse than doing nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.representation import (
    FailureSignature,
    Invariant,
    Observation,
    Representation,
    Transition,
)


class PixelRepresentation(Representation):
    """
    Identity representation.

    encode(grid) = grid
    decode(grid) = grid

    This is the baseline. It represents the observation exactly as given,
    with no structural abstraction.
    """

    @property
    def name(self) -> str:
        return "pixel"

    @property
    def version(self) -> str:
        return "0.1.0"

    # ── Encoding ──────────────────────────────────────────────────

    def encode(self, observation: Observation) -> np.ndarray:
        """Identity encoding — return the raw grid."""
        return observation.grid.copy()

    def decode(self, encoded: np.ndarray) -> Observation:
        """Identity decoding — wrap back into Observation."""
        return Observation(grid=encoded.copy())

    # ── Properties ────────────────────────────────────────────────

    def complexity(self, observation: Observation) -> float:
        """
        Complexity = number of pixels × log2(10).

        Each pixel is an independent choice from 10 colors.
        No structure is exploited → maximum description length.
        """
        return float(observation.height * observation.width * np.log2(10))

    def invariants(self, observation: Observation) -> List[Invariant]:
        """
        Pixel representation detects NO invariants.

        It sees every pixel as independent. There is no grouping,
        no pattern detection, no structural insight at this level.
        """
        # The only "invariant" is trivial: the grid has H×W pixels
        grid = observation.grid
        return [
            Invariant(
                name="grid_size",
                description=f"Grid is {observation.height}×{observation.width}",
                predicate=f"shape == ({observation.height}, {observation.width})",
            ),
            Invariant(
                name="color_palette",
                description=f"Colors used: {sorted(set(grid.flatten()))}",
                predicate=f"colors ⊆ {{0..9}}",
            ),
        ]

    # ── Reasoning support ─────────────────────────────────────────

    def hypothesis_space_size(self, observation: Observation) -> int:
        """
        Every pixel can independently change to any of 10 colors.

        |H| = 10^(H×W)

        This is astronomically large for any non-trivial grid.
        For a 10×10 grid: |H| = 10^100.

        This is WHY pixel representation is bad for reasoning.
        """
        n_pixels = observation.height * observation.width
        # Cap at something representable (10^100 is too large for int)
        if n_pixels > 30:
            return 10**30  # capped for practical computation
        return 10**n_pixels

    def estimated_reasoning_cost(self, observation: Observation) -> float:
        """
        Without structure, reasoning is brute-force search over 10^(H×W).

        Cost ≈ H × W × log2(10) — linear in pixel count, but the SEARCH
        space is exponential. This metric captures the search space,
        not the pixel count.
        """
        n_pixels = observation.height * observation.width
        return float(n_pixels * np.log2(10))

    # ── Applicability ─────────────────────────────────────────────

    def applicability(self, observation: Observation) -> float:
        """
        Pixel representation fits EVERYTHING — but poorly.

        α = 1.0 always. This is why pixel is the fallback, not the choice.
        High applicability but terrible quality on all other dimensions.
        """
        return 1.0

    # ── Failure detection ─────────────────────────────────────────

    def failure_detect(
        self, observation: Observation, reasoning_progress: Dict[str, Any]
    ) -> Optional[FailureSignature]:
        """
        Pixel representation fails when:
          - The grid is not pure noise (i.e., structure EXISTS)
          - Reasoning space is so large that no progress is possible

        Since pixel representation treats everything as independent,
        it fails on ANY structured task. The failure mode is "no_structure"
        — meaning the representation imposes no structure, not that the
        observation lacks it.
        """
        steps = reasoning_progress.get("steps_taken", 0)
        constraints = reasoning_progress.get("constraints_found", 0)

        # If we've tried reasoning and found nothing, the space is too big
        if steps > 0 and constraints == 0:
            return FailureSignature(
                mode="no_structure",
                detail=(
                    "Pixel representation treats all pixels as independent. "
                    "Hypothesis space is 10^(H×W). Brute-force reasoning "
                    "is computationally infeasible."
                ),
                confidence=1.0,
                evidence={"hypothesis_space": self.hypothesis_space_size(observation)},
            )

        # Even before reasoning: if grid is larger than trivial, pixel rep is inadequate
        if observation.height * observation.width > 4:
            return FailureSignature(
                mode="no_structure",
                detail="Grid has structure that pixel representation cannot exploit.",
                confidence=0.9,
            )

        return None

    def transition_candidates(
        self, observation: Observation, failure: FailureSignature
    ) -> List[Transition]:
        """
        From pixel, the natural transitions are to representations
        that impose some structure: object, region, or symmetry.

        The transition is triggered by failure (no_structure).
        """
        candidates = [
            Transition(
                source="pixel",
                target="object",
                trigger="failure",
                failure_signature=failure,
                rationale=(
                    "Object representation groups pixels into discrete entities, "
                    "massively reducing the hypothesis space."
                ),
            ),
            Transition(
                source="pixel",
                target="region",
                trigger="failure",
                failure_signature=failure,
                rationale=(
                    "Region representation segments the grid into spatial zones, "
                    "capturing adjacency structure."
                ),
            ),
            Transition(
                source="pixel",
                target="symmetry",
                trigger="failure",
                failure_signature=failure,
                rationale=(
                    "Symmetry representation looks for invariant transformations, "
                    "exploiting redundancy in the grid."
                ),
            ),
        ]
        return candidates

    # ── Explainability ─────────────────────────────────────────────

    def explain(self, observation: Observation) -> str:
        """
        Pixel representation has nothing to explain — it applies no
        structural interpretation to the observation.
        """
        return (
            f"PixelRepresentation encodes the observation as a raw "
            f"{observation.height}×{observation.width} grid of {len(np.unique(observation.grid))} "
            f"distinct colors. No structural grouping or abstraction is applied. "
            f"Each pixel is treated as an independent degree of freedom. "
            f"Hypothesis space size: 10^{observation.height * observation.width}."
        )
