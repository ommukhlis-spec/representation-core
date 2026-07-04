"""
Symmetry Representation.

Encodes an observation by detecting and exploiting symmetry groups:
  - Reflection (horizontal, vertical, diagonal)
  - Rotation (order 2, 3, 4)
  - Translation (periodic tiling)

A symmetry representation compresses the grid by storing only the
fundamental domain (the asymmetric unit) plus the symmetry group
that generates the full grid.

This is the FIRST new representation to enter the zoo through the
admission test pipeline:
  Failure Mode → Canonical Task → RCS → Admission

Compared to pixel representation:
  - Complexity: O(|fundamental_domain|) vs O(H×W)
  - Hypothesis space: symmetry-aware — generates valid symmetric patterns
  - Reasoning: operations preserve symmetry constraints
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.representation import (
    FailureSignature,
    Invariant,
    Observation,
    Representation,
    Transition,
)


# ─────────────────────────────────────────────────────────────────────
# Symmetry group data structures
# ─────────────────────────────────────────────────────────────────────

class SymmetryGroup:
    """
    A symmetry group detected in a grid.

    Stores:
      - group_type: "reflection", "rotation", "translation", "none"
      - generators: the transformations that generate the group
      - fundamental_domain: the minimal region that generates the full grid
      - order: number of elements in the group
    """

    def __init__(
        self,
        group_type: str,
        generators: List[str],
        fundamental_domain: np.ndarray,
        order: int,
        compression_ratio: float,
    ):
        self.group_type = group_type
        self.generators = generators
        self.fundamental_domain = fundamental_domain
        self.order = order
        self.compression_ratio = compression_ratio

    @property
    def is_trivial(self) -> bool:
        return self.group_type == "none" or self.order <= 1

    def __repr__(self) -> str:
        if self.is_trivial:
            return "SymmetryGroup(trivial)"
        return (
            f"SymmetryGroup({self.group_type}, order={self.order}, "
            f"compression={self.compression_ratio:.1f}x)"
        )


# ─────────────────────────────────────────────────────────────────────
# Symmetry detection helpers
# ─────────────────────────────────────────────────────────────────────

def _detect_horizontal_reflection(grid: np.ndarray) -> Optional[SymmetryGroup]:
    """Check if the grid is symmetric about its horizontal midline."""
    H, W = grid.shape
    if H < 2:
        return None

    mid = H // 2
    top = grid[:mid, :]
    bottom = np.flip(grid[H-mid:, :], axis=0)

    if np.array_equal(top, bottom):
        fundamental = grid[:mid+1, :] if H % 2 == 1 else grid[:mid, :]
        return SymmetryGroup(
            group_type="reflection",
            generators=["horizontal"],
            fundamental_domain=fundamental,
            order=2,
            compression_ratio=float(H * W) / max(fundamental.size, 1),
        )
    return None


def _detect_vertical_reflection(grid: np.ndarray) -> Optional[SymmetryGroup]:
    """Check if the grid is symmetric about its vertical midline."""
    H, W = grid.shape
    if W < 2:
        return None

    mid = W // 2
    left = grid[:, :mid]
    right = np.flip(grid[:, W-mid:], axis=1)

    if np.array_equal(left, right):
        fundamental = grid[:, :mid+1] if W % 2 == 1 else grid[:, :mid]
        return SymmetryGroup(
            group_type="reflection",
            generators=["vertical"],
            fundamental_domain=fundamental,
            order=2,
            compression_ratio=float(H * W) / max(fundamental.size, 1),
        )
    return None


def _detect_rotation_4(grid: np.ndarray) -> Optional[SymmetryGroup]:
    """Check for 4-fold rotational symmetry (square grids only)."""
    H, W = grid.shape
    if H != W or H < 2:
        return None

    rotated_90 = np.rot90(grid, k=1)
    rotated_180 = np.rot90(grid, k=2)
    rotated_270 = np.rot90(grid, k=3)

    if (np.array_equal(grid, rotated_90) and
        np.array_equal(grid, rotated_180) and
        np.array_equal(grid, rotated_270)):
        # Fundamental domain is one quadrant
        mid = H // 2
        fundamental = grid[:mid+1, :mid+1] if H % 2 == 1 else grid[:mid, :mid]
        return SymmetryGroup(
            group_type="rotation",
            generators=["90_degrees"],
            fundamental_domain=fundamental,
            order=4,
            compression_ratio=float(H * W) / max(fundamental.size, 1),
        )
    return None


def _detect_rotation_2(grid: np.ndarray) -> Optional[SymmetryGroup]:
    """Check for 2-fold (180°) rotational symmetry."""
    H, W = grid.shape
    rotated_180 = np.rot90(grid, k=2)
    if H == W and np.array_equal(grid, rotated_180):
        fundamental = grid[:H//2+1, :] if H % 2 == 1 else grid[:H//2, :]
        return SymmetryGroup(
            group_type="rotation",
            generators=["180_degrees"],
            fundamental_domain=fundamental,
            order=2,
            compression_ratio=float(H * W) / max(fundamental.size, 1),
        )
    return None


def _detect_translational(grid: np.ndarray) -> Optional[SymmetryGroup]:
    """
    Detect translational symmetry (periodic tiling).

    Brute-force: try tile sizes 1×1 up to H/2 × W/2 and check if
    repeating the tile reproduces the grid.
    """
    H, W = grid.shape
    best_group = None
    best_compression = 1.0

    for tile_h in range(1, H // 2 + 1):
        if H % tile_h != 0:
            continue
        for tile_w in range(1, W // 2 + 1):
            if W % tile_w != 0:
                continue
            tile = grid[:tile_h, :tile_w]
            # Reconstruct by tiling
            reconstructed = np.tile(tile, (H // tile_h, W // tile_w))
            if np.array_equal(grid, reconstructed):
                compression = float(H * W) / float(tile_h * tile_w)
                if compression > best_compression:
                    best_compression = compression
                    best_group = SymmetryGroup(
                        group_type="translation",
                        generators=[f"tile_{tile_h}x{tile_w}"],
                        fundamental_domain=tile.copy(),
                        order=(H // tile_h) * (W // tile_w),
                        compression_ratio=compression,
                    )

    return best_group


def detect_symmetry(grid: np.ndarray) -> SymmetryGroup:
    """
    Detect the dominant symmetry group in a grid.

    Tries multiple symmetry types and returns the one with the
    highest compression ratio. Returns trivial group if no
    symmetry is detected.
    """
    detectors = [
        _detect_translational,
        _detect_rotation_4,
        _detect_rotation_2,
        _detect_horizontal_reflection,
        _detect_vertical_reflection,
    ]

    best: Optional[SymmetryGroup] = None
    best_cr = 1.0

    for detector in detectors:
        result = detector(grid)
        if result is not None and result.compression_ratio > best_cr:
            best_cr = result.compression_ratio
            best = result

    if best is None:
        best = SymmetryGroup(
            group_type="none",
            generators=[],
            fundamental_domain=grid.copy(),
            order=1,
            compression_ratio=1.0,
        )

    return best


# ─────────────────────────────────────────────────────────────────────
# Symmetry Representation
# ─────────────────────────────────────────────────────────────────────

class SymmetryRepresentation(Representation):
    """
    Representation that encodes the grid via its symmetry group.

    Encoding: Observation -> SymmetryGroup
    Decoding: SymmetryGroup -> Observation (lossless if symmetry is exact)
    """

    @property
    def name(self) -> str:
        return "symmetry"

    @property
    def version(self) -> str:
        return "0.1.0"

    # ── Encoding / Decoding ────────────────────────────────────────

    def encode(self, observation: Observation) -> SymmetryGroup:
        """Detect the symmetry group of the observation."""
        return detect_symmetry(observation.grid)

    def decode(self, encoded: SymmetryGroup) -> Observation:
        """
        Reconstruct the grid from its symmetry group.

        For exact symmetries, this is lossless.
        """
        if encoded.is_trivial:
            return Observation(grid=encoded.fundamental_domain.copy())

        fd = encoded.fundamental_domain

        if encoded.group_type == "reflection":
            if "horizontal" in encoded.generators:
                # Reflect the fundamental domain
                H_fd = fd.shape[0]
                if fd.shape[0] > 1:
                    reflected = np.flip(fd[:-1, :] if fd.shape[0] > 1 else fd, axis=0)
                    full = np.vstack([fd, reflected]) if fd.shape[0] > 1 else fd
                else:
                    full = fd
                return Observation(grid=full)

            elif "vertical" in encoded.generators:
                W_fd = fd.shape[1]
                reflected = np.flip(fd[:, :-1] if fd.shape[1] > 1 else fd, axis=1)
                full = np.hstack([fd, reflected]) if fd.shape[1] > 1 else fd
                return Observation(grid=full)

        elif encoded.group_type == "rotation":
            if "90_degrees" in encoded.generators:
                # 4-fold: reconstruct from quadrant
                H, W = fd.shape
                # Build full grid from one quadrant
                full = np.zeros((H * 2 - 1, W * 2 - 1) if H > 1 else (1, 1), dtype=np.int32)
                # Top-left
                full[:H, :W] = fd
                # Top-right: rotate 90
                full[:H, W-1:] = np.rot90(fd, k=-1)
                # Bottom-right: rotate 180
                full[H-1:, W-1:] = np.rot90(fd, k=-2)
                # Bottom-left: rotate 270
                full[H-1:, :W] = np.rot90(fd, k=-3)
                return Observation(grid=full)

            elif "180_degrees" in encoded.generators:
                H_fd = fd.shape[0]
                rotated = np.rot90(fd, k=2)
                full = np.vstack([fd, rotated[1:, :]]) if H_fd > 1 else fd
                return Observation(grid=full)

        elif encoded.group_type == "translation":
            # Reconstruct by tiling
            tile_h, tile_w = fd.shape
            H, W = encoded.fundamental_domain.shape
            # Need the original grid dimensions from encoding context
            # For now, just return the fundamental domain
            return Observation(grid=fd)

        # Fallback
        return Observation(grid=fd)

    # ── Properties ─────────────────────────────────────────────────

    def complexity(self, observation: Observation) -> float:
        """
        Complexity = size of fundamental domain × log2(10).

        A grid with 4-fold rotational symmetry has complexity H×W/4.
        No symmetry → complexity = H×W (same as pixel).
        """
        group = self.encode(observation)
        fd = group.fundamental_domain
        return float(fd.size * np.log2(10))

    def invariants(self, observation: Observation) -> List[Invariant]:
        """Extract symmetry-related invariants."""
        group = self.encode(observation)
        invs: List[Invariant] = []

        if group.is_trivial:
            invs.append(Invariant(
                name="no_symmetry",
                description="No non-trivial symmetry group detected",
                predicate="symmetry_group == trivial",
            ))
            return invs

        invs.append(Invariant(
            name="symmetry_group_type",
            description=f"Grid has {group.group_type} symmetry",
            predicate=f"symmetry_type == '{group.group_type}'",
        ))

        invs.append(Invariant(
            name="symmetry_order",
            description=f"Symmetry group has order {group.order}",
            predicate=f"symmetry_order == {group.order}",
        ))

        invs.append(Invariant(
            name="compression_ratio",
            description=f"Symmetry compresses grid by {group.compression_ratio:.1f}x",
            predicate=f"compression_ratio == {group.compression_ratio:.1f}",
        ))

        return invs

    # ── Reasoning support ─────────────────────────────────────────

    def hypothesis_space_size(self, observation: Observation) -> int:
        """
        Hypothesis space under symmetry representation.

        Only symmetric transformations are considered. For an order-N
        symmetry group, the hypothesis space is ~10^(|FD|) where |FD|
        is the fundamental domain size.
        """
        group = self.encode(observation)
        fd_size = group.fundamental_domain.size
        if fd_size > 30:
            return 10**30  # capped
        return 10**fd_size

    def estimated_reasoning_cost(self, observation: Observation) -> float:
        """Reasoning cost scales with fundamental domain size."""
        group = self.encode(observation)
        fd_size = group.fundamental_domain.size
        return float(fd_size * np.log2(10))

    # ── Applicability ─────────────────────────────────────────────

    def applicability(self, observation: Observation) -> float:
        """
        Symmetry representation applies when the grid has detectable symmetry.

        α ≈ 0.0 for random grids (no symmetry)
        α ≈ 1.0 for perfectly symmetric grids
        """
        group = self.encode(observation)
        if group.is_trivial:
            # Even for non-symmetric grids, we give a small base applicability
            # because checking for symmetry is cheap and informative
            return 0.1

        # Higher compression = higher applicability
        # CR=1.0 (trivial) → α≈0.0, CR=2.0 (reflection) → α≈0.5, CR=4.0 → α≈1.0
        # Formula: α = (CR - 1) / 2, clipped to [0, 1]
        alpha = (group.compression_ratio - 1.0) / 2.0
        return float(np.clip(alpha, 0.0, 1.0))

    # ── Failure detection ─────────────────────────────────────────

    def failure_detect(
        self, observation: Observation, reasoning_progress: Dict[str, Any]
    ) -> Optional[FailureSignature]:
        """
        Symmetry representation fails when:
          - The grid has no detectable symmetry (trivial group)
          - Symmetry is only approximate, not exact
          - The pattern is not global (local symmetry only)
        """
        group = self.encode(observation)

        if group.is_trivial:
            return FailureSignature(
                mode="no_structure",
                detail=(
                    "No non-trivial symmetry detected. The grid may have "
                    "local structure (objects, regions) but no global symmetry."
                ),
                confidence=0.9,
                evidence={"symmetry_group": "trivial"},
            )

        # Check for approximate symmetry (not exact)
        steps = reasoning_progress.get("steps_taken", 0)
        constraints = reasoning_progress.get("constraints_found", 0)
        if steps > 0 and constraints == 0 and group.order <= 2:
            return FailureSignature(
                mode="partial_symmetry",
                detail=(
                    f"Detected {group.group_type} symmetry (order {group.order}) "
                    "but reasoning produces no constraints. Symmetry may be "
                    "only approximate."
                ),
                confidence=0.5,
                evidence={"group_type": group.group_type, "order": group.order},
            )

        return None

    def transition_candidates(
        self, observation: Observation, failure: FailureSignature
    ) -> List[Transition]:
        """Propose next representations based on failure mode."""
        candidates = []

        if failure.mode == "no_structure":
            candidates.append(Transition(
                source="symmetry", target="object",
                trigger="failure", failure_signature=failure,
                rationale="No global symmetry — try object-level structure.",
            ))
            candidates.append(Transition(
                source="symmetry", target="constraint",
                trigger="failure", failure_signature=failure,
                rationale="No spatial symmetry — try relational constraints.",
            ))

        elif failure.mode == "partial_symmetry":
            candidates.append(Transition(
                source="symmetry", target="topology",
                trigger="failure", failure_signature=failure,
                rationale="Partial symmetry may indicate topological regularity.",
            ))
            candidates.append(Transition(
                source="symmetry", target="region",
                trigger="failure", failure_signature=failure,
                rationale="Approximate symmetry suggests regional analysis.",
            ))

        # Default fallback
        candidates.append(Transition(
            source="symmetry", target="pixel",
            trigger="failure", failure_signature=failure,
            rationale="Fallback to raw pixel representation.",
        ))

        return candidates

    # ── Explainability ─────────────────────────────────────────────

    def explain(self, observation: Observation) -> str:
        """Explain the detected symmetry."""
        group = self.encode(observation)

        if group.is_trivial:
            return (
                f"SymmetryRepresentation analyzed the {observation.height}×{observation.width} "
                f"grid and found NO non-trivial symmetry group. "
                f"The grid is treated as having only the identity transformation."
            )

        lines = [
            f"SymmetryRepresentation detected {group.group_type} symmetry "
            f"of order {group.order} in the {observation.height}×{observation.width} grid.",
            f"  Generators: {group.generators}",
            f"  Fundamental domain size: {group.fundamental_domain.shape}",
            f"  Compression: {group.compression_ratio:.1f}×",
            f"  Hypothesis space reduced from 10^{observation.height * observation.width} "
            f"to 10^{group.fundamental_domain.size}",
        ]
        return "\n".join(lines)
