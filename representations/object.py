"""
Object Representation.

Encodes an observation as a collection of discrete objects connected
by spatial relationships.

An "object" is a maximal connected component of same-colored pixels
(4-connectivity or 8-connectivity).

This representation is the first step above raw pixels:
  - Groups pixels into entities (compression)
  - Extracts object-level properties (invariants)
  - Builds a relational graph (structure for reasoning)

Compared to pixel representation:
  - Complexity: O(N_objects) vs O(N_pixels)
  - Hypothesis space: exponentially smaller
  - Reasoning: operates on objects, not pixels
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from core.representation import (
    FailureSignature,
    Invariant,
    Observation,
    Representation,
    Transition,
)


# ─────────────────────────────────────────────────────────────────────
# Object data structures
# ─────────────────────────────────────────────────────────────────────

class Obj:
    """A discrete object in the grid."""

    def __init__(
        self,
        obj_id: int,
        color: int,
        pixels: List[Tuple[int, int]],
        bbox: Tuple[int, int, int, int],  # (min_r, min_c, max_r, max_c)
    ):
        self.id = obj_id
        self.color = color
        self.pixels = pixels  # list of (r, c)
        self.bbox = bbox  # inclusive

    @property
    def size(self) -> int:
        return len(self.pixels)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def centroid(self) -> Tuple[float, float]:
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (np.mean(rows), np.mean(cols))

    @property
    def aspect_ratio(self) -> float:
        return self.height / max(self.width, 1)

    def __repr__(self) -> str:
        return (
            f"Obj(id={self.id}, color={self.color}, "
            f"size={self.size}, pos=({self.centroid[0]:.1f}, {self.centroid[1]:.1f}))"
        )


class ObjectGraph:
    """A graph of objects and their spatial relationships."""

    def __init__(self, objects: List[Obj], grid_shape: Tuple[int, int]):
        self.objects = objects
        self.grid_shape = grid_shape
        self._adjacency: Optional[Dict[int, List[int]]] = None  # lazy

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def adjacency(self) -> Dict[int, List[int]]:
        """Build object adjacency graph (objects sharing a border)."""
        if self._adjacency is not None:
            return self._adjacency

        # Map pixel -> object id
        pixel_to_obj: Dict[Tuple[int, int], int] = {}
        for obj in self.objects:
            for p in obj.pixels:
                pixel_to_obj[p] = obj.id

        adj: Dict[int, set] = {obj.id: set() for obj in self.objects}
        for obj in self.objects:
            for (r, c) in obj.pixels:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    neighbor_id = pixel_to_obj.get((nr, nc))
                    if neighbor_id is not None and neighbor_id != obj.id:
                        adj[obj.id].add(neighbor_id)
                        adj[neighbor_id].add(obj.id)

        self._adjacency = {k: sorted(v) for k, v in adj.items()}
        return self._adjacency

    def get_object(self, obj_id: int) -> Obj:
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        raise KeyError(f"No object with id {obj_id}")

    def __repr__(self) -> str:
        return f"ObjectGraph(n={self.n_objects}, objects={self.objects})"


# ─────────────────────────────────────────────────────────────────────
# Object Representation
# ─────────────────────────────────────────────────────────────────────

class ObjectRepresentation(Representation):
    """
    Representation that segments the grid into discrete objects.

    Encoding: Observation -> ObjectGraph
    Decoding: ObjectGraph -> Observation (lossless for non-overlapping objects)
    """

    def __init__(self, connectivity: int = 4):
        """
        Args:
            connectivity: 4 (von Neumann) or 8 (Moore) for connected components.
        """
        self._connectivity = connectivity

    @property
    def name(self) -> str:
        return "object"

    @property
    def version(self) -> str:
        return "0.1.0"

    # ── Encoding / Decoding ────────────────────────────────────────

    def encode(self, observation: Observation) -> ObjectGraph:
        """Segment the grid into connected components (objects)."""
        grid = observation.grid

        # Define connectivity structure
        if self._connectivity == 4:
            structure = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ], dtype=np.int32)
        else:
            structure = np.ones((3, 3), dtype=np.int32)

        # Label connected components per color
        objects: List[Obj] = []
        obj_id = 0

        unique_colors = [c for c in np.unique(grid) if c != 0]  # skip background

        for color in unique_colors:
            color_mask = (grid == color).astype(np.int32)
            labeled, n_features = ndimage.label(color_mask, structure=structure)

            for label_idx in range(1, n_features + 1):
                coords = np.argwhere(labeled == label_idx)
                pixels = [(int(r), int(c)) for r, c in coords]
                rows = coords[:, 0]
                cols = coords[:, 1]
                bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

                objects.append(Obj(
                    obj_id=obj_id,
                    color=int(color),
                    pixels=pixels,
                    bbox=bbox,
                ))
                obj_id += 1

        return ObjectGraph(objects=objects, grid_shape=observation.shape)

    def decode(self, encoded: ObjectGraph) -> Observation:
        """Reconstruct grid from object graph (lossless)."""
        grid = np.zeros(encoded.grid_shape, dtype=np.int32)
        for obj in encoded.objects:
            for (r, c) in obj.pixels:
                grid[r, c] = obj.color
        return Observation(grid=grid)

    # ── Properties ─────────────────────────────────────────────────

    def complexity(self, observation: Observation) -> float:
        """
        Complexity = number of objects × cost per object.

        Each object requires describing: color, size, position, shape.
        This is ~5-10 values per object vs H×W values for pixels.
        """
        graph = self.encode(observation)
        n = graph.n_objects
        if n == 0:
            return 1.0  # empty grid
        # Each object: color (log2(10)), size, centroid_x, centroid_y, bbox (4 values)
        cost_per_object = np.log2(10) + 6.0  # ~9.3 bits per object
        return float(n * cost_per_object)

    def invariants(self, observation: Observation) -> List[Invariant]:
        """Extract invariants from object structure."""
        graph = self.encode(observation)
        invs: List[Invariant] = []

        # Object count
        invs.append(Invariant(
            name="object_count",
            description=f"Grid contains {graph.n_objects} discrete objects",
            predicate=f"n_objects == {graph.n_objects}",
        ))

        if graph.n_objects == 0:
            return invs

        # Color distribution
        colors = [obj.color for obj in graph.objects]
        unique_colors = sorted(set(colors))
        invs.append(Invariant(
            name="color_palette",
            description=f"Objects use colors: {unique_colors}",
            predicate=f"object_colors ⊆ {set(unique_colors)}",
        ))

        # Size statistics
        sizes = [obj.size for obj in graph.objects]
        if len(set(sizes)) == 1:
            invs.append(Invariant(
                name="uniform_size",
                description=f"All objects have size {sizes[0]}",
                predicate=f"all(obj.size == {sizes[0]})",
            ))

        # Color counts
        from collections import Counter
        color_counts = Counter(colors)
        if len(set(color_counts.values())) == 1:
            count_val = list(color_counts.values())[0]
            invs.append(Invariant(
                name="uniform_color_distribution",
                description=f"Each color appears exactly {count_val} times",
                predicate=f"all(count == {count_val} for count in color_counts.values())",
            ))

        # Adjacency structure
        adj = graph.adjacency()
        n_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
        invs.append(Invariant(
            name="adjacency_graph",
            description=f"Object adjacency graph has {n_edges} edges",
            predicate="adjacency_structure_preserved",
        ))

        return invs

    # ── Reasoning support ─────────────────────────────────────────

    def hypothesis_space_size(self, observation: Observation) -> int:
        """
        With object representation, |H| is determined by:
          - Number of objects (n)
          - Possible transformations per object (move, recolor, copy, delete, etc.)
          - Relational constraints between objects

        Rough estimate: O(k^n) where k ~ 10 (operations per object).
        Much smaller than 10^(H×W) for pixel representation.
        """
        graph = self.encode(observation)
        n = graph.n_objects
        if n == 0:
            return 1
        # Each object can be: same, moved, recolored, copied, deleted, resized (~10 ops)
        # Objects interact: adjacency relations can change
        ops_per_object = 10
        return ops_per_object ** n

    def estimated_reasoning_cost(self, observation: Observation) -> float:
        """Reasoning cost ≈ n_objects × log(ops_per_object)."""
        graph = self.encode(observation)
        n = graph.n_objects
        if n == 0:
            return 0.0
        return float(n * np.log2(10))

    # ── Applicability ─────────────────────────────────────────────

    def applicability(self, observation: Observation) -> float:
        """
        Object representation applies when discrete objects exist.

        α is high when:
          - Connected components are well-separated
          - Objects have reasonable sizes (not fragmented)
          - The grid is not continuous/noise

        α is low when:
          - No discrete objects (empty or continuous gradient)
          - Objects are heavily fragmented (many tiny components)
        """
        graph = self.encode(observation)
        n = graph.n_objects

        if n == 0:
            return 0.0  # no objects at all

        # Fragmentation penalty: too many tiny objects suggests
        # object representation may not be the right lens
        sizes = [obj.size for obj in graph.objects]
        avg_size = np.mean(sizes)
        fragmentation = min(avg_size / 4.0, 1.0)  # normalize: 4+ pixels = full score

        # Overlap penalty: overlapping bounding boxes suggest
        # objects are not well-separated
        bboxes = [obj.bbox for obj in graph.objects]
        overlap_count = 0
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if _bboxes_overlap(bboxes[i], bboxes[j]):
                    overlap_count += 1
        max_overlaps = max(n * (n - 1) // 2, 1)
        overlap_penalty = 1.0 - (overlap_count / max_overlaps)

        # Combine
        alpha = 0.5 * fragmentation + 0.5 * overlap_penalty
        return float(np.clip(alpha, 0.0, 1.0))

    # ── Failure detection ─────────────────────────────────────────

    def failure_detect(
        self, observation: Observation, reasoning_progress: Dict[str, Any]
    ) -> Optional[FailureSignature]:
        """
        Object representation fails when:
          - Objects overlap (same pixel belongs to multiple objects)
          - No discrete objects exist (continuous gradient)
          - Objects are perceptually there but not by connected-component definition
          - Too many tiny objects (fragmentation)
        """
        graph = self.encode(observation)

        # Check for object fragmentation (too many tiny objects)
        if graph.n_objects > 0:
            avg_size = np.mean([obj.size for obj in graph.objects])
            if avg_size < 2.0 and graph.n_objects > 10:
                return FailureSignature(
                    mode="non_discrete",
                    detail=(
                        f"Grid fragmented into {graph.n_objects} objects "
                        f"with avg size {avg_size:.1f}. Object boundaries are ambiguous."
                    ),
                    confidence=0.8,
                    evidence={"n_objects": graph.n_objects, "avg_size": avg_size},
                )

        # Check for overlapping bounding boxes (potential overlap)
        bboxes = [obj.bbox for obj in graph.objects]
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if _bboxes_overlap(bboxes[i], bboxes[j]):
                    # Check if same pixel claimed by both (shouldn't happen in our encoding)
                    return FailureSignature(
                        mode="overlap",
                        detail=f"Objects {i} and {j} have overlapping bounding boxes.",
                        confidence=0.6,
                        evidence={"obj_i": i, "obj_j": j},
                    )

        # Check for empty (no objects = all background)
        if graph.n_objects == 0:
            steps = reasoning_progress.get("steps_taken", 0)
            if steps > 0:
                return FailureSignature(
                    mode="no_structure",
                    detail="No objects found — grid may be empty or require region/topology view.",
                    confidence=0.7,
                )

        return None

    def transition_candidates(
        self, observation: Observation, failure: FailureSignature
    ) -> List[Transition]:
        """Propose next representations based on failure mode."""
        candidates = []

        if failure.mode == "overlap":
            candidates.append(Transition(
                source="object", target="region",
                trigger="failure", failure_signature=failure,
                rationale="Region representation handles overlapping/ambiguous boundaries.",
            ))
            candidates.append(Transition(
                source="object", target="topology",
                trigger="failure", failure_signature=failure,
                rationale="Topology representation ignores exact boundaries, focuses on connectivity.",
            ))

        elif failure.mode == "non_discrete":
            candidates.append(Transition(
                source="object", target="region",
                trigger="failure", failure_signature=failure,
                rationale="Region representation is better for non-discrete structures.",
            ))
            candidates.append(Transition(
                source="object", target="constraint",
                trigger="failure", failure_signature=failure,
                rationale="Constraint representation captures rules without object segmentation.",
            ))

        elif failure.mode == "no_structure":
            candidates.append(Transition(
                source="object", target="pixel",
                trigger="failure", failure_signature=failure,
                rationale="Fallback to pixel representation.",
            ))

        # Default candidates
        candidates.append(Transition(
            source="object", target="symmetry",
            trigger="failure", failure_signature=failure,
            rationale="Symmetry may capture patterns that object segmentation misses.",
        ))

        return candidates

    # ── Explainability ─────────────────────────────────────────────

    def explain(self, observation: Observation) -> str:
        """Explain the object segmentation."""
        graph = self.encode(observation)

        if graph.n_objects == 0:
            return "No discrete objects found. The grid is empty (all background)."

        lines = [
            f"ObjectRepresentation segments the {observation.height}×{observation.width} "
            f"grid into {graph.n_objects} discrete objects using "
            f"{self._connectivity}-connectivity:",
        ]

        for obj in graph.objects:
            lines.append(
                f"  - {obj}: bbox={obj.bbox}"
            )

        # Describe adjacency
        adj = graph.adjacency()
        if adj:
            edges_desc = []
            for src, neighbors in adj.items():
                for dst in neighbors:
                    if src < dst:  # avoid duplicates
                        edges_desc.append(f"{src}-{dst}")
            lines.append(f"Adjacency edges: {', '.join(edges_desc)}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _bboxes_overlap(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
) -> bool:
    """Check if two bounding boxes overlap."""
    r1_min, c1_min, r1_max, c1_max = bbox1
    r2_min, c2_min, r2_max, c2_max = bbox2
    return not (
        r1_max < r2_min or r2_max < r1_min or
        c1_max < c2_min or c2_max < c1_min
    )
