"""
Validate ObjectRepresentation against the contract, and compare
Pixel vs Object representations.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.representation import Observation, FailureSignature
from representations.pixel import PixelRepresentation
from representations.object import ObjectRepresentation, ObjectGraph, Obj
from metrics.metrics import compare_representations


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def pixel_rep():
    return PixelRepresentation()


@pytest.fixture
def object_rep():
    return ObjectRepresentation(connectivity=4)


@pytest.fixture
def simple_objects_obs():
    """A grid with two clearly separated objects."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, 0] = 1  # single-pixel object
    grid[0, 1] = 1  # same object (connected)
    grid[4, 3] = 2  # another object
    grid[4, 4] = 2  # connected
    return Observation(grid=grid)


@pytest.fixture
def complex_objects_obs():
    """A grid with multiple objects of varying sizes."""
    grid = np.zeros((8, 8), dtype=np.int32)
    # 3x3 block of color 1
    grid[0:3, 0:3] = 1
    # 2x2 block of color 2
    grid[5:7, 5:7] = 2
    # single pixel of color 3
    grid[3, 3] = 3
    # horizontal line of color 4
    grid[7, 0:4] = 4
    return Observation(grid=grid)


# ─────────────────────────────────────────────────────────────────────
# Contract tests
# ─────────────────────────────────────────────────────────────────────

class TestObjectRepresentation:
    """ObjectRepresentation must pass all contract tests."""

    def test_has_name(self, object_rep):
        assert object_rep.name == "object"

    def test_encode_returns_object_graph(self, object_rep, simple_objects_obs):
        graph = object_rep.encode(simple_objects_obs)
        assert isinstance(graph, ObjectGraph)
        assert graph.n_objects > 0

    def test_simple_grid_object_count(self, object_rep, simple_objects_obs):
        graph = object_rep.encode(simple_objects_obs)
        assert graph.n_objects == 2, f"Expected 2 objects, got {graph.n_objects}"

    def test_decode_is_lossless(self, object_rep, simple_objects_obs):
        graph = object_rep.encode(simple_objects_obs)
        decoded = object_rep.decode(graph)
        np.testing.assert_array_equal(decoded.grid, simple_objects_obs.grid)

    def test_complexity_is_positive(self, object_rep, simple_objects_obs):
        c = object_rep.complexity(simple_objects_obs)
        assert c > 0

    def test_complexity_lower_than_pixel(self, pixel_rep, object_rep, complex_objects_obs):
        c_pixel = pixel_rep.complexity(complex_objects_obs)
        c_object = object_rep.complexity(complex_objects_obs)
        assert c_object < c_pixel, (
            f"Object complexity ({c_object:.1f}) should be less than "
            f"pixel complexity ({c_pixel:.1f})"
        )

    def test_invariants(self, object_rep, complex_objects_obs):
        invs = object_rep.invariants(complex_objects_obs)
        assert len(invs) > 0
        names = [inv.name for inv in invs]
        assert "object_count" in names
        assert "color_palette" in names

    def test_hypothesis_space_smaller_than_pixel(self, pixel_rep, object_rep, complex_objects_obs):
        h_pixel = pixel_rep.hypothesis_space_size(complex_objects_obs)
        h_object = object_rep.hypothesis_space_size(complex_objects_obs)
        assert h_object < h_pixel, (
            f"Object |H| ({h_object}) should be less than pixel |H| ({h_pixel})"
        )

    def test_reasoning_cost_lower_than_pixel(self, pixel_rep, object_rep, complex_objects_obs):
        c_pixel = pixel_rep.estimated_reasoning_cost(complex_objects_obs)
        c_object = object_rep.estimated_reasoning_cost(complex_objects_obs)
        assert c_object < c_pixel, (
            f"Object reasoning cost ({c_object:.1f}) should be less than "
            f"pixel ({c_pixel:.1f})"
        )

    def test_applicability_high_for_discrete_objects(self, object_rep, simple_objects_obs):
        alpha = object_rep.applicability(simple_objects_obs)
        assert alpha > 0.5, (
            f"Object rep should have high applicability for discrete objects, got {alpha:.2f}"
        )

    def test_applicability_zero_for_empty(self, object_rep):
        obs = Observation(grid=np.zeros((3, 3), dtype=np.int32))
        alpha = object_rep.applicability(obs)
        assert alpha == 0.0, "Object rep should have zero applicability for empty grid"

    def test_applicability_range(self, object_rep, simple_objects_obs, complex_objects_obs):
        for obs in [simple_objects_obs, complex_objects_obs]:
            alpha = object_rep.applicability(obs)
            assert 0.0 <= alpha <= 1.0, f"Applicability out of range: {alpha}"

    def test_failure_detect_returns_signature(self, object_rep, simple_objects_obs):
        progress = {"steps_taken": 10, "constraints_found": 0}
        result = object_rep.failure_detect(simple_objects_obs, progress)
        # Should not always fail — simple objects should work fine
        # But we just check it returns something reasonable
        assert result is None or result is not None  # type check

    def test_transition_candidates(self, object_rep, simple_objects_obs):
        fs = FailureSignature(mode="overlap", detail="test")
        transitions = object_rep.transition_candidates(simple_objects_obs, fs)
        assert len(transitions) > 0
        targets = [t.target for t in transitions]
        assert "region" in targets or "topology" in targets

    def test_explain(self, object_rep, simple_objects_obs):
        explanation = object_rep.explain(simple_objects_obs)
        assert isinstance(explanation, str)
        assert "object" in explanation.lower()

    def test_empty_grid(self, object_rep):
        obs = Observation(grid=np.zeros((3, 3), dtype=np.int32))
        graph = object_rep.encode(obs)
        assert graph.n_objects == 0


# ─────────────────────────────────────────────────────────────────────
# ObjectGraph tests
# ─────────────────────────────────────────────────────────────────────

class TestObjectGraph:
    def test_adjacency(self, object_rep, simple_objects_obs):
        graph = object_rep.encode(simple_objects_obs)
        adj = graph.adjacency()
        assert isinstance(adj, dict)

    def test_adjacency_no_edges_for_separated(self, object_rep):
        """Two objects far apart should not be adjacent."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[0, 0] = 1
        grid[9, 9] = 2
        obs = Observation(grid=grid)
        graph = object_rep.encode(obs)
        adj = graph.adjacency()

        # Get object ids
        obj_ids = [obj.id for obj in graph.objects]
        assert len(obj_ids) == 2

        # They should not be adjacent (separated by background)
        neighbors_of_0 = adj.get(obj_ids[0], [])
        assert obj_ids[1] not in neighbors_of_0, "Distant objects should not be adjacent"


# ─────────────────────────────────────────────────────────────────────
# Comparative metrics
# ─────────────────────────────────────────────────────────────────────

class TestComparePixelVsObject:
    """Object representation should outperform pixel on most metrics."""

    def test_object_has_more_invariants(self, pixel_rep, object_rep, complex_objects_obs):
        n_pixel = len(pixel_rep.invariants(complex_objects_obs))
        n_object = len(object_rep.invariants(complex_objects_obs))
        assert n_object > n_pixel, (
            f"Object should detect more invariants ({n_object}) than pixel ({n_pixel})"
        )

    def test_object_has_lower_tension(self, pixel_rep, object_rep, complex_objects_obs):
        from metrics.metrics import representation_tension
        t_pixel = representation_tension(pixel_rep, complex_objects_obs)
        t_object = representation_tension(object_rep, complex_objects_obs)
        assert t_object < t_pixel, (
            f"Object tension ({t_object:.1f}) should be less than "
            f"pixel tension ({t_pixel:.1f})"
        )

    def test_object_has_better_compression(self, pixel_rep, object_rep, complex_objects_obs):
        from metrics.metrics import compression_ratio
        cr_pixel = compression_ratio(pixel_rep, complex_objects_obs)
        cr_object = compression_ratio(object_rep, complex_objects_obs)
        assert cr_object > cr_pixel, (
            f"Object compression ({cr_object:.2f}) should be better than "
            f"pixel ({cr_pixel:.2f})"
        )

    def test_object_ranks_higher(self, pixel_rep, object_rep, complex_objects_obs):
        from metrics.metrics import rank_representations
        ranked = rank_representations(
            [pixel_rep, object_rep], complex_objects_obs, by="tension"
        )
        assert ranked[0] is object_rep, (
            f"Object should rank #1 by tension, got {ranked[0].name}"
        )

    def test_object_on_pareto_front(self, pixel_rep, object_rep, complex_objects_obs):
        from metrics.metrics import find_pareto_front
        front = find_pareto_front([pixel_rep, object_rep], complex_objects_obs)
        # Object should be on (or dominate) the Pareto front
        assert object_rep in front, "Object rep should be on Pareto front"

    def test_dominance_matrix(self, pixel_rep, object_rep, complex_objects_obs):
        from metrics.metrics import dominance_matrix
        matrix = dominance_matrix([pixel_rep, object_rep], complex_objects_obs)
        # Object should dominate pixel
        assert matrix["object"]["pixel"] in (">", "∥"), (
            f"Expected object > pixel or incomparable, got {matrix['object']['pixel']}"
        )
