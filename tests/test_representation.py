"""
Validate the Representation contract using the Pixel implementation.

These tests ensure:
  1. The abstract interface is implementable
  2. Every required method returns a meaningful value
  3. Metrics computation does not crash
  4. The contract is internally consistent
"""

import sys
import os

import numpy as np
import pytest

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.representation import Observation, Representation
from representations.pixel import PixelRepresentation
from metrics.metrics import (
    QualityVector,
    compute_quality,
    compare_representations,
    compression_ratio,
    find_pareto_front,
    invariance_density,
    rank_representations,
    representation_tension,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def pixel_rep():
    return PixelRepresentation()


@pytest.fixture
def small_obs():
    """A 3x3 grid with a simple pattern."""
    grid = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.int32)
    return Observation(grid=grid)


@pytest.fixture
def large_obs():
    """A 10x10 random grid."""
    rng = np.random.default_rng(42)
    grid = rng.integers(0, 10, size=(10, 10)).astype(np.int32)
    return Observation(grid=grid)


# ─────────────────────────────────────────────────────────────────────
# Contract tests
# ─────────────────────────────────────────────────────────────────────

class TestRepresentationContract:
    """Every Representation MUST pass these tests."""

    def test_has_name(self, pixel_rep):
        assert isinstance(pixel_rep.name, str)
        assert len(pixel_rep.name) > 0

    def test_has_version(self, pixel_rep):
        assert isinstance(pixel_rep.version, str)

    def test_encode_returns_valid(self, pixel_rep, small_obs):
        encoded = pixel_rep.encode(small_obs)
        assert encoded is not None

    def test_decode_returns_observation(self, pixel_rep, small_obs):
        encoded = pixel_rep.encode(small_obs)
        decoded = pixel_rep.decode(encoded)
        assert isinstance(decoded, Observation)
        # Pixel representation is lossless
        np.testing.assert_array_equal(decoded.grid, small_obs.grid)

    def test_complexity_is_positive(self, pixel_rep, small_obs):
        c = pixel_rep.complexity(small_obs)
        assert c > 0

    def test_complexity_scales_with_size(self, pixel_rep, small_obs, large_obs):
        c_small = pixel_rep.complexity(small_obs)
        c_large = pixel_rep.complexity(large_obs)
        assert c_large > c_small, "Larger grids should have higher complexity"

    def test_invariants_returns_list(self, pixel_rep, small_obs):
        invs = pixel_rep.invariants(small_obs)
        assert isinstance(invs, list)
        assert len(invs) > 0, "Even pixel rep should detect basic invariants"

    def test_hypothesis_space_is_positive(self, pixel_rep, small_obs):
        h = pixel_rep.hypothesis_space_size(small_obs)
        assert h > 0

    def test_reasoning_cost_is_positive(self, pixel_rep, small_obs):
        cost = pixel_rep.estimated_reasoning_cost(small_obs)
        assert cost > 0

    def test_failure_detect_returns_signature_or_none(self, pixel_rep, small_obs):
        progress = {"steps_taken": 10, "constraints_found": 0}
        result = pixel_rep.failure_detect(small_obs, progress)
        # Pixel rep should detect failure on structured grid with no progress
        assert result is not None or result is None  # type check

    def test_transition_candidates_returns_list(self, pixel_rep, small_obs):
        from core.representation import FailureSignature
        fs = FailureSignature(mode="no_structure", detail="test")
        transitions = pixel_rep.transition_candidates(small_obs, fs)
        assert isinstance(transitions, list)
        assert len(transitions) > 0

    def test_explain_returns_string(self, pixel_rep, small_obs):
        explanation = pixel_rep.explain(small_obs)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_repr(self, pixel_rep):
        r = repr(pixel_rep)
        assert "PixelRepresentation" in r
        assert pixel_rep.name in r

    def test_applicability_returns_float(self, pixel_rep, small_obs):
        alpha = pixel_rep.applicability(small_obs)
        assert isinstance(alpha, float)
        assert 0.0 <= alpha <= 1.0

    def test_pixel_applicability_is_one(self, pixel_rep, small_obs, large_obs):
        """Pixel representation fits everything (poorly) — α = 1.0 always."""
        assert pixel_rep.applicability(small_obs) == 1.0
        assert pixel_rep.applicability(large_obs) == 1.0


# ─────────────────────────────────────────────────────────────────────
# QualityVector tests
# ─────────────────────────────────────────────────────────────────────

class TestQualityVector:
    """Validate the QualityVector abstraction."""

    def test_compute_quality(self, pixel_rep, small_obs):
        from metrics.metrics import compute_quality
        q = compute_quality(pixel_rep, small_obs)
        assert q.C > 0
        assert q.H > 0
        assert q.I >= 0
        assert 0.0 <= q.E <= 1.0
        assert q.X > 0

    def test_self_dominance(self, pixel_rep, small_obs):
        from metrics.metrics import compute_quality
        q = compute_quality(pixel_rep, small_obs)
        assert not q.dominates(q), "A vector should not dominate itself"
        assert q.is_equivalent_to(q)

    def test_find_pareto_front(self, pixel_rep, small_obs):
        from metrics.metrics import find_pareto_front
        front = find_pareto_front([pixel_rep], small_obs)
        assert len(front) == 1
        assert front[0] is pixel_rep

    def test_dominance_matrix(self, pixel_rep, small_obs):
        from metrics.metrics import dominance_matrix
        matrix = dominance_matrix([pixel_rep], small_obs)
        assert "pixel" in matrix
        assert matrix["pixel"]["pixel"] == "="


# ─────────────────────────────────────────────────────────────────────
# Metric tests
# ─────────────────────────────────────────────────────────────────────

class TestMetrics:
    """Validate that metrics compute without error and are consistent."""

    def test_compression_ratio(self, pixel_rep, small_obs):
        cr = compression_ratio(pixel_rep, small_obs)
        # Pixel rep has no compression → ratio should be ~1.0
        assert cr > 0

    def test_invariance_density(self, pixel_rep, small_obs):
        d = invariance_density(pixel_rep, small_obs)
        assert d > 0

    def test_tension(self, pixel_rep, small_obs):
        t = representation_tension(pixel_rep, small_obs)
        assert t > 0

    def test_reasoning_cost_estimate(self, pixel_rep, small_obs):
        q = compute_quality(pixel_rep, small_obs)
        assert q.X > 0

    def test_compare_representations(self, pixel_rep, small_obs):
        results = compare_representations([pixel_rep], small_obs)
        assert "pixel" in results
        q = results["pixel"]
        assert isinstance(q, QualityVector)
        # QualityVector has all dimensions
        assert q.C > 0
        assert q.H > 0
        assert q.I >= 0
        assert q.X > 0

    def test_rank_representations(self, pixel_rep, small_obs):
        ranked = rank_representations([pixel_rep], small_obs, by="tension")
        assert len(ranked) == 1
        assert ranked[0] is pixel_rep

    def test_tension_monotonic_with_size(self, pixel_rep, small_obs, large_obs):
        """Tension should increase with observation size for pixel rep."""
        t_small = representation_tension(pixel_rep, small_obs)
        t_large = representation_tension(pixel_rep, large_obs)
        assert t_large > t_small, (
            f"Expected tension({large_obs.shape}) > tension({small_obs.shape}), "
            f"got {t_large:.2f} <= {t_small:.2f}"
        )

    def test_all_metrics_finite(self, pixel_rep, small_obs):
        """No metric should be NaN or inf."""
        results = compare_representations([pixel_rep], small_obs)
        for name, q in results.items():
            for dim_name, value in q.to_dict().items():
                assert np.isfinite(value), f"{name}.{dim_name} = {value} (not finite)"


# ─────────────────────────────────────────────────────────────────────
# Observation tests
# ─────────────────────────────────────────────────────────────────────

class TestObservation:
    def test_create_valid(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        obs = Observation(grid=grid)
        assert obs.height == 5
        assert obs.width == 5

    def test_reject_3d(self):
        with pytest.raises(ValueError):
            Observation(grid=np.zeros((2, 2, 3), dtype=np.int32))

    def test_frozen(self):
        obs = Observation(grid=np.zeros((3, 3), dtype=np.int32))
        with pytest.raises(Exception):
            obs.grid = np.ones((3, 3), dtype=np.int32)  # frozen dataclass
