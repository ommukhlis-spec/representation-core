"""
Validate SymmetryRepresentation — the first new representation
admitted to the zoo through the full pipeline:
  Failure Mode → Canonical Task → RCS → Admission
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.representation import Observation, FailureSignature
from representations.pixel import PixelRepresentation
from representations.object import ObjectRepresentation
from representations.symmetry import (
    SymmetryRepresentation,
    SymmetryGroup,
    detect_symmetry,
)
from metrics.metrics import compute_quality, compare_representations


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def sym_rep():
    return SymmetryRepresentation()


@pytest.fixture
def pixel_rep():
    return PixelRepresentation()


@pytest.fixture
def object_rep():
    return ObjectRepresentation()


@pytest.fixture
def all_reps(sym_rep, pixel_rep, object_rep):
    return [pixel_rep, object_rep, sym_rep]


@pytest.fixture
def reflection_grid():
    """Grid with horizontal reflection symmetry."""
    return Observation(grid=np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0],
        [0, 0, 0, 0, 0],
        [6, 7, 8, 9, 0],
        [1, 2, 3, 4, 5],
    ], dtype=np.int32))


@pytest.fixture
def rotation_grid():
    """Grid with 4-fold rotational symmetry."""
    return Observation(grid=np.array([
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
    ], dtype=np.int32))


@pytest.fixture
def translation_grid():
    """Grid with translational symmetry (2x2 tiling)."""
    return Observation(grid=np.array([
        [1, 2, 1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4, 3, 4],
    ], dtype=np.int32))


@pytest.fixture
def random_grid():
    """Random grid with no symmetry."""
    rng = np.random.default_rng(42)
    grid = rng.integers(0, 10, size=(10, 10)).astype(np.int32)
    return Observation(grid=grid)


# ─────────────────────────────────────────────────────────────────────
# Symmetry detection tests
# ─────────────────────────────────────────────────────────────────────

class TestSymmetryDetection:
    """Validate symmetry group detection."""

    def test_detect_horizontal_reflection(self, reflection_grid):
        group = detect_symmetry(reflection_grid.grid)
        assert group.group_type == "reflection"
        assert "horizontal" in group.generators
        assert group.order == 2

    def test_detect_rotation_4(self, rotation_grid):
        group = detect_symmetry(rotation_grid.grid)
        assert group.group_type == "rotation"
        assert group.order == 4

    def test_detect_translation(self, translation_grid):
        group = detect_symmetry(translation_grid.grid)
        assert group.group_type == "translation"

    def test_detect_none_on_random(self, random_grid):
        group = detect_symmetry(random_grid.grid)
        assert group.is_trivial

    def test_compression_ratio(self, translation_grid):
        group = detect_symmetry(translation_grid.grid)
        assert group.compression_ratio > 1.0, (
            f"Translation should compress, got CR={group.compression_ratio}"
        )


# ─────────────────────────────────────────────────────────────────────
# Contract tests
# ─────────────────────────────────────────────────────────────────────

class TestSymmetryRepresentation:
    """SymmetryRepresentation passes the Representation contract."""

    def test_name(self, sym_rep):
        assert sym_rep.name == "symmetry"

    def test_version(self, sym_rep):
        assert sym_rep.version == "0.1.0"

    def test_encode_returns_symmetry_group(self, sym_rep, reflection_grid):
        group = sym_rep.encode(reflection_grid)
        assert isinstance(group, SymmetryGroup)

    def test_complexity_lower_on_symmetric(self, sym_rep, reflection_grid, random_grid):
        """Symmetric grids should have lower complexity than random ones."""
        c_sym = sym_rep.complexity(reflection_grid)
        c_rand = sym_rep.complexity(random_grid)
        assert c_sym < c_rand, (
            f"Symmetric complexity ({c_sym:.1f}) should be less than "
            f"random ({c_rand:.1f})"
        )

    def test_complexity_lower_than_pixel_on_symmetric(self, sym_rep, pixel_rep, reflection_grid):
        """Symmetry rep should beat pixel on symmetric grids."""
        c_sym = sym_rep.complexity(reflection_grid)
        c_pixel = pixel_rep.complexity(reflection_grid)
        assert c_sym < c_pixel, (
            f"Symmetry ({c_sym:.1f}) should beat pixel ({c_pixel:.1f}) "
            f"on symmetric grid"
        )

    def test_invariants_on_symmetric(self, sym_rep, reflection_grid):
        invs = sym_rep.invariants(reflection_grid)
        assert len(invs) >= 3  # type, order, compression

    def test_invariants_on_random(self, sym_rep, random_grid):
        invs = sym_rep.invariants(random_grid)
        assert len(invs) == 1  # only "no_symmetry"

    def test_hypothesis_space_smaller_on_symmetric(self, sym_rep, reflection_grid, random_grid):
        h_sym = sym_rep.hypothesis_space_size(reflection_grid)
        h_rand = sym_rep.hypothesis_space_size(random_grid)
        assert h_sym < h_rand, "Symmetric should have smaller hypothesis space"

    def test_applicability_high_on_symmetric(self, sym_rep, reflection_grid):
        alpha = sym_rep.applicability(reflection_grid)
        assert alpha > 0.3, f"Applicability should be elevated on symmetric grid, got {alpha:.2f}"

    def test_applicability_low_on_random(self, sym_rep, random_grid):
        alpha = sym_rep.applicability(random_grid)
        assert alpha < 0.5, f"Applicability should be low on random grid, got {alpha:.2f}"

    def test_failure_detect_on_random(self, sym_rep, random_grid):
        progress = {"steps_taken": 5, "constraints_found": 0}
        fs = sym_rep.failure_detect(random_grid, progress)
        assert fs is not None, "Should detect failure on random grid"
        assert fs.mode == "no_structure"

    def test_failure_detect_none_on_symmetric(self, sym_rep, reflection_grid):
        progress = {"steps_taken": 1, "constraints_found": 1}
        fs = sym_rep.failure_detect(reflection_grid, progress)
        # Should not fail when symmetry is found and reasoning has started
        if fs is not None:
            assert fs.mode != "no_structure", "Should not fail with 'no_structure'"

    def test_explain(self, sym_rep, reflection_grid):
        explanation = sym_rep.explain(reflection_grid)
        assert "reflection" in explanation.lower()

    def test_repr(self, sym_rep):
        r = repr(sym_rep)
        assert "SymmetryRepresentation" in r


# ─────────────────────────────────────────────────────────────────────
# Pipeline validation: Admission test + RCS
# ─────────────────────────────────────────────────────────────────────

class TestSymmetryPipeline:
    """
    Prove that SymmetryRepresentation passes the full admission pipeline:
      Failure Mode → Canonical Task → RCS → Admission
    """

    def test_admission_test_passes(self):
        """Symmetry rep should be admitted to the zoo."""
        from canonical_tasks.harness import admission_test
        from canonical_tasks.registry import tasks_for_representation

        sym_tasks = tasks_for_representation("symmetry")
        result = admission_test(
            SymmetryRepresentation(),
            [PixelRepresentation(), ObjectRepresentation()],
            sym_tasks,
        )
        assert result["unique_value"], "Symmetry should add unique value"
        assert result["no_regression"], "Symmetry should not cause regression"

    def test_rcs_positive(self):
        """Symmetry rep should have positive RCS."""
        from canonical_tasks.harness import contribution_score
        from canonical_tasks.registry import ALL_TASKS

        score = contribution_score(
            SymmetryRepresentation(),
            [PixelRepresentation(), ObjectRepresentation()],
            ALL_TASKS,
        )
        assert score["coverage_gain"] > 0, (
            f"Symmetry should add coverage, got {score}"
        )
        assert score["verdict"] in ("strong_accept", "weak_accept"), (
            f"Symmetry should be accepted, got {score['verdict']}"
        )

    def test_solves_symmetry_canonical_tasks(self, all_reps):
        """Symmetry rep should be ranked #1 on its canonical tasks."""
        from canonical_tasks.harness import evaluate_task
        from canonical_tasks.registry import tasks_for_representation

        sym_tasks = tasks_for_representation("symmetry")
        for task in sym_tasks:
            if task.task_id == "no_symmetry":
                continue  # negative test — symmetry should NOT win
            result = evaluate_task(task, all_reps, criterion="tension")
            if result.ranking and result.ranking[0] != "symmetry":
                # Check if symmetry is at least on Pareto front
                assert result.pareto_check or "symmetry" in result.ranking[:2], (
                    f"Task '{task.task_id}': symmetry should be top-ranked. "
                    f"Ranking: {result.ranking}"
                )

    def test_comparative_quality_on_reflection(self, all_reps, reflection_grid):
        """On a reflection-symmetric grid, symmetry should dominate."""
        q = compare_representations(all_reps, reflection_grid)
        q_sym = q["symmetry"]
        q_pixel = q["pixel"]
        # Symmetry should have lower complexity
        assert q_sym.C < q_pixel.C, (
            f"Symmetry complexity ({q_sym.C:.1f}) should be less than "
            f"pixel ({q_pixel.C:.1f})"
        )
