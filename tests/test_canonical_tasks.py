"""
Validate the canonical task harness.

Currently we only have Pixel and Object representations.
Expected results:
  - Object tasks (counting): SHOULD pass (object rep exists)
  - Symmetry tasks: WILL fail (no symmetry rep yet)
  - Topology tasks: WILL fail (no topology rep yet)
  - Region tasks: WILL fail (no region rep yet)
  - Constraint tasks: WILL fail (no constraint rep yet)

These failures are NOT bugs — they are the roadmap for Phase 1 (zoo expansion).
Each failure tells us exactly which representation to build next.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.representation import Observation
from representations.pixel import PixelRepresentation
from representations.object import ObjectRepresentation
from canonical_tasks.registry import ALL_TASKS, tasks_for_representation, reason_to_exist
from canonical_tasks.harness import evaluate_task, evaluate_suite, admission_test
from canonical_tasks.specification import CanonicalTask, grid


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def available_reps():
    """Representations currently in the zoo."""
    return [PixelRepresentation(), ObjectRepresentation()]


@pytest.fixture
def object_tasks():
    """Tasks where object representation should win."""
    return tasks_for_representation("object")


# ─────────────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────────────

class TestRegistry:
    """Validate the task registry structure."""

    def test_all_tasks_have_unique_ids(self):
        ids = [t.task_id for t in ALL_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs: {ids}"

    def test_all_tasks_have_justification(self):
        for task in ALL_TASKS:
            assert task.justification, f"Task {task.task_id} missing justification"
            assert len(task.justification) > 50, (
                f"Task {task.task_id} justification too short: {len(task.justification)} chars"
            )

    def test_all_tasks_have_train_inputs(self):
        for task in ALL_TASKS:
            assert len(task.train_inputs) > 0, f"Task {task.task_id} has no training inputs"

    def test_all_tasks_have_best_rep(self):
        for task in ALL_TASKS:
            assert task.best_representation, f"Task {task.task_id} has no best_representation"

    def test_task_count(self):
        """We should have at least 20 canonical tasks."""
        assert len(ALL_TASKS) >= 20, f"Only {len(ALL_TASKS)} tasks — need >= 20"

    def test_all_categories_covered(self):
        categories = set(t.category for t in ALL_TASKS)
        expected = {"counting", "symmetry", "topology", "region", "constraint"}
        missing = expected - categories
        assert not missing, f"Missing categories: {missing}"

    def test_reason_to_exist(self):
        reasons = reason_to_exist()
        assert "object" in reasons, "Object rep should have canonical tasks"
        assert "symmetry" in reasons, "Symmetry rep should have canonical tasks"
        assert "topology" in reasons, "Topology rep should have canonical tasks"
        assert "region" in reasons, "Region rep should have canonical tasks"
        assert "constraint" in reasons, "Constraint rep should have canonical tasks"


# ─────────────────────────────────────────────────────────────────────
# Harness tests with CURRENT representations (pixel + object only)
# ─────────────────────────────────────────────────────────────────────

class TestHarnessWithCurrentReps:
    """
    Test the harness with only pixel and object representations.

    Object tasks should pass. All others are expected to fail because
    the required representation doesn't exist yet.
    """

    def test_object_tasks_pass(self, available_reps, object_tasks):
        """Every object canonical task should pass when object rep is available."""
        results = []
        for task in object_tasks:
            result = evaluate_task(task, available_reps, criterion="tension")
            results.append(result)
            assert result.passed, (
                f"Object task '{task.task_id}' FAILED: {result.notes}\n"
                f"Ranking: {result.ranking}"
            )

    def test_full_suite_runs(self, available_reps):
        """The full suite should run without crashes."""
        suite_result = evaluate_suite(ALL_TASKS, available_reps, criterion="tension")
        assert suite_result.total == len(ALL_TASKS)
        # Object tasks = 4/20 should pass
        assert suite_result.passed >= 4, (
            f"Expected at least 4 passing (object tasks), got {suite_result.passed}"
        )

    def test_suite_summary(self, available_reps):
        """Suite summary should produce a readable string."""
        suite_result = evaluate_suite(ALL_TASKS, available_reps, criterion="tension")
        summary = suite_result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "PASS" in summary or "FAIL" in summary

    def test_failures_are_expected(self, available_reps):
        """
        Tasks that fail should be those where the best representation
        doesn't exist yet. This test DOCUMENTS the roadmap.
        """
        suite_result = evaluate_suite(ALL_TASKS, available_reps, criterion="tension")

        # Group failures by best_representation
        failures_by_rep = {}
        for r in suite_result.results:
            if not r.passed:
                rep = r.task.best_representation
                if rep not in failures_by_rep:
                    failures_by_rep[rep] = []
                failures_by_rep[rep].append(r.task.task_id)

        # Object tasks should NOT fail (we have object rep)
        assert "object" not in failures_by_rep, (
            f"Object tasks should not fail! Got: {failures_by_rep.get('object', [])}"
        )

        # All other categories SHOULD fail (missing reps)
        # This is EXPECTED and is the roadmap for Phase 1
        for rep in ["symmetry", "topology", "region", "constraint"]:
            if rep in failures_by_rep:
                pass  # Expected — this rep doesn't exist yet

        # Print the roadmap
        print(f"\nPhase 1 roadmap (from test failures):")
        for rep, task_ids in sorted(failures_by_rep.items()):
            print(f"  Build {rep} representation → solves: {task_ids}")


# ─────────────────────────────────────────────────────────────────────
# Admission test
# ─────────────────────────────────────────────────────────────────────

class TestAdmissionTest:
    """Validate the admission test logic."""

    def test_existing_rep_passes_own_tasks(self, available_reps):
        """Object rep should be admitted for its own canonical tasks."""
        result = admission_test(
            ObjectRepresentation(),
            [PixelRepresentation()],
            tasks_for_representation("object"),
        )
        assert result["unique_value"], "Object rep should add unique value"
        assert result["no_regression"], "Object rep should not cause regression"

    def test_duplicate_rep_not_admitted(self, available_reps):
        """A second pixel rep should not be admitted (no unique value)."""
        result = admission_test(
            PixelRepresentation(),
            [PixelRepresentation()],
            ALL_TASKS,
        )
        # A duplicate pixel rep provides no unique value
        assert not result["admitted"], "Duplicate rep should not be admitted"


# ─────────────────────────────────────────────────────────────────────
# Task specification tests
# ─────────────────────────────────────────────────────────────────────

class TestTaskSpecification:
    """Validate individual task specifications."""

    def test_counting_task_grid(self):
        from canonical_tasks.tasks.counting import COUNT_DISCRETE_OBJECTS
        obs = COUNT_DISCRETE_OBJECTS.train_inputs[0]
        assert obs.height == 10
        assert obs.width == 10

    def test_symmetry_task_grid(self):
        from canonical_tasks.tasks.symmetry import HORIZONTAL_REFLECTION
        obs = HORIZONTAL_REFLECTION.train_inputs[0]
        assert obs.height == 5
        assert obs.width == 5

    def test_topology_task_grid(self):
        from canonical_tasks.tasks.topology import SINGLE_HOLE
        obs = SINGLE_HOLE.train_inputs[0]
        assert obs.height == 7
        assert obs.width == 7

    def test_region_task_grid(self):
        from canonical_tasks.tasks.region import OVERLAPPING_REGIONS
        obs = OVERLAPPING_REGIONS.train_inputs[0]
        assert obs.height == 7
        assert obs.width == 7

    def test_constraint_task_grid(self):
        from canonical_tasks.tasks.constraint import COLOR_CONSERVATION
        obs = COLOR_CONSERVATION.train_inputs[0]
        assert obs.height == 4
        assert obs.width == 4

    def test_no_symmetry_is_negative_test(self):
        """NO_SYMMETRY should have object as best (symmetry should lose)."""
        from canonical_tasks.tasks.symmetry import NO_SYMMETRY
        assert NO_SYMMETRY.best_representation != "symmetry", (
            "NO_SYMMETRY should have best_rep != symmetry (it's a negative test)"
        )
        assert NO_SYMMETRY.worst_representation == "symmetry"


# ─────────────────────────────────────────────────────────────────────
# Failure Mode mapping tests
# ─────────────────────────────────────────────────────────────────────

class TestFailureModeMapping:
    """Validate the Failure Mode → Task → Representation mapping."""

    def test_all_tasks_have_failure_mode(self):
        for task in ALL_TASKS:
            assert task.failure_mode, (
                f"Task {task.task_id} is missing failure_mode"
            )

    def test_failure_modes_are_unique_per_category(self):
        """Within each category, failure modes should be distinct."""
        from canonical_tasks.registry import failure_mode_mapping
        mapping = failure_mode_mapping()
        by_category = {}
        for row in mapping:
            cat = next(t.category for t in ALL_TASKS if t.task_id == row["task_id"])
            if cat not in by_category:
                by_category[cat] = set()
            by_category[cat].add(row["failure_mode"])
        for cat, modes in by_category.items():
            assert len(modes) == 4, (
                f"Category '{cat}' has {len(modes)} failure modes, expected 4"
            )

    def test_failure_mode_mapping_table(self):
        from canonical_tasks.registry import failure_mode_mapping, print_failure_mode_table

        # Verify it runs without error
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        print_failure_mode_table()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert "Failure Mode" in output
        assert "Canonical Task" in output
        assert "Required Representation" in output

    def test_mapping_coverage(self):
        """Every representation should have failure modes mapped to it."""
        from canonical_tasks.registry import failure_mode_mapping
        mapping = failure_mode_mapping()
        reps_covered = set(row["required_representation"] for row in mapping)
        expected = {"object", "symmetry", "topology", "region", "constraint"}
        assert reps_covered == expected, (
            f"Missing reps in mapping: {expected - reps_covered}"
        )


# ─────────────────────────────────────────────────────────────────────
# RCS (Representation Contribution Score) tests
# ─────────────────────────────────────────────────────────────────────

class TestContributionScore:
    """Validate the RCS scoring function."""

    def test_rcs_runs(self, available_reps, object_tasks):
        """RCS should compute without errors."""
        from canonical_tasks.harness import contribution_score
        score = contribution_score(
            ObjectRepresentation(),
            [PixelRepresentation()],
            object_tasks,
        )
        assert "coverage_gain" in score
        assert "redundancy" in score
        assert "total_score" in score
        assert "verdict" in score
        assert score["verdict"] in ("strong_accept", "weak_accept", "reject")

    def test_rcs_for_object_on_counting(self, available_reps):
        """Object rep should get positive RCS on counting tasks."""
        from canonical_tasks.harness import contribution_score
        from canonical_tasks.registry import tasks_for_representation
        counting_tasks = tasks_for_representation("object")
        score = contribution_score(
            ObjectRepresentation(),
            [PixelRepresentation()],
            counting_tasks,
        )
        assert score["coverage_gain"] > 0, (
            f"Object rep should add coverage on counting tasks, got {score}"
        )

    def test_rcs_duplicate_is_rejected(self, available_reps):
        """A duplicate pixel rep should get low/negative RCS."""
        from canonical_tasks.harness import contribution_score
        score = contribution_score(
            PixelRepresentation(),
            [PixelRepresentation()],
            ALL_TASKS,
        )
        assert score["coverage_gain"] == 0, "Duplicate rep should not add coverage"
        assert score["verdict"] == "reject", (
            f"Duplicate should be rejected, got {score['verdict']}"
        )


# ─────────────────────────────────────────────────────────────────────
# JSON export tests
# ─────────────────────────────────────────────────────────────────────

class TestJsonExport:
    """Validate JSON dataset export."""

    def test_export_runs(self, tmp_path):
        """Export should create JSON files without errors."""
        from canonical_tasks.registry import export_tasks_json
        import os

        export_dir = str(tmp_path / "canonical_tasks_export")
        result = export_tasks_json(export_dir)

        # Should have created manifest + 20 task files
        assert os.path.exists(os.path.join(export_dir, "manifest.json"))
        task_files = [f for f in os.listdir(export_dir) if f.endswith(".json")]
        assert len(task_files) == 21  # 20 tasks + 1 manifest

    def test_exported_json_is_valid(self, tmp_path):
        """Exported JSON should be parseable and contain required fields."""
        from canonical_tasks.registry import export_tasks_json
        import json, os

        export_dir = str(tmp_path / "canonical_tasks_export")
        export_tasks_json(export_dir)

        # Verify one task file
        fpath = os.path.join(export_dir, "count_discrete_objects.json")
        with open(fpath) as f:
            data = json.load(f)

        required_fields = [
            "task_id", "name", "category", "failure_mode",
            "justification", "best_representation", "train_inputs",
        ]
        for field in required_fields:
            assert field in data, f"Missing field '{field}' in exported JSON"
