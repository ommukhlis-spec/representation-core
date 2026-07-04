"""
Canonical Representation Tasks — Phase 1.5.

Each task is a synthetic puzzle with KNOWN ground truth about which
representation should perform best. This is the scientific benchmark
for validating representation theory.

ARC is the consumer of this benchmark, not the benchmark itself.
"""

from .specification import CanonicalTask, grid, empty_grid
from .registry import (
    ALL_TASKS,
    tasks_by_category,
    tasks_for_representation,
    reason_to_exist,
    failure_mode_mapping,
    print_failure_mode_table,
    print_registry,
    export_tasks_json,
)
from .harness import (
    evaluate_task,
    evaluate_suite,
    admission_test,
    contribution_score,
    TaskResult,
    SuiteResult,
)

__all__ = [
    "CanonicalTask",
    "grid",
    "empty_grid",
    "ALL_TASKS",
    "tasks_by_category",
    "tasks_for_representation",
    "reason_to_exist",
    "failure_mode_mapping",
    "print_failure_mode_table",
    "print_registry",
    "export_tasks_json",
    "evaluate_task",
    "evaluate_suite",
    "admission_test",
    "contribution_score",
    "TaskResult",
    "SuiteResult",
]
