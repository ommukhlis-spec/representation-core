"""
Canonical Task Validation Harness.

Evaluates whether the representation framework correctly identifies
the best and worst representations for each canonical task.

A framework "passes" a task if:
  1. The declared best_representation ranks #1 by tension (or is on the
     Pareto front and not dominated by alternatives)
  2. The declared worst_representation ranks last (or is dominated)
  3. The justification is consistent with the observed metrics

This is NOT about solving ARC. It's about validating that our theory
of representation quality corresponds to ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.representation import Representation
from metrics.metrics import (
    QualityVector,
    compute_quality,
    find_pareto_front,
    dominance_matrix,
    rank_representations,
)
from .specification import CanonicalTask


# ─────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """Result of evaluating representations on a single canonical task."""

    task: CanonicalTask
    passed: bool
    best_correct: bool = False      # best_rep ranked #1
    worst_correct: bool = False     # worst_rep ranked last
    pareto_check: bool = False      # best_rep on Pareto front
    quality_vectors: Dict[str, QualityVector] = field(default_factory=dict)
    ranking: List[str] = field(default_factory=list)
    dominance: Dict[str, Dict[str, str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [f"{status} | {self.task.task_id}: {self.task.name}"]
        lines.append(f"  Best: {self.task.best_representation} → ranked #{self._rank_of(self.task.best_representation)}")
        lines.append(f"  Worst: {self.task.worst_representation} → ranked #{self._rank_of(self.task.worst_representation)}")
        if self.notes:
            for note in self.notes:
                lines.append(f"  ⚠ {note}")
        return "\n".join(lines)

    def _rank_of(self, name: str) -> str:
        if name in self.ranking:
            return str(self.ranking.index(name) + 1)
        return "N/A"


@dataclass
class SuiteResult:
    """Aggregate results across all canonical tasks."""

    results: List[TaskResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def best_correct_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return sum(1 for r in self.results if r.best_correct) / self.total

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"CANONICAL TASK SUITE RESULTS: {self.passed}/{self.total} passed ({self.pass_rate:.0%})",
            f"Best-representation accuracy: {self.best_correct_rate:.0%}",
            "=" * 60,
        ]
        for r in self.results:
            lines.append(r.summary())
        lines.append("=" * 60)

        # Per-representation diagnostic
        lines.append("\nPer-representation failures:")
        rep_failures: Dict[str, List[str]] = {}
        for r in self.results:
            if not r.passed:
                rep = r.task.best_representation
                if rep not in rep_failures:
                    rep_failures[rep] = []
                rep_failures[rep].append(r.task.task_id)

        for rep, tasks in sorted(rep_failures.items()):
            lines.append(f"  {rep}: failed on {tasks}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────

def evaluate_task(
    task: CanonicalTask,
    representations: List[Representation],
    criterion: str = "tension",
) -> TaskResult:
    """
    Evaluate whether the framework correctly identifies the best and worst
    representations for a canonical task.

    Args:
        task: The canonical task to evaluate.
        representations: Available representations (at least 2).
        criterion: Ranking criterion ("tension", "execution_cost", "complexity").

    Returns:
        TaskResult with pass/fail and diagnostics.
    """
    # Use the first training example as the observation
    if not task.train_inputs:
        return TaskResult(
            task=task,
            passed=False,
            notes=["No training inputs provided"],
        )

    obs = task.train_inputs[0]

    # Compute quality vectors
    quality_vectors = {
        rep.name: compute_quality(rep, obs) for rep in representations
    }

    # Rank by tension
    ranking = rank_representations(representations, obs, by=criterion)
    ranking_names = [r.name for r in ranking]

    # Pareto front
    pareto = find_pareto_front(representations, obs)
    pareto_names = [r.name for r in pareto]

    # Dominance matrix
    dom = dominance_matrix(representations, obs)

    # Checks
    notes: List[str] = []
    best_correct = False
    worst_correct = False
    pareto_check = False

    # Check 1: Is best_representation ranked #1?
    if ranking_names and ranking_names[0] == task.best_representation:
        best_correct = True
    elif task.best_representation in pareto_names:
        # It's not #1 but it's on the Pareto front — acceptable
        pareto_check = True
        notes.append(
            f"best='{task.best_representation}' not #1 by {criterion} "
            f"(#1 is '{ranking_names[0]}'), but on Pareto front"
        )
    else:
        best_rank = ranking_names.index(task.best_representation) + 1 if task.best_representation in ranking_names else -1
        notes.append(
            f"best='{task.best_representation}' ranked #{best_rank}, "
            f"expected #1"
        )

    # Check 2: Is worst_representation ranked last (or dominated)?
    if task.worst_representation:
        if ranking_names and ranking_names[-1] == task.worst_representation:
            worst_correct = True
        elif task.worst_representation in ranking_names:
            worst_rank = ranking_names.index(task.worst_representation) + 1
            notes.append(
                f"worst='{task.worst_representation}' ranked #{worst_rank}, "
                f"expected last (#{len(ranking_names)})"
            )
        else:
            notes.append(f"worst='{task.worst_representation}' not found in ranking")

    # Check 3: Also acceptable alternatives
    if task.also_acceptable:
        acceptable_ranked = any(
            r.name in task.also_acceptable for r in ranking[:2]
        )
        if not acceptable_ranked and not best_correct:
            notes.append(
                f"Neither '{task.best_representation}' nor "
                f"{task.also_acceptable} in top 2"
            )

    # Overall pass
    passed = best_correct or pareto_check

    return TaskResult(
        task=task,
        passed=passed,
        best_correct=best_correct,
        worst_correct=worst_correct,
        pareto_check=pareto_check,
        quality_vectors=quality_vectors,
        ranking=ranking_names,
        dominance=dom,
        notes=notes,
    )


def evaluate_suite(
    tasks: List[CanonicalTask],
    representations: List[Representation],
    criterion: str = "tension",
) -> SuiteResult:
    """
    Evaluate the full suite of canonical tasks.

    Returns a SuiteResult aggregating all task results.
    """
    results = [evaluate_task(task, representations, criterion) for task in tasks]
    return SuiteResult(results=results)


def admission_test(
    new_representation: Representation,
    existing_representations: List[Representation],
    tasks: List[CanonicalTask],
) -> Dict[str, bool]:
    """
    Admission test for a new representation.

    A new representation must:
      1. Be the BEST on at least one canonical task where no existing
         representation is best
      2. Not regress: it should not cause existing representations
         to fail their canonical tasks

    Returns:
        {"admitted": bool, "unique_value": bool, "no_regression": bool}
    """
    all_reps = existing_representations + [new_representation]

    # Evaluate without the new representation
    old_result = evaluate_suite(tasks, existing_representations)

    # Evaluate with the new representation
    new_result = evaluate_suite(tasks, all_reps)

    # Check unique value: does new_rep become best on any task
    # where it wasn't before?
    unique_value = False
    for new_r, old_r in zip(new_result.results, old_result.results):
        if not old_r.best_correct and new_r.best_correct:
            if new_r.ranking and new_r.ranking[0] == new_representation.name:
                unique_value = True
                break

    # Check no regression
    no_regression = new_result.pass_rate >= old_result.pass_rate

    admitted = unique_value and no_regression

    return {
        "admitted": admitted,
        "unique_value": unique_value,
        "no_regression": no_regression,
    }


def contribution_score(
    new_representation: Representation,
    existing_representations: List[Representation],
    tasks: List[CanonicalTask],
) -> Dict[str, float]:
    """
    Representation Contribution Score (RCS).

    RCS(R) = coverage_gain - redundancy_penalty - complexity_penalty

    Unlike the binary admission test, RCS is a continuous score that:
      - Rewards solving tasks that no existing representation can solve
      - Penalizes overlap with existing representations (redundancy)
      - Penalizes internal complexity (maintenance cost of transition edges)

    This allows the zoo to grow to dozens of representations without
    becoming a "collection of ideas."

    Returns:
        {
            "coverage_gain": int,      # new tasks solved
            "redundancy": float,        # 0-1, overlap with existing reps
            "complexity_penalty": float, # 0-1, normalized by max complexity
            "total_score": float,        # RCS
            "verdict": str,             # "strong_accept", "weak_accept", "reject"
        }
    """
    all_reps = existing_representations + [new_representation]
    new_name = new_representation.name

    # ── Coverage gain: new tasks where new_rep ranks #1 ────────────
    old_result = evaluate_suite(tasks, existing_representations)
    new_result = evaluate_suite(tasks, all_reps)

    coverage_gain = 0
    tasks_solved: List[str] = []
    for old_r, new_r in zip(old_result.results, new_result.results):
        was_solved = old_r.passed
        is_solved = new_r.passed
        is_new_best = (
            not was_solved
            and is_solved
            and new_r.ranking
            and new_r.ranking[0] == new_name
        )
        if is_new_best:
            coverage_gain += 1
            tasks_solved.append(new_r.task.task_id)

    # ── Redundancy: how much does new_rep overlap with existing? ───
    # Count tasks where new_rep ties with an existing rep as #1
    overlap_count = 0
    total_tasks = len(tasks)
    for r in new_result.results:
        if r.ranking and r.ranking[0] == new_name:
            # Check if an existing rep was already solving this
            old_r = next(
                (o for o in old_result.results if o.task.task_id == r.task.task_id),
                None,
            )
            if old_r and old_r.passed:
                overlap_count += 1

    redundancy = overlap_count / max(total_tasks, 1)

    # ── Complexity penalty: cost of maintaining this rep ───────────
    # Proxy: number of transition edges this rep would need to/from existing reps
    # Each new rep adds edges to every existing rep → O(|existing|)
    n_existing = len(existing_representations)
    max_transitions = max(n_existing, 1)  # edges to all existing reps
    complexity_penalty = min(n_existing / 10.0, 1.0)  # cap at 1.0

    # ── Composite score ────────────────────────────────────────────
    total_score = coverage_gain - (2.0 * redundancy) - (0.5 * complexity_penalty)

    # Verdict
    if total_score >= 3.0 and coverage_gain >= 2:
        verdict = "strong_accept"
    elif total_score >= 1.0:
        verdict = "weak_accept"
    else:
        verdict = "reject"

    return {
        "coverage_gain": coverage_gain,
        "tasks_solved": tasks_solved,
        "redundancy": round(redundancy, 3),
        "complexity_penalty": round(complexity_penalty, 3),
        "total_score": round(total_score, 2),
        "verdict": verdict,
    }
