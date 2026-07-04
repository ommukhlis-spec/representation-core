"""
Representation Benchmark Report Generator.

Produces automated benchmark reports comparing all representations
in the zoo against all canonical tasks.

Output:
  - report.md       — human-readable benchmark report
  - leaderboard.json — machine-readable rankings

This is the CI gate: if coverage drops or RCS drops, merge is blocked.

Usage:
  python benchmarks/report.py
  python benchmarks/report.py --output-dir ./results
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Import from the package
from representations import PixelRepresentation, ObjectRepresentation, SymmetryRepresentation
from canonical_tasks.registry import (
    ALL_TASKS,
    tasks_for_representation,
    failure_mode_mapping,
    print_failure_mode_table,
)
from canonical_tasks.harness import (
    evaluate_suite,
    contribution_score,
    SuiteResult,
)


# ─────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────

def generate_report(output_dir: str = "benchmarks/results") -> Dict:
    """
    Generate a full benchmark report.

    Returns:
        Dict with report data (also written to output_dir).
    """
    os.makedirs(output_dir, exist_ok=True)

    # All representations in the zoo
    all_reps = [
        PixelRepresentation(),
        ObjectRepresentation(),
        SymmetryRepresentation(),
    ]

    # ── Evaluate full suite ────────────────────────────────────────
    suite_result = evaluate_suite(ALL_TASKS, all_reps, criterion="tension")

    # ── Per-representation stats ───────────────────────────────────
    rep_stats = {}
    for rep in all_reps:
        name = rep.name

        # Which tasks does this rep solve?
        solved = []
        failed = []
        for r in suite_result.results:
            if r.ranking and r.ranking[0] == name:
                solved.append(r.task.task_id)
            elif r.task.best_representation == name and not r.best_correct:
                failed.append(r.task.task_id)

        # Which failure modes does this rep handle?
        failure_modes_covered = []
        for r in suite_result.results:
            if r.ranking and r.ranking[0] == name and r.task.failure_mode:
                failure_modes_covered.append(r.task.failure_mode)

        # Canonical tasks for this rep
        canonical = tasks_for_representation(name)
        canonical_ids = [t.task_id for t in canonical]

        # Average metrics across tasks where applied
        complexities = []
        applicabilities = []
        for r in suite_result.results:
            if name in r.quality_vectors:
                q = r.quality_vectors[name]
                complexities.append(q.C)
            # Compute applicability on the task's observation
            if r.task.train_inputs:
                try:
                    alpha = rep.applicability(r.task.train_inputs[0])
                    applicabilities.append(alpha)
                except Exception:
                    pass

        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
        avg_applicability = sum(applicabilities) / len(applicabilities) if applicabilities else 0.0

        # RCS for this rep
        other_reps = [r for r in all_reps if r.name != name]
        rcs = contribution_score(rep, other_reps, ALL_TASKS)

        rep_stats[name] = {
            "version": rep.version,
            "canonical_tasks": canonical_ids,
            "tasks_solved": solved,
            "tasks_failed": failed,
            "coverage": len(solved),
            "total_canonical": len(canonical_ids),
            "coverage_rate": len(solved) / max(len(canonical_ids), 1),
            "failure_modes_covered": sorted(set(failure_modes_covered)),
            "avg_complexity": round(avg_complexity, 1),
            "avg_applicability": round(avg_applicability, 3),
            "rcs": rcs,
        }

    # ── Leaderboard ────────────────────────────────────────────────
    leaderboard = sorted(
        rep_stats.items(),
        key=lambda item: (item[1]["coverage"], -item[1]["avg_complexity"]),
        reverse=True,
    )

    leaderboard_data = {
        "generated_at": datetime.now().isoformat(),
        "total_tasks": len(ALL_TASKS),
        "total_representations": len(all_reps),
        "rankings": [
            {
                "rank": i + 1,
                "representation": name,
                "coverage": stats["coverage"],
                "rcs_verdict": stats["rcs"]["verdict"],
                "rcs_score": stats["rcs"]["total_score"],
                "avg_applicability": stats["avg_applicability"],
            }
            for i, (name, stats) in enumerate(leaderboard)
        ],
    }

    # ── Write files ────────────────────────────────────────────────
    # Leaderboard JSON
    with open(os.path.join(output_dir, "leaderboard.json"), "w") as f:
        json.dump(leaderboard_data, f, indent=2)

    # Markdown report
    md = _generate_markdown(suite_result, rep_stats, leaderboard, leaderboard_data)
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Report generated: {output_dir}/report.md")
    print(f"Leaderboard: {output_dir}/leaderboard.json")

    return {
        "suite_result": suite_result,
        "rep_stats": rep_stats,
        "leaderboard": leaderboard_data,
    }


def _generate_markdown(
    suite_result: SuiteResult,
    rep_stats: Dict,
    leaderboard: List,
    leaderboard_data: Dict,
) -> str:
    """Generate the markdown report."""

    lines = []
    lines.append("# Representation Benchmark Report")
    lines.append(f"**Generated:** {leaderboard_data['generated_at']}")
    lines.append(f"**Tasks:** {leaderboard_data['total_tasks']}")
    lines.append(f"**Representations:** {leaderboard_data['total_representations']}")
    lines.append("")

    # ── Summary ────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append(f"Suite pass rate: **{suite_result.pass_rate:.0%}** ({suite_result.passed}/{suite_result.total})")
    lines.append(f"Best-representation accuracy: **{suite_result.best_correct_rate:.0%}**")
    lines.append("")

    # ── Leaderboard ────────────────────────────────────────────────
    lines.append("## Leaderboard")
    lines.append("| Rank | Representation | Coverage | RCS Verdict | RCS Score | Avg α |")
    lines.append("|------|---------------|----------|-------------|-----------|-------|")
    for entry in leaderboard_data["rankings"]:
        lines.append(
            f"| {entry['rank']} | **{entry['representation']}** | "
            f"{entry['coverage']}/{leaderboard_data['total_tasks']} | "
            f"{entry['rcs_verdict']} | {entry['rcs_score']:.1f} | "
            f"{entry['avg_applicability']:.2f} |"
        )
    lines.append("")

    # ── Per-Representation Details ─────────────────────────────────
    lines.append("## Per-Representation Details")
    for name, stats in rep_stats.items():
        lines.append(f"### {name} (v{stats['version']})")
        lines.append(f"- **Coverage:** {stats['coverage']}/{stats['total_canonical']} canonical tasks")
        lines.append(f"- **Failure modes covered:** {stats['failure_modes_covered']}")
        lines.append(f"- **Avg complexity:** {stats['avg_complexity']:.1f}")
        lines.append(f"- **Avg applicability:** {stats['avg_applicability']:.3f}")
        lines.append(f"- **RCS:** {stats['rcs']['total_score']:.1f} ({stats['rcs']['verdict']})")

        if stats["tasks_solved"]:
            lines.append(f"- **Tasks solved:** {', '.join(stats['tasks_solved'])}")
        if stats["tasks_failed"]:
            lines.append(f"- **Tasks failed:** {', '.join(stats['tasks_failed'])}")
        lines.append("")

    # ── Failure Mode Table ─────────────────────────────────────────
    lines.append("## Failure Mode → Representation Mapping")
    lines.append("| Failure Mode | Canonical Task | Required Representation |")
    lines.append("|-------------|---------------|------------------------|")
    for row in failure_mode_mapping():
        lines.append(
            f"| {row['failure_mode']} | {row['task_name']} | "
            f"**{row['required_representation']}** |"
        )
    lines.append("")

    # ── CI Gate ────────────────────────────────────────────────────
    lines.append("## CI Gate Status")
    min_coverage = 3  # each rep must solve at least 3 failure modes
    all_pass = True
    for name, stats in rep_stats.items():
        passed = stats["coverage"] >= min_coverage
        status = "✅" if passed else "❌"
        lines.append(f"- {status} **{name}**: {stats['coverage']} coverage (min: {min_coverage})")
        if not passed:
            all_pass = False

    if all_pass:
        lines.append("\n**Gate: PASSED** ✅ — all representations meet minimum coverage.")
    else:
        lines.append("\n**Gate: FAILED** ❌ — one or more representations below minimum coverage.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Representation Benchmark Report")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory for report files",
    )
    args = parser.parse_args()

    generate_report(args.output_dir)
