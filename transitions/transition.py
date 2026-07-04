"""
Transition dynamics between representations.

This module implements the "Representation Dynamics" layer:
  - Detect when a representation is failing
  - Propose candidate next representations
  - Execute transitions based on FailureSignature + EmergingInvariants

The transition logic itself is NOT an AI. It is a deterministic engine
that encodes our hypotheses about how representational change works.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.representation import (
    FailureSignature,
    Invariant,
    Observation,
    Representation,
    Transition,
)

# ─────────────────────────────────────────────────────────────────────
# Transition graph (static knowledge)
# ─────────────────────────────────────────────────────────────────────

# Maps failure modes to plausible next representations.
# This is the "topology of representation space" encoded as explicit knowledge.
# In a full system, this would be learned. For now, it's hand-designed
# based on our theoretical framework.

FAILURE_TO_REPRESENTATION: Dict[str, List[str]] = {
    "overlap": ["region", "topology"],
    "non_discrete": ["region", "symmetry", "topology"],
    "partial_symmetry": ["topology", "constraint"],
    "ambiguous_topology": ["object", "region"],
    "sequential_only": ["constraint", "topology"],
    "too_many_exceptions": ["constraint", "symmetry"],
    "no_structure": ["pixel"],  # fallback
}

# Reverse: which emerging invariants suggest which representation
INVARIANT_TO_REPRESENTATION: Dict[str, List[str]] = {
    "object_permanence": ["object"],
    "color_conservation": ["constraint", "symmetry"],
    "spatial_symmetry": ["symmetry"],
    "connectivity": ["topology", "region"],
    "sequential_pattern": ["constraint"],
    "size_invariance": ["object", "constraint"],
    "topological_invariance": ["topology"],
    "repetition": ["symmetry", "constraint"],
}


# ─────────────────────────────────────────────────────────────────────
# Transition engine
# ─────────────────────────────────────────────────────────────────────

def detect_stagnation(
    reasoning_progress: Dict[str, Any],
    patience: int = 5,
    complexity_growth_threshold: float = 2.0,
) -> bool:
    """
    Detect whether reasoning has stagnated.

    Signals:
      - No new constraints found in `patience` steps
      - Complexity (description length) is growing instead of shrinking
      - Contradictions detected

    Returns True if stagnation is detected.
    """
    steps = reasoning_progress.get("steps_taken", 0)
    constraints = reasoning_progress.get("constraints_found", 0)
    contradictions = reasoning_progress.get("contradictions", 0)
    last_progress_step = reasoning_progress.get("last_progress_step", 0)

    # No progress in recent steps
    no_progress = (steps - last_progress_step) >= patience

    # Contradictions found
    has_contradictions = contradictions > 0

    # Complexity is growing
    complexity_trend = reasoning_progress.get("complexity_trend", 0.0)
    complexity_growing = complexity_trend > complexity_growth_threshold

    return no_progress or has_contradictions or complexity_growing


def build_failure_signature(
    rep: Representation,
    observation: Observation,
    reasoning_progress: Dict[str, Any],
) -> Optional[FailureSignature]:
    """
    Build a structured failure signature from the representation's
    own failure detection and the reasoning progress signals.
    """
    # First, ask the representation to self-diagnose
    own_failure = rep.failure_detect(observation, reasoning_progress)
    if own_failure is not None:
        return own_failure

    # Fallback: use generic stagnation detection
    if detect_stagnation(reasoning_progress):
        return FailureSignature(
            mode="unknown_stagnation",
            detail=f"Reasoning stagnated after {reasoning_progress.get('steps_taken', 0)} steps",
            confidence=0.5,
            evidence=reasoning_progress,
        )

    return None


def propose_transitions(
    rep: Representation,
    observation: Observation,
    failure: FailureSignature,
    available_representations: List[Representation],
    emerging_invariants: Optional[List[Invariant]] = None,
) -> List[Transition]:
    """
    Propose candidate next representations.

    Combines:
      - PUSH: failure signature → which representations to leave FOR
      - PULL: emerging invariants → which representations are ATTRACTED

    Returns list of Transitions, ordered by plausibility.
    """
    candidates: List[Transition] = []

    # PUSH: from failure
    push_targets = FAILURE_TO_REPRESENTATION.get(failure.mode, [])
    for target in push_targets:
        if target != rep.name and _representation_available(target, available_representations):
            candidates.append(
                Transition(
                    source=rep.name,
                    target=target,
                    trigger="failure",
                    failure_signature=failure,
                    rationale=f"Failure mode '{failure.mode}' suggests {target}",
                )
            )

    # PULL: from invariants
    if emerging_invariants:
        for inv in emerging_invariants:
            pull_targets = INVARIANT_TO_REPRESENTATION.get(inv.name, [])
            for target in pull_targets:
                if target != rep.name and _representation_available(target, available_representations):
                    candidates.append(
                        Transition(
                            source=rep.name,
                            target=target,
                            trigger="invariant",
                            emerging_invariant=inv,
                            rationale=f"Emerging invariant '{inv.name}' attracts {target}",
                        )
                    )

    # Deduplicate and order: both-failure-and-invariant first
    seen = set()
    ordered: List[Transition] = []
    for t in candidates:
        key = (t.source, t.target)
        if key not in seen:
            seen.add(key)
            ordered.append(t)

    # Both triggers = strongest signal
    both = [t for t in ordered if t.trigger == "invariant" and any(
        t2.trigger == "failure" and t2.target == t.target for t2 in ordered
    )]
    rest = [t for t in ordered if t not in both]
    return both + rest


def _representation_available(name: str, available: List[Representation]) -> bool:
    """Check if a representation with the given name is in the available list."""
    return any(r.name == name for r in available)


def execute_transition(
    transition: Transition,
    available_representations: List[Representation],
) -> Optional[Representation]:
    """
    Execute a transition: return the target Representation object.

    Returns None if target is not available.
    """
    for rep in available_representations:
        if rep.name == transition.target:
            return rep
    return None
