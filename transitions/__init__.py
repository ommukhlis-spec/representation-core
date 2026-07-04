"""
Transition dynamics between representations.
"""

from .transition import (
    FAILURE_TO_REPRESENTATION,
    INVARIANT_TO_REPRESENTATION,
    build_failure_signature,
    detect_stagnation,
    execute_transition,
    propose_transitions,
)

__all__ = [
    "FAILURE_TO_REPRESENTATION",
    "INVARIANT_TO_REPRESENTATION",
    "detect_stagnation",
    "build_failure_signature",
    "propose_transitions",
    "execute_transition",
]
