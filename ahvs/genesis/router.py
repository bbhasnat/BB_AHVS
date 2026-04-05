"""Problem router — maps a natural-language problem description to a solver.

Currently simple keyword matching. When a second solver is added, this
can be upgraded to LLM-based classification.
"""

from __future__ import annotations

import logging
import re

from ahvs.genesis.registry import SolverRegistry

logger = logging.getLogger(__name__)

# Keyword → problem type mapping (checked in order)
_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["classif", "categoriz", "label", "sentiment", "intent", "detect", "spam", "toxic"], "classification"),
    (["multi-label", "multilabel", "tagging"], "multi-label"),
    (["ner", "entity", "extraction"], "extraction"),
    (["summariz", "generat", "paraphras"], "generation"),
]


class ProblemRouter:
    """Route a problem description to the best available solver.

    Uses keyword matching against registered solver problem_types.
    Falls back to the first registered solver if no keywords match
    (since there is currently only one solver).
    """

    def __init__(self, registry: SolverRegistry) -> None:
        self._registry = registry

    def route(self, problem: str) -> str:
        """Return the solver name best matching the problem description."""
        problem_lower = problem.lower()

        # Try keyword matching
        for keywords, problem_type in _KEYWORD_MAP:
            if any(kw in problem_lower for kw in keywords):
                solver = self._registry.get_for_problem_type(problem_type)
                if solver:
                    logger.info(
                        "Routed problem to %s (matched type: %s)",
                        solver.name,
                        problem_type,
                    )
                    return solver.name

        # Fallback: return the first registered solver
        available = self._registry.list_solvers()
        if available:
            logger.info(
                "No keyword match — defaulting to first solver: %s", available[0]
            )
            return available[0]

        raise RuntimeError("No solvers registered. Check solvers.yaml.")
