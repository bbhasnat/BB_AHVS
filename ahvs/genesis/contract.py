"""Genesis solver contract — defines what solvers produce and how they're called."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class GenesisResult:
    """What every solver must produce.

    ``baseline_metric`` is written directly as ``.ahvs/baseline_metric.json``
    in the project directory. It must contain at minimum:
    ``primary_metric``, the metric name's value, and ``eval_command``.
    """

    project_dir: Path
    baseline_metric: dict          # written as .ahvs/baseline_metric.json
    eval_command: str              # shell command to re-measure the metric
    model_path: str | None = None  # path to trained model (if applicable)
    summary: str = ""              # human-readable summary of what was built
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and bool(self.eval_command)


@runtime_checkable
class Solver(Protocol):
    """Interface that all solvers implement.

    A solver takes a natural-language problem description plus data and
    produces a working project with a measurable baseline metric.
    """

    name: str
    problem_types: list[str]  # e.g. ["classification", "multi-label"]

    def solve(
        self,
        problem: str,
        data_path: str,
        target_metric: str,
        output_dir: str,
        config_overrides: dict | None = None,
    ) -> GenesisResult: ...
