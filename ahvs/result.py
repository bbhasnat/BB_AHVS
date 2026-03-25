"""HypothesisResult — the tool-agnostic output contract for AHVS execution.

All execution paths (Promptfoo, DSPy, Phoenix, sandbox + custom script)
write to this dataclass. Everything downstream — report, memory, verify —
reads only this format, never tool-specific output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HypothesisResult:
    """Normalised outcome of executing one hypothesis."""

    hypothesis_id: str          # "H1", "H2", etc.
    hypothesis_type: str        # "prompt_rewrite", "architecture_change", etc.
    primary_metric: str         # metric name (from baseline_metric.json)
    metric_value: float         # measured value after applying hypothesis
    baseline_value: float       # value from baseline_metric.json
    delta: float                # metric_value - baseline_value
    delta_pct: float            # (delta / baseline_value) * 100
    regression_guard_passed: bool  # True if guard exited 0, or not configured
    eval_method: str            # "promptfoo" | "phoenix" | "custom_script" | "dspy+promptfoo"
    artifact_paths: list[str] = field(default_factory=list)  # paths relative to cycle_dir
    raw_output_path: str = ""   # full eval output directory for audit
    duration_seconds: float = 0.0
    skill_planned: str | None = None  # skill name from validation plan (plan-time, not runtime)
    error: str | None = None    # None on success
    measurement_status: str = "not_executed"  # "measured" | "extraction_failed" | "sandbox_error" | "not_executed"
    kept: bool = False          # set by operator after reviewing cycle_summary
    worktree_path: str = ""     # path to kept worktree (empty if cleaned up)
    patch_path: str = ""        # path to .patch file relative to cycle_dir
    execution_mode: str = "repo_grounded"  # "repo_grounded" | "no_worktree"

    @classmethod
    def make_error(
        cls,
        hypothesis_id: str,
        hypothesis_type: str,
        primary_metric: str,
        baseline_value: float,
        error: str,
    ) -> "HypothesisResult":
        """Construct a failed result placeholder."""
        return cls(
            hypothesis_id=hypothesis_id,
            hypothesis_type=hypothesis_type,
            primary_metric=primary_metric,
            metric_value=baseline_value,
            baseline_value=baseline_value,
            delta=0.0,
            delta_pct=0.0,
            regression_guard_passed=False,
            eval_method="none",
            error=error,
            measurement_status="sandbox_error",
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def improved(self) -> bool:
        """True if this hypothesis beats the baseline with a valid measurement."""
        return (
            self.delta > 0
            and self.regression_guard_passed
            and self.error is None
            and self.measurement_status == "measured"
        )


def save_results(results: list[HypothesisResult], path: Path) -> None:
    """Save results, merging with any existing results (newer wins by hypothesis_id)."""
    merged: dict[str, HypothesisResult] = {}
    if path.exists():
        for existing in load_results(path):
            merged[existing.hypothesis_id] = existing
    for r in results:
        merged[r.hypothesis_id] = r
    path.write_text(
        json.dumps([r.to_dict() for r in merged.values()], indent=2),
        encoding="utf-8",
    )


def load_results(path: Path) -> list[HypothesisResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        # Migrate old skill_used → skill_planned (v4 rename)
        if "skill_used" in item and "skill_planned" not in item:
            item["skill_planned"] = item.pop("skill_used")
    return [HypothesisResult(**item) for item in data]
