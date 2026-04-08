"""AHVS Data Analyst — goal-directed ML data analysis, selection, and augmentation.

Public API:
    analyze()      — one-call analysis pipeline (Profile → Plan → Execute → Report)
    profile_data() — Phase 1 only (data profiling)
    DataProfile, AnalysisPlan, ModuleResult, AnalysisReport — core models
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ahvs.data_analyst.models import (
    AnalysisPlan,
    AnalysisReport,
    DataProfile,
    ModuleResult,
)
from ahvs.data_analyst.profiler import profile_data

__all__ = [
    "analyze",
    "profile_data",
    "DataProfile",
    "AnalysisPlan",
    "ModuleResult",
    "AnalysisReport",
]

logger = logging.getLogger(__name__)


def analyze(
    data_path: str,
    *,
    goal: str = "",
    task: str = "",
    modules: list[str] | None = None,
    output_dir: str | None = None,
    label_hint: str | None = None,
    input_hint: list[str] | None = None,
    nrows: int | None = None,
    llm_client: Any | None = None,
) -> AnalysisReport:
    """Run the full 4-phase analysis pipeline.

    Args:
        data_path: Path to data file (CSV, Parquet, JSON, JSONL).
        goal: Natural-language goal (e.g., "build ABSA sentiment classifier").
        task: Shorthand task type (e.g., "classification"). Used if goal is empty.
        modules: Explicit module list. If provided, skips LLM planning.
        output_dir: Where to write results. Defaults to ``analysis_<timestamp>/``.
        label_hint: Force a specific column as the label.
        input_hint: Force specific columns as inputs.
        nrows: Limit rows for quick profiling.
        llm_client: Optional LLM client for Phase 2 planning.

    Returns:
        AnalysisReport with paths to markdown/JSON reports and all artifacts.
    """
    from ahvs.data_analyst import executor as exec_mod
    from ahvs.data_analyst import registry
    from ahvs.data_analyst.planner import plan
    from ahvs.data_analyst.synthesizer import synthesize

    # Ensure modules are discovered
    registry.discover()

    # Phase 1: Profile
    logger.info("Phase 1: Profiling %s", data_path)
    profile = profile_data(
        data_path, nrows=nrows, label_hint=label_hint, input_hint=input_hint
    )

    # Phase 2: Plan
    effective_goal = goal or task or "general data analysis"
    logger.info("Phase 2: Planning for goal '%s'", effective_goal)
    analysis_plan = plan(
        profile,
        effective_goal,
        llm_client=llm_client,
        modules_override=modules,
    )

    # Resolve output directory
    if output_dir:
        out = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(f"analysis_{ts}")
    out.mkdir(parents=True, exist_ok=True)

    # Load the FULL DataFrame for execution (nrows only affects profiling)
    from ahvs.data_analyst.profiler import _load_data, _detect_format

    p = Path(data_path).resolve()
    df = _load_data(p, _detect_format(p))

    # Phase 3: Execute
    logger.info("Phase 3: Executing %d modules", len(analysis_plan.modules))
    results = exec_mod.execute(df, profile, analysis_plan, out)

    # Phase 4: Synthesize
    logger.info("Phase 4: Synthesizing report")
    report = synthesize(profile, analysis_plan, results, out)

    logger.info(
        "Analysis complete. Completeness: %.0f%%. Report: %s",
        report.completeness_score(),
        report.markdown_path,
    )
    return report
