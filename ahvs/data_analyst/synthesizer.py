"""Phase 4: Report synthesis — markdown + JSON output.

Adapted from KD MarkdownReporter and JSONExporter. Adds visualization
references and structured recommendations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ahvs.data_analyst.models import (
    AnalysisPlan,
    AnalysisReport,
    DataProfile,
    ModuleResult,
    ValidationResult,
)
from ahvs.data_analyst.validators import recommend, validate

logger = logging.getLogger(__name__)


def synthesize(
    profile: DataProfile,
    plan: AnalysisPlan,
    module_results: list[ModuleResult],
    output_dir: Path,
) -> AnalysisReport:
    """Synthesize all module results into a final report.

    Args:
        profile: DataProfile from Phase 1.
        plan: AnalysisPlan from Phase 2.
        module_results: Results from Phase 3 execution.
        output_dir: Root output directory.

    Returns:
        AnalysisReport with paths to generated files.
    """
    report = AnalysisReport(
        output_dir=output_dir,
        profile=profile,
        plan=plan,
        module_results=module_results,
    )

    # Run validations — include module failures in status (M6 fix)
    validations = validate(profile)
    # Add validation entries for failed modules
    for r in module_results:
        if r.status == "error":
            validations.append(
                ValidationResult(
                    id=f"module_failure_{r.module_name}",
                    passed=False,
                    severity="error",
                    message=f"Module '{r.module_name}' failed: {r.error_message}",
                )
            )
    status = recommend(validations)

    # Generate recommendations
    report.recommendations = _build_recommendations(
        profile, module_results, validations, status
    )

    # Write markdown report
    md_path = output_dir / "analysis_report.md"
    md_content = _render_markdown(profile, plan, module_results, validations, status, report.recommendations, output_dir)
    md_path.write_text(md_content, encoding="utf-8")
    report.markdown_path = md_path

    # Write JSON report
    json_path = output_dir / "analysis_report.json"
    json_content = _render_json(profile, plan, module_results, validations, status, report.recommendations)
    json_content = _sanitize_for_json(json_content)
    json_path.write_text(
        json.dumps(json_content, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    report.json_path = json_path

    logger.info("Report written to %s", output_dir)
    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(
    profile: DataProfile,
    plan: AnalysisPlan,
    results: list[ModuleResult],
    validations: list,
    status: str,
    recommendations: list[str],
    output_dir: Path | None = None,
) -> str:
    lines: list[str] = []

    # Header
    lines.append("# Data Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Status:** {status}")
    lines.append(f"**Source:** `{profile.source_path}`")
    lines.append(f"**Task type:** {plan.task_type or 'auto-detected'}")
    lines.append("")

    # Dataset overview
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- **Rows:** {profile.total_rows}")
    lines.append(f"- **Columns:** {profile.total_columns}")
    lines.append(f"- **Input columns:** {', '.join(profile.input_columns) or 'none detected'}")
    lines.append(f"- **Label column:** {profile.label_column or 'none detected'}")
    lines.append(f"- **Quality score:** {profile.quality_score:.0f}/100")
    lines.append("")

    # Column details
    lines.append("### Columns")
    lines.append("")
    lines.append("| Column | Type | Role | Cardinality | Null % |")
    lines.append("|--------|------|------|-------------|--------|")
    for ci in profile.columns:
        lines.append(
            f"| {ci.name} | {ci.dtype} | {ci.role} | {ci.cardinality} | {ci.null_pct:.1f}% |"
        )
    lines.append("")

    # Class distribution
    if profile.class_distribution:
        lines.append("### Class Distribution")
        lines.append("")
        lines.append("| Class | Count | % |")
        lines.append("|-------|-------|---|")
        for cls, count in sorted(
            profile.class_distribution.items(), key=lambda x: -x[1]
        ):
            pct = count / profile.total_rows * 100 if profile.total_rows else 0
            lines.append(f"| {cls} | {count} | {pct:.1f}% |")
        lines.append("")

    # Module results
    lines.append("## Analysis Results")
    lines.append("")
    for r in results:
        icon = {"success": "+", "skipped": "~", "error": "!"}.get(r.status, "?")
        lines.append(f"### [{icon}] {r.module_name} ({r.status})")
        lines.append("")
        if r.narrative:
            lines.append(r.narrative)
            lines.append("")
        if r.warnings:
            for w in r.warnings:
                lines.append(f"- **Warning:** {w}")
            lines.append("")
        if r.error_message:
            lines.append(f"- **Error:** {r.error_message}")
            lines.append("")
        if r.figures:
            for fig in r.figures:
                # M3: render paths relative to output_dir so links work
                if output_dir:
                    try:
                        rel = fig.relative_to(output_dir)
                    except ValueError:
                        rel = fig.name
                else:
                    rel = fig.name
                lines.append(f"![{fig.stem}]({rel})")
            lines.append("")
        if r.artifacts:
            lines.append("**Artifacts:**")
            for a in r.artifacts:
                lines.append(f"- `{a}`")
            lines.append("")

    # Validations
    errors = [v for v in validations if v.severity == "error" and not v.passed]
    warnings_list = [v for v in validations if v.severity == "warning"]
    if errors or warnings_list:
        lines.append("## Validations")
        lines.append("")
        for v in errors:
            lines.append(f"- **ERROR:** {v.message}")
        for v in warnings_list:
            lines.append(f"- **WARNING:** {v.message}")
        lines.append("")

    # Recommendations
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for r in recommendations:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


def _render_json(
    profile: DataProfile,
    plan: AnalysisPlan,
    results: list[ModuleResult],
    validations: list,
    status: str,
    recommendations: list[str],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(),
        "status": status,
        "profile": profile.to_dict(),
        "plan": {
            "task_type": plan.task_type,
            "input_columns": plan.input_columns,
            "label_column": plan.label_column,
            "modules": [
                {"name": m.name, "params": m.params, "reason": m.reason}
                for m in plan.modules
            ],
        },
        "module_results": [
            {
                "module": r.module_name,
                "status": r.status,
                "summary": r.summary,
                "narrative": r.narrative,
                "warnings": r.warnings,
                "error_message": r.error_message,
                "figures": [str(f) for f in r.figures],
                "artifacts": [str(a) for a in r.artifacts],
            }
            for r in results
        ],
        "validations": [
            {"id": v.id, "passed": v.passed, "severity": v.severity, "message": v.message}
            for v in validations
        ],
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# JSON sanitization (convert NaN/inf to None for strict JSON)
# ---------------------------------------------------------------------------

import math


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/inf floats with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Recommendations builder
# ---------------------------------------------------------------------------


def _build_recommendations(
    profile: DataProfile,
    results: list[ModuleResult],
    validations: list,
    status: str,
) -> list[str]:
    recs: list[str] = []

    if status == "NOT_RECOMMENDED":
        recs.append("Address validation errors before proceeding with model training.")

    # Data size
    if profile.total_rows < 500:
        recs.append(
            f"Dataset is small ({profile.total_rows} rows). "
            "Consider synthetic augmentation or collecting more data."
        )

    # Class balance
    if profile.class_distribution:
        counts = list(profile.class_distribution.values())
        if counts:
            ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
            if ratio > 5:
                recs.append(
                    "Significant class imbalance detected. Consider class-balanced "
                    "sampling or synthetic augmentation for minority classes."
                )

    # Duplicates
    for r in results:
        if r.module_name == "duplicates" and r.status == "success":
            exact = r.summary.get("exact_duplicates", 0)
            if exact > 0:
                recs.append(f"Remove {exact} exact duplicate rows before training.")
            for col, info in r.summary.get("fuzzy_duplicates", {}).items():
                pct = info.get("fuzzy_duplicate_pct", 0)
                if pct > 10:
                    recs.append(
                        f"Column '{col}' has {pct}% fuzzy duplicates. Consider deduplication."
                    )

    # Quality
    if profile.quality_score < 70:
        recs.append(
            f"Data quality score is {profile.quality_score:.0f}/100. "
            "Review missing values and encoding issues."
        )

    if not recs:
        recs.append("Data looks good. Proceed with model training.")

    return recs
