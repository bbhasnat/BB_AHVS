"""Core data models for the AHVS data analyst agent.

All models use stdlib dataclasses to match AHVS conventions (not Pydantic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Column role classification
# ---------------------------------------------------------------------------

@dataclass
class ColumnInfo:
    """Metadata about a single column in the dataset."""

    name: str
    dtype: str  # pandas dtype as string
    role: Literal[
        "text_input", "numeric_input", "categorical_input",
        "label", "id", "timestamp", "metadata", "unknown",
    ] = "unknown"
    cardinality: int = 0
    null_count: int = 0
    null_pct: float = 0.0
    sample_values: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1 output — DataProfile
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """Complete profile of a dataset produced by Phase 1 (profiler)."""

    # File metadata
    source_path: str = ""
    file_format: str = ""  # csv, parquet, json, jsonl, sql, huggingface
    total_rows: int = 0
    total_columns: int = 0

    # Column info
    columns: list[ColumnInfo] = field(default_factory=list)

    # Detected roles (resolved column names)
    input_columns: list[str] = field(default_factory=list)
    label_column: str | None = None
    id_columns: list[str] = field(default_factory=list)

    # Quick stats (populated during profiling)
    class_distribution: dict[str, int] = field(default_factory=dict)
    quality_score: float = 0.0  # 0-100
    warnings: list[str] = field(default_factory=list)

    # Timestamps
    profiled_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON export / LLM context."""
        return {
            "source_path": self.source_path,
            "file_format": self.file_format,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "role": c.role,
                    "cardinality": c.cardinality,
                    "null_pct": round(c.null_pct, 2),
                }
                for c in self.columns
            ],
            "input_columns": self.input_columns,
            "label_column": self.label_column,
            "class_distribution": self.class_distribution,
            "quality_score": round(self.quality_score, 1),
            "warnings": self.warnings,
        }

    def summary_for_llm(self) -> str:
        """Short text summary suitable for sending to the LLM planner."""
        lines = [
            f"File: {self.source_path} ({self.file_format})",
            f"Shape: {self.total_rows} rows x {self.total_columns} columns",
            "",
            "Columns:",
        ]
        for c in self.columns:
            lines.append(
                f"  {c.name}: {c.dtype} (role={c.role}, "
                f"cardinality={c.cardinality}, null={c.null_pct:.1f}%)"
            )
        if self.label_column and self.class_distribution:
            lines.append("")
            lines.append(f"Label column: {self.label_column}")
            lines.append("Class distribution:")
            for cls, count in sorted(
                self.class_distribution.items(), key=lambda x: -x[1]
            ):
                pct = count / self.total_rows * 100 if self.total_rows else 0
                lines.append(f"  {cls}: {count} ({pct:.1f}%)")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2 output — AnalysisPlan
# ---------------------------------------------------------------------------

@dataclass
class ModuleSpec:
    """A single module to execute, with its configuration."""

    name: str  # registry key, e.g. "eda", "class_balance"
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # why the planner selected this module


@dataclass
class AnalysisPlan:
    """Ordered plan of modules to execute, produced by Phase 2 (planner)."""

    task_type: str = ""  # e.g. "multiclass_classification", "binary_classification"
    input_columns: list[str] = field(default_factory=list)
    label_column: str | None = None
    modules: list[ModuleSpec] = field(default_factory=list)
    notes: str = ""  # planner notes / reasoning

    def module_names(self) -> list[str]:
        return [m.name for m in self.modules]


# ---------------------------------------------------------------------------
# Module I/O contract
# ---------------------------------------------------------------------------

@dataclass
class ModuleInput:
    """Standard input passed to every analysis module."""

    df: Any  # pd.DataFrame — typed as Any to avoid import at module level
    profile: DataProfile
    plan: AnalysisPlan
    task_type: str = ""
    input_cols: list[str] = field(default_factory=list)
    label_col: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("."))


@dataclass
class ModuleResult:
    """Standard output returned by every analysis module."""

    module_name: str = ""
    status: Literal["success", "skipped", "error"] = "success"
    summary: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    figures: list[Path] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_message: str = ""
    # Optional: modules that transform data (subsample, split) can set this
    # so downstream modules operate on the modified DataFrame.
    transformed_df: Any = None  # pd.DataFrame | None

    @staticmethod
    def make_error(module_name: str, error: str) -> ModuleResult:
        return ModuleResult(
            module_name=module_name,
            status="error",
            error_message=error,
        )

    @staticmethod
    def make_skipped(module_name: str, reason: str) -> ModuleResult:
        return ModuleResult(
            module_name=module_name,
            status="skipped",
            narrative=reason,
        )


# ---------------------------------------------------------------------------
# Phase 4 output — AnalysisReport
# ---------------------------------------------------------------------------

@dataclass
class AnalysisReport:
    """Final report produced by Phase 4 (synthesizer)."""

    output_dir: Path = field(default_factory=lambda: Path("."))
    profile: DataProfile = field(default_factory=DataProfile)
    plan: AnalysisPlan = field(default_factory=AnalysisPlan)
    module_results: list[ModuleResult] = field(default_factory=list)
    markdown_path: Path | None = None
    json_path: Path | None = None
    recommendations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def completeness_score(self) -> float:
        """Percentage of requested modules that succeeded."""
        if not self.module_results:
            return 0.0
        ok = sum(1 for r in self.module_results if r.status == "success")
        return ok / len(self.module_results) * 100


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a single validation check."""

    id: str
    passed: bool
    severity: Literal["error", "warning", "info"] = "info"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
