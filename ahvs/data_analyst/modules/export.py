"""Export module — save processed dataset in multiple formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)


def run(inp: ModuleInput) -> ModuleResult:
    """Export the dataset (or a filtered view) to the requested format(s)."""
    df = inp.df
    params = inp.params
    output_dir = inp.output_dir / "export"
    output_dir.mkdir(parents=True, exist_ok=True)

    formats: list[str] = params.get("formats", ["parquet"])
    if isinstance(formats, str):
        formats = [formats]

    # Optional column selection
    columns = params.get("columns")
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return ModuleResult.make_error(
                "export", f"Requested columns not found: {missing}"
            )
        df = df[columns]

    # Optional row filter via column-value dict (safe — no eval/exec)
    filters: dict[str, Any] | None = params.get("filters")
    if filters and isinstance(filters, dict):
        for col, value in filters.items():
            if col not in df.columns:
                return ModuleResult.make_error(
                    "export", f"Filter column '{col}' not found in data."
                )
            if isinstance(value, list):
                df = df[df[col].isin(value)]
            else:
                df = df[df[col] == value]

    artifacts: list[Path] = []
    warnings: list[str] = []

    for fmt in formats:
        fmt = fmt.lower().strip()
        try:
            path = _export(df, fmt, output_dir)
            artifacts.append(path)
        except Exception as exc:
            warnings.append(f"Export to {fmt} failed: {exc}")

    # H4: report error if no artifacts were produced
    if not artifacts:
        return ModuleResult.make_error(
            "export",
            f"All export formats failed: {', '.join(warnings)}" if warnings
            else "No formats specified or all failed.",
        )

    summary: dict[str, Any] = {
        "rows_exported": len(df),
        "columns_exported": len(df.columns),
        "formats": [str(a.suffix).lstrip(".") for a in artifacts],
        "files": [str(a) for a in artifacts],
    }

    return ModuleResult(
        module_name="export",
        status="success",
        summary=summary,
        narrative=f"Exported {len(df)} rows to {', '.join(summary['formats'])}.",
        artifacts=artifacts,
        warnings=warnings,
    )


def _export(df: pd.DataFrame, fmt: str, output_dir: Path) -> Path:
    if fmt == "parquet":
        path = output_dir / "data.parquet"
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        path = output_dir / "data.csv"
        df.to_csv(path, index=False)
    elif fmt == "json":
        path = output_dir / "data.json"
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    elif fmt == "jsonl":
        path = output_dir / "data.jsonl"
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return path
