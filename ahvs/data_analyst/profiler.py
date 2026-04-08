"""Phase 1: Data profiling — load, infer schema, classify columns, detect labels.

Adapted from the KD pipeline's DataLoader, ColumnMatcher, and
data_intelligence modules. Generalized beyond NLP classification to support
tabular, text, CV, regression, and multi-label tasks.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahvs.data_analyst.models import ColumnInfo, DataProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_data(
    data_path: str,
    *,
    nrows: int | None = None,
    label_hint: str | None = None,
    input_hint: list[str] | None = None,
) -> DataProfile:
    """Profile a dataset file and return a DataProfile.

    Args:
        data_path: Path to data file (CSV, Parquet, JSON, JSONL).
        nrows: Optional row limit for quick profiling of large files.
        label_hint: If provided, use this column as the label instead of
            auto-detecting.
        input_hint: If provided, use these columns as inputs instead of
            auto-detecting.

    Returns:
        DataProfile with schema, column roles, class distribution, and
        quality information.
    """
    path = Path(data_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    file_format = _detect_format(path)
    df = _load_data(path, file_format, nrows=nrows)

    profile = DataProfile(
        source_path=str(path),
        file_format=file_format,
        total_rows=len(df),
        total_columns=len(df.columns),
    )

    # Step 1: Infer column metadata
    profile.columns = _infer_columns(df)

    # Step 2: Classify column roles
    _classify_roles(profile, df, label_hint=label_hint, input_hint=input_hint)

    # Step 3: Compute class distribution if label detected
    if profile.label_column and profile.label_column in df.columns:
        labels = df[profile.label_column].dropna().astype(str)
        profile.class_distribution = dict(Counter(labels))

    # Step 4: Compute quality score
    profile.quality_score = _compute_quality_score(df, profile)

    # Step 5: Generate warnings
    profile.warnings = _generate_warnings(profile)

    logger.info(
        "Profiled %s: %d rows x %d cols, label=%s, inputs=%s",
        path.name,
        profile.total_rows,
        profile.total_columns,
        profile.label_column,
        profile.input_columns,
    )
    return profile


# ---------------------------------------------------------------------------
# Data loading (adapted from KD DataLoader)
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    fmt_map = {
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".json": "json",
        ".jsonl": "jsonl",
    }
    return fmt_map.get(suffix, "csv")


def _load_data(
    path: Path, file_format: str, *, nrows: int | None = None
) -> pd.DataFrame:
    """Load data from disk into a DataFrame."""
    if file_format == "parquet":
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.head(nrows)
        return df

    if file_format in ("json", "jsonl"):
        lines = file_format == "jsonl"
        df = pd.read_json(path, lines=lines, nrows=nrows)
        return df

    # CSV / TSV — multi-encoding, multi-delimiter (from KD DataLoader)
    return _load_csv(path, nrows=nrows)


def _load_csv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load CSV with encoding detection and delimiter inference."""
    encoding = _detect_encoding(path)
    encodings = [encoding, "utf-8", "latin-1"]
    # deduplicate while preserving order
    seen: set[str] = set()
    unique_encodings = []
    for e in encodings:
        if e not in seen:
            seen.add(e)
            unique_encodings.append(e)

    last_error: Exception | None = None
    for enc in unique_encodings:
        for delim in [",", ";", "\t"]:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    delimiter=delim,
                    low_memory=False,
                    nrows=nrows,
                )
                if len(df.columns) > 1:
                    return df
            except Exception as exc:
                last_error = exc
                continue
        # Last-ditch: default delimiter
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, nrows=nrows)
            if not df.empty:
                return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not load CSV {path}: {last_error}"
    )


def _detect_encoding(path: Path) -> str:
    """Detect file encoding via chardet (best-effort)."""
    try:
        import chardet

        with open(path, "rb") as f:
            raw = f.read(10_000)
        result = chardet.detect(raw)
        enc = result.get("encoding", "utf-8") or "utf-8"
        if enc.lower() == "ascii":
            enc = "utf-8"
        if (result.get("confidence") or 0) < 0.7:
            enc = "utf-8"
        return enc
    except ImportError:
        return "utf-8"


# ---------------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------------


def _infer_columns(df: pd.DataFrame) -> list[ColumnInfo]:
    """Build ColumnInfo list for every column in the DataFrame."""
    infos: list[ColumnInfo] = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        cardinality = int(non_null.nunique()) if len(non_null) > 0 else 0
        sample_vals = [str(v) for v in non_null.head(5).tolist()]

        infos.append(
            ColumnInfo(
                name=col,
                dtype=str(series.dtype),
                cardinality=cardinality,
                null_count=null_count,
                null_pct=null_count / len(df) * 100 if len(df) else 0.0,
                sample_values=sample_vals,
            )
        )
    return infos


# ---------------------------------------------------------------------------
# Column role classification (generalized from KD ColumnMatcher)
# ---------------------------------------------------------------------------

# Patterns that suggest an ID column
_ID_PATTERNS = re.compile(
    r"^(id|idx|index|row_?id|sample_?id|doc_?id|uuid|_id)$", re.I
)
# Patterns that suggest a timestamp column
_TS_PATTERNS = re.compile(
    r"(date|time|timestamp|created|updated|_at$|_on$)", re.I
)
# Patterns that suggest a label column
_LABEL_PATTERNS = re.compile(
    r"(label|target|class|category|sentiment|tag|annotation|ground_?truth|y$)", re.I
)
# Patterns that suggest metadata (not useful as features)
_META_PATTERNS = re.compile(
    r"(file_?name|path|url|source|origin|split|fold|partition)", re.I
)


def _classify_roles(
    profile: DataProfile,
    df: pd.DataFrame,
    *,
    label_hint: str | None = None,
    input_hint: list[str] | None = None,
) -> None:
    """Classify each column's role and populate profile fields."""
    n = len(df)

    for col_info in profile.columns:
        col_info.role = _guess_role(col_info, n)

    # Override with user hints
    if label_hint:
        for ci in profile.columns:
            if ci.name == label_hint:
                ci.role = "label"

    if input_hint:
        for ci in profile.columns:
            if ci.name in input_hint:
                if ci.role not in ("label",):
                    ci.role = (
                        "text_input"
                        if ci.dtype == "object"
                        else "numeric_input"
                    )

    # Auto-detect label if no hint
    if not label_hint:
        label_col = _detect_label_column(profile.columns, df)
        if label_col:
            for ci in profile.columns:
                if ci.name == label_col:
                    ci.role = "label"

    # Resolve final lists
    profile.label_column = next(
        (c.name for c in profile.columns if c.role == "label"), None
    )
    if input_hint:
        profile.input_columns = input_hint
    else:
        profile.input_columns = [
            c.name
            for c in profile.columns
            if c.role in ("text_input", "numeric_input", "categorical_input")
        ]
    profile.id_columns = [
        c.name for c in profile.columns if c.role == "id"
    ]


def _guess_role(col: ColumnInfo, n_rows: int) -> str:
    """Heuristic role guess for a single column."""
    name = col.name

    # ID columns
    if _ID_PATTERNS.search(name):
        return "id"
    # High-cardinality integer with sequential-looking values → ID
    if col.dtype in ("int64", "int32") and n_rows > 0:
        if col.cardinality > 0.9 * n_rows:
            return "id"

    # Timestamps
    if _TS_PATTERNS.search(name):
        return "timestamp"
    if "datetime" in col.dtype:
        return "timestamp"

    # Metadata
    if _META_PATTERNS.search(name):
        return "metadata"

    # Label candidates
    if _LABEL_PATTERNS.search(name):
        if col.dtype == "object" and 1 < col.cardinality <= 100:
            return "label"

    # Text input: object dtype with long-ish values
    if col.dtype == "object":
        if col.cardinality > 100 or col.cardinality == 0:
            return "text_input"
        if 2 < col.cardinality <= 100:
            # Could be categorical input or label — mark categorical for now
            return "categorical_input"
        return "text_input"

    # Numeric input
    if col.dtype in ("int64", "int32", "float64", "float32"):
        if col.cardinality <= 20 and n_rows > 0:
            return "categorical_input"
        return "numeric_input"

    return "unknown"


def _detect_label_column(
    columns: list[ColumnInfo], df: pd.DataFrame
) -> str | None:
    """Auto-detect the most likely label column.

    Adapted from KD's find_label_column — generalized beyond task_id pattern
    matching to work with any dataset.
    """
    candidates: list[tuple[str, float]] = []

    for ci in columns:
        score = 0.0

        # Pattern match on name
        if _LABEL_PATTERNS.search(ci.name):
            score += 0.5

        # Categorical with reasonable cardinality
        if ci.dtype == "object" and 2 <= ci.cardinality <= 50:
            score += 0.3
        elif ci.dtype in ("int64", "int32") and 2 <= ci.cardinality <= 20:
            score += 0.2

        # Low null rate
        if ci.null_pct < 5:
            score += 0.1

        # Penalize high cardinality (unlikely to be a label)
        if ci.cardinality > 100:
            score -= 0.5

        # Penalize if it looks like an ID
        if _ID_PATTERNS.search(ci.name):
            score -= 1.0

        if score > 0:
            candidates.append((ci.name, score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Quality scoring (adapted from KD DataQualityAnalyzer)
# ---------------------------------------------------------------------------


def _compute_quality_score(df: pd.DataFrame, profile: DataProfile) -> float:
    """Composite quality score (0-100)."""
    if len(df) == 0:
        return 0.0

    score = 100.0
    n = len(df)

    # Penalize missing values (average across columns)
    null_pcts = [c.null_pct for c in profile.columns]
    if null_pcts:
        avg_null = sum(null_pcts) / len(null_pcts)
        score -= min(avg_null * 0.5, 30)

    # Penalize encoding issues
    encoding_issues = 0
    for col in df.select_dtypes(include=["object"]).columns:
        encoding_issues += int(df[col].astype(str).str.contains("\ufffd", regex=False).sum())
    if encoding_issues:
        score -= min((encoding_issues / n) * 100 * 0.2, 20)

    # Penalize empty strings
    empty_count = 0
    for col in df.select_dtypes(include=["object"]).columns:
        empty_count += int((df[col].astype(str).str.strip() == "").sum())
    if empty_count:
        score -= min((empty_count / n) * 100 * 0.1, 15)

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def _generate_warnings(profile: DataProfile) -> list[str]:
    """Generate human-readable warnings based on the profile."""
    warnings: list[str] = []

    if profile.total_rows < 100:
        warnings.append(
            f"Very small dataset ({profile.total_rows} rows). "
            "Results may not be statistically meaningful."
        )
    elif profile.total_rows < 500:
        warnings.append(
            f"Small dataset ({profile.total_rows} rows). "
            "Consider whether this is sufficient for your task."
        )

    if not profile.label_column:
        warnings.append("No label column detected. Provide label_hint if applicable.")

    if not profile.input_columns:
        warnings.append("No input columns detected. Provide input_hint if applicable.")

    # Check class imbalance
    if profile.class_distribution:
        counts = list(profile.class_distribution.values())
        if counts:
            max_c, min_c = max(counts), min(counts)
            if min_c > 0:
                ratio = max_c / min_c
                if ratio > 10:
                    warnings.append(
                        f"Severe class imbalance (ratio {ratio:.1f}:1). "
                        "Consider subsampling or augmentation."
                    )
                elif ratio > 3:
                    warnings.append(
                        f"Moderate class imbalance (ratio {ratio:.1f}:1)."
                    )

    # High null columns
    for ci in profile.columns:
        if ci.null_pct > 50:
            warnings.append(f"Column '{ci.name}' is >50% null ({ci.null_pct:.1f}%).")

    if profile.quality_score < 50:
        warnings.append(
            f"Low data quality score ({profile.quality_score:.0f}/100)."
        )

    return warnings
