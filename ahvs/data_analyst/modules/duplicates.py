"""Duplicate detection module — MinHash LSH fuzzy deduplication.

Adapted from KD deduplicate_lsh(). Text column dedup via MinHash; exact-match
dedup for non-text data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)


def run(inp: ModuleInput) -> ModuleResult:
    """Detect and report duplicates in the dataset."""
    df = inp.df
    params = inp.params
    output_dir = inp.output_dir / "duplicates"
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = params.get("lsh_threshold", 0.85)
    num_perm = params.get("lsh_num_perm", 128)

    summary: dict[str, Any] = {}
    warnings: list[str] = []
    artifacts: list[Path] = []
    narrative_parts: list[str] = []

    # --- Exact duplicates (all columns) ---
    exact_dups = int(df.duplicated().sum())
    summary["exact_duplicates"] = exact_dups
    if exact_dups:
        narrative_parts.append(f"{exact_dups} exact duplicate rows found.")
        warnings.append(f"{exact_dups} exact duplicates detected.")

    # --- Fuzzy duplicates via MinHash LSH on text columns ---
    text_cols = [
        c for c in inp.input_cols if c in df.columns and df[c].dtype == "object"
    ]
    fuzzy_results: dict[str, dict[str, Any]] = {}

    for col in text_cols:
        try:
            dup_indices = _lsh_duplicates(df, col, threshold, num_perm)
            n_fuzzy = len(dup_indices)
            pct = round(n_fuzzy / len(df) * 100, 2) if len(df) else 0.0
            fuzzy_results[col] = {
                "fuzzy_duplicates": n_fuzzy,
                "fuzzy_duplicate_pct": pct,
                "threshold": threshold,
            }
            narrative_parts.append(
                f"Column '{col}': {n_fuzzy} fuzzy duplicates ({pct}%) at threshold {threshold}."
            )
            if pct > 10:
                warnings.append(
                    f"Column '{col}' has {pct}% fuzzy duplicates — consider deduplication."
                )
        except ImportError:
            fuzzy_results[col] = {"error": "datasketch not installed"}
            warnings.append(
                f"datasketch not installed — skipping fuzzy dedup for '{col}'."
            )
        except Exception as exc:
            fuzzy_results[col] = {"error": str(exc)}
            warnings.append(f"Fuzzy dedup failed for '{col}': {exc}")

    summary["fuzzy_duplicates"] = fuzzy_results

    if not narrative_parts:
        narrative_parts.append("No duplicates detected.")

    return ModuleResult(
        module_name="duplicates",
        status="success",
        summary=summary,
        narrative=" ".join(narrative_parts),
        artifacts=artifacts,
        warnings=warnings,
    )


def _lsh_duplicates(
    df: pd.DataFrame, col: str, threshold: float, num_perm: int
) -> set[int]:
    """Return row indices that are fuzzy duplicates via MinHash LSH.

    Every row is inserted into the LSH index so that later rows can match
    against any earlier row (not just the first representative of a cluster).
    The first occurrence of each duplicate group is kept; subsequent matches
    are flagged as duplicates.
    """
    from datasketch import MinHash, MinHashLSH

    texts = df[col].fillna("").astype(str).tolist()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicate_indices: set[int] = set()

    for idx, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for word in text.lower().split():
            m.update(word.encode("utf8"))

        candidates = lsh.query(m)
        if candidates:
            # This row is similar to an already-indexed row → duplicate
            duplicate_indices.add(idx)
        # Always insert so later rows can match against this one too
        try:
            lsh.insert(idx, m)
        except ValueError:
            # Duplicate key — already inserted (shouldn't happen but guard)
            pass

    return duplicate_indices
