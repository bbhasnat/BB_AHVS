"""Text statistics module — token length, vocabulary, per-class breakdown.

Adapted from KD TextStatisticsAnalyzer. Adds per-class stats and percentile analysis.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)


def run(inp: ModuleInput) -> ModuleResult:
    """Compute text statistics for input text columns."""
    df = inp.df
    text_cols = [
        c for c in inp.input_cols if c in df.columns and df[c].dtype == "object"
    ]
    if not text_cols:
        return ModuleResult.make_skipped("text_stats", "No text columns found.")

    output_dir = inp.output_dir / "text_stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    narrative_parts: list[str] = []
    warnings: list[str] = []
    figures: list[Path] = []

    for col in text_cols:
        texts = df[col].dropna().astype(str)
        total = len(texts)
        if total == 0:
            continue

        # Character lengths
        lengths = texts.str.len()
        # Word counts
        word_counts = texts.str.split().str.len()

        # Vocabulary (streaming set — avoids materializing full token list)
        vocab: set[str] = set()
        for t in texts:
            vocab.update(t.lower().split())
        vocab_size = len(vocab)

        # Duplicates
        unique_count = texts.nunique()
        dup_count = total - unique_count
        dup_pct = round(dup_count / total * 100, 2) if total else 0.0

        # Empty texts
        empty_count = int((texts.str.strip() == "").sum())

        col_stats: dict[str, Any] = {
            "total_texts": total,
            "char_length": {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": round(float(lengths.mean()), 1),
                "median": round(float(lengths.median()), 1),
                "std": round(float(lengths.std()), 1),
                "p25": int(lengths.quantile(0.25)),
                "p75": int(lengths.quantile(0.75)),
                "p95": int(lengths.quantile(0.95)),
            },
            "word_count": {
                "min": int(word_counts.min()),
                "max": int(word_counts.max()),
                "mean": round(float(word_counts.mean()), 1),
                "median": round(float(word_counts.median()), 1),
                "std": round(float(word_counts.std()), 1),
            },
            "vocabulary_size": vocab_size,
            "unique_texts": unique_count,
            "duplicate_count": dup_count,
            "duplicate_pct": dup_pct,
            "empty_texts": empty_count,
        }

        # Per-class breakdown
        if inp.label_col and inp.label_col in df.columns:
            per_class: dict[str, dict[str, Any]] = {}
            for cls, grp in df.groupby(inp.label_col):
                cls_texts = grp[col].dropna().astype(str)
                if len(cls_texts) == 0:
                    continue
                cls_wc = cls_texts.str.split().str.len()
                per_class[str(cls)] = {
                    "count": len(cls_texts),
                    "avg_words": round(float(cls_wc.mean()), 1),
                    "avg_chars": round(float(cls_texts.str.len().mean()), 1),
                }
            col_stats["per_class"] = per_class

        summary[col] = col_stats

        narrative_parts.append(
            f"Column '{col}': {total} texts, avg {col_stats['word_count']['mean']} words, "
            f"vocab {vocab_size}, {dup_count} duplicates ({dup_pct}%)."
        )
        if dup_pct > 10:
            warnings.append(f"Column '{col}' has {dup_pct}% duplicate texts.")
        if empty_count > 0:
            warnings.append(f"Column '{col}' has {empty_count} empty texts.")

        # Plot word-count distribution
        try:
            figures.extend(_plot_word_dist(word_counts, col, output_dir))
        except Exception:
            pass

    return ModuleResult(
        module_name="text_stats",
        status="success",
        summary=summary,
        narrative=" ".join(narrative_parts),
        figures=figures,
        warnings=warnings,
    )


def _plot_word_dist(
    word_counts: pd.Series, col_name: str, output_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(word_counts.dropna(), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Word Count Distribution — {col_name}")
    plt.tight_layout()

    path = output_dir / f"word_dist_{col_name}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]
