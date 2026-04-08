"""Class balance module — label frequency, imbalance metrics, entropy.

Adapted from KD ClassDistributionAnalyzer.
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
    """Analyse class distribution of the label column."""
    label_col = inp.label_col
    if not label_col or label_col not in inp.df.columns:
        return ModuleResult.make_skipped(
            "class_balance", "No label column available."
        )

    df = inp.df
    output_dir = inp.output_dir / "class_balance"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = df[label_col].dropna().astype(str)
    total = len(labels)
    class_counts = dict(Counter(labels))
    num_classes = len(class_counts)

    if total == 0:
        return ModuleResult.make_skipped(
            "class_balance", "Label column is entirely null."
        )

    # Percentages
    class_pct = {
        cls: round(count / total * 100, 2) for cls, count in class_counts.items()
    }

    # Imbalance ratio
    counts_list = list(class_counts.values())
    max_c, min_c = max(counts_list), min(counts_list)
    imbalance_ratio = max_c / min_c if min_c > 0 else float("inf")

    # Shannon entropy
    probs = [c / total for c in counts_list if c > 0]
    entropy = float(-sum(p * np.log2(p) for p in probs)) if probs else 0.0

    # Narrative
    lines = [
        f"{num_classes} classes, {total} labelled samples.",
        f"Imbalance ratio: {imbalance_ratio:.2f}:1.",
        f"Shannon entropy: {entropy:.3f}.",
    ]
    warnings: list[str] = []
    if imbalance_ratio > 10:
        warnings.append(f"Severe imbalance ({imbalance_ratio:.1f}:1). Consider augmentation or downsampling.")
    elif imbalance_ratio > 3:
        warnings.append(f"Moderate imbalance ({imbalance_ratio:.1f}:1).")

    # Per-class table in narrative
    lines.append("")
    lines.append("Class distribution:")
    for cls in sorted(class_counts, key=lambda c: -class_counts[c]):
        lines.append(f"  {cls}: {class_counts[cls]} ({class_pct[cls]}%)")

    summary: dict[str, Any] = {
        "total_labelled": total,
        "num_classes": num_classes,
        "class_counts": class_counts,
        "class_percentages": class_pct,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "entropy": round(entropy, 4),
    }

    # Plot
    figures: list[Path] = []
    try:
        figures.extend(_plot_distribution(class_counts, output_dir))
    except Exception as exc:
        warnings.append(f"Plot failed: {exc}")

    return ModuleResult(
        module_name="class_balance",
        status="success",
        summary=summary,
        narrative="\n".join(lines),
        figures=figures,
        warnings=warnings,
    )


def _plot_distribution(
    class_counts: dict[str, int], output_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(class_counts, key=lambda c: -class_counts[c])
    counts = [class_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.6), 4))
    bars = ax.bar(range(len(classes)), counts, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    plt.tight_layout()

    path = output_dir / "class_distribution.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]
