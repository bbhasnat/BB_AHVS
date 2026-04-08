"""Subsample module — intelligent data selection strategies.

Includes stratified, class-balanced, and diversity sampling. Diversity sampling
adapted from KD sample_diverse(). Additional strategies (uncertainty, coreset)
are v2 placeholders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)

Strategy = Literal[
    "stratified",
    "class_balanced",
    "diversity",
    "random",
]


def run(inp: ModuleInput) -> ModuleResult:
    """Select a subsample of the dataset using the configured strategy."""
    df = inp.df
    params = inp.params
    output_dir = inp.output_dir / "subsample"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = params.get("target_size")
    strategy: str = params.get("strategy", "stratified")
    label_col = inp.label_col

    if target_size is None:
        return ModuleResult.make_skipped(
            "subsample",
            "No target_size specified in params. Skipping subsampling.",
        )

    target_size = int(target_size)
    if target_size >= len(df):
        return ModuleResult.make_skipped(
            "subsample",
            f"target_size ({target_size}) >= dataset size ({len(df)}). No subsampling needed.",
        )

    warnings: list[str] = []
    actual_strategy = strategy

    if strategy == "stratified" and label_col and label_col in df.columns:
        sampled = _stratified_sample(df, label_col, target_size)
    elif strategy == "class_balanced" and label_col and label_col in df.columns:
        sampled = _class_balanced_sample(df, label_col, target_size)
    elif strategy == "diversity":
        if "cluster_label" not in df.columns:
            warnings.append(
                "No cluster_label column — falling back to random sampling."
            )
            actual_strategy = "random"
            sampled = df.sample(n=target_size, random_state=42)
        else:
            sampled = _diversity_sample(df, target_size)
    else:
        # Fallback to random
        if strategy not in ("random",):
            warnings.append(
                f"Strategy '{strategy}' requires a label column or special setup. "
                "Falling back to random sampling."
            )
            actual_strategy = "random"
        sampled = df.sample(n=target_size, random_state=42)

    # Enforce exact target_size with final trim or top-up
    sampled = _enforce_target_size(sampled, df, target_size)

    # Save sampled data
    out_path = output_dir / "subsample.parquet"
    sampled.to_parquet(out_path, index=False)

    # Summary
    summary: dict[str, Any] = {
        "strategy_requested": strategy,
        "strategy_used": actual_strategy,
        "original_size": len(df),
        "sampled_size": len(sampled),
        "reduction_pct": round((1 - len(sampled) / len(df)) * 100, 1),
    }
    if label_col and label_col in sampled.columns:
        summary["sampled_class_distribution"] = (
            sampled[label_col].value_counts().to_dict()
        )

    narrative = (
        f"Subsampled {len(df)} → {len(sampled)} rows "
        f"({summary['reduction_pct']}% reduction) using '{actual_strategy}' strategy."
    )

    return ModuleResult(
        module_name="subsample",
        status="success",
        summary=summary,
        narrative=narrative,
        artifacts=[out_path],
        warnings=warnings,
        transformed_df=sampled,  # H2: pass to downstream modules
    )


# ---------------------------------------------------------------------------
# Size enforcement (M7 fix)
# ---------------------------------------------------------------------------


def _enforce_target_size(
    sampled: pd.DataFrame, original: pd.DataFrame, target: int
) -> pd.DataFrame:
    """Trim or top-up the sample to match the exact target size."""
    if len(sampled) == target:
        return sampled
    if len(sampled) > target:
        return sampled.sample(n=target, random_state=42)
    # Top-up: fill remaining from original rows not already in sample
    deficit = target - len(sampled)
    remaining = original.loc[~original.index.isin(sampled.index)]
    if len(remaining) == 0:
        return sampled
    topup = remaining.sample(n=min(deficit, len(remaining)), random_state=42)
    return pd.concat([sampled, topup], ignore_index=True)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _stratified_sample(
    df: pd.DataFrame, label_col: str, target_size: int
) -> pd.DataFrame:
    """Random sample preserving class ratios."""
    frac = target_size / len(df)
    groups = []
    allocated = 0
    class_groups = list(df.groupby(label_col))

    for _, grp in class_groups:
        n = max(1, round(len(grp) * frac))
        n = min(n, len(grp))
        groups.append(grp.sample(n=n, random_state=42))
        allocated += n

    return pd.concat(groups, ignore_index=True)


def _class_balanced_sample(
    df: pd.DataFrame, label_col: str, target_size: int
) -> pd.DataFrame:
    """Equal samples per class (downsample majority)."""
    classes = df[label_col].dropna().unique()
    per_class = max(1, target_size // len(classes))
    remainder = target_size - per_class * len(classes)

    groups = []
    for i, cls in enumerate(sorted(classes)):
        grp = df[df[label_col] == cls]
        n = per_class + (1 if i < remainder else 0)  # distribute remainder
        groups.append(grp.sample(n=min(n, len(grp)), random_state=42))
    return pd.concat(groups, ignore_index=True)


def _diversity_sample(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    """Cluster-based diversity sampling (adapted from KD sample_diverse)."""
    cluster_groups = df.groupby("cluster_label")
    cluster_sizes = cluster_groups.size()
    noise_count = cluster_sizes.get(-1, 0)
    n_clustered = len(df) - noise_count

    sampled_dfs: list[pd.DataFrame] = []
    remaining = target_size

    for cluster_label, size in cluster_sizes.items():
        if cluster_label == -1:
            continue
        proportion = size / n_clustered if n_clustered > 0 else 0
        n_sample = max(1, round(target_size * proportion))
        n_sample = min(n_sample, size, remaining)
        if n_sample > 0:
            grp = cluster_groups.get_group(cluster_label)
            s = grp.sample(n=min(n_sample, len(grp)), random_state=42)
            sampled_dfs.append(s)
            remaining -= len(s)

    # Fill remaining from noise
    if remaining > 0 and noise_count > 0:
        noise_df = cluster_groups.get_group(-1)
        sampled_dfs.append(
            noise_df.sample(n=min(remaining, len(noise_df)), random_state=42)
        )

    if sampled_dfs:
        return pd.concat(sampled_dfs, ignore_index=True)
    return df.sample(n=min(target_size, len(df)), random_state=42)
