"""EDA module — descriptive statistics, data quality, and distribution plots.

Adapted from KD DataQualityAnalyzer. Adds visualization and numeric stats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)


def run(inp: ModuleInput) -> ModuleResult:
    """Run exploratory data analysis on the dataset."""
    df = inp.df
    output_dir = inp.output_dir / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    figures: list[Path] = []
    warnings: list[str] = []
    narrative_parts: list[str] = []

    # ---- Basic shape ----
    summary["shape"] = {"rows": len(df), "columns": len(df.columns)}
    narrative_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")

    # ---- Dtype breakdown ----
    dtype_counts: dict[str, int] = {}
    for dtype in df.dtypes:
        key = str(dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    summary["dtype_counts"] = dtype_counts

    # ---- Missing values ----
    missing: dict[str, Any] = {}
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        if n_miss > 0:
            missing[col] = {
                "count": n_miss,
                "pct": round(n_miss / len(df) * 100, 2),
            }
    summary["missing_values"] = missing
    if missing:
        worst = max(missing.items(), key=lambda x: x[1]["pct"])
        narrative_parts.append(
            f"{len(missing)} columns have missing values. "
            f"Worst: '{worst[0]}' ({worst[1]['pct']}% null)."
        )

    # ---- Numeric summary ----
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        # Convert numpy types for JSON safety
        summary["numeric_summary"] = {
            col: {k: float(v) for k, v in stats.items()}
            for col, stats in desc.items()
        }
        narrative_parts.append(f"{len(numeric_cols)} numeric columns analysed.")

    # ---- Categorical summary ----
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_summary: dict[str, Any] = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        cat_summary[col] = {
            "unique": int(vc.shape[0]),
            "top": str(vc.index[0]) if len(vc) > 0 else None,
            "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
        }
    summary["categorical_summary"] = cat_summary
    if cat_cols:
        narrative_parts.append(f"{len(cat_cols)} categorical columns analysed.")

    # ---- Encoding issues ----
    encoding_issues = 0
    for col in df.select_dtypes(include=["object"]).columns:
        encoding_issues += int(
            df[col].astype(str).str.contains("\ufffd", regex=False).sum()
        )
    summary["encoding_issues"] = encoding_issues
    if encoding_issues:
        warnings.append(f"{encoding_issues} encoding issue(s) detected (replacement characters).")

    # ---- Empty strings ----
    empty_count = 0
    for col in df.select_dtypes(include=["object"]).columns:
        empty_count += int((df[col].astype(str).str.strip() == "").sum())
    summary["empty_strings"] = empty_count
    if empty_count:
        warnings.append(f"{empty_count} empty string(s) found across text columns.")

    # ---- Quality score ----
    summary["quality_score"] = inp.profile.quality_score

    # ---- Visualizations (matplotlib, optional) ----
    try:
        figures.extend(_generate_plots(df, numeric_cols, cat_cols, output_dir))
    except ImportError:
        logger.debug("matplotlib not available — skipping plots")
    except Exception as exc:
        warnings.append(f"Plot generation failed: {exc}")

    return ModuleResult(
        module_name="eda",
        status="success",
        summary=summary,
        narrative=" ".join(narrative_parts),
        figures=figures,
        warnings=warnings,
    )


def _generate_plots(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
    output_dir: Path,
) -> list[Path]:
    """Generate basic EDA plots. Returns list of saved figure paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[Path] = []

    # Missing-value heatmap (if any missing)
    if df.isna().any().any():
        fig, ax = plt.subplots(figsize=(min(20, len(df.columns) * 0.4 + 2), 4))
        cols_with_na = [c for c in df.columns if df[c].isna().any()]
        ax.barh(cols_with_na, [df[c].isna().sum() for c in cols_with_na])
        ax.set_xlabel("Missing count")
        ax.set_title("Missing Values by Column")
        plt.tight_layout()
        path = output_dir / "missing_values.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        figures.append(path)

    # Numeric histograms (up to 12 columns)
    plot_numeric = numeric_cols[:12]
    if plot_numeric:
        ncols = min(4, len(plot_numeric))
        nrows = (len(plot_numeric) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
        axes_flat = [axes] if nrows == 1 and ncols == 1 else np.array(axes).flatten()
        for i, col in enumerate(plot_numeric):
            axes_flat[i].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
            axes_flat[i].set_title(col, fontsize=9)
        for j in range(len(plot_numeric), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Numeric Distributions", fontsize=11)
        plt.tight_layout()
        path = output_dir / "numeric_distributions.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        figures.append(path)

    # Top-10 bar charts for categoricals (up to 6 columns)
    plot_cat = [c for c in cat_cols if df[c].nunique() <= 30][:6]
    if plot_cat:
        ncols = min(3, len(plot_cat))
        nrows = (len(plot_cat) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes_flat = [axes] if nrows == 1 and ncols == 1 else np.array(axes).flatten()
        for i, col in enumerate(plot_cat):
            vc = df[col].value_counts().head(10)
            axes_flat[i].barh(vc.index.astype(str), vc.values)
            axes_flat[i].set_title(col, fontsize=9)
        for j in range(len(plot_cat), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Categorical Distributions", fontsize=11)
        plt.tight_layout()
        path = output_dir / "categorical_distributions.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        figures.append(path)

    return figures
