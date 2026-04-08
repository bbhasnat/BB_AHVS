"""Split module — train/val/test split with stratification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)


def run(inp: ModuleInput) -> ModuleResult:
    """Split dataset into train / val / test partitions."""
    df = inp.df
    params = inp.params
    output_dir = inp.output_dir / "split"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = params.get("train", 0.8)
    val_ratio = params.get("val", 0.1)
    test_ratio = params.get("test", 0.1)
    seed = params.get("seed", 42)
    label_col = inp.label_col

    warnings: list[str] = []
    artifacts: list[Path] = []

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        return ModuleResult.make_error(
            "split", "Split ratios sum to zero. Provide positive ratios."
        )
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Check if stratification is feasible
    can_stratify = False
    if label_col and label_col in df.columns:
        class_counts = df[label_col].value_counts()
        # Need at least 2 samples per class for stratified split
        min_class_count = class_counts.min() if len(class_counts) > 0 else 0
        holdout_ratio = val_ratio + test_ratio
        min_holdout = max(2, int(len(df) * holdout_ratio))
        if min_class_count >= 2 and len(df) >= 4:
            can_stratify = True
        else:
            warnings.append(
                f"Cannot stratify: smallest class has {min_class_count} samples. "
                "Using unstratified split."
            )

    try:
        from sklearn.model_selection import train_test_split

        stratify_col = df[label_col] if can_stratify else None

        # First split: train vs (val + test)
        holdout_ratio = val_ratio + test_ratio
        try:
            train_df, holdout_df = train_test_split(
                df,
                test_size=holdout_ratio,
                random_state=seed,
                stratify=stratify_col,
            )
        except ValueError as exc:
            # Fall back to unstratified if stratification fails
            if can_stratify:
                warnings.append(f"Stratified split failed ({exc}). Using unstratified.")
                can_stratify = False
            train_df, holdout_df = train_test_split(
                df,
                test_size=holdout_ratio,
                random_state=seed,
            )

        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0 and len(holdout_df) >= 2:
            val_frac = val_ratio / holdout_ratio
            stratify_holdout = (
                holdout_df[label_col]
                if can_stratify and label_col in holdout_df.columns
                else None
            )
            try:
                val_df, test_df = train_test_split(
                    holdout_df,
                    test_size=1 - val_frac,
                    random_state=seed,
                    stratify=stratify_holdout,
                )
            except ValueError:
                try:
                    val_df, test_df = train_test_split(
                        holdout_df,
                        test_size=1 - val_frac,
                        random_state=seed,
                    )
                except ValueError:
                    # Holdout too small to split — assign all to val
                    warnings.append("Holdout set too small to split into val+test.")
                    val_df = holdout_df
                    test_df = pd.DataFrame(columns=df.columns)
        elif val_ratio > 0 and test_ratio > 0:
            # holdout_df too small to split (< 2 rows)
            warnings.append("Holdout set too small to split into val+test.")
            val_df = holdout_df
            test_df = pd.DataFrame(columns=df.columns)
        elif val_ratio > 0:
            val_df = holdout_df
            test_df = pd.DataFrame(columns=df.columns)
        else:
            test_df = holdout_df
            val_df = pd.DataFrame(columns=df.columns)

    except ImportError:
        warnings.append("scikit-learn not installed — using manual shuffle split.")
        shuffled = df.sample(frac=1, random_state=seed)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = shuffled.iloc[:n_train]
        val_df = shuffled.iloc[n_train : n_train + n_val]
        test_df = shuffled.iloc[n_train + n_val :]

    # Save splits
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) > 0:
            path = output_dir / f"{name}.parquet"
            split_df.to_parquet(path, index=False)
            artifacts.append(path)

    summary: dict[str, Any] = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "ratios": {
            "train": round(train_ratio, 2),
            "val": round(val_ratio, 2),
            "test": round(test_ratio, 2),
        },
        "stratified": can_stratify,
    }

    narrative = (
        f"Split {len(df)} rows → train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)} "
        f"({'stratified' if can_stratify else 'random'})."
    )

    return ModuleResult(
        module_name="split",
        status="success",
        summary=summary,
        narrative=narrative,
        artifacts=artifacts,
        warnings=warnings,
    )
