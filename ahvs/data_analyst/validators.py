"""Consolidated validation framework.

Adapted from KD's 4-validator system (SampleSizeValidator, ClassBalanceValidator,
DataQualityValidator, TaskAlignmentValidator). Generalized to work with any ML task.
"""

from __future__ import annotations

import logging
from typing import Any

from ahvs.data_analyst.models import DataProfile, ValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (configurable via overrides dict)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "min_total_samples_error": 50,
    "min_total_samples_warn": 200,
    "min_per_class_error": 10,
    "min_per_class_warn": 50,
    "imbalance_ratio_error": 10.0,
    "imbalance_ratio_warn": 3.0,
    "max_null_pct_warn": 20.0,
    "max_null_pct_error": 50.0,
    "min_quality_score_warn": 60.0,
    "min_quality_score_error": 30.0,
}


def validate(
    profile: DataProfile,
    *,
    thresholds: dict[str, Any] | None = None,
) -> list[ValidationResult]:
    """Run all validators and return a list of results.

    Args:
        profile: DataProfile from Phase 1.
        thresholds: Optional overrides for default thresholds.

    Returns:
        List of ValidationResult objects.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    results: list[ValidationResult] = []
    results.extend(_check_sample_size(profile, t))
    results.extend(_check_class_balance(profile, t))
    results.extend(_check_data_quality(profile, t))
    results.extend(_check_column_alignment(profile))
    return results


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------


def _check_sample_size(
    profile: DataProfile, t: dict[str, Any]
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    n = profile.total_rows

    if n < t["min_total_samples_error"]:
        results.append(
            ValidationResult(
                id="sample_size_total",
                passed=False,
                severity="error",
                message=f"Only {n} total samples (minimum {t['min_total_samples_error']}).",
            )
        )
    elif n < t["min_total_samples_warn"]:
        results.append(
            ValidationResult(
                id="sample_size_total",
                passed=True,
                severity="warning",
                message=f"{n} total samples — consider collecting more (recommend {t['min_total_samples_warn']}+).",
            )
        )
    else:
        results.append(
            ValidationResult(
                id="sample_size_total",
                passed=True,
                severity="info",
                message=f"{n} total samples.",
            )
        )

    # Per-class minimums
    if profile.class_distribution:
        for cls, count in profile.class_distribution.items():
            if count < t["min_per_class_error"]:
                results.append(
                    ValidationResult(
                        id=f"sample_size_class_{cls}",
                        passed=False,
                        severity="error",
                        message=f"Class '{cls}' has only {count} samples (minimum {t['min_per_class_error']}).",
                    )
                )
            elif count < t["min_per_class_warn"]:
                results.append(
                    ValidationResult(
                        id=f"sample_size_class_{cls}",
                        passed=True,
                        severity="warning",
                        message=f"Class '{cls}' has {count} samples (recommend {t['min_per_class_warn']}+).",
                    )
                )

    return results


def _check_class_balance(
    profile: DataProfile, t: dict[str, Any]
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    if not profile.class_distribution:
        return results

    counts = list(profile.class_distribution.values())
    if not counts:
        return results

    max_c, min_c = max(counts), min(counts)
    if min_c == 0:
        results.append(
            ValidationResult(
                id="class_balance",
                passed=False,
                severity="error",
                message="At least one class has zero samples.",
            )
        )
        return results

    ratio = max_c / min_c
    if ratio > t["imbalance_ratio_error"]:
        results.append(
            ValidationResult(
                id="class_balance",
                passed=False,
                severity="error",
                message=f"Severe class imbalance (ratio {ratio:.1f}:1, threshold {t['imbalance_ratio_error']}:1).",
            )
        )
    elif ratio > t["imbalance_ratio_warn"]:
        results.append(
            ValidationResult(
                id="class_balance",
                passed=True,
                severity="warning",
                message=f"Moderate class imbalance (ratio {ratio:.1f}:1).",
            )
        )
    else:
        results.append(
            ValidationResult(
                id="class_balance",
                passed=True,
                severity="info",
                message=f"Class balance OK (ratio {ratio:.1f}:1).",
            )
        )

    return results


def _check_data_quality(
    profile: DataProfile, t: dict[str, Any]
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    # Quality score
    qs = profile.quality_score
    if qs < t["min_quality_score_error"]:
        results.append(
            ValidationResult(
                id="quality_score",
                passed=False,
                severity="error",
                message=f"Data quality score is {qs:.0f}/100 (minimum {t['min_quality_score_error']}).",
            )
        )
    elif qs < t["min_quality_score_warn"]:
        results.append(
            ValidationResult(
                id="quality_score",
                passed=True,
                severity="warning",
                message=f"Data quality score is {qs:.0f}/100.",
            )
        )

    # High-null columns
    for ci in profile.columns:
        if ci.null_pct > t["max_null_pct_error"]:
            results.append(
                ValidationResult(
                    id=f"null_pct_{ci.name}",
                    passed=False,
                    severity="error",
                    message=f"Column '{ci.name}' has {ci.null_pct:.1f}% null values.",
                )
            )
        elif ci.null_pct > t["max_null_pct_warn"]:
            results.append(
                ValidationResult(
                    id=f"null_pct_{ci.name}",
                    passed=True,
                    severity="warning",
                    message=f"Column '{ci.name}' has {ci.null_pct:.1f}% null values.",
                )
            )

    return results


def _check_column_alignment(profile: DataProfile) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    if not profile.input_columns:
        results.append(
            ValidationResult(
                id="column_alignment_input",
                passed=False,
                severity="error",
                message="No input columns detected or specified.",
            )
        )

    if not profile.label_column:
        results.append(
            ValidationResult(
                id="column_alignment_label",
                passed=True,
                severity="warning",
                message="No label column detected. Unsupervised analysis only.",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def recommend(validations: list[ValidationResult]) -> str:
    """Return a recommendation status string based on validation results."""
    errors = [v for v in validations if v.severity == "error" and not v.passed]
    warnings = [v for v in validations if v.severity == "warning"]

    if errors:
        return "NOT_RECOMMENDED"
    if len(warnings) >= 3:
        return "PROCEED_WITH_CAUTION"
    if warnings:
        return "ACCEPTABLE"
    return "RECOMMENDED"
