"""AHVS context loader — combines EvolutionStore lessons + baseline metric.

Produces context_bundle.json, the input to hypothesis generation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ahvs.evolution import EvolutionStore

BASELINE_REQUIRED_FIELDS = ("primary_metric", "recorded_at", "eval_command")


def load_baseline_metric(baseline_path: Path) -> dict:
    """Load and validate .ahvs/baseline_metric.json."""
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline metric file not found: {baseline_path}\n"
            "Create it by running your eval once and recording the result.\n"
            "Required format:\n"
            '  {"primary_metric": "answer_relevance", "answer_relevance": 0.74,\n'
            '   "recorded_at": "2026-03-17T10:00:00Z", "commit": "abc1234",\n'
            '   "eval_command": "promptfoo eval --config .ahvs/eval/baseline.yaml"}'
        )
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline_metric.json is not valid JSON: {exc}") from exc

    for f in BASELINE_REQUIRED_FIELDS:
        if f not in data:
            raise ValueError(
                f"baseline_metric.json missing required field: '{f}'"
            )
    metric_key = data["primary_metric"]
    if metric_key not in data:
        raise ValueError(
            f"baseline_metric.json must contain the metric value under key '{metric_key}'"
        )
    return data


def _extract_rejected_approaches(
    lessons: list["LessonEntry"],
) -> list[str]:
    """Pull rejected/failed entries from structured LessonEntry objects.

    A lesson counts as "rejected" if its severity is warning/error
    (these are written for failed or non-improving hypotheses at Stage 7).
    """
    from ahvs.evolution import LessonEntry  # noqa: F811

    rejected = []
    for lesson in lessons:
        if lesson.severity in ("warning", "error"):
            rejected.append(lesson.description)
    return rejected[:10]


def _extract_prior_lessons(
    lessons: list["LessonEntry"],
) -> list[str]:
    """Extract positive/informational lessons from structured LessonEntry objects.

    Lessons with severity "info" are successful outcomes or neutral observations.
    """
    from ahvs.evolution import LessonEntry  # noqa: F811

    result = []
    for lesson in lessons:
        if lesson.severity == "info" and len(lesson.description) > 20:
            result.append(lesson.description)
    return result[:15]


def _infer_domain_tags(repo_path: Path) -> list[str]:
    """Scan repo dependency files to infer its domain."""
    tags: list[str] = []
    candidates = [
        repo_path / "requirements.txt",
        repo_path / "pyproject.toml",
        repo_path / "setup.py",
        repo_path / "package.json",
    ]
    content = ""
    for f in candidates:
        if f.exists():
            try:
                content += f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass

    content_lower = content.lower()

    llm_indicators = [
        "openai", "anthropic", "langchain", "llama", "transformers",
        "promptfoo", "dspy", "litellm", "huggingface", "ollama",
    ]
    rag_indicators = [
        "chromadb", "pinecone", "weaviate", "qdrant", "faiss",
        "retrieval", "embedding", "vector", "langchain",
    ]
    ml_indicators = [
        "torch", "tensorflow", "sklearn", "xgboost", "optuna", "mlflow",
    ]

    if any(ind in content_lower for ind in llm_indicators):
        tags.append("llm")
    if any(ind in content_lower for ind in rag_indicators):
        tags.append("rag")
    if any(ind in content_lower for ind in ml_indicators):
        tags.append("ml")

    # Detect prompt files
    try:
        prompt_files = (
            list(repo_path.rglob("*system_prompt*"))
            + list(repo_path.rglob("*.prompts.*"))
        )
        if prompt_files:
            tags.append("prompt-driven")
    except OSError:
        pass

    return tags or ["general"]


def load_context_bundle(
    repo_path: Path,
    question: str,
    evolution_dir: Path,
    baseline_path: Path,
) -> dict:
    """Build the context_bundle.json payload.

    Combines:
    - baseline metric (from .ahvs/baseline_metric.json)
    - cross-cycle memory (from EvolutionStore)
    - domain tags (inferred from repo dependencies)
    """
    baseline = load_baseline_metric(baseline_path)

    from ahvs.evolution import LessonEntry

    lessons: list[LessonEntry] = []
    if evolution_dir.exists():
        try:
            store = EvolutionStore(evolution_dir)
            # Query with "ahvs_execution" — this is the stage_name used when
            # writing lessons at Stage 7 (report/memory).  Using the same name
            # ensures the EvolutionStore 2x boost applies to direct matches,
            # giving cross-cycle lessons proper retrieval priority.
            lessons = store.query_for_stage("ahvs_execution", max_lessons=12)
        except Exception:  # noqa: BLE001
            pass  # Non-fatal — first cycle has no history

    metric_key = baseline["primary_metric"]

    # Forward enriched onboarding fields from baseline_metric.json.
    # These fields are documented in README_AHVS.md and improve hypothesis
    # quality by giving the LLM richer operator intent context.
    enriched_fields = {}
    for field in (
        "optimization_goal",
        "regression_floor",
        "constraints",
        "system_levers",
        "prior_experiments",
        "notes",
    ):
        if field in baseline and baseline[field] is not None:
            enriched_fields[field] = baseline[field]

    return {
        "question": question,
        "baseline": {
            "primary_metric": metric_key,
            "value": float(baseline[metric_key]),
            "eval_command": baseline.get("eval_command", ""),
            "recorded_at": baseline.get("recorded_at", ""),
            "commit": baseline.get("commit", ""),
        },
        "enriched_context": enriched_fields,
        "prior_lessons": _extract_prior_lessons(lessons),
        "rejected_approaches": _extract_rejected_approaches(lessons),
        "domain_tags": _infer_domain_tags(repo_path),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
