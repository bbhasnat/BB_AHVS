"""Self-evolution system for the AHVS pipeline.

Records lessons from each AHVS cycle (failures, slow stages, quality issues)
and injects them into future cycles as prompt overlays.  Inspired by Sibyl's
time-weighted evolution mechanism.

Architecture
------------
* ``LessonCategory`` — 6 issue categories for classification.
* ``LessonEntry`` — single lesson (stage, category, severity, description, ts).
* ``EvolutionStore`` — JSONL-backed persistent store with append + query.
* ``extract_lessons()`` — auto-extract lessons from ``StageResult`` lists.
* ``build_overlay()`` — generate per-stage prompt overlay text.

Usage
-----
::

    from ahvs.evolution import EvolutionStore, extract_lessons

    store = EvolutionStore(Path("evolution"))
    lessons = extract_lessons(results)
    store.append_many(lessons)
    overlay = store.build_overlay("hypothesis_gen", max_lessons=5)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LessonCategory(str, Enum):
    """Issue classification for extracted lessons."""

    SYSTEM = "system"          # Environment / network / timeout
    EXPERIMENT = "experiment"  # Code validation, sandbox timeout
    WRITING = "writing"        # Paper quality issues
    ANALYSIS = "analysis"      # Weak analysis, missing comparison
    LITERATURE = "literature"  # Search / verification failures
    PIPELINE = "pipeline"      # Stage orchestration issues


@dataclass
class LessonEntry:
    """A single lesson extracted from a pipeline run."""

    stage_name: str
    stage_num: int
    category: str
    severity: str  # "info", "warning", "error"
    description: str
    timestamp: str  # ISO 8601
    run_id: str = ""
    cycle_status: str = "complete"  # "complete", "failed", "partial"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LessonEntry:
        return cls(
            stage_name=str(data.get("stage_name", "")),
            stage_num=int(data.get("stage_num", 0)),
            category=str(data.get("category", "pipeline")),
            severity=str(data.get("severity", "info")),
            description=str(data.get("description", "")),
            timestamp=str(data.get("timestamp", "")),
            run_id=str(data.get("run_id", "")),
            cycle_status=str(data.get("cycle_status", "complete")),
        )


# ---------------------------------------------------------------------------
# Lesson classification keywords
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    LessonCategory.SYSTEM: [
        "timeout", "connection", "network", "oom", "memory",
        "permission", "ssh", "socket", "dns",
    ],
    LessonCategory.EXPERIMENT: [
        "sandbox", "validation", "import", "syntax", "subprocess",
        "experiment", "code", "execution",
    ],
    LessonCategory.WRITING: [
        "paper", "draft", "outline", "revision", "review",
        "template", "latex",
    ],
    LessonCategory.ANALYSIS: [
        "analysis", "metric", "statistic", "comparison", "baseline",
    ],
    LessonCategory.LITERATURE: [
        "search", "citation", "verify", "hallucin", "arxiv",
        "semantic_scholar", "literature", "collect",
    ],
}


def _classify_error(stage_name: str, error_text: str) -> str:
    """Classify an error into a LessonCategory based on keywords."""
    combined = f"{stage_name} {error_text}".lower()
    best_category = LessonCategory.PIPELINE
    best_score = 0
    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


# ---------------------------------------------------------------------------
# Lesson extraction from pipeline results
# ---------------------------------------------------------------------------

# Stage name mapping (import-free to avoid circular deps)
_STAGE_NAMES: dict[int, str] = {
    1: "ahvs_setup", 2: "ahvs_context_load", 3: "ahvs_hypothesis_gen",
    4: "ahvs_human_selection", 5: "ahvs_validation_plan", 6: "ahvs_execution",
    7: "ahvs_report_memory", 8: "ahvs_cycle_verify",
}


def extract_lessons(
    results: list[object],
    run_id: str = "",
    run_dir: Path | None = None,
) -> list[LessonEntry]:
    """Extract lessons from a list of StageResult objects.

    Detects:
    - Failed stages → error lesson
    - Blocked stages → pipeline lesson
    - Decision pivots/refines → pipeline lesson (with rationale if available)
    - Runtime warnings from experiment stderr → code_bug lesson
    - Metric anomalies (NaN, identical convergence) → metric_anomaly lesson
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lessons: list[LessonEntry] = []

    for result in results:
        stage_num = int(getattr(result, "stage", 0))
        stage_name = _STAGE_NAMES.get(stage_num, f"stage_{stage_num}")
        status = str(getattr(result, "status", ""))
        error = getattr(result, "error", None)
        decision = str(getattr(result, "decision", "proceed"))

        # Failed stages
        if "failed" in status.lower() and error:
            category = _classify_error(stage_name, str(error))
            lessons.append(LessonEntry(
                stage_name=stage_name,
                stage_num=stage_num,
                category=category,
                severity="error",
                description=f"Stage {stage_name} failed: {str(error)[:300]}",
                timestamp=now,
                run_id=run_id,
            ))

        # Blocked stages
        if "blocked" in status.lower():
            lessons.append(LessonEntry(
                stage_name=stage_name,
                stage_num=stage_num,
                category=LessonCategory.PIPELINE,
                severity="warning",
                description=f"Stage {stage_name} blocked awaiting approval",
                timestamp=now,
                run_id=run_id,
            ))

        # PIVOT / REFINE decisions — extract rationale if available
        if decision in ("pivot", "refine"):
            rationale = _extract_decision_rationale(run_dir) if run_dir else ""
            desc = f"Research decision was {decision.upper()}"
            if rationale:
                desc += f": {rationale[:200]}"
            else:
                desc += " — prior hypotheses/experiments were insufficient"
            lessons.append(LessonEntry(
                stage_name=stage_name,
                stage_num=stage_num,
                category=LessonCategory.PIPELINE,
                severity="warning",
                description=desc,
                timestamp=now,
                run_id=run_id,
            ))

    # --- Extract lessons from experiment artifacts ---
    if run_dir is not None:
        lessons.extend(_extract_runtime_lessons(run_dir, now, run_id))

    return lessons


def _extract_decision_rationale(run_dir: Path) -> str:
    """Extract rationale from the most recent decision_structured.json.

    Supports multiple field formats:
    - ``rationale`` or ``reason`` key (direct)
    - ``raw_text_excerpt`` containing ``## Justification`` section (LLM output)
    """
    for stage_dir in sorted(run_dir.glob("stage-15*"), reverse=True):
        decision_file = stage_dir / "decision_structured.json"
        if decision_file.exists():
            try:
                data = json.loads(decision_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                # Try direct rationale/reason keys first
                direct = data.get("rationale", "") or data.get("reason", "")
                if direct:
                    return str(direct)
                # Parse raw_text_excerpt for Justification section
                raw = data.get("raw_text_excerpt", "")
                if raw:
                    return _parse_justification_from_excerpt(str(raw))
            except (json.JSONDecodeError, OSError):
                pass
    return ""


def _parse_justification_from_excerpt(text: str) -> str:
    """Extract the Justification/Rationale section from LLM decision text."""
    import re

    # Match ## Justification, ## Rationale, or similar headings
    pattern = re.compile(
        r"##\s*(?:Justification|Rationale|Reason)\s*\n(.*?)(?=\n##|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()[:300]
    # Fallback: skip the first line (## Decision / **REFINE**) and return the rest
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Skip heading lines starting with ## or **
    content_lines = [
        l for l in lines
        if not l.startswith("##") and not (l.startswith("**") and l.endswith("**"))
    ]
    if content_lines:
        return " ".join(content_lines)[:300]
    return ""


def _extract_runtime_lessons(
    run_dir: Path, timestamp: str, run_id: str
) -> list[LessonEntry]:
    """Extract fine-grained lessons from experiment run artifacts."""
    import math

    lessons: list[LessonEntry] = []

    # Check sandbox run results for stderr warnings and NaN
    for runs_dir in run_dir.glob("stage-*/runs"):
        for run_file in runs_dir.glob("*.json"):
            if run_file.name == "results.json":
                continue
            try:
                payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue

            # Check stderr for runtime warnings
            stderr = payload.get("stderr", "")
            if stderr and any(
                kw in stderr for kw in ("Warning", "Error", "divide", "overflow", "invalid value")
            ):
                lessons.append(LessonEntry(
                    stage_name="experiment_run",
                    stage_num=12,
                    category=LessonCategory.EXPERIMENT,
                    severity="warning",
                    description=f"Runtime warning in experiment: {stderr[:200]}",
                    timestamp=timestamp,
                    run_id=run_id,
                ))

            # Check metrics for NaN/Inf
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                for key, val in metrics.items():
                    try:
                        fval = float(val)
                        if math.isnan(fval) or math.isinf(fval):
                            lessons.append(LessonEntry(
                                stage_name="experiment_run",
                                stage_num=12,
                                category=LessonCategory.EXPERIMENT,
                                severity="error",
                                description=f"Metric '{key}' was {val} — code bug (division by zero or overflow)",
                                timestamp=timestamp,
                                run_id=run_id,
                            ))
                    except (TypeError, ValueError):
                        pass

    return lessons


# ---------------------------------------------------------------------------
# Time-decay weighting
# ---------------------------------------------------------------------------

HALF_LIFE_DAYS: float = 30.0
MAX_AGE_DAYS: float = 90.0


def _time_weight(timestamp_iso: str) -> float:
    """Compute exponential decay weight for a lesson based on age.

    Uses 30-day half-life: weight = exp(-age_days * ln(2) / 30).
    Returns 0.0 for lessons older than 90 days.
    """
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        age_days = age.total_seconds() / 86400.0
        if age_days > MAX_AGE_DAYS:
            return 0.0
        return math.exp(-age_days * math.log(2) / HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Evolution store
# ---------------------------------------------------------------------------


class EvolutionStore:
    """JSONL-backed store for pipeline lessons."""

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lessons_path = self._dir / "lessons.jsonl"

    @property
    def lessons_path(self) -> Path:
        return self._lessons_path

    def append(self, lesson: LessonEntry) -> None:
        """Append a single lesson to the store."""
        with self._lessons_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

    def append_many(self, lessons: list[LessonEntry]) -> None:
        """Append multiple lessons atomically."""
        if not lessons:
            return
        with self._lessons_path.open("a", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Appended %d lessons to evolution store", len(lessons))

    def load_all(self) -> list[LessonEntry]:
        """Load all lessons from disk."""
        if not self._lessons_path.exists():
            return []
        lessons: list[LessonEntry] = []
        for line in self._lessons_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                lessons.append(LessonEntry.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                continue
        return lessons

    def query_for_stage(
        self,
        stage_name: str,
        *,
        max_lessons: int = 5,
        max_cycles: int = 0,
    ) -> list[LessonEntry]:
        """Return the most relevant lessons for a stage, weighted by recency.

        Includes lessons that directly match the stage, plus high-severity
        lessons from related stages.

        Args:
            stage_name: Stage name to boost (2x weight for direct matches).
            max_lessons: Maximum number of lessons to return.
            max_cycles: Only include lessons from the K most recent
                **complete** cycles.  0 means no cycle cap (time-decay
                still applies).
        """
        all_lessons = self.load_all()

        # Filter out lessons from failed cycles (infrastructure crashes with
        # no hypothesis data).  Keep "complete" (Stage 7 archive), "partial"
        # (eager writes from Stage 6 — cycle may not have finished), and
        # legacy entries without the field (default to "complete").
        all_lessons = [
            l for l in all_lessons
            if l.cycle_status != "failed"
        ]

        # Cap to K most recent cycles by run_id (timestamp-based IDs sort chronologically)
        if max_cycles > 0:
            cycle_ids = sorted(
                {l.run_id for l in all_lessons if l.run_id},
                reverse=True,
            )[:max_cycles]
            cycle_id_set = set(cycle_ids)
            all_lessons = [
                l for l in all_lessons
                if l.run_id in cycle_id_set or not l.run_id
            ]

        scored: list[tuple[float, LessonEntry]] = []
        for lesson in all_lessons:
            weight = _time_weight(lesson.timestamp)
            if weight <= 0.0:
                continue
            # Boost direct stage matches
            if lesson.stage_name == stage_name:
                weight *= 2.0
            # Boost errors over warnings/info
            if lesson.severity == "error":
                weight *= 1.5
            scored.append((weight, lesson))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_lessons]]

    def build_overlay(
        self,
        stage_name: str,
        *,
        max_lessons: int = 5,
        skills_dir: str = "",
    ) -> str:
        """Generate a prompt overlay string for a given stage.

        Combines two sources:
        1. Current-run lessons from ``lessons.jsonl`` (intra-run learning).
        2. Cross-run ``arc-*`` skills from *skills_dir* (inter-run learning).

        Returns empty string if no relevant lessons or skills exist.
        """
        parts: list[str] = []

        # --- Section 1: intra-run lessons ---
        lessons = self.query_for_stage(stage_name, max_lessons=max_lessons)
        if lessons:
            parts.append("## Lessons from Prior Runs")
            for i, lesson in enumerate(lessons, 1):
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                    lesson.severity, "•"
                )
                parts.append(
                    f"{i}. {severity_icon} [{lesson.category}] {lesson.description}"
                )
            parts.append(
                "\nUse these lessons to avoid repeating past mistakes."
            )

        # --- Section 2: cross-run arc-* skills ---
        if skills_dir:
            from pathlib import Path as _Path

            sd = _Path(skills_dir).expanduser()
            if sd.is_dir():
                arc_skills: list[str] = []
                for skill_dir in sorted(sd.iterdir()):
                    if skill_dir.is_dir() and skill_dir.name.startswith("arc-"):
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.is_file():
                            try:
                                text = skill_file.read_text(encoding="utf-8").strip()
                                if text:
                                    arc_skills.append(text)
                            except OSError:
                                continue
                if arc_skills:
                    parts.append("\n## Learned Skills from Prior Runs")
                    for skill_text in arc_skills[:5]:
                        parts.append(skill_text)
                    parts.append(
                        "\nApply these skills proactively to improve quality."
                    )

        return "\n".join(parts)

    def compact(self) -> int:
        """Remove expired (>90 days) and duplicate lessons.

        Deduplicates by ``(run_id, description[:100])``.  When duplicates
        exist, prefers ``cycle_status="complete"`` over ``"partial"`` so
        that Stage 7 archive entries supersede eager Stage 6 writes.
        Also drops lessons whose time-decay weight has reached zero
        (older than ``MAX_AGE_DAYS``).

        Returns the number of entries removed.
        """
        all_lessons = self.load_all()
        before = len(all_lessons)
        if before == 0:
            return 0

        # Drop expired entries
        kept: list[LessonEntry] = [
            l for l in all_lessons if _time_weight(l.timestamp) > 0.0
        ]

        # Deduplicate by (run_id, description prefix).
        # Prefer "complete" over "partial" when both exist for the same key.
        best: dict[tuple[str, str], LessonEntry] = {}
        _STATUS_RANK = {"complete": 2, "partial": 1}
        for lesson in kept:
            key = (lesson.run_id, lesson.description[:100])
            existing = best.get(key)
            if existing is None:
                best[key] = lesson
            elif _STATUS_RANK.get(lesson.cycle_status, 0) > _STATUS_RANK.get(
                existing.cycle_status, 0
            ):
                best[key] = lesson
        deduped = list(best.values())

        removed = before - len(deduped)
        if removed > 0:
            self._lessons_path.write_text(
                "".join(
                    json.dumps(l.to_dict(), ensure_ascii=False) + "\n"
                    for l in deduped
                ),
                encoding="utf-8",
            )
            logger.info(
                "Compacted evolution store: %d → %d entries (%d removed)",
                before, len(deduped), removed,
            )
        return removed

    @staticmethod
    def cleanup_cycles(
        cycles_dir: Path,
        *,
        keep_complete: int = 3,
        exclude: Path | None = None,
    ) -> list[str]:
        """Clean up cycle directories that are no longer needed.

        Removes, in order:
        1. **Stage-1 failures** — crashed at setup, no artifacts.
        2. **Orphan dirs** — no checkpoint file (manual test dirs, etc.).
        3. **Partial cycles** — reached execution but never completed
           Stage 8.  With eager lesson writes, their hypothesis results
           are already in ``lessons.jsonl``.
        4. **Old complete cycles** beyond the *keep_complete* retention
           window.  The most recent *keep_complete* complete cycles are
           preserved for auditing.

        Args:
            cycles_dir: Path to ``<repo>/.ahvs/cycles/``.
            keep_complete: Number of most-recent complete cycles to retain
                (default 3).  Set to 0 to keep all complete cycles.
            exclude: If given, this cycle directory is never removed
                (used to protect the currently-running cycle).

        Returns list of removed directory names.
        """
        import shutil

        removed: list[str] = []
        if not cycles_dir.is_dir():
            return removed

        # Classify every cycle dir
        failed: list[Path] = []
        orphan: list[Path] = []
        partial: list[Path] = []
        complete: list[Path] = []

        exclude_resolved = exclude.resolve() if exclude else None
        for cycle_dir in sorted(cycles_dir.iterdir()):
            if not cycle_dir.is_dir():
                continue
            if exclude_resolved and cycle_dir.resolve() == exclude_resolved:
                continue
            checkpoint = cycle_dir / "ahvs_checkpoint.json"
            if not checkpoint.exists():
                orphan.append(cycle_dir)
                continue
            try:
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                orphan.append(cycle_dir)
                continue

            status = data.get("status", "")
            stage_num = data.get("stage_num", 0)

            if status == "failed" and stage_num <= 1:
                failed.append(cycle_dir)
            elif status == "done" and stage_num >= 8:
                complete.append(cycle_dir)
            else:
                partial.append(cycle_dir)

        # 1–3: Remove all failed, orphan, and partial dirs
        for d in failed + orphan + partial:
            try:
                shutil.rmtree(d)
                removed.append(d.name)
            except OSError:
                continue

        # 4: Trim old complete cycles beyond the retention window.
        #    complete list is already sorted by name (timestamp-based),
        #    so the last `keep_complete` entries are the most recent.
        if keep_complete > 0 and len(complete) > keep_complete:
            to_remove = complete[:-keep_complete]
            for d in to_remove:
                try:
                    shutil.rmtree(d)
                    removed.append(d.name)
                except OSError:
                    continue

        if removed:
            logger.info(
                "Cleaned up %d cycle dir(s): %s",
                len(removed), ", ".join(removed),
            )
        return removed

    def count(self) -> int:
        """Return total number of stored lessons."""
        return len(self.load_all())
