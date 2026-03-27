# AHVS Memory Management System

This document describes the full memory management architecture of AHVS: how lessons are written, stored, queried, compacted, verified, summarised, and shared across projects.

---

## Table of Contents

1. [Three-Tier Memory Model](#1-three-tier-memory-model)
2. [LessonEntry Schema](#2-lessonentry-schema)
3. [Lesson Write Paths](#3-lesson-write-paths)
4. [Time-Decay Weighting](#4-time-decay-weighting)
5. [Query and Retrieval](#5-query-and-retrieval)
6. [Compaction and Deduplication](#6-compaction-and-deduplication)
7. [Stage 8 Verification Feedback](#7-stage-8-verification-feedback)
8. [Friction Log and Session Memory Compaction](#8-friction-log-and-session-memory-compaction)
9. [Historical Digest (Context Window Expansion)](#9-historical-digest-context-window-expansion)
10. [Cross-Project Learning](#10-cross-project-learning)
11. [Memory Flow Across the 8-Stage Cycle](#11-memory-flow-across-the-8-stage-cycle)
12. [Configuration Reference](#12-configuration-reference)
13. [Backward Compatibility](#13-backward-compatibility)

---

## 1. Three-Tier Memory Model

All AHVS memory lives in the **target repository** (not the AHVS framework repo or the user's home directory), making it portable across machines and co-located with the code it describes.

| Tier | Location | Format | Purpose |
|------|----------|--------|---------|
| **Evolution Store** | `<repo>/.ahvs/evolution/lessons.jsonl` | JSONL | Machine-queryable lessons fed into hypothesis generation |
| **Friction Logs** | `<repo>/.ahvs/cycles/<id>/friction_log.md` | Markdown | Per-cycle execution errors and operator notes |
| **Session Memory** | `<repo>/.ahvs/memory/` | Markdown | Human-readable cross-session summaries and observations |

A fourth, optional tier exists for cross-project learning:

| Tier | Location | Format | Purpose |
|------|----------|--------|---------|
| **Global Store** | `~/.ahvs/global/evolution/lessons.jsonl` | JSONL | Framework-level lessons shared across all target repos |

---

## 2. LessonEntry Schema

Each lesson is a `LessonEntry` dataclass serialized as a single JSON line in `lessons.jsonl`.

### Core fields (always present)

| Field | Type | Description |
|-------|------|-------------|
| `stage_name` | `str` | Stage where the lesson was recorded (e.g. `"ahvs_execution"`) |
| `stage_num` | `int` | Numeric stage identifier (6 for execution) |
| `category` | `str` | Classification: `system`, `experiment`, `writing`, `analysis`, `literature`, `pipeline` |
| `severity` | `str` | `"info"`, `"warning"`, or `"error"` |
| `description` | `str` | Human-readable lesson text |
| `timestamp` | `str` | ISO 8601 UTC timestamp |
| `run_id` | `str` | Cycle directory name (timestamp-based, sorts chronologically) |
| `cycle_status` | `str` | `"complete"`, `"partial"`, or `"failed"` |

### Structured outcome fields (optional, backward-compatible)

These fields enable quantitative analysis, semantic deduplication, and cross-project learning. All have defaults so that old JSONL entries without them deserialize cleanly.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hypothesis_id` | `str` | `""` | e.g. `"H1"`, `"H2"` |
| `hypothesis_type` | `str` | `""` | e.g. `"prompt_rewrite"`, `"code_change"` |
| `metric_name` | `str` | `""` | e.g. `"answer_relevance"`, `"precision"` |
| `metric_baseline` | `float \| None` | `None` | Metric value before this hypothesis |
| `metric_after` | `float \| None` | `None` | Metric value after this hypothesis |
| `metric_delta` | `float \| None` | `None` | `metric_after - metric_baseline` |
| `files_changed` | `list[str] \| None` | `None` | Relative paths modified by this hypothesis |
| `eval_method` | `str` | `""` | e.g. `"promptfoo"`, `"custom_script"` |
| `verified` | `str` | `""` | `""`, `"kept"`, or `"reverted"` (set by Stage 8) |
| `source_repo` | `str` | `""` | Populated only in the global cross-project store |

### Category classification

Lessons are classified into 6 categories via keyword matching against the stage name and error text:

- **SYSTEM** -- environment, network, timeout, OOM, permissions
- **EXPERIMENT** -- sandbox, validation, import, syntax, code execution
- **WRITING** -- paper, draft, outline, revision
- **ANALYSIS** -- metrics, statistics, comparison, baseline anomalies
- **LITERATURE** -- search, citation, verification, hallucination
- **PIPELINE** -- stage orchestration, decision pivots/refines

---

## 3. Lesson Write Paths

Lessons are written at two points in the cycle, implementing a crash-safe dual-write pattern:

### Stage 6: Eager writes (`_write_eager_lesson`)

Immediately after each hypothesis is measured, a lesson is appended to `lessons.jsonl` with `cycle_status="partial"`. This ensures results survive even if the cycle crashes before Stage 7.

Four cases are handled:

| Condition | Severity | Meaning |
|-----------|----------|---------|
| `result.improved` | `info` | Hypothesis improved the metric |
| `result.error` | `warning` | Execution failed with an error |
| `measurement_status != "measured"` | `warning` | Infrastructure failure (sandbox, extraction) -- hypothesis was not tested, NOT marked as rejected |
| Measured but no improvement | `info` | Rejected approach |

Each eager write populates all structured outcome fields from the `HypothesisResult` (`hypothesis_id`, `hypothesis_type`, `metric_name`, `metric_baseline`, `metric_after`, `metric_delta`, `eval_method`).

### Stage 7: Final archival (`_execute_report_and_memory`)

After all hypotheses have been measured, Stage 7 writes the same lessons again with `cycle_status="complete"` (the default). It also generates:

- `report.md` -- LLM-generated cycle summary
- `friction_log.md` -- auto-generated error/measurement sections + operator notes template

The next compaction pass deduplicates partial and complete entries for the same lesson, keeping the complete version.

---

## 4. Time-Decay Weighting

Lessons are weighted by recency using exponential decay:

```
weight = exp(-age_days * ln(2) / HALF_LIFE_DAYS)
```

| Constant | Value | Effect |
|----------|-------|--------|
| `HALF_LIFE_DAYS` | 30 | A 30-day-old lesson has weight 0.5 |
| `MAX_AGE_DAYS` | 90 | Lessons older than 90 days have weight 0.0 |

Time decay is applied at **query time**, not write time. This means lessons remain in storage until explicitly compacted but are naturally deprioritized as they age.

---

## 5. Query and Retrieval

`EvolutionStore.query_for_stage()` retrieves the most relevant lessons for hypothesis generation:

```python
store.query_for_stage(
    "ahvs_execution",
    max_lessons=12,      # top N by weight
    max_cycles=5,        # from last K recent non-failed cycle IDs (0 = unlimited)
)
```

### Weight boosts

Each lesson's time-decay weight is modified by multipliers:

| Condition | Multiplier | Reason |
|-----------|------------|--------|
| Direct stage name match | 2.0x | Prioritize lessons from the same pipeline stage |
| Severity = `"error"` | 1.5x | Remember failures longer than successes |
| Verified = `"kept"` | 1.5x | Proven improvements are more valuable |
| Verified = `"reverted"` | 0.5x | Deprioritize approaches that were ultimately rejected |

### Filtering

- Lessons with `cycle_status="failed"` are excluded (infrastructure crashes with no hypothesis data)
- Lessons with `cycle_status="partial"` are included (eager writes from Stage 6)
- When `max_cycles > 0`, only lessons from the K most recent non-failed cycle IDs are considered

### Consumption in Stage 2

`load_context_bundle()` calls `query_for_stage("ahvs_execution", max_lessons=12, max_cycles=5)` and splits the results into:

- **`prior_lessons`** -- lessons with `severity="info"` (positive outcomes, up to 15)
- **`rejected_approaches`** -- lessons with `severity="warning"` or `"error"` (failed ideas, up to 10)
- **`historical_digest`** -- aggregated statistics from older cycles beyond the recent window (see [Section 9](#9-historical-digest-context-window-expansion))

These are injected into the Stage 3 (hypothesis generation) prompt.

---

## 6. Compaction and Deduplication

`EvolutionStore.compact()` runs automatically at the start of each new cycle (in `runner.py`). It performs two deduplication passes:

### Phase 1: Exact dedup

Key: `(run_id, description[:100])`

When duplicates exist (e.g. both a partial Stage 6 entry and a complete Stage 7 entry for the same hypothesis), the entry with the higher `cycle_status` rank is kept:

| Status | Rank |
|--------|------|
| `"complete"` | 2 |
| `"partial"` | 1 |

This ensures Stage 7 entries supersede Stage 6 eager writes within the same cycle.

### Phase 2: Semantic dedup (cross-cycle)

After exact dedup, remaining entries are grouped by a **semantic fingerprint**:

**Structured entries** (with `hypothesis_type` and `metric_name` populated):
```
fingerprint = hash(hypothesis_type | metric_name | round(metric_delta, 2))
```

This collapses paraphrased lessons from different cycles that describe the same experiment type with the same outcome. `run_id` is excluded from the fingerprint so that cross-cycle duplicates collapse.

**Legacy entries** (no structured fields):
```
fingerprint = hash(sorted_tokens | severity | run_id)
```

Legacy entries include `run_id` in the fingerprint to be conservative, since text alone can't reliably distinguish different hypotheses from paraphrases.

Within each fingerprint group, `_pick_best_from_group()` keeps the single best entry by:
1. Highest severity (`error` > `warning` > `info`)
2. Most recent timestamp
3. Most complete structured data (has `hypothesis_type`, has `verified`)

### Expiration

Before deduplication, entries older than `MAX_AGE_DAYS` (90 days) are dropped entirely.

### Cycle directory cleanup

`EvolutionStore.cleanup_cycles()` also runs at cycle start. It removes directories in order:

1. **Stage-1 failures** -- crashed at setup, no artifacts
2. **Orphan dirs** -- no `ahvs_checkpoint.json` (manual test dirs, etc.)
3. **Partial cycles** -- any non-complete cycle that is not a Stage-1 failure or orphan (results may already be in `lessons.jsonl` via eager writes)
4. **Old complete cycles** -- beyond the retention window (keeps the 3 most recent complete cycles for auditing)

---

## 7. Stage 8 Verification Feedback

After Stage 8 (`AHVS_CYCLE_VERIFY`) builds `cycle_summary.json` with a keep/revert recommendation, it feeds that decision back into the evolution store via `_update_lesson_verification()`:

- The best hypothesis's lessons are marked `verified="kept"`
- All other hypotheses' lessons are marked `verified="reverted"`
- Lessons from other cycles are untouched

This creates a closed feedback loop: the keep/revert decision at the end of one cycle influences the weight of those lessons when queried by future cycles. Kept lessons get a 1.5x boost; reverted lessons get a 0.5x penalty (see [Section 5](#5-query-and-retrieval)).

---

## 8. Friction Log and Session Memory Compaction

In addition to evolution store compaction, AHVS manages the lifecycle of friction logs and session memory files.

### Friction log summarization

`EvolutionStore.compact_friction_logs()` reads all `friction_log.md` files from retained cycle directories, extracts error patterns and operator notes, deduplicates recurring themes (normalizing hypothesis IDs like `H1` to `H?` for matching), and writes a consolidated `friction_summary.md` to the evolution directory.

This runs automatically at the start of each new cycle.

### Session memory lifecycle

`EvolutionStore.compact_memory_files()` manages files in `<repo>/.ahvs/memory/`:

| Age | Action |
|-----|--------|
| < `memory_stale_days` (default 60) | No change |
| >= `memory_stale_days` | Prepend `> [STALE]` marker (idempotent -- won't double-mark) |
| >= `memory_archive_days` (default 120) | Move to `memory/archive/` |

This prevents unbounded accumulation of session memory while preserving the content for audit purposes.

---

## 9. Historical Digest (Context Window Expansion)

Raw lessons are limited to 12 entries from 5 recent cycles. To surface institutional knowledge from older cycles, AHVS builds a **historical digest** -- a compact statistical summary.

`EvolutionStore.build_historical_digest()` loads all lessons, excludes the most recent K cycles (those are already surfaced as raw `prior_lessons`), and aggregates the remainder by `hypothesis_type`:

```json
{
  "by_hypothesis_type": {
    "prompt_rewrite": {
      "total": 8,
      "improved": 5,
      "avg_delta": 0.025,
      "best_delta": 0.045,
      "kept_count": 3,
      "reverted_count": 2,
      "representative": "H1 (prompt_rewrite) improved answer_relevance by +0.0450..."
    },
    "code_change": {
      "total": 4,
      "improved": 1,
      "avg_delta": -0.005,
      "best_delta": 0.012,
      "representative": "..."
    }
  },
  "total_lessons": 12,
  "total_cycles": 6,
  "oldest_lesson": "2026-01-15T10:30:00Z"
}
```

This is formatted into a compact text block and injected into the hypothesis generation prompt under "Historical Performance Digest". It allows the LLM to reason about long-term patterns (e.g. "prompt_rewrite has a 62% improvement rate while code_change only 25%") without consuming raw lesson context window slots.

For legacy entries without structured fields, the digest gracefully degrades to count-only statistics.

---

## 10. Cross-Project Learning

By default, all memory is scoped to the target repo. The **GlobalEvolutionStore** enables framework-level insights to transfer across repos.

### Global store location

`~/.ahvs/global/evolution/lessons.jsonl` (configurable via `global_evolution_dir`).

### Promotion criteria

At the end of a successful cycle, `GlobalEvolutionStore.promote_lessons()` copies qualifying lessons from the local store to the global store. A lesson qualifies if:

- Its category is `SYSTEM` or `PIPELINE` (framework-level, not repo-specific), **OR**
- Its `verified` field is `"kept"` (proven improvement, transferable pattern)

Promoted lessons are tagged with `source_repo` for provenance and deduplicated by semantic fingerprint to prevent re-promotion of the same insight.

### Query with exclusion

`GlobalEvolutionStore.query_cross_project()` retrieves lessons from the global store while excluding the current repo to avoid self-reinforcement:

```python
global_store.query_cross_project(
    "ahvs_execution",
    max_lessons=3,
    exclude_repo="current_repo_name",
)
```

### Integration into context bundle

During Stage 2, if `enable_cross_project=True` (the default) and the global store exists, up to 3 global lessons are merged into the context bundle. Local lessons are loaded first (up to 12); global lessons only fill any remaining capacity and are deduplicated against local lessons using semantic fingerprints (the same mechanism used by `compact()` and `promote_lessons()`).

### Opting out

Set `enable_cross_project=False` in `AHVSConfig` to disable all global store interaction.

---

## 11. Memory Flow Across the 8-Stage Cycle

```
Stage 1: SETUP
  No memory access

Stage 2: CONTEXT_LOAD
  Read:  lessons.jsonl (filtered, time-weighted, boosted)
  Read:  global lessons (if enabled, up to 3, excluding current repo)
  Write: context_bundle.json (prior_lessons, rejected_approaches, historical_digest)

Stage 3: HYPOTHESIS_GEN
  Read:  context_bundle (lessons + historical digest as prompt context)
  LLM generates hypotheses, guided away from rejected approaches

Stage 4: HUMAN_SELECTION
  No memory access

Stage 5: VALIDATION_PLAN
  No memory access

Stage 6: EXECUTION
  Write: eager lessons with cycle_status="partial" (per hypothesis, immediate)
         All structured fields populated from HypothesisResult

Stage 7: REPORT_MEMORY
  Write: report.md (LLM-generated cycle summary)
  Write: friction_log.md (auto-generated errors + operator notes)
  Write: final lessons with cycle_status="complete"

Stage 8: CYCLE_VERIFY
  Write: cycle_summary.json (keep/revert recommendation)
  Write: verified="kept"/"reverted" back into lessons.jsonl

[Cycle Start Housekeeping]
  cleanup_cycles() -- remove stale/orphan/partial/old dirs
  compact() -- expire old lessons + exact dedup + semantic dedup
  compact_friction_logs() -- summarize retained friction logs
  compact_memory_files() -- stale-mark and archive old memory files

[End of Successful Cycle]
  promote_lessons() -- copy qualifying lessons to global store
```

---

## 12. Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `max_lesson_cycles` | `5` | Load lessons from last K recent non-failed cycle IDs (0 = unlimited) |
| `memory_stale_days` | `60` | Prepend `[STALE]` marker to old memory files |
| `memory_archive_days` | `120` | Move very old memory files to `archive/` |
| `enable_cross_project` | `True` | Enable global cross-project lesson sharing |
| `global_evolution_dir` | `~/.ahvs/global/evolution` | Path to global evolution store |

### Constants (in `evolution.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `HALF_LIFE_DAYS` | `30.0` | Exponential decay half-life for lesson weighting |
| `MAX_AGE_DAYS` | `90.0` | Hard cutoff -- lessons older than this are expired |

---

## 13. Backward Compatibility

All structured outcome fields on `LessonEntry` have defaults. Old `lessons.jsonl` files without these fields deserialize cleanly via `from_dict()` using `.get()` with defaults. No migration is required.

- Old entries without `hypothesis_type` / `metric_name` use the text-based semantic fingerprint (conservative, includes `run_id`)
- Old entries without `verified` receive no query boost or penalty (weight multiplier = 1.0)
- Old entries without `source_repo` are treated as local-only
- The `historical_digest` key is new in `context_bundle.json`; prompt templates that lack the `{historical_digest}` placeholder leave it unresolved (the `_render` function in `prompts.py` preserves unmatched placeholders)
