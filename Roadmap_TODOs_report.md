# AHVS Roadmap Implementation Report

**Date:** 2026-04-08
**Branch:** `data_analyst`
**Basis:** README Section 16 (Roadmap) vs. current implementation audit

---

## Current State Summary

### Fully implemented and tested (ship-grade)

| Feature | Key Files | Tests |
|---------|-----------|-------|
| Core 8-stage cycle | `executor.py`, `runner.py`, `stages.py`, `contracts.py` | 284 tests in `test_ahvs.py` |
| LLM client layer (4 providers + ACP + cache) | `llm/client.py`, `anthropic_adapter.py`, `acp_client.py`, `cache.py` | Mocked throughout test suite |
| EvolutionStore / cross-cycle memory | `evolution.py` (1,074 lines) | Stage 7/8 tests |
| Genesis system (KD solver) | `genesis/contract.py`, `registry.py`, `router.py`, `solvers/kd_classifier.py` | Integration-heavy (no unit tests) |
| Data Analyst v1 (7 modules) | `data_analyst/` (executor, planner, profiler, synthesizer, validators, 7 modules) | Manual testing per git history |
| Hypothesis ops (add/edit/insert + browser GUI) | `hypothesis_ops.py`, `hypothesis_selector.py` | 36 tests in `test_hypothesis_ops.py` |
| Browser GUI forms (genesis, onboarding, multiagent) | `gui.py`, `gui_schemas.py` | 45 tests in `test_gui.py` |
| Installer / registry / CLI | `installer.py`, `registry.py`, `cli.py` | Covered in `test_ahvs.py` |
| AST-based splice + worktree isolation | `worktree.py` (741 lines) | Covered in `test_ahvs.py` |
| Domain packs (LLM + ML) | `domain_packs/ml_prompts.yaml`, `ml_skills.yaml` | Covered in `test_ahvs.py` |
| Multi-agent execution (team lead/executor/observer) | `.claude/skills/ahvs_multiagent/` | Skill-level (manual) |

---

## Scoring Axes

Each roadmap item is scored on three axes to determine implementation order:

| Axis | Meaning |
|------|---------|
| **Isolation** | Can it be added as a new module/file without modifying existing code? |
| **Testability** | Can it be unit tested independently with mocks? |
| **Disruption** | How much existing behavior changes? (lower is better) |

---

## Recommended Implementation Order

### Phase 1 — Safety & Efficiency (zero disruption)

#### 1. Constraint gating for text artifacts

**Roadmap category:** Intelligence & Evolution
**Description:** Size limits and growth bounds for evolved prompts/configs.

| Axis | Score |
|------|-------|
| Isolation | Very High |
| Testability | Very High |
| Disruption | Near zero |

**Rationale:** This is a safety feature. Right now, Claude Code can generate a 50KB prompt file that balloons to 200KB over successive cycles with no guard. This must be in place before adding any intelligence features that generate text artifacts (DSPy, prompt evolution). It is approximately 50 lines of code plus tests.

**Implementation sketch:**
- New file `ahvs/constraints.py`
- Function: `validate_artifact_growth(original: Path, modified: Path, max_growth_ratio: float = 2.0, max_size_bytes: int = 500_000) -> list[str]`
- Hook into `HypothesisWorktree.apply_files()` — if violations found, log warning and optionally reject the file
- Pure additive: a single validation function called from Stage 6 after files are applied but before eval runs
- No existing code changes needed beyond one hook point
- When gating is off (default), behavior is identical

**Testing:** Unit test: "given a file before/after, reject if growth > N% or size > M bytes." Completely deterministic.

---

#### 2. LLM-based lesson summarization

**Roadmap category:** Memory
**Description:** Consolidate related lessons into summaries to reduce prompt token usage.

| Axis | Score |
|------|-------|
| Isolation | Very High |
| Testability | Very High |
| Disruption | Low |

**Rationale:** The EvolutionStore already has `compact()` for dedup, but over 10+ cycles the lessons list grows linearly. Summarization is the natural next step and directly reduces token cost in Stage 2/3. This makes all subsequent features cheaper to run.

**Implementation sketch:**
- New method: `EvolutionStore.summarize_lessons(lessons: list, llm_client) -> list`
- Groups lessons by `hypothesis_type` + `stage`
- Calls LLM to merge each group into one lesson
- Preserves the highest-weight entry's metadata
- `context_loader.py` calls `store.query()` — add an optional `summarize=True` path that groups and summarizes before injecting into prompt
- New CLI flag: `--summarize-lessons`

**Testing:** Mock the LLM call. Assert that N lessons condense to 1 summary. Assert token count reduction.

---

### Phase 2 — UX & Intelligence (new modules only)

#### 3. Browser GUI for lesson/memory cleanup

**Roadmap category:** UX
**Description:** Interactive GUI (similar to hypothesis selector) for inspecting, filtering, and deleting individual lessons from `lessons.jsonl` and memory files from `.ahvs/memory/`.

| Axis | Score |
|------|-------|
| Isolation | Very High |
| Testability | Very High |
| Disruption | Zero |

**Rationale:** After summarization lands, users need a way to inspect what got summarized, delete bad lessons, and clean stale memory files. The hypothesis selector proves the browser-GUI pattern works. This is the same pattern applied to a different data source.

**Implementation sketch:**
- New standalone module: `ahvs/lesson_browser.py`
- Follows the exact pattern of `hypothesis_selector.py` (pure stdlib, no pip deps)
- Read `lessons.jsonl` and render as a table with columns: stage, category, summary, weight, verified, date
- Checkboxes for bulk delete
- Filter sidebar by stage/category/severity
- "Export selected" and "Delete selected" buttons
- Write back to `lessons.jsonl`
- Additional tab for `.ahvs/memory/*.md` files with preview and delete

**Testing:** Same test pattern as `test_gui.py` — render HTML, simulate POST, verify file mutations.

---

#### 4. LLM-as-judge secondary evaluator

**Roadmap category:** Intelligence & Evolution
**Description:** Fast LLM-judge heuristic for intermediate hypothesis filtering.

| Axis | Score |
|------|-------|
| Isolation | High |
| Testability | High |
| Disruption | Low |

**Rationale:** Stage 6 is the most expensive stage (Claude Code invocation + eval per hypothesis). Filtering weak hypotheses before execution saves real money and time. It also prepares the architecture for more intelligence features later.

**Implementation sketch:**
- New module: `ahvs/llm_judge.py`
- Function: `score_hypotheses(hypotheses: list, context: dict, llm_client) -> list[tuple[Hypothesis, float]]`
- Optional hook between Stage 5 (validation plan) and Stage 6 (execution)
- Uses a cheap/fast model (gpt-4.1-mini per cost policy)
- Scores each hypothesis on feasibility, expected impact, novelty
- Drops anything below threshold
- New CLI flags: `--llm-judge` / `--judge-threshold 0.5`
- When disabled (default), flow is unchanged

**Testing:** Mock the LLM. Assert scoring and threshold filtering. Deterministic with fixed responses.

---

### Phase 3 — Domain Expansion (extend existing patterns)

#### 5. Knowledge distillation as hypothesis type

**Roadmap category:** Platform Integration
**Description:** `knowledge_distillation` as a first-class AHVS hypothesis type, `--domain kd` domain pack, and cross-system memory fields.

| Axis | Score |
|------|-------|
| Isolation | High |
| Testability | High |
| Disruption | Low |

**Rationale:** This bridges the gap between Genesis (which scaffolds KD projects) and the optimization cycle (which improves them). Right now, after `ahvs genesis` creates a KD project, the cycle has no domain-specific knowledge about KD. This closes that loop.

**Implementation sketch:**
- New files: `ahvs/domain_packs/kd_prompts.yaml` with KD-specific hypothesis generation prompts (distillation temperature, teacher selection, student architecture, label strategy)
- New files: `ahvs/domain_packs/kd_skills.yaml` with skills for running KD stages
- Add `"knowledge_distillation"` to `VALID_TYPES` in `hypothesis_ops.py` and the strategy map in `executor.py`
- The KD solver already exists at the Genesis layer — this extends the pattern, not modifies it
- The 7 existing hypothesis types are untouched

**Testing:** Unit test the new type's execution strategy. Integration test with the KD pipeline (already in the adjacent repo).

---

#### 6. End-to-end examples

**Roadmap category:** Domain Expansion
**Description:** Worked examples for text classification, regression, data preprocessing.

| Axis | Score |
|------|-------|
| Isolation | Perfect |
| Testability | High |
| Disruption | Zero |

**Rationale:** Examples are most valuable after constraint gating, summarization, and the KD type are in place — they can showcase the full feature set. Writing examples too early means rewriting them when new features land. Each example is a self-contained script that can be run as a CI integration test.

**Implementation sketch:**
- New directory: `examples/`
- `examples/text_classification/` — full walkthrough: onboard, genesis, cycle, data_analyst
- `examples/regression/` — numeric target, custom eval
- `examples/data_preprocessing/` — data_analyst pipeline standalone
- Each example includes: README, sample data, expected output, runnable script

---

### Phase 4 — Platform (new integration points)

#### 7. MLOps export

**Roadmap category:** Platform Integration
**Description:** Push cycle results to W&B, MLflow, or Comet.

| Axis | Score |
|------|-------|
| Isolation | High |
| Testability | High |
| Disruption | Low |

**Rationale:** By this point the cycle produces richer artifacts (summarized lessons, gated artifacts, judged hypotheses). Exporting to MLOps makes them visible to the broader team.

**Implementation sketch:**
- New module: `ahvs/mlops_export.py`
- Hook into Stage 7/8 after report generation
- Adapter pattern: `WandbExporter`, `MLflowExporter`, `CometExporter` — each implements `export(cycle_summary: dict, artifacts: list[Path])`
- New CLI flag: `--export wandb|mlflow|comet`
- Optional dependency: `wandb`, `mlflow`, or `comet-ml` (graceful import failure)

**Testing:** Mock the W&B/MLflow APIs. Assert correct payload shape.

---

#### 8. MCP server for AHVS

**Roadmap category:** Platform Integration
**Description:** Expose AHVS as a Model Context Protocol service.

| Axis | Score |
|------|-------|
| Isolation | Very High |
| Testability | High |
| Disruption | Zero |

**Rationale:** MCP is a distribution mechanism, not a capability. It makes sense to stabilize the feature set first, then expose it. This is a wrapper over existing code.

**Implementation sketch:**
- New module: `ahvs/mcp_server.py`
- Wraps existing Python API (`AHVSConfig`, `execute_ahvs_cycle`, `EvolutionStore.query`)
- MCP tools: `run_cycle`, `list_repos`, `query_lessons`, `get_cycle_summary`, `run_data_analyst`
- MCP resources: `baseline_metric.json`, `lessons.jsonl`, `cycle_summary.json`

**Testing:** Mock the transport layer, test tool definitions and dispatch.

---

#### 9. GitHub CI integration

**Roadmap category:** Platform Integration
**Description:** PR-triggered AHVS cycles, eval-quality gates.

| Axis | Score |
|------|-------|
| Isolation | High |
| Testability | Medium |
| Disruption | Low |

**Rationale:** Depends on having a stable, well-tested feature set. Also benefits from MCP server being available as an alternative integration point.

**Implementation sketch:**
- New workflow YAML: `.github/workflows/ahvs-cycle.yml`
- New trigger module: `ahvs/ci_trigger.py`
- PR comment trigger: `/ahvs run --question "..."`
- Posts cycle summary as PR comment
- Eval-quality gate: fail PR if metric regresses beyond threshold

**Testing:** Needs GitHub Actions environment or `act` for local testing.

---

### Phase 5 — Core Contract Changes (careful, tested)

#### 10. Multi-metric optimization

**Roadmap category:** Domain Expansion
**Description:** Pareto-optimal selection across multiple metrics (precision *and* recall, accuracy *and* latency).

| Axis | Score |
|------|-------|
| Isolation | Medium |
| Testability | High |
| Disruption | Medium |

**Rationale:** Important feature but touches the deepest contract in the system (`baseline_metric.json`). By this point 9 items have shipped without modifying the metric contract, so the codebase is well-exercised and stable.

**Implementation sketch:**
- Extend `baseline_metric.json` to support `metrics: [{name, value, direction, weight}]` (backward-compatible: single `metric_name`/`current_value` still works)
- Pareto dominance function: `is_pareto_dominant(a: dict, b: dict) -> bool`
- Selection logic in Stage 4/7/8 uses Pareto front instead of single-metric comparison
- New CLI flag: `--multi-metric`

**Testing:** Pareto dominance is a pure function — well-testable with edge cases (ties, single-metric fallback, weighted preferences).

---

## Later Tier (higher disruption or dependencies)

These items should be implemented after Phase 5, in roughly this order:

| Order | Item | Category | Why Later |
|-------|------|----------|-----------|
| 11 | **Data Analyst v2** | Domain Expansion | Extends 7 existing modules — medium disruption. Includes: synthetic text augmentation, correlation analysis, outlier detection, uncertainty sampling. Do after core is stable. |
| 12 | **Parallel hypothesis execution** | Execution & Runtime | Changes Stage 6 execution loop fundamentally (`fcntl.flock` + `concurrent.futures`). Also conflicts with preference for sequential GPU execution. Reconsider when GPU contention is solved. |
| 13 | **Multi-agent decomposition** | Execution & Runtime | Architectural change to agent structure (narrower agents: plan-validator, code-reviewer, test-runner). Do after the simpler intelligence features prove their value. |
| 14 | **DSPy/GEPA prompt evolution** | Intelligence & Evolution | New subsystem for iteratively evolving prompts/configs over N generations. Depends on constraint gating (#1) being in place first. |
| 15 | **Jupyter notebook-style execution** | Execution & Runtime | New execution mode — fix only the failing cell and resume. High complexity, niche use case. |

---

## Defer (experimental / high-dependency)

| Item | Category | Reason to Defer |
|------|----------|-----------------|
| **Databricks integration** | Platform Integration | Platform-specific, requires Databricks environment for development and testing |
| **AutoResearch priors** | Intelligence & Evolution | Depends on external literature search infrastructure |
| **Synthetic eval dataset generation** | Intelligence & Evolution | Research-grade feature, needs careful design of generation quality guarantees |
| **Recursive self-improvement** | Intelligence & Evolution | Meta-system — AHVS optimizing its own strategies. Needs all other features stable first |
| **Data Analyst v3+** | Domain Expansion | Depends on v2 (CV support, regression tasks, NER, multi-file datasets, active learning) |
| **Complex algorithmic tasks** | Domain Expansion | Experimental — knowledge graph improvement, narrative quality. Unclear contract |
| **Improve unattended throughput** | Execution & Runtime | Depends on parallel execution (#12) being implemented first |

---

## Visual Summary

```
Phase 1 — Safety & Efficiency (zero disruption)
  1. Constraint gating          <- guard rails before intelligence
  2. Lesson summarization       <- reduce token cost for everything after

Phase 2 — UX & Intelligence (new modules only)
  3. Lesson/memory browser GUI  <- inspect what summarization produces
  4. LLM-as-judge evaluator     <- save money on Stage 6

Phase 3 — Domain Expansion (extend existing patterns)
  5. KD hypothesis type         <- close the genesis->cycle loop
  6. End-to-end examples        <- showcase the feature set

Phase 4 — Platform (new integration points)
  7. MLOps export               <- W&B / MLflow push
  8. MCP server                 <- expose as MCP service
  9. GitHub CI                  <- PR-triggered cycles

Phase 5 — Core Contract Changes (careful, tested)
  10. Multi-metric optimization <- first change to metric contract
```

Each phase builds on the previous. No item requires modifying a feature from a later phase. Every item can be shipped, tested, and validated independently before moving to the next.
