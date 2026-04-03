# AHVS × KD Integration Plan

**Purpose:** Detailed TODOs for integrating the KD system as a "solver" in AHVS's genesis layer, and adding `knowledge_distillation` as a hypothesis type. All work happens in the AHVS repo (`BB_AHVS`), branch `KD_integration`.

**Prerequisite:** KD bug fixes from `KD_improvement_fix.md` should be completed and merged to KD's `main` branch first. The KD repo must be pip-installable or accessible via Python path.

**Architecture principle:** AHVS is HOM (Hypothesis-driven Optimization with Memory). KD is a solver. They talk through a contract, never through shared code. Changes in AHVS never affect KD.

**LLM Policy:**
- **Default: ACP (Claude Code / Codex subscription)** — use for all agent orchestration, hypothesis generation, validation planning, code execution, and reasoning tasks. This maximizes subscription value and avoids per-call API costs.
- **When external LLM API is needed** (KD pipeline annotation, prompt building, prompt selection): use **gpt-4.1-mini** (never gpt-4o) or **Gemini 2.5 Flash Lite**.
- **Never use gpt-4o** — cost-prohibitive for annotation/labeling workloads.
- AHVS's `--provider acp` should remain the default for all AHVS operations.

---

## Phase 1: Solver Contract + KD Adapter

**Goal:** Define the solver protocol and build a thin adapter that calls KD's pipeline, producing AHVS-compatible output. No changes to AHVS core.

### Task 1.1: Create Genesis Module Structure

Create the following directory structure:

```
ahvs/genesis/
├── __init__.py              # Exports: GenesisResult, Solver, SolverRegistry
├── contract.py              # GenesisResult dataclass + Solver protocol
├── registry.py              # SolverRegistry: loads solver configs from YAML
├── router.py                # ProblemRouter: LLM-based problem → solver routing
└── solvers/
    ├── __init__.py
    └── kd_classifier.py     # KD adapter: calls KD pipeline, returns GenesisResult
```

### Task 1.2: Define the Solver Contract (`contract.py`)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional

@dataclass
class GenesisResult:
    """What every solver must produce."""
    project_dir: Path              # where the solution lives
    baseline_metric: dict          # {primary_metric, <metric_name>: value, eval_command, ...}
    eval_command: str              # shell command to re-measure the metric
    model_path: Optional[str]      # path to trained model (if applicable)
    summary: str                   # human-readable summary of what was built

class Solver(Protocol):
    """Interface that all solvers implement."""
    name: str
    problem_types: list[str]       # e.g., ["classification", "multi-label"]

    def solve(
        self,
        problem: str,              # natural language problem description
        data_path: str,            # path to user's data
        target_metric: str,        # metric to optimize (e.g., "f1_weighted")
        output_dir: str,           # where to put the solution
        config_overrides: dict | None = None,  # optional solver-specific overrides
    ) -> GenesisResult:
        ...
```

**Key design decisions:**
- `GenesisResult.baseline_metric` must be a dict that can be written directly as `.ahvs/baseline_metric.json`
- `eval_command` must be a shell command that, when run in the project_dir, outputs the metric value
- The Solver protocol is minimal — easy for future solvers to implement

### Task 1.3: Build KD Classifier Adapter (`solvers/kd_classifier.py`)

This is the bridge between AHVS and KD. Two implementation options — choose based on KD state:

**Option A: Direct Python API (simpler, works today)**

```python
def solve(self, problem, data_path, target_metric, output_dir, config_overrides=None):
    # 1. Generate task spec YAML from problem description (using LLM)
    # 2. Build pipeline config YAML
    # 3. Call run_auto_ml_pipeline(config_path, spec_path, data_path, root_output_dir=output_dir)
    # 4. Extract metrics from AutoMLPipelineResult
    # 5. Write .ahvs/baseline_metric.json
    # 6. Construct eval_command (e.g., "python -m src.tml_classifier.main --spec ... --evaluate-only")
    # 7. Return GenesisResult
```

**Requires:** KD installable via `pip install -e ../hackathon_knowledge_distillation` or sys.path manipulation.

**Option B: Agent SDK (more powerful, requires KD agent fixes)**

```python
def solve(self, problem, data_path, target_metric, output_dir, config_overrides=None):
    # 1. from src.kd_agent.server import create_kd_agent
    # 2. Compose prompt: f"Build a classifier: {problem}. Data: {data_path}. Optimize for: {target_metric}."
    # 3. server, options = create_kd_agent(cwd=kd_repo_path)
    # 4. result = await query(prompt=prompt, options=options)
    # 5. Parse agent output → extract model_path, metrics
    # 6. Write .ahvs/baseline_metric.json
    # 7. Return GenesisResult
```

**Requires:** KD agent bugs fixed (KD_improvement_fix.md), `claude-agent-sdk` installed.

**Recommendation:** Start with Option A for reliability. Switch to Option B once KD agent is battle-tested.

### Task 1.4: Build Solver Registry (`registry.py`)

YAML-driven registry so new solvers can be added without code changes:

```yaml
# ahvs/genesis/solvers.yaml
solvers:
  kd_classifier:
    module: ahvs.genesis.solvers.kd_classifier
    class: KDClassifierSolver
    problem_types: [classification, multi-label, sentiment, intent]
    config:
      kd_repo_path: /home/ubuntu/vision/hackathon_knowledge_distillation
      default_model: distilbert-base-uncased
      default_annotation_model: gpt-4o
      conda_env: py11
```

The registry loads solver configs, instantiates solver classes, and provides a `get_solver(problem_type)` method.

### Task 1.5: Build Problem Router (`router.py`)

LLM-based routing that maps a problem description to a solver:

```python
def route(self, problem: str) -> str:
    """Returns solver name (e.g., 'kd_classifier') based on problem description."""
    # Use LLM to classify problem type from description
    # Match against registered solvers' problem_types
    # Return best match
```

For now, this can be simple — if only one solver exists (kd_classifier), just return it. The router becomes important when solver #2 is added.

### Task 1.6: Test the Adapter Independently

Write a test script (not a unit test — a manual integration test):

```python
# test_genesis_kd.py
from ahvs.genesis.solvers.kd_classifier import KDClassifierSolver

solver = KDClassifierSolver(kd_repo_path="/home/ubuntu/vision/hackathon_knowledge_distillation")
result = solver.solve(
    problem="Classify tweet sentiment into 4 categories: neutral, happy, anger, excited",
    data_path="/home/ubuntu/vision/hackathon_knowledge_distillation/sample_data/train_sentiment_small.csv",
    target_metric="f1_weighted",
    output_dir="/tmp/genesis_test",
)
print(f"Success: {result.project_dir}")
print(f"Baseline: {result.baseline_metric}")
print(f"Eval command: {result.eval_command}")
```

**Acceptance:** Running this script produces a trained model and a valid `baseline_metric.json`.

---

## Phase 2: Genesis CLI Entry Point

**Goal:** Add `ahvs genesis` subcommand. No changes to the 8-stage AHVS loop.

### Task 2.1: Add Genesis Subcommand to CLI (`ahvs/cli.py`)

Add a new subcommand (or flag) to the existing CLI:

```bash
ahvs genesis \
  --problem "Classify customer emails into intent categories" \
  --data /path/to/emails.csv \
  --target-metric f1_weighted \
  --output-dir /path/to/new_project \
  --solver kd_classifier
```

**Implementation:**
- Add `cmd_genesis()` function in `cli.py` (similar pattern to existing `cmd_ahvs()`)
- Parse arguments: `--problem`, `--data`, `--target-metric`, `--output-dir`, `--solver`
- Load solver from registry
- Call `solver.solve(...)`
- Write `.ahvs/baseline_metric.json` in the output directory
- Register the new project in `~/.ahvs/registry.json`
- Print success message with next-step instructions

### Task 2.2: Auto-Generate eval_command

The adapter must produce an `eval_command` that AHVS can run in future cycles. This is tricky because:
- TML classifier eval requires the correct spec YAML, data path, and model path
- The command must be self-contained (runnable from the project directory)

**Approach:**
- Write a small eval script in the project_dir: `eval_metric.py`
- This script loads the trained model, runs inference on test data, prints the metric
- `eval_command` = `python eval_metric.py`

### Task 2.3: Test End-to-End (Genesis → AHVS Cycle)

```bash
# Step 1: Genesis creates a new project
ahvs genesis --problem "sentiment classification" \
  --data emails.csv --target-metric f1_weighted \
  --output-dir /tmp/my_classifier

# Step 2: Normal AHVS cycle improves it
ahvs --repo /tmp/my_classifier \
  --question "improve f1_weighted to 0.85" \
  --domain ml
```

**Acceptance:** Step 1 produces a project with baseline_metric.json. Step 2 runs a normal AHVS optimization cycle against that baseline.

---

## Phase 3: Genesis as Optional Pre-Stage

**Goal:** Single-command experience: `ahvs --problem ... --question ...` creates + optimizes. Minimal changes to AHVS core.

### Task 3.1: Add `--problem` and `--data` Flags to Main CLI

In `cli.py:cmd_ahvs()`, add optional flags:
- `--problem` — natural language problem description
- `--data` — path to input data
- `--solver` — which solver to use (default: auto-detect via router)

### Task 3.2: Add Genesis Logic Before Cycle Start

In `cli.py`, between config creation and `execute_ahvs_cycle()` call:

```python
if args.problem and not config.baseline_path.exists():
    from ahvs.genesis import SolverRegistry, ProblemRouter
    registry = SolverRegistry()
    solver_name = ProblemRouter().route(args.problem) if not args.solver else args.solver
    solver = registry.get(solver_name)
    result = solver.solve(
        problem=args.problem,
        data_path=args.data,
        target_metric=...,  # extract from --question or default
        output_dir=str(config.repo_path),
    )
    # baseline_metric.json now exists → continue to normal AHVS cycle
```

**Key:** If baseline already exists, skip genesis entirely (zero-cost pass-through).

### Task 3.3: Test Single-Command Flow

```bash
ahvs --repo /tmp/new_project \
  --problem "classify customer intent" \
  --data emails.csv \
  --question "improve f1_weighted to 0.90" \
  --domain ml
```

**Acceptance:** Creates project, trains initial model, measures baseline, then runs optimization cycle — all in one command.

---

## Phase 4: `knowledge_distillation` Hypothesis Type

**Goal:** AHVS can generate hypotheses that use KD as an optimization action (e.g., "distill this component into a fine-tuned model").

### Task 4.1: Create KD Domain Pack

Create `ahvs/domain_packs/kd_prompts.yaml`:

```yaml
hypothesis_types:
  knowledge_distillation:
    description: >
      Distill a component (e.g., a scoring function, classifier, or LLM call)
      into a smaller, faster model using the KD pipeline.
    execution_strategy: >
      1. Identify the component to distill (function signature, I/O format)
      2. Generate a task spec YAML from the component's behavior
      3. Generate training data by running the existing component on sample inputs
      4. Run KD pipeline (auto_ml) to train a distilled model
      5. Swap the distilled model into the system
      6. Re-measure the target metric
    constraints:
      - The distilled model must be measurable by the existing eval_command
      - Quality regression must be within regression_floor
      - Report both quality metric AND latency/cost improvement
    success_criteria: >
      target_metric >= baseline - regression_floor AND
      (latency_improvement > 2x OR cost_reduction > 50%)
```

Create `ahvs/domain_packs/kd_skills.yaml`:

```yaml
skills:
  kd_auto_ml:
    description: Run KD auto_ml pipeline to produce a trained classifier
    command: "python -m src.auto_ml.main {config} --spec {spec} --data {data}"
    requires: [hackathon_knowledge_distillation]
  kd_annotate:
    description: Label data using LLM via KD annotator
    command: "python -m src.annotator.main --data {data} --prompt {prompt} --output {output}"
    requires: [hackathon_knowledge_distillation]
```

### Task 4.2: Add KD Hypothesis Execution Logic

In `ahvs/executor.py`, add handling for `hypothesis_type == "knowledge_distillation"`:

- During validation planning (Stage 5): generate KD-specific implementation plan
- During execution (Stage 6): invoke KD pipeline in worktree, extract metrics
- This is the most complex task — the executor needs to:
  1. Generate a task spec from the hypothesis description
  2. Generate training data from the existing system
  3. Call KD pipeline
  4. Extract trained model and metrics
  5. Measure end-to-end metric in the worktree

**Note:** This phase should be deferred until Phases 1-3 are proven. The adapter must be reliable before AHVS trusts it inside the optimization loop.

### Task 4.3: Add `--domain kd` Flag

In `cli.py`, extend the `--domain` choices to include `kd`, which loads the KD domain pack:

```python
if args.domain == "kd":
    config.prompts_override_path = DOMAIN_PACKS / "kd_prompts.yaml"
    config.skill_registry_path = DOMAIN_PACKS / "kd_skills.yaml"
```

---

## Phase 5: Cross-System Memory (Flywheel)

**Goal:** Lessons from KD-driven cycles accumulate in AHVS's evolution store with KD-specific structured fields, enabling the system to learn which KD configurations work for which task types.

### Task 5.1: Extend LessonEntry with KD Fields

In `ahvs/evolution.py`, add optional fields to LessonEntry:

```python
# Optional KD-specific fields (only populated for knowledge_distillation hypotheses)
kd_config: dict | None = None       # {backend, model, dspy, annotation_model, ...}
task_domain: str | None = None       # "sentiment", "intent", "ner", ...
data_size: int | None = None         # number of training samples
```

### Task 5.2: Enrich Context Loader with KD Lessons

In `ahvs/context_loader.py`, when building the historical digest, include KD-specific aggregations:

```
## KD-Specific Insights (from prior knowledge_distillation hypotheses)
- DistilBERT on sentiment (5 attempts): avg F1 delta +0.034, best with DSPy annotation
- gpt-4.1-nano fine-tune on intent (3 attempts): avg F1 delta +0.021, best with >10k samples
```

### Task 5.3: Promote KD Lessons to Global Store

In `ahvs/runner.py`, when promoting lessons to `~/.ahvs/global/evolution/lessons.jsonl`, include KD lessons with `category: "KD_PIPELINE"` so they transfer across projects.

---

## Dependency Graph

```
Phase 1 (Solver + Adapter)
    │
    ├── Phase 2 (Genesis CLI) ── depends on Phase 1
    │       │
    │       └── Phase 3 (Pre-stage) ── depends on Phase 2
    │
    └── Phase 4 (KD Hypothesis Type) ── depends on Phase 1
            │
            └── Phase 5 (Cross-System Memory) ── depends on Phase 4
```

Phases 2 and 4 can be developed in parallel after Phase 1 is complete.

---

## File Change Summary

| Phase | Files Created | Files Modified |
|---|---|---|
| 1 | `ahvs/genesis/__init__.py`, `contract.py`, `registry.py`, `router.py`, `solvers/__init__.py`, `solvers/kd_classifier.py`, `solvers.yaml` | None |
| 2 | None | `ahvs/cli.py` (add `cmd_genesis`) |
| 3 | None | `ahvs/cli.py` (add `--problem`, `--data` flags) |
| 4 | `ahvs/domain_packs/kd_prompts.yaml`, `kd_skills.yaml` | `ahvs/executor.py` (add KD execution), `ahvs/cli.py` (add `--domain kd`) |
| 5 | None | `ahvs/evolution.py` (extend LessonEntry), `ahvs/context_loader.py` (KD digest), `ahvs/runner.py` (KD promotion) |

---

## LLM Usage Architecture

**Cost policy:** Default to ACP (Claude Code / Codex subscription) for all reasoning and orchestration. Use cheap external LLMs only for bulk labeling work.

```
AHVS Genesis / Optimization
    │
    │ Uses Claude via ACP (default, subscription-based, no per-call cost)
    │ for: hypothesis generation, validation planning, code execution
    │
    ▼
KD Agent (claude-agent-sdk, orchestrator)
    │
    │ Uses Claude via ACP (subscription-based)
    │ for: conversational pipeline orchestration, decision-making
    │
    ▼
KD Pipeline Stages (labeling workload)
    │
    │ Uses gpt-4.1-mini or Gemini 2.5 Flash Lite (cheap, per-call)
    │ for: prompt building, prompt selection, data annotation
    │ NEVER use gpt-4o (cost-prohibitive for bulk labeling)
    │
    ▼
Trained Model (output)
```

**Required credentials:**
- Claude: via ACP (Claude Code session) — no separate API key needed
- OpenAI: `OPENAI_API_KEY` in KD repo's `.env` — for gpt-4.1-mini labeling
- Google: `GOOGLE_API_KEY` in KD repo's `.env` — if using Gemini 2.5 Flash Lite instead

**Default model configs to update in KD repo:**
- `src/kd_agent/config_builder.py` — change default `annotator.model` from `gpt-4o` → `gpt-4.1-mini`
- `src/kd_agent/prompts.py` — if any tool defaults reference `gpt-4o`, change to `gpt-4.1-mini`
- `examples/auto_ml/config.yaml` — update `prompt_builder.model` and `annotator.model`
- `examples/auto_label/config.yaml` — same updates
