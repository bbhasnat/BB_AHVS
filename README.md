# AHVS — Adaptive Hypothesis Validation System

AHVS is a standalone cyclic hypothesis-validation pipeline that autonomously generates, selects, executes, and evaluates improvement hypotheses for target AI/ML systems. Point it at a repo with a measurable metric, ask a question like *"How can we improve test_accuracy by 5%?"*, and it tests each hypothesis in an isolated git worktree, measures the result, and archives what it learned. Lessons from past cycles steer future experiments away from dead ends and toward approaches that have worked before — so the system gets smarter with every run.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [The 8-Stage Cycle](#3-the-8-stage-cycle)
4. [Four Ways to Use AHVS](#4-four-ways-to-use-ahvs)
5. [Quick Start](#5-quick-start)
6. [Onboarding a Target Repository](#6-onboarding-a-target-repository)
7. [CLI Reference](#7-cli-reference)
8. [Hypothesis Types](#8-hypothesis-types)
9. [Skill Library](#9-skill-library)
10. [Cross-Cycle Memory](#10-cross-cycle-memory)
11. [Python API](#11-python-api)
12. [Configuration Reference](#12-configuration-reference)
13. [Directory Layout](#13-directory-layout)
14. [Advanced Usage](#14-advanced-usage)
    - [Model Selection Strategy](#model-selection-strategy)
    - [Budget Planning for Experiments](#budget-planning-for-experiments)
    - [LLM Response Cache](#llm-response-cache)
    - [Prompt Engineering Tips](#prompt-engineering-tips)
    - [Multi-Agent Execution](#multi-agent-execution-with-claude-code-agent-teams)
    - [Roadmap / TODOs](#roadmap--todos)
    - [Browser-Based Hypothesis Selector](#browser-based-hypothesis-selector)
    - [AST-Based Partial Output Merging](#ast-based-partial-output-merging-splice_functions)
    - [Test Coverage](#test-coverage)

---

## 1. Overview

Given a target repository and a single question — *"How can we improve answer\_relevance by 5%?"* — AHVS:

1. Reads the current baseline metric and all lessons from previous cycles
2. Asks an LLM to generate concrete, typed hypotheses
3. Pauses for a human (or runs non-interactively) to select which to test
4. Builds a detailed implementation + evaluation plan per hypothesis
5. Invokes **Claude Code** to implement each hypothesis, applies the generated files to a **git worktree** of the target repo, and runs the `eval_command` against it
6. Writes a cycle report and archives every lesson into the **EvolutionStore**
7. Produces a `cycle_summary.json` with a clear keep/revert recommendation and the path to the kept worktree/patch

Each cycle is self-contained and idempotent. The accumulated lessons steer future cycles away from dead ends and toward approaches that have worked before.

---

## 2. Architecture

AHVS is a self-contained package with no external framework dependencies. Its core components are:

| Component | Module | Purpose |
|-----------|--------|---------|
| **Claude Code** | `ahvs.executor` | Implements hypotheses via Claude Code CLI (`claude -p`) |
| **EvolutionStore** | `ahvs.evolution` | Cross-cycle memory — persists lessons across cycles |
| **LLM Client** | `ahvs.llm` | Provider-agnostic LLM client factory (Anthropic, OpenAI, OpenRouter, DeepSeek, ACP) with content-addressable response cache |
| **Stage Pipeline** | `ahvs.stages`, `ahvs.executor`, `ahvs.runner` | 8-stage cycle orchestration with checkpoint/resume |
| **Hypothesis Worktree** | `ahvs.worktree` | Git worktree lifecycle, file application, and AST-based splice |
| **Prompt Manager** | `ahvs.prompts` | Stage prompts with YAML overrides |

### Code agent: Claude Code

AHVS uses Claude Code CLI (`claude -p`) as its code execution backend. Claude Code runs **inside the hypothesis worktree** (not the live target repo), so the main repository is never modified during execution. It operates as a subprocess with Read/Edit/Glob/Grep tools scoped to the worktree directory.

Before granting Claude Code access, AHVS runs a **secret-scan preflight** that checks for `.env`, `*.pem`, `*.key`, `credentials.json`, and other common secret-file patterns. If potential secrets are detected, a visible warning is emitted.

Claude Code receives:

- The hypothesis description and implementation plan
- The available **skill library** (which tools exist and how to invoke them)
- A required output contract (`result.json` with the metric as a float)
- The full source tree (via the worktree, which is a copy of the repo at HEAD)
- The `eval_command` that will measure the result in the worktree

After Claude Code edits files in the worktree, AHVS captures **all** modified files (Python, YAML, JSON, config, templates — not just `.py`), reverts the worktree to clean state, then re-applies the changes through its safety pipeline (forbidden-file filter, syntax validation, AST-based splice). The `eval_command` from `baseline_metric.json` is then executed inside the worktree to produce a real measurement.

### Worktree execution model

Each hypothesis gets its own worktree under `<cycle_dir>/worktrees/<ID>/`. This ensures:

- **Repo-grounded execution:** Code is tested against the actual repo, not in an isolated sandbox
- **No branch pollution:** Worktrees are detached (no branches created)
- **Safe concurrency:** Each hypothesis has its own copy of the repo (note: parallel execution is not yet supported — see [Parallel execution](#parallel-hypothesis-execution-future-work) for status)
- **Path containment:** All generated file paths are validated by a shared `validate_safe_relpath()` utility before writing — to both `tool_runs/` and worktree directories. Absolute paths, `..` traversal, and symlink escapes are rejected. The containment check uses `Path.is_relative_to()` (not string-prefix matching) to prevent false-positive bypasses
- **Audit trail:** A `.patch` file is saved for every hypothesis
- **Smart merging:** When Claude Code produces partial file output (only some functions), AST-based `splice_functions` merges changes into the existing file rather than overwriting it — preserving untouched code, replacing modified definitions, and appending new ones

After all hypotheses run, AHVS identifies the best improvement and keeps its worktree. All other worktrees are cleaned up. The kept worktree path and all patch paths are recorded in `cycle_summary.json`.

If the target path is not a git repository, AHVS auto-initializes one at Stage 1 (`git init && git add -A && git commit`). This is logged, recorded in `cycle_manifest.json`, and the `.git` directory persists after the cycle. No manual setup required.

---

## 3. The 8-Stage Cycle

```
Stage 1  AHVS_SETUP            Auto-init git if needed, pre-flight checks (baseline, clean repo, LLM), cycle dir init
Stage 2  AHVS_CONTEXT_LOAD     Load baseline + enriched context + EvolutionStore -> context_bundle.json
Stage 3  AHVS_HYPOTHESIS_GEN   LLM generates 1-5 typed hypotheses
Stage 4  AHVS_HUMAN_SELECTION  -- GATE -- operator selects which to run
Stage 5  AHVS_VALIDATION_PLAN  LLM writes per-hypothesis implementation plan
Stage 6  AHVS_EXECUTION        Claude Code executes each hypothesis; worktree + eval_command
Stage 7  AHVS_REPORT_MEMORY    LLM writes cycle report; lessons -> EvolutionStore
Stage 8  AHVS_CYCLE_VERIFY     Validate all artifacts; write cycle_summary.json
```

For ML or other resource-heavy evals, Stage 6 now tears down timed-out or crashed eval subprocesses before moving on. This helps prevent orphan workers from holding GPU or CPU memory after a hypothesis fails.

The gate at Stage 4 pauses for human input. It supports four selection modes:

1. **Pre-specified** — If `selection.json` already exists in the cycle directory (e.g. written by a conversational Claude Code session), the gate honours it and skips all prompts. This is how conversational mode works: Claude shows you the hypotheses, you say which to run, Claude writes `selection.json`, and the executor respects your choice.
2. **Auto-approve** (`--auto-approve`) — Selects all hypotheses. For CI/scripted runs.
3. **Interactive** (default for CLI) — Prompts on stdin. Enter IDs (e.g. `H1 H3`), `all`, or `none` to abort.
4. **Browser GUI** — Run hypothesis generation first with `--until-stage AHVS_HYPOTHESIS_GEN`, then launch the browser-based selector:
   ```bash
   python -m ahvs.hypothesis_selector <cycle_dir>
   ```
   This opens a dark-themed web UI on localhost with checkboxes for each hypothesis. After submitting, it writes `selection.json` into the cycle directory. Resume with `--from-stage AHVS_HUMAN_SELECTION`.

You can also use `--selection H1,H3` on the CLI to pre-specify which hypotheses to run without interactive prompts.

If the operator aborts, the cycle can be resumed from Stage 3 to regenerate hypotheses.

Every stage writes a checkpoint. A failed stage stops the cycle; later stages are not run. You can also stop early with `--until-stage STAGE_NAME` to run only up to a specific stage (e.g. `--until-stage AHVS_HYPOTHESIS_GEN` for generation only), then resume later with `--from-stage`.

> **Clean repo required:** Stage 1 pre-flight **fails** if the target repo has uncommitted changes. This is a hard requirement because AHVS creates hypothesis worktrees from committed `HEAD` — uncommitted changes in the working tree would not be included in the experiment. Commit or stash changes before starting a cycle.

---

## 4. Four Ways to Use AHVS

AHVS supports four usage modes depending on your workflow. All four produce identical results — same 8-stage cycle, same artifacts, same cross-cycle memory.

### Mode 1: Conversational in Claude Code (recommended for most users)

Just describe what you want in natural language. No commands to memorize.

**Onboarding a new repo:**
```
> Onboard /path/to/my-project for AHVS — I want to improve precision without tanking F1
```

The `ahvs_onboarding` skill will scan your repo, create a headless eval script if needed, write `.ahvs/baseline_metric.json`, and verify everything works.

**Running a cycle:**
```
> Run AHVS on /path/to/my-project — improve precision above 0.80 using only
  Gemini 3.1 Flash Lite. Don't let F1 drop below 0.62. Propose algorithmic
  changes, not just prompt tweaks. Use 2 hypotheses.
```

**Reviewing results:**
```
> Show me the results from the last AHVS cycle on /path/to/my-project
```

**Iterating:**
```
> Run another AHVS cycle — the keyword strategy worked well last time,
  focus on that direction with 3 hypotheses
```

Claude Code reads `.ahvs/baseline_metric.json` for all the details (eval command, constraints, system levers, prior experiments) so you don't need to repeat them every time.

### Mode 2: CLI command

For scripting, CI integration, or when you prefer explicit commands:

```bash
ahvs \
  --repo /path/to/my-project \
  --question "Improve precision above 0.80 while keeping F1 >= 0.62" \
  --max-hypotheses 2 \
  --provider openrouter \
  --model anthropic/claude-sonnet-4-6 \
  --api-key-env OPENROUTER_API_KEY
```

Add `--auto-approve` for unattended runs. See [CLI Reference](#7-cli-reference) for all flags.

### Mode 3: Python API

For integration into existing scripts or notebooks:

```python
from ahvs import AHVSConfig, execute_ahvs_cycle

config = AHVSConfig(
    repo_path="/path/to/my-project",
    question="Improve precision above 0.80 while keeping F1 >= 0.62",
    max_hypotheses=2,
    llm_model="claude-sonnet-4-6",
    llm_api_key_env="OPENROUTER_API_KEY",
)

results = execute_ahvs_cycle(config, auto_approve=True)
for r in results:
    print(r.stage.name, r.status.value)
```

See [Python API](#11-python-api) for resumption, callbacks, and result inspection.

### Mode 4: Multi-agent with GUI selection

For supervised execution with multiple Claude Code agents (executor + observer) and human hypothesis selection via browser GUI. Say this to Claude Code:

```
Run AHVS on /path/to/my-project with multi-agent supervision.
Use 3 hypotheses. Show me the GUI for selection.
```

Claude Code (as team lead) will:
1. Generate hypotheses (Stages 1-3)
2. Open a browser GUI for you to select which to run
3. Spawn an executor agent (runs hypotheses) and an observer agent (verifies results, fixes framework bugs)
4. Run each hypothesis one at a time with verification between them
5. Generate the final report

Your only manual step is clicking checkboxes in the browser. For fully automatic execution, say "auto-approve all hypotheses" to skip the GUI entirely.

### Which mode should I use?

| Scenario | Recommended mode |
|----------|-----------------|
| First time using AHVS | **Conversational** — Claude guides you through onboarding |
| Exploring what to optimize | **Conversational** — describe goals in natural language |
| Running a specific experiment | **CLI** or **Conversational** — both work equally |
| CI/CD integration | **CLI** with `--auto-approve` |
| Supervised execution with bug-fixing | **Multi-agent** — observer catches and fixes framework bugs |
| Scripting multi-cycle campaigns | **CLI** (bash loop) or **Python API** |
| Embedding in existing workflows | **Python API** |

---

## 5. Quick Start

### Prerequisites

- Python 3.11+
- `pip install ahvs`
- **API mode:** An API key for a Claude model (or any OpenAI-compatible endpoint)
- **ACP mode:** A local ACP-compatible agent CLI (Claude Code, Codex, etc.) + `acpx`

AHVS supports two LLM modes for its own orchestration calls (hypothesis generation, validation planning, reporting):

| Mode | Flag | API key needed? | What runs the LLM? |
|------|------|-----------------|---------------------|
| API provider (default) | `--provider anthropic` | Yes | Direct API call |
| ACP local agent | `--provider acp` | No (for AHVS) | Claude Code / Codex via acpx |

> **Control-plane vs runtime inference:** The `--provider` flag only controls AHVS's own orchestration LLM calls. Runtime inference inside the target repo's evaluated code (e.g. an OpenAI-powered RAG pipeline) still uses whatever credentials that codebase requires.

### Step 1a — API mode (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Step 1b — ACP mode (no API key needed for AHVS)

```bash
# Ensure acpx is installed
npm install -g acpx

# Use a local agent for AHVS orchestration
ahvs \
  --repo /path/to/repo \
  --question "How can we improve answer_relevance?" \
  --provider acp \
  --acp-agent claude   # or codex, gemini, etc.
```

### Step 2 — Onboard your target repo

**Option A: Conversational onboarding (recommended)**

If you're using Claude Code, just tell it what you want:

> "Onboard this repo for AHVS — I want to improve precision without tanking F1"

The `ahvs_onboarding` skill will:
1. Deep-scan your repo to understand its evaluation pipeline
2. Create a headless CLI eval script if evaluation only exists in notebooks
3. Ask about your optimization goals, budget constraints, and hypothesis diversity
4. Gather prior experiment results to prevent repeating dead ends
5. Write an enriched `.ahvs/baseline_metric.json` with metrics, constraints, system levers, and prior experiments
6. Verify the eval command produces parseable output

It refuses to proceed until the setup is verified. See `.claude/skills/ahvs_onboarding/SKILL.md` for details.

**Option B: Manual setup**

Create `.ahvs/baseline_metric.json` in your target repository (minimal):

```json
{
  "primary_metric": "answer_relevance",
  "answer_relevance": 0.74,
  "recorded_at": "2026-03-18T10:00:00Z",
  "commit": "abc1234",
  "eval_command": "python scripts/eval.py --dataset data/test.jsonl"
}
```

For better hypothesis quality, include the enriched fields (see [Section 6.1](#61-baseline-metric-file)).

### Step 3 — Run a cycle

**Option A: Conversational (in Claude Code)**

```
> Run AHVS on /path/to/your-rag-project — improve answer_relevance by at least 5%.
  Use 3 hypotheses.
```

**Option B: CLI**

```bash
ahvs \
  --repo /path/to/your-rag-project \
  --question "How can we improve answer_relevance by at least 5%?" \
  --max-hypotheses 3
```

Both modes pause at Stage 4 to display generated hypotheses. In conversational mode, Claude asks which hypotheses to run and writes your choice to `selection.json`. In CLI mode, enter the IDs you want to test (e.g. `H1 H3`) or `all`, or use `--selection H1,H3` to pre-specify. Pass `--auto-approve` to skip this gate entirely.

### Step 4 — Review results

```bash
cat /path/to/your-rag-project/.ahvs/cycles/<timestamp>/cycle_summary.json
cat /path/to/your-rag-project/.ahvs/cycles/<timestamp>/report.md
```

Or conversationally: `> Show me the results from the last AHVS cycle`

---

## 6. Onboarding a Target Repository

AHVS needs four things from a target repo:

### 6.1 Baseline metric file

`.ahvs/baseline_metric.json` — required fields:

| Field | Description |
|---|---|
| `primary_metric` | Name of the metric to optimise (e.g. `precision`) |
| `<primary_metric>` | Current numeric value of that metric (float) |
| `recorded_at` | ISO-8601 timestamp when this baseline was measured |
| `eval_command` | Headless shell command that prints `metric_name: value` to stdout |
| `commit` | Git commit SHA when baseline was recorded. *(Recommended — AHVS emits a pre-flight warning if absent, since it cannot verify the baseline matches the current repo state.)* |

Enriched fields (optional but strongly recommended — improves hypothesis quality):

| Field | Description |
|---|---|
| `optimization_goal` | Plain English description of what to optimize and constraints |
| `regression_floor` | Secondary metrics with minimum acceptable values, e.g. `{"f1_score": 0.62}` |
| `constraints` | Budget limits, model restrictions, hypothesis scope requirements |
| `system_levers` | Tunable parameters: strategies, modes, algorithmic areas with file paths |
| `prior_experiments` | Results from past experiments with config details and identified problems |
| `notes` | Additional context (dataset size, key insights, known hard cases) |

**Minimal example:**
```json
{
  "primary_metric": "f1_score",
  "f1_score": 0.81,
  "recorded_at": "2026-03-18T09:00:00Z",
  "commit": "d4e5f6a",
  "eval_command": "python scripts/eval.py --dataset data/test.jsonl"
}
```

**Enriched example** (produces better hypotheses):
```json
{
  "primary_metric": "precision",
  "precision": 0.7128,
  "f1_score": 0.6699,
  "recorded_at": "2026-03-19T10:00:00Z",
  "commit": "9c41b35",
  "eval_command": "cd /path/to/project && python -m package.run_eval --eval-only",
  "optimization_goal": "Maximize precision while keeping f1_score >= 0.62",
  "regression_floor": {"f1_score": 0.62},
  "constraints": {
    "model_budget": "Gemini 3.1 Flash Lite only. No expensive models.",
    "hypothesis_scope": "Must include algorithmic changes, not just prompt rewrites."
  },
  "system_levers": {
    "strategies": ["random", "keyword", "semantic"],
    "algorithmic_areas": ["post_selector.py: selection strategy", "parsing.py: threshold tuning"]
  },
  "prior_experiments": {
    "best_precision": {"config": "gpt54_kw_prv2", "precision": 0.76, "f1": 0.57, "problem": "F1 too low"}
  }
}
```

### 6.2 Evaluation setup

Your `eval_command` is **executed in a git worktree** of the target repo after Claude Code's generated files are applied. It must be reproducible and must write a numeric result that can be parsed.

AHVS extracts metrics using a two-path strategy depending on whether `eval_command` is configured:

**When `eval_command` is configured (recommended):**

| Tier | Source | Behavior |
|---|---|---|
| **0** | `eval_command` stdout (run in worktree) | **Only trusted source.** Metric parsed from stdout. |
| — | `extraction_failed` | eval_command did not produce a valid metric |

This is the authoritative-eval policy: when `eval_command` exists, it is the single source of truth.

**When `eval_command` is empty or missing:**

| Tier | Source | When used |
|---|---|---|
| 1 | `result.json` in work_dir or `agent_runs/*/` | Fallback metric source |
| 2 | `extraction_failed` | No valid metric found — hypothesis is treated as **failed** |

**Important:** A hypothesis with `measurement_status="extraction_failed"` is treated as an invalid experiment — it cannot count as "improved" even if the baseline value happens to produce `delta > 0`. If *all* hypotheses in a cycle fail measurement, Stage 8 marks the entire cycle as **FAILED** with an "INVALID CYCLE" recommendation.

### 6.3 Regression guard (optional but recommended)

A shell script that exits 0 if a result passes quality checks, non-zero if it regresses. The guard receives the path to a **canonical `result.json`** as its first argument — this file is always written after metric extraction (from any tier), so the guard never inspects a stale or missing file. **When configured, the guard is fail-closed:** if the script is missing, times out, or throws an error, AHVS treats the guard as failed and rejects the hypothesis.

```bash
# .ahvs/regression_guard.sh
#!/bin/bash
RESULT=$(jq '.answer_relevance' "$1")
# Fail if more than 5% below baseline
python -c "import sys; sys.exit(0 if float('$RESULT') >= 0.70 else 1)"
```

Pass it to AHVS:
```bash
ahvs --repo . --question "..." \
  --regression-guard .ahvs/regression_guard.sh
```

### 6.4 Domain context (automatic)

AHVS infers domain tags (`llm`, `rag`, `ml`, `prompt-driven`) by scanning `requirements.txt`, `pyproject.toml`, and `package.json`. These tags guide hypothesis generation without any manual setup.

---

## 7. CLI Reference

### Genesis — bootstrap a new project from data

```bash
ahvs genesis \
  --problem "Classify customer emails into intent categories" \
  --data /path/to/emails.csv \
  --target-metric f1_weighted \
  --output-dir /path/to/new_project \
  --mode pipeline
```

| Flag | Default | Description |
|---|---|---|
| `--problem`, `-p` | *(required)* | Natural language problem description |
| `--data`, `-d` | *(required)* | Path to input data file (CSV, TSV, Parquet) |
| `--output-dir`, `-o` | *(required)* | Output directory — never auto-generated |
| `--target-metric`, `-m` | `f1_weighted` | Metric to optimize |
| `--mode` | `pipeline` | `pipeline` (deterministic) or `agent` (KD Agent via claude-agent-sdk, smarter) |
| `--solver`, `-s` | *(auto-detect)* | Solver name (e.g. `kd_classifier`) |
| `--classes` | *(none)* | Classification classes (e.g. `--classes pos neg neutral`) |
| `--input-column` | `text` | Text column name in the data file |
| `--annotation-model` | `gpt-4.1-mini` | LLM for annotation (never gpt-4o) |
| `--solver-registry` | *(built-in)* | Path to custom solvers.yaml |

Genesis creates a project with `.ahvs/baseline_metric.json`, registers it in `~/.ahvs/registry.json`, and git-inits the directory. Use `/ahvs_genesis` in Claude Code for an interactive walkthrough.

**Step-by-step workflow:**

```bash
# Step 1: Genesis creates a new project
ahvs genesis -p "sentiment classification" -d emails.csv -o /tmp/my_classifier

# Step 2: Inspect the result
cat /tmp/my_classifier/.ahvs/baseline_metric.json

# Step 3: Run AHVS optimization (when ready)
ahvs --repo my_classifier --question "improve f1_weighted to 0.85" --domain ml
```

### Cycle — run an AHVS optimization cycle

```
ahvs [options]
```

| Flag | Default | Description |
|---|---|---|
| `--repo`, `-r` | *(required)* | Path or registered short name of target repository (see `--list-repos`) |
| `--list-repos` | off | List all registered AHVS repos and exit |
| `--unregister` | none | Remove a repo from the registry by name and exit |
| `--question`, `-q` | *(required)* | The cycle question (what to improve) |
| `--max-hypotheses` | `3` | How many hypotheses to generate (max 5) |
| `--max-lesson-cycles` | `5` | Load lessons from last K recent non-failed cycle IDs (0 = unlimited) |
| `--auto-approve` | off | Skip interactive gate; run all hypotheses |
| `--selection` | none | Pre-specify hypotheses to run (e.g. `H1,H3`). For conversational/agent-driven mode. |
| `--from-stage` | *(stage 1)* | Resume from a specific stage name |
| `--until-stage` | *(last stage)* | Stop after this stage (e.g. `AHVS_HYPOTHESIS_GEN`). Useful for split workflows — generate hypotheses, select via GUI, then resume. |
| `--resume` | off | Resume from last written checkpoint |
| `--regression-guard` | none | Path to regression guard shell script |
| `--domain` | *(none)* | Domain pack: `llm` (default) or `ml` (traditional ML). Sets prompts + skills automatically. |
| `--skill-registry` | none | Path to custom skill registry YAML |
| `--prompts` | none | Path to AHVS prompts override YAML |
| `--model` | `claude-opus-4-6` | LLM model ID |
| `--api-key-env` | `ANTHROPIC_API_KEY` | Env var holding the API key |
| `--base-url` | *(per provider)* | Override LLM base URL (required for `openai-compatible`) |
| `--provider` | `anthropic` | LLM provider: `anthropic`, `openai`, `openai-compatible`, `openrouter`, `deepseek`, `acp` |
| `--acp-agent` | `claude` | ACP agent CLI name (only with `--provider acp`) |
| `--acpx-command` | *(auto-detect)* | Path to acpx binary (only with `--provider acp`) |
| `--acp-session-name` | `ahvs` | ACP session name (only with `--provider acp`) |
| `--acp-timeout` | `1800` | ACP per-prompt timeout in seconds (only with `--provider acp`) |
| `--apply-best` | off | Auto-apply best improving hypothesis patch and update baseline |
| `--run-dir` | `<repo>/.ahvs/cycles/<ts>` | Override cycle output directory |
| `--no-cache` | off | Disable LLM response cache (also controllable via `LLM_CACHE_ENABLED=false`) |

### Split workflow: generate -> GUI select -> execute

Use `--until-stage` and `--from-stage` together for a human-in-the-loop workflow:

```bash
# Step 1: Generate hypotheses only
ahvs --repo . --question "..." \
  --until-stage AHVS_HYPOTHESIS_GEN

# Step 2: Open browser GUI to select hypotheses
python -m ahvs.hypothesis_selector \
  .ahvs/cycles/<timestamp>

# Step 3: Resume from selection (selection.json already written by GUI)
ahvs --repo . --question "..." \
  --from-stage AHVS_HUMAN_SELECTION \
  --run-dir .ahvs/cycles/<timestamp>
```

This is particularly useful when running AHVS via multi-agent teams (see [Multi-Agent Execution](#multi-agent-execution-with-claude-code-agent-teams)).

### Resuming a failed cycle

```bash
# From a specific stage:
ahvs --repo . --question "..." \
  --from-stage AHVS_HYPOTHESIS_GEN

# From last checkpoint (auto-detects latest cycle under .ahvs/cycles/):
ahvs --repo . --question "..." --resume

# Or from a specific cycle directory:
ahvs --repo . --question "..." \
  --run-dir .ahvs/cycles/20260318_120000 \
  --resume
```

### Non-interactive / CI mode

```bash
ahvs \
  --repo . \
  --question "Can we reduce latency while keeping answer_relevance above 0.74?" \
  --auto-approve \
  --max-hypotheses 2
```

---

## 8. Hypothesis Types

AHVS generates hypotheses of these types. Each type controls what instructions, constraints, and package hints Claude Code receives:

| Type | Focus area | Required tools |
|---|---|---|
| `prompt_rewrite` | System prompts, few-shot examples, instruction wording | `promptfoo` |
| `model_comparison` | Swap model IDs, compare across providers | `promptfoo` |
| `config_change` | Hyperparameters (temperature, chunk_size, top_k) | `promptfoo` |
| `dspy_optimize` | DSPy modules for programmatic prompt optimization | `dspy`, `promptfoo` |
| `code_change` | Algorithms, retrieval logic, ranking functions, data processing | *(none)* |
| `architecture_change` | Pipeline redesign, new components, caching, re-rankers | *(none)* |
| `multi_llm_judge` | Multi-model consensus or judge-based evaluation | *(none)* |
| `phoenix_eval` | Arize Phoenix evaluation | `arize-phoenix` |

### How types affect execution

All hypothesis types share the same execution engine: **Claude Code CLI**. There is no separate executor per type. Instead, each type injects **type-specific execution strategies** (defined in `_TYPE_EXECUTION_STRATEGIES` in `executor.py`) that shape Claude Code's behaviour:

- **System instructions** — type-specific guidance appended to the prompt (e.g., `code_change` is told to focus on algorithms and retrieval logic, not prompt wording)
- **Constraints** — hard boundaries on what the type may modify (e.g., `prompt_rewrite` is scoped to prompt template files only)
- **Success guidance** — how to measure whether the change achieved its goal
- **Package hints** — which pip packages and tools to prefer

This means the taxonomy controls *what Claude Code is instructed to do*, not *which runtime engine runs*. A `code_change` hypothesis produces genuinely different code than a `prompt_rewrite` because Claude Code receives different instructions, constraints, and success criteria.

### Pre-flight tool checks

AHVS runs a secondary pre-flight check *after* hypothesis selection (Stage 4) to verify the tools needed for selected hypothesis types are available. This check skips the LLM connectivity test (already verified at Stage 1) and focuses only on tool availability. If a tool is missing, AHVS warns and asks for confirmation before proceeding.

### Domain packs: using AHVS with traditional ML models

By default, AHVS is tuned for LLM/RAG optimization. To use it with traditional ML models (classifiers, regressors, ranking models), use the `--domain ml` flag:

```bash
ahvs \
  --repo /path/to/my-classifier \
  --question "Improve F1 score on the churn prediction model" \
  --domain ml \
  --max-hypotheses 3
```

This loads the ML domain pack, which:
- **Overrides hypothesis generation prompts** — focuses on feature engineering, hyperparameter tuning, algorithm selection, sampling strategies, and pipeline architecture instead of LLM prompts and retrieval
- **Restricts hypothesis types** to `code_change`, `config_change`, and `architecture_change` (excludes LLM-specific types like `prompt_rewrite`, `dspy_optimize`, etc.)
- **Provides ML-specific skill templates** — `sklearn_eval`, `hyperparameter_sweep`, `feature_importance`, `confusion_matrix`, `cross_validate`

The domain pack files live in `ahvs/domain_packs/`:
- `ml_prompts.yaml` — hypothesis generation and validation plan prompts
- `ml_skills.yaml` — ML tool descriptions for Claude Code's context

You can also use them explicitly:
```bash
ahvs --repo . --question "..." \
  --prompts ahvs/domain_packs/ml_prompts.yaml \
  --skill-registry ahvs/domain_packs/ml_skills.yaml
```

The core pipeline (worktrees, eval_command, metric extraction, cross-cycle memory) works identically for ML and LLM targets. Only the hypothesis generation and skill context differ.

---

## 9. Skill Library

Skills are pre-built guidance templates injected into Claude Code's prompt context. Claude Code reads them, picks the right approach for the hypothesis type, and invokes the underlying tools directly in its generated code. Skills are **purely advisory** — they describe what tools are available and how to use them, but AHVS does not enforce, dispatch, or resolve skill invocations at runtime. The `skill_planned` field in `HypothesisResult` reflects the plan's declared skill, not a runtime observation.

### Built-in skills

| Skill | Applicable types |
|---|---|
| `promptfoo_eval` | `prompt_rewrite`, `model_comparison`, `config_change` |
| `dspy_compile` | `dspy_optimize` |
| `phoenix_eval` | `prompt_rewrite`, `model_comparison`, `config_change`, `code_change` |
| `local_run` | `code_change`, `architecture_change`, `multi_llm_judge` |
| `regression_guard` | all types |
| `metric_capture` | all types |

### Adding custom skills

Create a YAML file and pass it with `--skill-registry`:

```yaml
# my_skills.yaml
skills:
  - name: my_custom_eval
    description: >
      Run our internal evaluation harness at scripts/eval.py.
      Reads from data/test.jsonl and writes {"answer_relevance": <float>} to stdout.
    invocation_template: |
      SKILL: my_custom_eval
        entry_point: scripts/eval.py
        dataset: data/test.jsonl
        output_path: tool_runs/{hypothesis_id}/result.json
    applicable_types:
      - prompt_rewrite
      - code_change
    required_tools:
      - python
```

---

## 10. Cross-Cycle Memory

> **Full reference:** [docs/memory_management_system.md](docs/memory_management_system.md) covers the complete memory architecture including the LessonEntry schema, all weight boosts, compaction algorithms, and configuration reference.

AHVS uses a **three-tier memory model** to persist and leverage lessons across cycles:

| Tier | Location | Purpose |
|------|----------|---------|
| **Evolution Store** | `<repo>/.ahvs/evolution/lessons.jsonl` | Machine-queryable lessons fed into hypothesis generation |
| **Friction Logs** | `<repo>/.ahvs/cycles/<id>/friction_log.md` | Per-cycle execution errors and operator notes |
| **Session Memory** | `<repo>/.ahvs/memory/` | Human-readable cross-session summaries |

All memory lives in the **target repository**, making it portable across machines.

### How lessons are recorded

- **Successful hypotheses** are recorded as positive lessons — future cycles know what worked
- **Failed attempts** (measured but not improved) are marked as rejected approaches — the LLM is explicitly told not to repeat them
- **Infrastructure failures** (`extraction_failed`, `sandbox_error`) are recorded as warnings noting the hypothesis was never actually measured — they do *not* count as rejected approaches, so the idea can be retried in future cycles
- **Errors** during execution are logged as warnings with enough context to diagnose

Each lesson carries **structured outcome fields** (`hypothesis_type`, `metric_name`, `metric_delta`, `eval_method`, etc.) alongside the description text. These enable quantitative analysis, semantic deduplication, and cross-project learning while remaining backward-compatible with older JSONL entries.

### Eager writes and crash safety

Each hypothesis result is written to `lessons.jsonl` immediately after measurement (Stage 6), with `cycle_status="partial"`. This ensures lessons survive even if the cycle crashes before Stage 7. When Stage 7 runs successfully, it writes the same lessons with `cycle_status="complete"`. The next compaction pass deduplicates and upgrades partial entries to complete.

### Memory management

The `--max-lesson-cycles K` flag (default: 5) caps how many recent non-failed cycle IDs feed into hypothesis generation (partial-cycle lessons are included). Set to 0 for unlimited (time-decay with 30-day half-life still applies; lessons older than 90 days are expired).

**Automatic housekeeping** runs at the start of every new cycle:

1. **Cycle directory cleanup** — Removes stale dirs (stage-1 failures, orphans, partial cycles, old complete cycles beyond the 3-cycle retention window)
2. **Lesson compaction** — Two-phase deduplication:
   - *Exact dedup:* Collapses partial/complete pairs within a cycle (keeps complete)
   - *Semantic dedup:* Collapses paraphrased lessons across cycles using a fingerprint of `(hypothesis_type, metric_name, rounded delta)`. Keeps the best entry per group (highest severity, most recent, most structured data)
3. **Friction log summarization** — Extracts recurring error patterns and operator notes from retained friction logs, writes a consolidated `friction_summary.md`
4. **Memory file lifecycle** — Marks files in `.ahvs/memory/` older than 60 days as `[STALE]`; archives files older than 120 days to `memory/archive/`

### Verification feedback

After Stage 8 determines which hypothesis to keep or revert, it **feeds that decision back** into the evolution store. Lessons from the kept hypothesis are marked `verified="kept"` (1.5x weight boost in future queries); all others are marked `verified="reverted"` (0.5x penalty). This creates a closed feedback loop where proven improvements carry more weight.

### Historical digest

Beyond the 12 raw recent lessons, AHVS builds a **historical digest** — aggregated statistics from older cycles grouped by hypothesis type (total attempts, improvement rate, average/best delta, kept/reverted counts). This compact summary is injected into the hypothesis generation prompt, allowing the LLM to reason about long-term patterns without consuming raw lesson context slots.

### Cross-project learning

A **GlobalEvolutionStore** at `~/.ahvs/global/evolution/` enables framework-level insights to transfer across repos. At the end of each successful cycle, qualifying lessons are promoted to the global store:

- **SYSTEM** and **PIPELINE** category lessons (infrastructure issues, not repo-specific)
- **Verified-kept** lessons (proven improvements, transferable patterns)

During Stage 2, local lessons are loaded first (up to 12); global lessons only fill any remaining capacity (up to 3, excluding the current repo) and are deduplicated against local lessons using semantic fingerprints. This can be disabled with `enable_cross_project=False`.

At Stage 2 (`AHVS_CONTEXT_LOAD`), AHVS queries the top 12 lessons from the store using the stage name `"ahvs_execution"` — the same name used when writing lessons at Stage 7. This ensures the EvolutionStore's 2x relevance boost applies correctly to AHVS-specific lessons. Lessons with severity `"info"` are treated as positive outcomes; those with `"warning"` or `"error"` are surfaced as rejected approaches. The LLM sees both what has worked and what has been ruled out, producing increasingly targeted hypotheses over time.

### Repo registry

AHVS maintains a **global repo registry** at `~/.ahvs/registry.json` that maps short names to filesystem paths. This enables the CLI to accept `--repo autoresearch` instead of requiring a full path, and ensures future sessions can reliably locate onboarded repos without filesystem searching.

The registry is updated automatically:
- **During onboarding** — the `ahvs_onboarding` skill registers the repo after writing `baseline_metric.json`
- **After each cycle** — `runner.py` writes the `last_cycle` ID back to the registry entry

```bash
# List all registered repos
ahvs --list-repos

# Run a cycle using a short name
ahvs --repo autoresearch --question "How can we improve test_accuracy?"

# Remove a repo from the registry
ahvs --unregister old-project
```

Each entry stores the repo path, primary metric, baseline value, onboarding timestamp, and the last cycle ID. The registry is user-local (`~/.ahvs/`), not checked into any repo.

### Enriched onboarding context

Stage 2 also forwards enriched fields from `baseline_metric.json` into the hypothesis-generation prompt. These fields — `optimization_goal`, `regression_floor`, `constraints`, `system_levers`, `prior_experiments`, `notes` — are written during onboarding (see [Section 6.1](#61-baseline-metric-file)) and appear in the prompt under "Operator Context". This gives the LLM richer intent signals without additional inference calls.

### Hypothesis output format

Stage 3 accepts hypotheses in **either** structured JSON or markdown format. When the LLM returns a JSON array of hypothesis objects (with `id`, `type`, `description` fields), AHVS parses it directly with schema validation. When the LLM returns markdown (the `## H1` / `**Type:**` format), AHVS falls back to regex parsing. The same dual-format support applies to selection and validation plan outputs. This improves reliability while maintaining full backward compatibility.

### Eval-mode warnings

Stage 6 detects incompatible hypothesis-type/eval-command combinations. If a `prompt_rewrite` or `model_comparison` hypothesis is paired with an eval command containing `--eval-only`, AHVS prints a visible warning explaining that changes to prompts/models have no measurable effect when the eval pipeline reads frozen checkpoint data. This prevents the "all hypotheses at baseline" failure mode discovered during initial production cycles.

---

## 11. Python API

```python
from ahvs import AHVSConfig, execute_ahvs_cycle

config = AHVSConfig(
    repo_path="/path/to/your-rag-project",
    question="How can we improve answer_relevance by 5%?",
    max_hypotheses=3,
    llm_model="claude-opus-4-6",
    llm_api_key_env="ANTHROPIC_API_KEY",
)

results = execute_ahvs_cycle(config, auto_approve=True)

for r in results:
    print(r.stage.name, r.status.value)
```

### Stopping early with until_stage

```python
from ahvs import AHVSConfig, execute_ahvs_cycle
from ahvs.stages import AHVSStage

config = AHVSConfig(repo_path="/path/to/repo", question="...")

# Generate hypotheses only — stop before human selection
results = execute_ahvs_cycle(
    config,
    until_stage=AHVSStage.AHVS_HYPOTHESIS_GEN,
)

# Later: resume from human selection (after GUI or manual selection.json)
results = execute_ahvs_cycle(
    config,
    from_stage=AHVSStage.AHVS_HUMAN_SELECTION,
)
```

### Resuming from checkpoint

```python
from pathlib import Path
from ahvs import AHVSConfig, execute_ahvs_cycle, read_ahvs_checkpoint
from ahvs.stages import AHVS_NEXT_STAGE

cycle_dir = Path("/path/to/repo/.ahvs/cycles/20260318_120000")
config = AHVSConfig(
    repo_path="/path/to/repo",
    question="...",
    run_dir=cycle_dir,
)

last_done = read_ahvs_checkpoint(cycle_dir)
from_stage = AHVS_NEXT_STAGE.get(last_done) if last_done else None

results = execute_ahvs_cycle(config, from_stage=from_stage, auto_approve=True)
```

### Stage completion callback

```python
def on_done(stage_result):
    print(f"Stage {stage_result.stage.name}: {stage_result.status.value}")

results = execute_ahvs_cycle(config, on_stage_complete=on_done)
```

### Reading results programmatically

```python
from ahvs.result import load_results

results = load_results(cycle_dir / "results.json")
for r in results:
    print(f"{r.hypothesis_id}: delta={r.delta:+.4f} improved={r.improved}")
```

### Result Fields: `measurement_status`

Each `HypothesisResult` includes a `measurement_status` field that tracks whether the metric was successfully captured. This prevents silent fallback to the baseline value when measurement fails.

| Value | Meaning |
|---|---|
| `"measured"` | Metric was successfully extracted from at least one source |
| `"extraction_failed"` | Hypothesis ran but no metric could be parsed from any source — treated as a **failed** hypothesis (cannot count as improved) |
| `"sandbox_error"` | Hypothesis execution raised an exception before metric extraction (legacy name retained for compatibility) |
| `"not_executed"` | Hypothesis was not executed (default state) |

**Metric extraction policy** (see [Section 6.2](#62-eval-command) for full details):

When `eval_command` is configured, it is the **only trusted measurement source**. When `eval_command` is not configured, AHVS falls back to `result.json` in the work directory. If no source produces a valid metric, `measurement_status` is set to `"extraction_failed"`.

**Diagnosing `extraction_failed`:**
- Check `tool_runs/<ID>/` for generated files — did Claude Code produce code?
- Check the eval_command stderr in the logs — did the eval crash?
- For GPU-heavy evals, confirm the machine had enough free VRAM before the run. AHVS cleans up timed-out/crashed eval workers, but external processes can still block a hypothesis from starting or finishing.
- Ensure the hypothesis code writes the metric in a parseable format (JSON or `key: value`)

---

## 12. Configuration Reference

### `AHVSConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `repo_path` | `Path` | *(required)* | Root of target repository |
| `question` | `str` | *(required)* | Cycle question |
| `run_dir` | `Path` | `<repo>/.ahvs/cycles/<ts>` | Cycle output directory |
| `max_hypotheses` | `int` | `3` | Max hypotheses to generate (hard cap: 5) |
| `max_lesson_cycles` | `int` | `5` | Load lessons from last K recent non-failed cycle IDs (0 = unlimited) |
| `regression_guard_path` | `Path \| None` | `None` | Path to regression guard script |
| `apply_best` | `bool` | `False` | Auto-apply best improving hypothesis patch after cycle |
| `skill_registry_path` | `Path \| None` | `None` | Custom skills YAML |
| `prompts_override_path` | `Path \| None` | `None` | AHVS prompts override YAML |
| `llm_provider` | `str` | `"anthropic"` | LLM provider (`anthropic`, `openai`, `openrouter`, `deepseek`, `acp`) |
| `llm_model` | `str` | `"claude-opus-4-6"` | LLM model ID |
| `llm_api_key_env` | `str` | `"ANTHROPIC_API_KEY"` | API key env var name |
| `llm_base_url` | `str` | `""` | Override LLM base URL |
| `llm_api_key` | `str` | `""` | Inline API key (prefer env var) |
| `acp_agent` | `str` | `"claude"` | ACP agent CLI name (only with `provider=acp`) |
| `acp_cwd` | `str` | `"."` | ACP working directory (resolved to `repo_path`) |
| `acpx_command` | `str` | `""` | Path to acpx binary (auto-detect if empty) |
| `acp_session_name` | `str` | `"ahvs"` | ACP session name |
| `acp_timeout_sec` | `int` | `1800` | ACP per-prompt timeout in seconds |
| `enable_cross_project` | `bool` | `True` | Enable global cross-project lesson sharing |
| `global_evolution_dir` | `Path` | `~/.ahvs/global/evolution` | Path to global evolution store |
| `eval_timeout_sec` | `int` | `600` | Timeout for eval_command in seconds (overridden by `eval_timeout` in baseline_metric.json) |

### Derived paths (automatic)

| Path | Location |
|---|---|
| Baseline metric | `<repo>/.ahvs/baseline_metric.json` |
| EvolutionStore | `<repo>/.ahvs/evolution/` |
| Cycle artifacts | `<repo>/.ahvs/cycles/<timestamp>/` |
| Session memory | `<repo>/.ahvs/memory/` |

### Custom prompts override

To override any AHVS stage prompt without touching Python source, create a YAML file and pass it with `--prompts`:

```yaml
# ahvs_prompts.yaml
stages:
  ahvs_hypothesis_gen:
    system: >
      You are a specialist in RAG pipeline optimisation.
      Focus exclusively on retrieval-side improvements.
    max_tokens: 3000
  ahvs_report:
    max_tokens: 2000
```

Only the fields you specify are overridden; unspecified fields retain their defaults.

---

## 13. Directory Layout

### Package structure

```
ahvs/
├── __init__.py              # Public API: execute_ahvs_cycle, AHVSConfig, ...
├── stages.py                # AHVSStage IntEnum, AHVS_STAGE_SEQUENCE, gate rules
├── contracts.py             # AHVSStageContract (input/output/DoD per stage)
├── config.py                # AHVSConfig dataclass
├── result.py                # HypothesisResult — tool-agnostic output contract
├── context_loader.py        # load_context_bundle() — baseline + EvolutionStore
├── health.py                # Pre-flight checks (tools, baseline, guard, branch)
├── skills.py                # SkillLibrary + 6 built-in skills
├── prompts.py               # AHVSPromptManager (3 stage prompts + YAML overrides)
├── worktree.py              # HypothesisWorktree — git worktree lifecycle + AST splice
├── hypothesis_selector.py   # Browser-based GUI for human hypothesis selection
├── executor.py              # 8 stage handlers + execute_ahvs_stage() dispatcher
├── runner.py                # execute_ahvs_cycle() — outer orchestration loop
├── evolution.py             # EvolutionStore — cross-cycle memory persistence
├── registry.py              # Repo registry (~/.ahvs/registry.json)
├── cli.py                   # CLI entry point (ahvs command + ahvs genesis)
├── genesis/                 # Genesis — solver-based project bootstrapping
│   ├── __init__.py          # Exports: GenesisResult, Solver, SolverRegistry, ProblemRouter
│   ├── contract.py          # GenesisResult dataclass + Solver protocol
│   ├── registry.py          # YAML-driven SolverRegistry
│   ├── router.py            # ProblemRouter (problem description → solver)
│   └── solvers/
│       ├── kd_classifier.py # KD adapter (pipeline + agent modes)
│       └── solvers.yaml     # Solver registry config
├── domain_packs/            # Domain-specific prompt + skill overrides
│   ├── ml_prompts.yaml      # Traditional ML hypothesis prompts (--domain ml)
│   └── ml_skills.yaml       # ML skill templates (sklearn, optuna, etc.)
├── llm/                     # LLM client factory + response cache
│   ├── __init__.py
│   ├── client.py            # Provider-agnostic LLM client
│   ├── anthropic_adapter.py # Anthropic API adapter
│   ├── acp_client.py        # ACP (Agent Communication Protocol) client
│   └── cache.py             # Content-addressable LLM response cache (SQLite, SHA-256)
└── utils/                   # Shared utilities
    ├── __init__.py
    └── thinking_tags.py     # LLM thinking-tag parsing

.claude/skills/ahvs_onboarding/    # Claude Code onboarding skill
├── SKILL.md                       # Conversational wizard: repo -> .ahvs/baseline_metric.json
└── references/                    # Policy docs loaded as needed
    ├── artifact_contract.md       # Baseline JSON schema
    ├── eval_command_policy.md     # Eval command acceptance rules
    └── git_mode_policy.md         # Git vs non-git trust model

.claude/skills/ahvs_multiagent/    # Claude Code multi-agent execution skill
├── SKILL.md                       # 5-phase flow: gen -> GUI -> team -> loop -> archive
└── references/
    ├── agent_prompts.md           # Executor + observer prompts with placeholders
    └── failure_classification.md  # FRAMEWORK_BUG / HYPOTHESIS_MISS / AMBIGUOUS rules

.claude/skills/ahvs_genesis/       # Claude Code genesis skill
└── SKILL.md                       # Interactive wizard: data -> project with baseline
```

### Per-repo `.ahvs/` directory

```
<repo>/.ahvs/
├── baseline_metric.json         # Required: baseline metric snapshot
├── evolution/                   # EvolutionStore: cumulative lessons across cycles
├── memory/                     # Project-specific session memory (portable across machines)
│   ├── INDEX.md                # One-line index of all memory files
│   ├── session_20260319.md     # Session summaries: what ran, what broke, what was learned
│   ├── bug_report_example.md   # Bug reports with root cause and fix details
│   └── ...
└── cycles/                         # Auto-cleaned: keeps last 3 complete cycles
    └── 20260318_120000/         # One directory per cycle run
        ├── ahvs_checkpoint.json      # Stage resumption checkpoint
        ├── cycle_manifest.json       # Cycle metadata + preflight results
        ├── context_bundle.json       # Stage 2 output: baseline + enriched context + lessons
        ├── hypotheses.md             # Stage 3 output: generated hypotheses
        ├── selection.md              # Stage 4 output: operator selection
        ├── selection.json            # Machine-readable selection
        ├── validation_plan.md        # Stage 5 output: per-hypothesis plans
        ├── results.json              # Stage 6 output: HypothesisResult list
        ├── report.md                 # Stage 7 output: LLM cycle report
        ├── friction_log.md           # Stage 7 output: errors + measurement issues + operator notes
        ├── cycle_summary.json        # Stage 8 output: keep/revert + worktree/patch refs
        ├── worktrees/
        │   └── H1/                   # Git worktree for hypothesis H1 (kept if best)
        └── tool_runs/
            ├── H1/                   # Claude Code workspace for hypothesis H1
            │   ├── result.json       # Canonical metric result (written after any-tier extraction)
            │   ├── H1.patch          # Diff of all changes applied to worktree
            │   └── <generated files>
            └── H2/
```

### User-local `~/.ahvs/` directory

```
~/.ahvs/
├── registry.json               # Repo registry: name→path mapping (auto-updated)
└── global/
    └── evolution/              # GlobalEvolutionStore: cross-project lessons
        └── lessons.jsonl
```

The registry is updated during onboarding and after each cycle. It is user-local and never checked into any repo.

### Memory model

AHVS uses a three-tier persistence model. All tiers write to the **target repo**, not to the AHVS package directory or Claude's machine-local storage. This ensures memory is portable across machines and stays with the project it describes.

| Tier | Location | Purpose | Written by |
|---|---|---|---|
| **Session memory** | `<repo>/.ahvs/memory/` | Human-readable session summaries, bug reports, and cross-session lessons | Team lead / observer agent |
| **Friction log** | `<repo>/.ahvs/cycles/<id>/friction_log.md` | Per-cycle operator notes, errors, measurement issues | Executor (auto) + observer |
| **Evolution lessons** | `<repo>/.ahvs/evolution/lessons.jsonl` | Machine-readable JSON lines fed into the next cycle's context loader (Stage 2) | Observer agent |

**Session memory** (`<repo>/.ahvs/memory/`) contains:
- Session records: what hypotheses ran, what improved, what bugs were found
- Bug reports: root cause, fix details, files changed
- Lessons that span multiple cycles (e.g., "prompt_rewrite hypotheses are unmeasurable with --eval-only")
- An `INDEX.md` file indexing all memory files with one-line descriptions

When AHVS agents start a cycle, they read `<repo>/.ahvs/memory/` to recall prior session context. When they find a bug or lesson, they write to it immediately — not at end of session.

---

## 14. Advanced Usage

### Running multiple cycles in sequence

```bash
for i in 1 2 3; do
  ahvs \
    --repo /path/to/project \
    --question "How can we further improve answer_relevance?" \
    --auto-approve \
    --max-hypotheses 2
  echo "--- Cycle $i complete ---"
done
```

Each cycle reads the lessons left by the previous one, building progressively more targeted hypotheses.

### Using with OpenAI or OpenRouter

```bash
export OPENAI_API_KEY=sk-...
ahvs \
  --repo . \
  --question "..." \
  --provider openai \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY
```

For OpenRouter:
```bash
export OPENROUTER_API_KEY=...
ahvs \
  --repo . \
  --question "..." \
  --provider openrouter \
  --model anthropic/claude-opus-4-6 \
  --api-key-env OPENROUTER_API_KEY
```

### Inspecting what Claude Code generated

Every hypothesis has its own workspace under `tool_runs/<ID>/`. You can inspect:

- The files Claude Code generated
- `result.json` — the numeric metric output
- Any intermediate artifacts from Promptfoo/DSPy/custom scripts

### Keeping a successful hypothesis

**Automatic promotion (recommended):**

Use `--apply-best` to let AHVS apply the winning patch and update the baseline in one step:

```bash
ahvs \
  --repo . --question "..." \
  --auto-approve --apply-best
```

When the cycle completes successfully and a hypothesis improved the metric, AHVS will:
1. Run `git apply` with the best hypothesis patch on your working tree
2. Update `.ahvs/baseline_metric.json` with the new metric value and provenance fields:
   - `commit` — the pre-patch HEAD (the patch is applied but not committed)
   - `commit_note` — explains that the working tree includes an uncommitted patch
   - `applied_patch` — relative path to the `.patch` file
   - `applied_hypothesis` — ID of the winning hypothesis
   - `applied_from_cycle` — cycle directory name
3. Print a confirmation of what was applied

> **Note:** The recorded `commit` is the pre-patch HEAD because `--apply-best` does not auto-commit. Commit the changes yourself to make the baseline provenance fully clean.

**Manual promotion:**

If you prefer to review before applying, inspect `cycle_summary.json` and apply manually:

```bash
# Apply the patch
git apply <cycle_dir>/tool_runs/H1/H1.patch
```

The `cycle_summary.json` includes:
- `kept_worktree`: path to the git worktree of the best hypothesis (if one improved)
- `kept_patch`: path to its `.patch` file (relative to cycle_dir)
- `all_patches`: list of `.patch` paths for every hypothesis (audit trail)
- `per_hypothesis`: per-hypothesis details including `execution_mode`
- `all_unmeasured`: `true` if no hypothesis produced a valid measurement (cycle is invalid)

After manual application, update the baseline yourself or let `--apply-best` handle it on the next run.

### Model selection strategy

AHVS makes LLM calls at three points in the cycle, each with different quality requirements:

| Call site | What it does | Model recommendation |
|-----------|-------------|---------------------|
| **Hypothesis generation** (Stage 3) | Proposes concrete, diverse improvement strategies | Use the **strongest reasoning model** available (e.g. `claude-opus-4-6`, `o3`). This is the highest-leverage call — a weak model here produces shallow, repetitive hypotheses that waste the entire cycle budget. |
| **Validation planning** (Stage 5) | Writes implementation steps and eval criteria | Same strong model. The plan quality directly determines whether Claude Code produces useful code. |
| **Claude Code execution** (Stage 6) | Generates and runs implementation code | Strong model again. This is where real code is written — algorithms, retrieval pipelines, evaluation harnesses. A capable coding model produces implementations that actually test the hypothesis rather than trivial wrappers. |
| **Report writing** (Stage 7) | Summarizes results and extracts lessons | A lighter model is acceptable here (e.g. `claude-sonnet-4-6`, `gpt-4o-mini`) since it only summarizes data that already exists. |

**The general principle:** Invest in intelligence where it has multiplicative impact — hypothesis quality and code quality compound across cycles. A single good hypothesis from a strong model is worth more than five shallow ones from a cheap model.

### Budget planning for experiments

Before running AHVS cycles, estimate your costs:

**Per-cycle cost breakdown (typical):**

| Component | Token usage | Notes |
|-----------|------------|-------|
| Hypothesis generation | ~2K input + ~2K output | One LLM call |
| Validation planning | ~3K input + ~2.5K output | One LLM call |
| Claude Code execution | ~10-50K per hypothesis | Depends on complexity; `code_change` and `architecture_change` types use more tokens than `prompt_rewrite` |
| Report writing | ~3K input + ~1.5K output | One LLM call |
| **Total per cycle** | **~20-70K tokens** (with 2-3 hypotheses) | |

**Cost control levers:**

- `--max-hypotheses 1-2` for exploratory cycles; save `3-5` for when you have high-confidence directions
- Start with `prompt_rewrite` and `config_change` hypotheses (cheaper, faster) before moving to `code_change` and `architecture_change` (more tokens, longer runs)
- Use `--auto-approve` carefully — unattended cycles with 5 hypotheses can accumulate cost without human judgment on which ideas are worth testing
- Set up a regression guard early — it prevents wasted cycles on regressions before you inspect results

**Planning a multi-cycle experiment:**

| Phase | Cycles | Strategy |
|-------|--------|----------|
| **Exploration** (1-3 cycles) | 1-2 hypotheses per cycle | Cast a wide net: mix `prompt_rewrite`, `config_change`, `code_change`. Learn which levers move the metric. |
| **Exploitation** (3-5 cycles) | 2-3 hypotheses per cycle | Double down on the type that showed movement. Use prior lessons to refine. |
| **Diminishing returns** | Stop when delta < noise floor | If 2-3 consecutive cycles show no improvement, the metric may be saturated for this approach. Change the question or target a different metric. |

A typical improvement campaign runs 5-10 cycles. Budget accordingly — with a strong model at ~$15/M input tokens, a 5-cycle campaign with 3 hypotheses each costs roughly $5-15 in API calls.

### LLM response cache

AHVS caches LLM responses in a per-project SQLite database at `<repo>/.ahvs/.llm_cache/responses.db`. The cache key is a SHA-256 hash of the full call parameters (model, messages, system prompt, max_tokens, temperature), so any change in input produces a different key.

**What gets cached:** All 3 orchestration LLM calls (hypothesis generation, validation planning, report writing) are cached automatically. ACP calls are excluded because they maintain stateful sessions.

**When it helps most:** Multi-cycle campaigns where you re-run with the same question and baseline — hypothesis generation hits 100% cache on the prompt-identical calls. Downstream-only changes (new skill registry, different `--max-hypotheses`) that don't alter the LLM input also hit cache.

**Controls:**

| Control | Effect |
|---------|--------|
| `--no-cache` | Disable cache for this run |
| `LLM_CACHE_ENABLED=false` | Disable via environment variable |
| `LLM_CACHE_TTL_HOURS=168` | Auto-expire entries after N hours (default: no expiry) |
| `LLM_CACHE_STORE_MESSAGES=true` | Store input messages in cache (default: false for PII safety) |
| `rm -rf <repo>/.ahvs/.llm_cache/` | Clear cache entirely |

**Safety:** Only successful, non-truncated responses are cached (`finish_reason == "stop"` and non-empty content). Errors, rate-limit responses, and truncated outputs are never cached.

### Prompt engineering tips

- Keep `--question` specific and metric-anchored: *"Improve answer\_relevance from 0.74 to above 0.78 by improving the retrieval step"* generates better hypotheses than *"make the system better"*.
- Use `--max-hypotheses 2` for faster cycles during exploration; increase to 5 when you want broader coverage.
- The `--auto-approve` flag is safe for unattended runs but be sure your regression guard is set up — it prevents a bad hypothesis from being silently "improved".
- Reference concrete code paths in your question when possible: *"The chunking in `src/pipeline/splitter.py` uses fixed 512-token windows — can we improve answer\_relevance by switching to semantic chunking?"* gives Claude Code much better starting context.

### Multi-agent execution with Claude Code Agent Teams

AHVS can be orchestrated by a multi-agent team using Claude Code's Agent Teams feature. This enables a supervisory pattern where:

- **Team Lead** generates hypotheses and coordinates the cycle
- **Executor** (Sonnet) runs each hypothesis via the AHVS CLI
- **Observer** (Opus) verifies results, classifies failures, and fixes framework bugs

**To trigger multi-agent execution conversationally**, use the `ahvs_multiagent` skill (see `.claude/skills/ahvs_multiagent/SKILL.md`):

```
Run AHVS on /path/to/my-project with multi-agent supervision. 3 hypotheses.
```

The skill encodes the exact 5-phase flow so nothing is left to improvisation:

1. **Hypothesis generation** — Team Lead runs `--until-stage AHVS_HYPOTHESIS_GEN`, then opens the browser GUI for human selection (or auto-approves)
2. **Team setup** — Team Lead spawns executor (sonnet) and observer (opus) with full prompts
3. **Per-hypothesis loop** — Executor runs one hypothesis at a time; Observer verifies and fixes any framework bugs before the next hypothesis runs
4. **Archive** — Team Lead runs Stages 7-8, shuts down the team, and reports summary

#### Running multi-agent with tmux

The `ahvs_multiagent` skill handles everything — it spawns the executor and observer as subagents automatically via Claude Code's Agent Teams. You just need one Claude Code session. Use tmux so the session survives terminal disconnects and you can monitor progress:

```bash
# Start a tmux session
tmux new-session -s ahvs

# Activate your environment and launch Claude Code
conda activate py11
claude

# Inside Claude Code, say:
# "Run AHVS on /path/to/project with multi-agent supervision. 3 hypotheses."
```

The skill will:
1. Generate hypotheses (Stages 1-3)
2. Open the browser GUI for selection (or auto-approve)
3. Spawn executor + observer subagents automatically
4. Run each hypothesis with verification between them
5. Archive results and report summary

You can detach from the tmux session (`Ctrl-b d`) and re-attach later (`tmux attach -t ahvs`) — the cycle keeps running.

**tmux quick reference:**

| Shortcut | Action |
|---|---|
| `Ctrl-b d` | Detach (session keeps running in background) |
| `tmux attach -t ahvs` | Re-attach to session |
| `Ctrl-b [` | Scroll mode (navigate output history) |
| `q` | Exit scroll mode |

### Roadmap / TODOs

AHVS already has a strong generic execution contract: repo + baseline metric + `eval_command` + isolated worktrees. That foundation should stay stable. The next work falls into six categories.

#### UX & distribution *(priority)*

1. **CLI + GUI hypothesis add/insert/edit** *(priority)*
   Allow operators to manually add, insert, or edit hypotheses via CLI flags and the browser-based GUI before execution — not just select/deselect from LLM-generated candidates. This is the highest-priority UX improvement: it unblocks human-in-the-loop workflows where domain experts want to inject their own ideas alongside LLM-generated ones.
2. **Installable plugin / skills / installer**
   Develop an installable package (pip, brew, or standalone installer) with bundled skills so that others can install and use AHVS without cloning the repo. Expose as much functionality as possible through the installer — onboarding, cycle execution, results viewer.
3. **Package & distribution**
   pip extras, Docker image, one-command setup. The goal is `pip install ahvs` or `docker run ahvs` with zero manual configuration beyond a target repo path and eval command.
4. **Browser GUI for manual lesson/memory cleanup**
   Add an interactive browser-based GUI (similar to hypothesis selector) that lets the operator inspect, filter, and selectively delete individual lessons from `lessons.jsonl` and memory files from `.ahvs/memory/`. Useful for curating cross-cycle memory when automatic compaction is insufficient.

#### Execution & runtime

1. **Parallel hypothesis execution**
   Hypotheses currently run sequentially. Parallel execution is feasible — Claude Code already runs inside per-hypothesis worktrees (done), so two remaining items are needed:
   - Add `fcntl.flock` around `git worktree add/remove` to prevent metadata corruption
   - Use `concurrent.futures` or `asyncio.gather` over the hypothesis list
2. **Improve unattended throughput**
   `save_results` already supports merge-by-ID accumulation, so batch and unattended execution should benefit once parallelism is in place. Multi-agent supervised mode benefits less because the observer verifies between hypotheses.
3. **Jupyter notebook-style execution**
   When execution scripts encounter errors, the current approach reruns the entire script. A notebook-style execution model would allow fixing only the failing cell and resuming from that point — reducing iteration time and preserving expensive intermediate state (loaded models, processed data).
4. **Multi-agent decomposition**
   Decompose the current agent architecture into more agents with narrower responsibilities — each agent becomes smarter and faster at its specific task. For example, separate plan-validator, code-reviewer, and test-runner agents instead of a single monolithic executor. This improves both speed (parallel specialist agents) and quality (each agent is deeply focused).
5. ~~**LLM call deduplication / semantic cache**~~ **DONE** — Implemented in `ahvs/llm/cache.py`. Content-addressable SQLite cache with SHA-256 keys, WAL mode, TTL support, PII-safe defaults. Wraps all API-based LLM calls transparently via `CachedClientWrapper`. Disable with `--no-cache` or `LLM_CACHE_ENABLED=false`. See [LLM Response Cache](#llm-response-cache) for details.

#### Domain expansion

The `--domain` flag and YAML-based domain packs (`ahvs/domain_packs/`) provide the adapter mechanism. Two domain packs are available: `llm` (default, LLM/RAG optimization) and `ml` (traditional ML — classifiers, regressors, NLP, CV, time series, etc.). For specialized tool chains (e.g., Hugging Face Trainer, torchvision augmentation), use `--skill-registry` with a project-specific YAML. Remaining work:

1. **Multi-metric optimization** — Preserve the current primary-metric contract, but add first-class support for Pareto-optimal selection across multiple metrics (e.g., precision *and* recall, accuracy *and* latency)
2. **Complex algorithmic tasks** — Extend AHVS to narrative improvement algorithms, knowledge graph improvement procedures, and other algorithmic-style hypothesis validation. These are structurally similar to ML optimization but operate on graph structures and text quality metrics. Align with existing autonomous research tools like AI Scientist / AI Scholar for structured experimental workflows.
3. **Data analytics domain (exploratory)** — Investigate whether AHVS can drive data-focused hypothesis cycles: dataset selection, feature subset optimization, preprocessing pipeline tuning, data augmentation experiments. These fit the current loop well when there's a clear downstream metric (e.g., "which cleaning pipeline gives the best F1?"). Open-ended exploration (EDA, pattern discovery) would need a different output contract since AHVS currently requires a single numeric metric. A `data_prompts.yaml` pack would be the first step; structural changes only if prompt-level framing proves insufficient.
4. **End-to-end examples** — Add worked examples for non-RAG repos (text classification, regression, data preprocessing) so users can see the full onboarding → cycle → results flow

#### Platform integration

1. **Databricks integration**
   Large datasets live in the Databricks datalake and are impractical to download. AHVS needs to run natively on Databricks — via Databricks Asset Bundles or workflow jobs — so it can exploit datalake data directly on the platform for hypothesis execution and evaluation.
2. **Knowledge distillation integration**
   KD becomes a first-class AHVS hypothesis type (`knowledge_distillation`). AHVS wraps the KD pipeline, discovers optimal configurations (prompt strategy, model architecture, DSPy mode), and accumulates lessons across KD runs. See [Deep Analysis: AHVS × KD](Deep_Analysis_AHVS_KD.md) for the full integration architecture.
3. **MLOps export** — Push cycle results to W&B, MLflow, or Comet for experiment tracking, model registry integration, and team-wide visibility.
4. **GitHub CI integration** — PR-triggered AHVS cycles, eval-quality gates, Phoenix traces for observability.
5. **MCP server for AHVS** — Expose AHVS as a Model Context Protocol service so that other LLM agents and tools can invoke AHVS cycles, query the evolution store, and access lessons programmatically.

#### Intelligence & evolution

Several ideas from NousResearch's [Hermes Agent Self-Evolution](https://github.com/NousResearch/hermes-agent-self-evolution) pipeline (DSPy GEPA, LLM-as-judge scoring, synthetic eval generation) could strengthen AHVS:

1. **DSPy/GEPA prompt evolution** — Instead of one-shot LLM rewrites, use DSPy GEPA to iteratively evolve prompts/configs over N generations with trace-informed mutations. This gives structured exploration of the prompt space rather than single-sample guesses. Integrate as an optional `--optimizer gepa` flag on prompt-type hypotheses.
2. **LLM-as-judge secondary evaluator** — When `eval_command` is slow or expensive (e.g., full RAG pipeline runs), use a fast LLM-judge heuristic for intermediate hypothesis filtering, reserving the real eval for final holdout comparison. This could dramatically reduce eval cost per cycle.
3. **AutoResearch** — Inject literature-grounded priors into hypothesis generation so cycles start from published best practices rather than cold-start guesses.
4. **Synthetic eval dataset generation** — For repos that lack a formal eval harness, auto-generate train/val/holdout test cases from the repo's README, docstrings, and baseline metric description. This lowers the onboarding barrier for repos without existing test suites.
5. **OpenClaw / PaperClip-style advanced capabilities** — Autonomous recursive self-improvement beyond a single cycle, where AHVS applies hypothesis-driven improvement to its own strategies and hypothesis templates.
6. **Constraint gating for text artifacts** — Adopt size limits and growth bounds (e.g., max 15KB, max 2× baseline) for evolved prompts/configs to prevent unbounded prompt bloat across AHVS cycles.

#### Memory management *(mostly complete)*

The core memory management overhaul is implemented (see [docs/memory_management_system.md](docs/memory_management_system.md)): structured outcome fields on lessons, two-phase semantic deduplication, Stage 8 verification feedback, friction log summarization, session memory lifecycle (stale marking + archival), historical digest for context window expansion, and cross-project learning via GlobalEvolutionStore. Remaining items:

1. **LLM-based lesson summarization**
   Periodically consolidate clusters of related lessons into summary entries (e.g., "5 threshold-tweak hypotheses tried across cycles X-Y, max +0.8% improvement") instead of keeping every raw entry. Reduces prompt token usage while preserving signal.

### Browser-based hypothesis selector

The `hypothesis_selector.py` module provides a standalone web GUI for hypothesis selection (no pip dependencies — pure stdlib):

```bash
python -m ahvs.hypothesis_selector <cycle_dir> ["optional question"]
```

This:
1. Reads `hypotheses.md` from the cycle directory
2. Serves a dark-themed HTML page on `localhost` (auto-assigned port) with checkbox cards for each hypothesis
3. Opens the default browser automatically
4. Blocks until the human submits their selection
5. Writes `selection.json` in AHVS Mode 1 format (`{"selected": [...], "approved_by": "human"}`)

The selector can also be called programmatically:

```python
from pathlib import Path
from ahvs.hypothesis_selector import run_selector

selected_ids = run_selector(Path("path/to/cycle_dir"))
# Returns e.g. ['H1', 'H3']
```

### AST-based partial output merging (splice_functions)

When Claude Code produces a partial file (containing only some functions from an existing file), AHVS uses AST-based merging (`splice_functions` in `worktree.py`) rather than naive overwrite. This:

- **Replaces** matching function/class definitions with the new versions
- **Appends** genuinely new definitions
- **Preserves** existing code that wasn't in the partial output
- **Propagates** new imports from the partial file

If either the original or partial file has syntax errors, the merge falls back gracefully — returning the partial output (if the original can't be parsed) or the original (if the partial can't be parsed).

### Test coverage

284 unit and integration tests in `tests/test_ahvs.py` covering stage orchestration, config validation, health checks, skill matching, worktree lifecycle, eval execution, result serialization, AST splicing, memory management (cycle cleanup, lesson compaction, eager writes, cycle-status filtering, structured outcomes, semantic deduplication, verification feedback, friction log summarization, memory file lifecycle, historical digest, cross-project learning), repo registry, and regression tests for all known framework bugs.

Tests work from a fresh checkout — no `pip install -e .` or `PYTHONPATH` required (`conftest.py` bootstraps the import path):

```bash
python -m pytest tests/test_ahvs.py -v
```
