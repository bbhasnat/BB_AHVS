# AHVS Onboarding — Prepare a Repository for AHVS Cycles

Onboarding is the third stage of the AHVS pipeline. It takes an existing repository (either created by genesis or an existing project) and **wires it up for AHVS optimization** by creating the eval command, baseline metric file, and verifying everything works.

```
brainstorm → genesis → onboarding → ahvs cycles
                           ▲ you are here
```

---

## Table of Contents

1. [What Onboarding Does](#1-what-onboarding-does)
2. [When to Use Onboarding](#2-when-to-use-onboarding)
3. [The 8-Phase Workflow](#3-the-8-phase-workflow)
4. [Baseline Metric File](#4-baseline-metric-file)
5. [Evaluation Setup](#5-evaluation-setup)
6. [Regression Guard](#6-regression-guard)
7. [Return Contract](#7-return-contract)
8. [Git Mode Policy](#8-git-mode-policy)
9. [Failure Modes](#9-failure-modes)

---

## 1. What Onboarding Does

Onboarding inspects a target repository and produces everything AHVS needs to run optimization cycles:

1. **Scans the codebase** — finds eval scripts, metrics, models, configs, prior experiments
2. **Creates a headless eval command** — extracts eval logic from notebooks/CI into a standalone script if needed
3. **Writes `.ahvs/baseline_metric.json`** — the single source of truth for AHVS
4. **Registers the repo** in `~/.ahvs/registry.json` for short-name access
5. **Verifies the eval command** — runs it and confirms it produces parseable output

**Onboarding does NOT:**
- Invent baseline values or guess fake eval commands
- Start an AHVS cycle — that's a separate step after onboarding
- Modify git state without informing the user

## 2. When to Use Onboarding

| Scenario | Use onboarding? |
|----------|----------------|
| Genesis just created a project | Genesis already writes `baseline_metric.json` — **skip onboarding** unless you need enriched fields |
| Existing project without `.ahvs/` | **Yes** — onboarding creates everything from scratch |
| Project has `.ahvs/` but no eval command | **Yes** — onboarding creates the eval script |
| Want to add regression guards, system levers, prior experiments | **Yes** — onboarding enriches the baseline file |

## 3. The 8-Phase Workflow

### Phase 1: Discover — Deep Codebase Inspection

Scans the target repo comprehensively:

```
README*, pyproject.toml, requirements.txt, package.json, Makefile,
.github/workflows/*, .ahvs/*, promptfoo*, eval.py, scripts/eval*,
tests/, benchmarks/, *.yaml configs, *.ipynb notebooks,
checkpoints/, results/, experiments.*
```

Identifies:
- **Project type** — Python package, Node app, RAG pipeline, ML model, classifier
- **Evaluation pipeline** — how metrics are currently computed
- **Candidate metric names** — from eval output, checkpoint files, experiment registries
- **Existing experiments** — checkpoint files, experiment YAML, prior results
- **System levers** — tunable parameters (models, prompts, algorithms, configs)
- **Python environment** — which conda/venv has the project's dependencies
- **Data dependencies** — ground truth files, datasets, API keys needed

### Phase 2: Extract or Create Eval Command

AHVS needs a shell command that:
1. Runs headlessly (no notebook, no interactive input)
2. Prints `metric_name: numeric_value` to stdout
3. Is reproducible (same inputs = same output)

| Case | Action |
|------|--------|
| **CLI eval script exists** | Verify it prints metric to stdout, test it |
| **Eval in Jupyter notebook** | Extract into standalone `run_eval.py` with `--eval-only` mode |
| **Eval in CI/Makefile** | Extract the eval command, verify locally |
| **No evaluation exists** | Return `blocked` — AHVS needs a measurable metric |

### Phase 3: Align — Understand Optimization Goals

Asks the user:
1. **What metric should AHVS optimize?** — must be numeric
2. **Are there metrics that must not regress?** — regression guards
3. **What's your model/cost budget?** — acceptable models, cost-per-row limits
4. **Should hypotheses go beyond prompt changes?** — algorithmic changes, data strategies

### Phase 4: Gather Prior Experiments

Checks for existing experiment results:
- Experiment registry files (YAML, JSON)
- Checkpoint directories with results
- Reports or analysis documents
- Comments in config files about what was tried

Prior experiments prevent AHVS from repeating dead ends.

### Phase 5: Document System Levers

Identifies all tunable parameters so AHVS generates diverse hypotheses:
- Strategies, scoring modes, algorithmic areas
- Config parameters, thresholds, model choices
- File paths where each lever lives

### Phase 6: Write Artifacts and Register

Creates `.ahvs/` directory, writes `baseline_metric.json`, and registers the repo in `~/.ahvs/registry.json`.

### Phase 7: Verify — Test Everything

Runs the eval command and verifies:
1. Exits with code 0
2. stdout contains `<primary_metric>: <numeric_value>`
3. Value matches the baseline (or is close)

If verification fails, diagnoses and fixes before returning `ready`.

### Phase 8: Gate — Return Status

Presents a clear status block with all details.

## 4. Baseline Metric File

`.ahvs/baseline_metric.json` — the central artifact. Required fields:

| Field | Description |
|---|---|
| `primary_metric` | Name of the metric to optimize (e.g. `precision`) |
| `<primary_metric>` | Current numeric value (float) |
| `recorded_at` | ISO-8601 timestamp |
| `eval_command` | Headless shell command that prints `metric_name: value` to stdout |
| `commit` | Git commit SHA when baseline was recorded |

Enriched fields (optional but strongly recommended):

| Field | Description |
|---|---|
| `optimization_goal` | Plain English: what to optimize and constraints |
| `regression_floor` | Secondary metrics with minimum values, e.g. `{"f1_score": 0.62}` |
| `constraints` | Budget limits, model restrictions, hypothesis scope |
| `system_levers` | Tunable parameters: strategies, modes, algorithmic areas |
| `prior_experiments` | Past results with config details and problems |
| `notes` | Dataset size, key insights, known hard cases |

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

**Enriched example:**
```json
{
  "primary_metric": "precision",
  "precision": 0.7128,
  "recorded_at": "2026-03-19T10:00:00Z",
  "commit": "9c41b35",
  "eval_command": "cd /path/to/project && python -m package.run_eval --eval-only",
  "optimization_goal": "Maximize precision while keeping f1_score >= 0.62",
  "regression_floor": {"f1_score": 0.62},
  "constraints": {
    "model_budget": "Gemini 3.1 Flash Lite only.",
    "hypothesis_scope": "Must include algorithmic changes, not just prompt rewrites."
  },
  "system_levers": {
    "strategies": ["random", "keyword", "semantic"],
    "algorithmic_areas": ["post_selector.py: selection strategy"]
  },
  "prior_experiments": {
    "best_precision": {"config": "gpt54_kw_prv2", "precision": 0.76, "f1": 0.57}
  }
}
```

## 5. Evaluation Setup

The `eval_command` is executed **in a git worktree** of the target repo after hypothesis code changes are applied. It must:

1. Be reproducible (same inputs = same output)
2. Print `metric_name: numeric_value` to stdout
3. Run headlessly (no interactive prompts)

**Metric extraction policy:**

When `eval_command` is configured (recommended): it is the **single source of truth**. Metrics are parsed from stdout only.

When `eval_command` is missing: AHVS falls back to `result.json` in the work dir. If no valid metric is found, the hypothesis is treated as failed.

## 6. Regression Guard

Optional but recommended. A shell script that exits 0 if results pass quality checks, non-zero if they regress.

```bash
# .ahvs/regression_guard.sh
#!/bin/bash
RESULT=$(jq '.answer_relevance' "$1")
python -c "import sys; sys.exit(0 if float('$RESULT') >= 0.70 else 1)"
```

When configured, the guard is **fail-closed**: if missing, times out, or throws an error, AHVS rejects the hypothesis.

## 7. Return Contract

Every onboarding pass ends with one of three statuses:

| Status | Meaning | What happens next |
|--------|---------|-------------------|
| `ready` | All gates passed, artifacts written, eval verified | User can run `ahvs` |
| `needs_user_input` | Promising but missing info | Ask follow-up, re-evaluate |
| `blocked` | Cannot proceed safely | Explain blocker, what must change |

`ready` is returned **only** when all of these are true:
1. Target path exists and is understood
2. Optimization metric is clearly identified
3. A headless evaluation command exists and has been tested
4. Baseline value is known (from existing results or a verified run)
5. `.ahvs/baseline_metric.json` is internally consistent
6. The eval command stdout is parseable

## 8. Git Mode Policy

| Mode | Behavior |
|------|----------|
| **Git repo** | Record commit SHA in baseline. AHVS uses detached worktrees. Recommended. |
| **Not a git repo** | Do not block. AHVS auto-initializes git at Stage 1 if needed. Inform user. |

## 9. Failure Modes

| Condition | Status |
|-----------|--------|
| Target path does not exist | `blocked` |
| Metric too vague ("make it better") | `blocked` |
| No reproducible eval path and none can be created | `blocked` |
| No ground truth or benchmark data | `blocked` |
| Metric needs confirmation | `needs_user_input` |
| Found likely eval command, needs approval | `needs_user_input` |
| Baseline missing but can be measured | `needs_user_input` |
| Optimization goal or constraints unclear | `needs_user_input` |

---

## Usage

In Claude Code (terminal):
```
/ahvs_onboarding
```

In Claude Code (browser form):
```
/ahvs_onboarding:gui
```
