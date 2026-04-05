---
name: ahvs_genesis
description: >-
  Creates a new AHVS project from raw data using the Genesis system.
  Walks the user through: problem description, data path, target metric,
  output directory, and execution mode (pipeline or agent). Calls the
  appropriate solver (e.g., KD classifier) to label data, train a model,
  measure a baseline, and produce a ready-to-optimize AHVS project.
  Triggers on "genesis", "ahvs genesis", "create a new project",
  "build a classifier from data", "bootstrap a project", "start from scratch".
---

# AHVS Genesis

This skill creates a new AHVS-ready project from raw data. It is the
**step before** AHVS optimization — genesis builds the initial project
and baseline, then the user decides when to run AHVS cycles.

## What This Skill Does and Does Not Do

**Does:**
1. Collects required inputs from the user (problem, data, metric, output dir)
2. Validates inputs before running anything
3. Calls the appropriate genesis solver to build the project
4. Produces `.ahvs/baseline_metric.json` with a real measured baseline
5. Registers the project in `~/.ahvs/registry.json`
6. Reports results and next-step command

**Does not:**
1. Auto-generate or suggest the output directory — the user MUST provide it
2. Auto-chain into AHVS optimization cycles — genesis and optimization are separate steps
3. Modify any existing project or repository

## Required Inputs

Collect ALL of these from the user before proceeding:

| Input | Required | Example |
|-------|----------|---------|
| **Problem description** | YES | "Classify customer emails into intent categories" |
| **Data file path** | YES | `/path/to/emails.csv` |
| **Target metric** | YES (default: f1_weighted) | `f1_weighted`, `accuracy`, `f1_macro` |
| **Output directory** | YES — never suggest a default | `/home/user/projects/email_classifier` |
| **Execution mode** | YES — ask the user | `pipeline` or `agent` |
| **Classes** | Optional | `["urgent", "question", "feedback", "spam"]` |
| **Input column** | Optional (default: text) | The column name in the CSV containing text |

### Execution Mode Explanation

When asking the user which mode to use, explain:

- **Pipeline mode** (default): Fast and deterministic. Generates config/spec from
  your problem description and runs the KD pipeline directly. You need to know
  your classification classes upfront.

- **Agent mode**: Smarter but slower. Uses the KD Agent (claude-agent-sdk) to
  inspect your data, discover classes, generate optimal config, and drive all
  pipeline stages. Better for unfamiliar datasets.

## Workflow

### Step 1: Gather Inputs

Ask the user for each required input. Do NOT proceed until all are collected.
Validate:
- Data file exists and is CSV/TSV/Parquet
- Output directory path is provided (not empty, not auto-generated)
- Problem description is non-trivial

### Step 2: Confirm and Run

Show the user a summary of what will happen:
```
Problem:    <description>
Data:       <path>
Metric:     <metric>
Output:     <output_dir>
Mode:       <pipeline|agent>
Classes:    <classes or "auto-detect">
```

Ask for confirmation, then run:

```bash
ahvs genesis \
  --problem "<description>" \
  --data <path> \
  --target-metric <metric> \
  --output-dir <output_dir> \
  --mode <mode> \
  [--classes <class1> <class2> ...] \
  [--input-column <column>]
```

Or via Python:

```python
from ahvs.genesis import SolverRegistry, ProblemRouter

registry = SolverRegistry()
router = ProblemRouter(registry)
solver_name = router.route(problem)
solver = registry.get(solver_name)

result = solver.solve(
    problem=problem,
    data_path=data_path,
    target_metric=target_metric,
    output_dir=output_dir,
    config_overrides={"mode": mode, "classes": classes},
)
```

### Step 3: Report Results

On success, report:
- Project directory path
- Baseline metric name and value
- Model path (if applicable)
- Registry name
- The exact command to run AHVS optimization next

On failure, report the errors and suggest fixes.

### Step 4: Next Steps

After successful genesis, tell the user:

```
Your project is ready. To run AHVS optimization:

  ahvs --repo <name> --question "improve <metric> to <target>" --domain ml

Or inspect the baseline first:
  cat <output_dir>/.ahvs/baseline_metric.json
```

## LLM Cost Policy

- Annotation/labeling: use `gpt-4.1-mini` (never `gpt-4o`)
- AHVS orchestration: use ACP (Claude Code subscription)
- The `--annotation-model` flag controls which model labels data

## Important Rules

1. **Output directory is ALWAYS required** — never suggest, generate, or default a path
2. **Genesis does NOT chain into AHVS** — it is a separate step
3. **Ask about mode** — always explain pipeline vs agent and let the user choose
4. **Validate before running** — check data file exists, output dir is writable
5. **Never use gpt-4o** for annotation — cost-prohibitive for labeling
