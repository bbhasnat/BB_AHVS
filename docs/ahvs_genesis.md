# AHVS Genesis — Bootstrap a Project from Raw Data

Genesis is the second stage of the AHVS pipeline. It takes the decisions made during brainstorm (or the user's direct input) and **scaffolds a complete AHVS-ready project** with a trained model, measured baseline, and registered entry.

```
brainstorm → genesis → onboarding → ahvs cycles
                ▲ you are here
```

---

## Table of Contents

1. [What Genesis Does](#1-what-genesis-does)
2. [Required Inputs](#2-required-inputs)
3. [Execution Modes](#3-execution-modes)
4. [The 4-Step Workflow](#4-the-4-step-workflow)
5. [CLI Reference](#5-cli-reference)
6. [Python API](#6-python-api)
7. [What Genesis Produces](#7-what-genesis-produces)
8. [LLM Cost Policy](#8-llm-cost-policy)
9. [Important Rules](#9-important-rules)

---

## 1. What Genesis Does

Genesis takes raw data and a problem description, then:

1. Routes the problem to the appropriate solver (e.g., `kd_classifier` for text classification)
2. Labels data using an LLM (annotation step)
3. Trains a model (e.g., DistilBERT for KD classification)
4. Measures a baseline metric
5. Writes `.ahvs/baseline_metric.json`
6. Registers the project in `~/.ahvs/registry.json`
7. Git-initializes the output directory

After genesis, the project is ready for AHVS optimization cycles.

**Genesis does NOT:**
- Auto-chain into AHVS optimization — genesis and optimization are separate steps
- Auto-generate or suggest the output directory — the user must always provide it
- Modify any existing project or repository

## 2. Required Inputs

| Input | Required | Example |
|-------|----------|---------|
| **Problem description** | Yes | "Classify customer emails into intent categories" |
| **Data file path** | Yes | `/path/to/emails.csv` |
| **Target metric** | Yes (default: `f1_weighted`) | `f1_weighted`, `accuracy`, `f1_macro` |
| **Output directory** | Yes — never auto-generated | `/home/user/projects/email_classifier` |
| **Execution mode** | Yes — always ask user | `pipeline` or `agent` |
| **Classes** | Optional | `["urgent", "question", "feedback", "spam"]` |
| **Input column** | Optional (default: `text`) | Column name in the CSV containing text |

If coming from a brainstorm design doc, these values are pre-filled in the **Genesis Inputs** section.

## 3. Execution Modes

Genesis supports two modes. Always explain both and let the user choose:

### Pipeline mode (default)

Fast and deterministic. Generates config/spec from the problem description and runs the KD pipeline directly. You need to know your classification classes upfront.

**Best for:** well-understood problems where you know the classes.

### Agent mode

Smarter but slower. Uses the KD Agent (claude-agent-sdk) to inspect your data, discover classes, generate optimal config, and drive all pipeline stages.

**Best for:** unfamiliar datasets, class discovery, complex problems.

## 4. The 4-Step Workflow

### Step 1: Gather Inputs

Collect all required inputs from the user. Do not proceed until all are collected. Validate:
- Data file exists and is CSV/TSV/Parquet
- Output directory path is provided (not empty, not auto-generated)
- Problem description is non-trivial

### Step 2: Confirm and Run

Show a summary and ask for confirmation:

```
Problem:    Classify customer emails into intent categories
Data:       /path/to/emails.csv
Metric:     f1_weighted
Output:     /home/user/projects/email_classifier
Mode:       pipeline
Classes:    ["urgent", "question", "feedback", "spam"]
```

Then execute genesis.

### Step 3: Report Results

On success, report:
- Project directory path
- Baseline metric name and value
- Model path (if applicable)
- Registry name
- The exact command to run AHVS optimization next

On failure, report errors and suggest fixes.

### Step 4: Next Steps

```
Your project is ready. To run AHVS optimization:

  ahvs --repo <name> --question "improve <metric> to <target>" --domain ml

Or inspect the baseline first:
  cat <output_dir>/.ahvs/baseline_metric.json
```

## 5. CLI Reference

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
| `--mode` | `pipeline` | `pipeline` (deterministic) or `agent` (KD Agent) |
| `--solver`, `-s` | *(auto-detect)* | Solver name (e.g. `kd_classifier`) |
| `--classes` | *(none)* | Classification classes |
| `--input-column` | `text` | Text column name in the data file |
| `--annotation-model` | `gpt-4.1-mini` | LLM for annotation (never gpt-4o) |
| `--solver-registry` | *(built-in)* | Path to custom solvers.yaml |

## 6. Python API

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

## 7. What Genesis Produces

After a successful genesis run, the output directory contains:

```
<output_dir>/
├── .ahvs/
│   └── baseline_metric.json    # Baseline metric, eval command, constraints
├── .git/                       # Auto-initialized
├── model/                      # Trained model artifacts
├── data/                       # Processed data (labels, splits)
├── config/                     # Pipeline configuration
└── scripts/                    # Eval and utility scripts
```

The project is also registered in `~/.ahvs/registry.json`, enabling `ahvs --repo <short_name>` in future sessions.

## 8. LLM Cost Policy

- **Annotation/labeling:** `gpt-4.1-mini` (never `gpt-4o` — cost-prohibitive for labeling)
- **AHVS orchestration:** ACP (Claude Code subscription)
- The `--annotation-model` flag controls which model labels data

## 9. Important Rules

1. **Output directory is ALWAYS required** — never suggest, generate, or default a path
2. **Genesis does NOT chain into AHVS** — it is a separate step
3. **Always ask about mode** — explain pipeline vs agent and let the user choose
4. **Validate before running** — check data file exists, output dir is writable
5. **Never use gpt-4o** for annotation — cost-prohibitive for labeling

---

## Usage

In Claude Code (terminal):
```
/ahvs_genesis
```

In Claude Code (browser form):
```
/ahvs_genesis:gui
```

The GUI form collects all inputs via a browser-based dark-themed form with live path validation and conditional fields.
