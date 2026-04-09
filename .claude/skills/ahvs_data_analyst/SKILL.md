---
name: ahvs_data_analyst
description: >-
  Goal-directed ML data analysis — profiles datasets, detects quality issues,
  runs EDA / class balance / text stats / duplicates / subsampling / splits,
  and produces a markdown report with actionable recommendations. Use this skill
  when the user says "analyze my data", "profile this dataset", "check data
  quality", "run EDA", "data analysis", "data readiness", or asks about class
  balance, duplicates, text statistics, or train/test splits for a dataset.
---

# AHVS Data Analyst

## Subcommands

| Subcommand | Purpose |
|------------|---------|
| `/ahvs_data_analyst` | Terminal-based input collection + pipeline execution (default) |
| `/ahvs_data_analyst:gui` | Browser-based report viewer — renders analysis results beautifully in browser |

### /ahvs_data_analyst:gui — Browser-Based Report Viewer

Renders a data analyst markdown report as a styled, interactive HTML page in the browser with embedded figures.

Load: the gui command file for full protocol.

**What it does:**

1. **Collect report path** — ask the user for the path to an `analysis_report.md` or the analysis output directory
2. **Serve report** — converts markdown to styled HTML with embedded PNG figures and serves on localhost:8765
3. **Display** — user views the report in browser; server stays up until manually closed

**Usage:**
```
/ahvs_data_analyst:gui
```

---

## What This Skill Does

Runs the 4-phase data analysis pipeline on a user-provided dataset:

1. **Profile** — load data, infer schema, classify column roles
2. **Plan** — LLM-assisted (via ACP) or heuristic module selection based on user goal
3. **Execute** — run selected analysis modules (EDA, class balance, text stats, duplicates, subsample, split, export)
4. **Synthesize** — produce markdown + JSON report with figures and recommendations

## Collect Inputs

Ask the user for these values (only `data_path` is required):

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `data_path` | Yes | — | Path to CSV, Parquet, JSON, or JSONL file |
| `goal` | No | "general data analysis" | Natural-language goal (e.g., "build intent classifier") |
| `modules` | No | auto-selected | Comma-separated: eda, class_balance, text_stats, duplicates, subsample, split, export |
| `output_dir` | No | `analysis_<timestamp>/` | Where to write results |
| `label` | No | auto-detected | Force a specific column as the label |
| `inputs` | No | auto-detected | Comma-separated input column names |
| `nrows` | No | all rows | Limit rows for quick profiling |
| `use_llm` | No | yes | Whether to use ACP for smart module planning |

If the user provides a data path and goal in their message, proceed directly — don't re-ask.

## Execution

### Option A: Via CLI (preferred for full pipeline)

Build and run the `ahvs data_analyst` command:

```bash
ahvs data_analyst \
  --data <data_path> \
  --goal "<goal>" \
  --provider acp \
  --acp-agent claude \
  --acp-session-name ahvs-data-analyst \
  [--modules <modules>] \
  [--output <output_dir>] \
  [--label <label>] \
  [--inputs <inputs>] \
  [--nrows <nrows>] \
  --verbose
```

- Default to `--provider acp` (local Claude agent) per user preference.
- If user says "no LLM" or "heuristic only", omit `--provider` entirely.
- If user specifies a different provider (openai, anthropic, etc.), use that instead.

### Option B: Via Python API (for interactive refinement)

```python
from ahvs.data_analyst import analyze, profile_data

# Profile first to discuss with user
profile = profile_data("<data_path>", label_hint="<label>")
print(profile.summary_for_llm())

# Then run full analysis
from ahvs.llm.acp_client import ACPClient, ACPConfig
client = ACPClient(ACPConfig(agent="claude", session_name="ahvs-data-analyst"))

report = analyze(
    data_path="<data_path>",
    goal="<goal>",
    output_dir="<output_dir>",
    llm_client=client,
)
```

Use Option B when:
- The user wants to discuss the profile before deciding on modules
- The user wants to iteratively refine the analysis
- You need programmatic access to intermediate results

## After Execution

1. **Read the markdown report** and present key findings to the user:
   - Dataset shape, quality score, completeness
   - Class distribution (if applicable)
   - Key warnings or issues found
   - Recommendations
2. **Show figure paths** so the user can view visualizations
3. **Ask what's next** — e.g., "Want me to run specific modules in more detail?" or "Ready to proceed to model training?"

## Rules

- Always default to `--provider acp` unless user specifies otherwise
- If user provides `--modules`, those override LLM planning entirely
- Never auto-generate the output path — if not provided, let the CLI default to `analysis_<timestamp>/`
- For large files (>100k rows), suggest `--nrows 5000` for a quick first pass
- Present the report findings conversationally — don't just dump the raw report
