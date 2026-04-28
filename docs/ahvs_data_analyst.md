# AHVS Data Analyst — Goal-Directed ML Data Analysis

Data Analyst is a cross-cutting tool in the AHVS ecosystem. It analyses datasets for ML readiness — profiling, class balance, text statistics, deduplication, subsampling, and splitting — before you commit to training.

```
brainstorm → genesis → onboarding → ahvs cycles
                                        ↕
                               data_analyst (any time)
```

---

## Table of Contents

1. [When to Use Data Analyst](#1-when-to-use-data-analyst)
2. [The 4-Phase Pipeline](#2-the-4-phase-pipeline)
3. [Analysis Modules](#3-analysis-modules)
4. [CLI Reference](#4-cli-reference)
5. [Python API](#5-python-api)
6. [Module Interface](#6-module-interface)
7. [Data Selection Strategies](#7-data-selection-strategies)
8. [Report Output](#8-report-output)
9. [Adding Custom Modules](#9-adding-custom-modules)
10. [Design Principles](#10-design-principles)

---

## 1. When to Use Data Analyst

Use data analyst when you need to understand your data before building a model:

| Situation | What data analyst does |
|-----------|----------------------|
| New dataset, unknown quality | Profiles schema, detects columns, flags issues, reports class balance |
| Building a classifier | Detects labels, computes imbalance ratio, suggests subsampling or augmentation |
| Large dataset, need a subset | Selects a representative subsample using stratified, balanced, or diversity strategies |
| Preparing train/val/test splits | Produces stratified splits with configurable ratios |
| Checking for duplicates | Detects and removes duplicates — lexical (MinHash LSH), semantic (embeddings + DBSCAN), or hybrid |
| Exporting processed data | Writes filtered/subsampled data to CSV, Parquet, JSON, or JSONL |

Data analyst runs at **any point** in the AHVS lifecycle — before genesis, during onboarding, or between optimization cycles. It does not modify your data; it produces analysis reports and optional output files.

## 2. The 4-Phase Pipeline

Every analysis run follows four sequential phases:

```
User Goal + Data Path
        |
        v
+-------------------------------------------+
|  PHASE 1: DATA PROFILING  (local, 0 LLM)  |
|  Load data, infer schema, classify columns |
|  Detect labels, compute quality score      |
|  Output: DataProfile                       |
+-------------------+-----------------------+
                    v
+-------------------------------------------+
|  PHASE 2: GOAL ALIGNMENT  (1 LLM call)    |
|  Resolve column roles, pick task type      |
|  Select which modules to run               |
|  Output: AnalysisPlan                      |
+-------------------+-----------------------+
                    v
+-------------------------------------------+
|  PHASE 3: MODULE EXECUTION  (local)        |
|  Run each module sequentially              |
|  Modules that transform data pass it       |
|  downstream (e.g., subsample -> export)    |
|  Output: List[ModuleResult]                |
+-------------------+-----------------------+
                    v
+-------------------------------------------+
|  PHASE 4: SYNTHESIS & REPORT  (1 LLM call) |
|  Markdown + JSON report, recommendations   |
|  Output: analysis_report.md + .json        |
+-------------------------------------------+
```

**LLM usage:** 0-2 calls total. Phase 1 and 3 are fully local. Phase 2 uses an LLM if available (falls back to heuristic planner). Phase 4 generates reports from collected results without an LLM call unless explicitly configured.

**Privacy:** Raw data never leaves the machine. Only schema summaries and aggregated statistics are sent to the LLM planner.

## 3. Analysis Modules

Eight modules ship with v1. Each is independently runnable and follows a standard interface.

| Module | What it does | Requires label? | Produces figures? |
|--------|-------------|-----------------|-------------------|
| **eda** | Descriptive statistics, dtype breakdown, missing values, encoding issues, numeric histograms, categorical bar charts | No | Yes |
| **class_balance** | Class counts, percentages, imbalance ratio, Shannon entropy, distribution bar chart | Yes | Yes |
| **text_stats** | Character/word length distributions, vocabulary size, duplicates, empty texts, per-class breakdown | No (enhanced with label) | Yes |
| **duplicates** | Three dedup modes: *lexical* (MinHash LSH), *semantic* (sentence-transformers + DBSCAN), *hybrid* (lexical then semantic). Configurable via `dedup_config.yaml` or `--dedup-mode`. Auto-escalates to hybrid when subsample follows | No | No |
| **cluster** | Sentence-transformer embeddings + DBSCAN clustering. Adds `cluster_label` column for diversity sampling. Reuses embeddings from semantic dedup when available | No | No |
| **subsample** | Intelligent data selection: stratified, class-balanced, diversity, or random sampling | No (enhanced with label) | No |
| **split** | Train/val/test split with configurable ratios and stratification | No (stratified with label) | No |
| **export** | Save dataset to CSV, Parquet, JSON, or JSONL with optional column selection and filtering | No | No |

### Module composition

Modules run sequentially. When a module transforms data (e.g., `duplicates`, `cluster`, `subsample`), downstream modules receive the transformed DataFrame. The standard pipeline for large text datasets is:

```
eda → class_balance → text_stats → duplicates (hybrid) → cluster → subsample (diversity) → split → export
```

The planner automatically selects this order. When `duplicates` and `subsample` are both selected, dedup auto-escalates to `hybrid` and subsample uses `diversity` strategy (cluster-proportional sampling). GPU is required for the semantic pass and clustering — without GPU, both downgrade gracefully (dedup falls back to lexical, cluster is skipped, subsample falls back to stratified).

## 4. CLI Reference

```bash
ahvs data_analyst [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data`, `-d` | *(required)* | Path to data file (CSV, Parquet, JSON, JSONL) |
| `--goal`, `-g` | `""` | Natural-language goal (e.g., "build ABSA sentiment classifier") |
| `--task`, `-t` | `classification` | Shorthand task type |
| `--modules`, `-m` | *(auto-select)* | Comma-separated module list (e.g., `eda,class_balance,subsample`) |
| `--output`, `-o` | `analysis_<timestamp>/` | Output directory for results |
| `--label` | *(auto-detect)* | Force a specific column as the label |
| `--inputs` | *(auto-detect)* | Comma-separated list of input columns |
| `--nrows` | *(all)* | Limit rows for quick profiling (full data still used for execution) |
| `--verbose`, `-v` | off | Enable verbose logging |
| `--view` | *(none)* | Path to `analysis_report.md` or output directory — opens report in browser instead of running analysis |
| `--dedup-mode` | *(from config)* | Deduplication mode: `lexical` (MinHash LSH), `semantic` (embeddings + DBSCAN), or `hybrid` (both) |

#### LLM Provider Flags

These flags enable LLM-assisted planning in Phase 2. If `--provider` is omitted, the heuristic planner is used (no LLM call).

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | *(none — heuristic)* | LLM provider: `acp`, `anthropic`, `openai`, `openrouter`, `deepseek`, `openai-compatible` |
| `--model` | `claude-opus-4-6` | LLM model ID |
| `--api-key-env` | `ANTHROPIC_API_KEY` | Env var holding the LLM API key |
| `--base-url` | `""` | Override LLM base URL (required for `openai-compatible`) |
| `--acp-agent` | `claude` | ACP agent CLI name (only with `--provider acp`) |
| `--acpx-command` | *(auto-detect)* | Path to acpx binary (only with `--provider acp`) |
| `--acp-session-name` | `ahvs-data-analyst` | ACP session name (only with `--provider acp`) |
| `--acp-timeout` | `1800` | ACP per-prompt timeout in seconds (only with `--provider acp`) |

### Examples

```bash
# Full automatic analysis (heuristic planner, no LLM)
ahvs data_analyst --data emails.csv --goal "build intent classifier"

# With ACP — Claude Code/Codex plans which modules to run
ahvs data_analyst --data emails.csv --goal "build intent classifier" --provider acp

# With Anthropic API
ahvs data_analyst --data emails.csv --goal "build intent classifier" --provider anthropic

# Specific modules only (skips planning entirely)
ahvs data_analyst --data data.csv --modules eda,class_balance,text_stats

# With explicit column hints
ahvs data_analyst --data absa.parquet --label sentiment --inputs review_text

# Quick profile of a large file
ahvs data_analyst --data big_dataset.parquet --nrows 1000 --modules eda

# Hybrid dedup (lexical + semantic) — catches paraphrases
ahvs data_analyst --data social_media.parquet --inputs text --dedup-mode hybrid

# Semantic-only dedup with custom threshold
ahvs data_analyst --data articles.csv --inputs body --dedup-mode semantic
```

### Claude Code Skills

The `/ahvs_data_analyst` skill provides a conversational interface in Claude Code. It collects your data path and goal, runs the pipeline with ACP by default, and presents findings interactively.

```
/ahvs_data_analyst
```

The `/ahvs_data_analyst:gui` subcommand opens an existing analysis report in the browser as a styled HTML page with embedded figures and dark theme.

```
/ahvs_data_analyst:gui path/to/analysis_report.md
```

Or via CLI:

```bash
ahvs data_analyst --view analysis_20260408/analysis_report.md
ahvs data_analyst --view analysis_20260408/   # auto-finds analysis_report.md
```

## 5. Python API

### One-call analysis

```python
from ahvs.data_analyst import analyze

report = analyze(
    data_path="absa_test.parquet",
    goal="aspect-based sentiment analysis",
    label_hint="sentiment",
    input_hint=["review_text"],
    output_dir="analysis_output/",
    dedup_mode="hybrid",  # or "lexical", "semantic"
)

print(f"Completeness: {report.completeness_score():.0f}%")
print(f"Report: {report.markdown_path}")
```

### With ACP (Claude Code / Codex)

```python
from ahvs.data_analyst import analyze
from ahvs.llm.acp_client import ACPClient, ACPConfig

client = ACPClient(ACPConfig(
    agent="claude",
    session_name="ahvs-data-analyst",
))

report = analyze(
    data_path="data.csv",
    goal="build intent classifier",
    llm_client=client,
    output_dir="analysis_output/",
)
```

### With Anthropic API

```python
from ahvs.data_analyst import analyze
from ahvs.llm.client import LLMClient, LLMConfig

client = LLMClient(LLMConfig(
    base_url="https://api.anthropic.com",
    api_key="your-key",
    primary_model="claude-opus-4-6",
))

report = analyze(
    data_path="data.csv",
    goal="build intent classifier",
    llm_client=client,
)
```

### Phase-by-phase access

```python
from ahvs.data_analyst.profiler import profile_data
from ahvs.data_analyst.planner import plan
from ahvs.data_analyst.executor import execute
from ahvs.data_analyst.synthesizer import synthesize
from pathlib import Path

# Phase 1: Profile
profile = profile_data("data.csv", label_hint="label")
print(profile.summary_for_llm())

# Phase 2: Plan (heuristic, no LLM needed)
analysis_plan = plan(profile, goal="sentiment classification")
print(analysis_plan.module_names())

# Phase 3: Execute
import pandas as pd
df = pd.read_csv("data.csv")
results = execute(df, profile, analysis_plan, Path("output/"))

# Phase 4: Synthesize
report = synthesize(profile, analysis_plan, results, Path("output/"))
```

### Profiling only (no execution)

```python
from ahvs.data_analyst.profiler import profile_data

profile = profile_data("data.parquet")
print(f"Rows: {profile.total_rows}")
print(f"Label: {profile.label_column}")
print(f"Classes: {profile.class_distribution}")
print(f"Quality: {profile.quality_score}/100")
print(f"Warnings: {profile.warnings}")
```

## 6. Module Interface

Every module follows a standard contract:

```python
from ahvs.data_analyst.models import ModuleInput, ModuleResult

def run(inp: ModuleInput) -> ModuleResult:
    """Execute the module and return results."""
    ...
```

### ModuleInput

| Field | Type | Description |
|-------|------|-------------|
| `df` | `pd.DataFrame` | The dataset (or transformed subset from prior modules) |
| `profile` | `DataProfile` | Phase 1 output |
| `plan` | `AnalysisPlan` | Phase 2 output |
| `task_type` | `str` | e.g., "multiclass_classification" |
| `input_cols` | `list[str]` | Resolved input columns |
| `label_col` | `str \| None` | Resolved label column |
| `params` | `dict` | Module-specific parameters |
| `output_dir` | `Path` | Where to write artifacts |

### ModuleResult

| Field | Type | Description |
|-------|------|-------------|
| `module_name` | `str` | Registry key (e.g., "eda") |
| `status` | `"success" \| "skipped" \| "error"` | Outcome |
| `summary` | `dict` | Machine-readable findings |
| `narrative` | `str` | Human-readable paragraph |
| `figures` | `list[Path]` | Generated plots |
| `artifacts` | `list[Path]` | Output files |
| `warnings` | `list[str]` | Issues detected |
| `error_message` | `str` | Error details (if status is "error") |
| `transformed_df` | `pd.DataFrame \| None` | Modified data for downstream modules |

## 7. Data Selection Strategies

The `subsample` module supports four strategies:

| Strategy | When to use | How it works |
|----------|-------------|--------------|
| **stratified** (default) | Balanced classes, large dataset | Random sample preserving class ratios |
| **class_balanced** | Imbalanced classes | Equal samples per class with remainder distribution |
| **diversity** | Clustered data | Cluster-proportional sampling (requires `cluster_label` column) |
| **random** | Fallback / no labels | Simple random sample |

All strategies enforce exact `target_size` via a final trim/top-up pass.

### Example: subsample via Python API

```python
from ahvs.data_analyst import analyze

report = analyze(
    data_path="large_dataset.csv",
    modules=["subsample", "export"],
    label_hint="label",
)
# subsample selects a representative subset
# export writes it to disk
# downstream modules automatically receive the subsampled data
```

## 7b. Deduplication Modes

The `duplicates` module supports three dedup strategies, configurable via `--dedup-mode` or `ahvs/data_analyst/configs/dedup_config.yaml`.

| Mode | Algorithm | Speed | Catches |
|------|-----------|-------|---------|
| **lexical** (default) | MinHash LSH (Jaccard similarity on whitespace tokens) | Fast (~90s for 96K rows) | Near-identical texts (retweets, copy-paste) |
| **semantic** | Sentence-transformer embeddings + DBSCAN (cosine distance) | Moderate (~80s with GPU) | Paraphrases and semantic near-duplicates |
| **hybrid** | Lexical first, then semantic on survivors | Slower (~180s) | Both lexical and semantic duplicates |

### Configuration

All parameters live in `ahvs/data_analyst/configs/dedup_config.yaml`:

```yaml
dedup_mode: lexical        # lexical | semantic | hybrid
deduplicate: true          # false = report only, don't remove

lexical:
  lsh_threshold: 0.85      # Jaccard similarity (higher = stricter)
  lsh_num_perm: 128        # MinHash permutations (higher = more accurate)

semantic:
  embedding_model: paraphrase-multilingual-MiniLM-L12-v2
  eps: 0.15                # DBSCAN cosine distance (lower = stricter)
  min_samples: 2
  batch_size: 64
  device: auto             # auto | cpu | cuda

hybrid:
  skip_semantic_if_lexical_removed_pct: 100
```

CLI override: `ahvs data_analyst --data file.parquet --dedup-mode hybrid`

Param override: `params={"dedup_mode": "semantic", "eps": 0.2}`

### Output artifacts

```
duplicates/
├── deduplicated.parquet      # Cleaned dataset (when deduplicate=true)
├── duplicate_groups.json     # Audit trail: group memberships + config used
└── embeddings_text.npy       # Saved embeddings (semantic/hybrid only)
```

The deduplication module sets `transformed_df`, so all downstream modules (cluster, subsample, split, export) automatically operate on the deduplicated data.

### Auto-escalation

When the planner selects both `duplicates` and `subsample` (i.e., large text datasets), dedup auto-escalates from `lexical` to `hybrid`. This ensures semantic paraphrases are removed before diversity sampling.

### GPU policy

Semantic dedup and clustering **require a GPU** by default:
- **GPU available:** hybrid dedup + embedding clustering run normally
- **GPU unavailable:** dedup downgrades to lexical, clustering is skipped, subsample falls back to stratified. A warning is logged.
- **User override:** `--dedup-mode hybrid` or `--dedup-mode semantic` forces CPU execution even without GPU

## 8. Report Output

Every analysis run produces a timestamped output directory:

```
analysis_20260408_143022/
+-- analysis_report.md        # Human-readable markdown report
+-- analysis_report.json      # Machine-readable JSON report
+-- eda/
|   +-- numeric_distributions.png
|   +-- categorical_distributions.png
|   +-- missing_values.png
+-- class_balance/
|   +-- class_distribution.png
+-- text_stats/
|   +-- word_dist_review_text.png
+-- duplicates/
|   +-- deduplicated.parquet
|   +-- duplicate_groups.json
|   +-- embeddings_text.npy         (semantic/hybrid only)
+-- cluster/
|   +-- embeddings_text.npy         (only if not reused from duplicates)
+-- split/
|   +-- train.parquet
|   +-- val.parquet
|   +-- test.parquet
+-- subsample/
|   +-- subsample.parquet
+-- export/
    +-- data.parquet
```

The markdown report includes:
- Dataset overview (shape, columns, types, quality score)
- Column table with roles, cardinality, and null percentages
- Class distribution table (if labels detected)
- Per-module results with narratives, warnings, and figure links
- Validation results (errors and warnings)
- Actionable recommendations

The JSON report mirrors the same structure for programmatic consumption. All float values are sanitized (NaN/inf replaced with null) for strict JSON compatibility.

## 9. Adding Custom Modules

To add a new analysis module:

**Step 1:** Create `ahvs/data_analyst/modules/my_module.py`:

```python
from ahvs.data_analyst.models import ModuleInput, ModuleResult

def run(inp: ModuleInput) -> ModuleResult:
    # Your analysis logic here
    return ModuleResult(
        module_name="my_module",
        status="success",
        summary={"key": "value"},
        narrative="Description of findings.",
    )
```

**Step 2:** Register it in `ahvs/data_analyst/registry.py`:

```python
_BUILTIN_MODULES = [
    "eda",
    "class_balance",
    ...
    "my_module",  # Add here
]
```

**Step 3:** Use it:

```bash
ahvs data_analyst --data file.csv --modules eda,my_module
```

Or register at runtime without modifying package code:

```python
from ahvs.data_analyst import registry

def my_custom_run(inp):
    ...

registry.register("my_module", my_custom_run)
```

## 10. Design Principles

- **Local-first** — all analysis runs locally. LLM used only for planning and synthesis. Raw data never leaves the machine.
- **Module-based** — each capability is a standalone function with a standard interface. Deterministic, testable, reproducible.
- **LLM-assisted, not LLM-dependent** — works without LLM for standard tasks. Falls back to heuristic planner when no LLM client is available.
- **Composable** — modules that transform data pass results downstream. `subsample -> export` just works.
- **Extensible** — new modules are a function + registry entry. No core changes needed.
- **AHVS-native** — integrates as a domain pack (`--domain data_analyst`), CLI subcommand, and Python API.

---

## Roadmap

| Version | Scope |
|---------|-------|
| **v1 (current)** | Profiling, EDA, class balance, text stats, duplicates (lexical/semantic/hybrid), embedding clustering, diversity subsample, split, export |
| **v2** | Synthetic text generation (augment), correlation analysis, outlier detection, uncertainty sampling |
| **v3** | CV support, regression tasks, NER, multi-file datasets |
| **v4** | Active learning loops with AHVS cycle integration |

---

## Usage

**CLI:**
```bash
ahvs data_analyst --data <file> --goal "..." [options]
```

**Python:**
```python
from ahvs.data_analyst import analyze
report = analyze(data_path="file.csv", goal="build classifier")
```

**Design doc:** [docs/ahvs/designs/2026-04-08-data-analyst-agent-design.md](ahvs/designs/2026-04-08-data-analyst-agent-design.md)
