# AHVS Data Analyst Agent — Design Document

**Date:** 2026-04-08
**Status:** Approved — 2026-04-08

---

## 1. Problem

Building ML classifiers (NLP, CV, etc.) requires substantial upfront data work:
profiling datasets, detecting label columns, assessing class balance, selecting
informative subsamples, and generating synthetic samples when data is insufficient.

Today this is done manually in Jupyter notebooks, augmented with basic tools
(pandas-profiling, sweetviz) and ad-hoc LLM sessions (ChatGPT Code Interpreter).
The results are **not reproducible**, **not reusable**, and **not integrated**
into the ML pipeline.

**Goal:** Build `ahvs_data_analyst` — an AHVS extension that automates
goal-directed data analysis, selection, and augmentation for ML tasks.

### Example interaction

```
User:  I want to build a complex aspect-based sentiment classifier
       with my data in absa_test.parquet.

Agent: Scanning absa_test.parquet...
       → 12,847 rows × 5 columns
       → Detected input column: "review_text" (str, avg 47 tokens)
       → Detected label column: "sentiment" (categorical: positive/negative/neutral)
       → Aspect column candidate: "aspect_category" (14 unique values)

       Class distribution:
         positive: 5,231 (40.7%)
         negative: 4,102 (31.9%)
         neutral:  3,514 (27.4%)

       ⚠ "neutral" class is 32% smaller than "positive"
       → Recommend: stratified subsample of 3,500/class (10,500 total)
         OR generate ~1,700 synthetic neutral samples to balance

       Statistical summary written to analysis_20260408_143022/
       Proceed with subsampling, synthetic generation, or full dataset?
```

---

## 2. Data

**Input formats supported:**
- CSV, TSV
- Parquet
- JSON / JSONL
- SQL tables (via SQLAlchemy connection string)
- HuggingFace datasets (by name)

**No assumptions about schema.** The agent auto-detects:
- Column types (text, numeric, categorical, datetime, ID-like)
- Input columns (text/feature columns suitable for the declared task)
- Label columns (categorical columns matching classification patterns)
- Metadata columns (IDs, timestamps, filenames — excluded from analysis)

---

## 3. Approach — Hybrid Modular Architecture

### 3.1 Design Principles

1. **Local-first** — all analysis runs locally via Python. LLM used only for
   planning, ambiguity resolution, and synthesis. Raw data never sent to LLM.
2. **Module-based** — each analysis capability is a standalone Python module
   with a standard interface. Deterministic, testable, reproducible.
3. **LLM-assisted, not LLM-dependent** — the agent works without LLM for
   standard tasks. LLM adds intelligence for goal interpretation, column
   disambiguation, and report generation.
4. **AHVS-native** — extends AHVS as a domain pack + skill library + CLI
   subcommand. Reuses AHVS config, memory, evolution, and cycle infrastructure.

### 3.2 Architecture

```
User Goal + Data Path
        │
        ▼
┌──────────────────────────────────────────────────┐
│  PHASE 1: DATA PROFILING  (local, zero LLM)      │
│                                                    │
│  • Load data (auto-detect format from extension)   │
│  • Schema inference (dtypes, cardinality, nulls)   │
│  • Column role classification:                     │
│      - text_input, numeric_input, categorical_input│
│      - label, aspect, id, timestamp, metadata      │
│  • Basic statistics per column                     │
│  • Class distribution (if labels detected)         │
│  • Data quality flags (nulls, duplicates, outliers)│
│                                                    │
│  Output: DataProfile (dataclass, serializable)     │
└──────────────────┬─────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────┐
│  PHASE 2: GOAL ALIGNMENT  (1 LLM call)           │
│                                                    │
│  Input: user goal + DataProfile summary            │
│  LLM resolves:                                     │
│    • Which columns are input vs label vs ignore    │
│    • Task type (binary clf, multiclass, multilabel,│
│      regression, NER, ABSA, etc.)                  │
│    • Whether data is sufficient for the task       │
│    • Recommended analysis modules to run           │
│                                                    │
│  Output: AnalysisPlan (ordered list of modules     │
│          with configured parameters)               │
└──────────────────┬─────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────┐
│  PHASE 3: MODULE EXECUTION  (local, zero LLM)    │
│                                                    │
│  Runs each module in the plan sequentially.        │
│  Each module: receives DataFrame + config,         │
│               returns ModuleResult (data + figs).  │
│                                                    │
│  Core modules (v1):                                │
│  ┌─────────────────────────────────────────────┐  │
│  │ eda          │ Descriptive stats, dist plots │  │
│  │ class_balance│ Label freq, imbalance metrics  │  │
│  │ text_stats   │ Token length, vocab, OOV rate  │  │
│  │ correlation  │ Feature correlation matrix     │  │
│  │ duplicates   │ Exact + near-duplicate detect  │  │
│  │ outliers     │ Statistical outlier detection   │  │
│  │ subsample    │ Stratified / importance-based   │  │
│  │ augment      │ LLM-based synthetic generation  │  │
│  │ split        │ Train/val/test split strategy   │  │
│  │ export       │ Save processed dataset          │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  Output: List[ModuleResult] + artifacts dir        │
└──────────────────┬─────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────┐
│  PHASE 4: SYNTHESIS & REPORT  (1 LLM call)       │
│                                                    │
│  Collects all ModuleResults. Produces:             │
│    • analysis_report.md — human-readable summary   │
│    • analysis_report.json — machine-readable       │
│    • recommendations.md — actionable next steps    │
│    • figures/ — all generated visualizations       │
│                                                    │
│  Output directory: analysis_<timestamp>/           │
└──────────────────────────────────────────────────┘
```

### 3.3 Module Interface

Every module follows a standard contract:

```python
@dataclass
class ModuleInput:
    df: pd.DataFrame           # The data (or subset)
    profile: DataProfile       # From Phase 1
    plan: AnalysisPlan         # From Phase 2
    task_type: str             # e.g., "multiclass_classification"
    input_cols: list[str]      # Resolved input columns
    label_col: str | None      # Resolved label column
    params: dict               # Module-specific parameters

@dataclass
class ModuleResult:
    module_name: str
    status: Literal["success", "skipped", "error"]
    summary: dict              # Key findings (machine-readable)
    narrative: str             # Human-readable paragraph
    figures: list[Path]        # Generated plots
    artifacts: list[Path]      # Output files (CSVs, etc.)
    warnings: list[str]        # Issues detected
```

New modules are added by:
1. Writing a function `def run(input: ModuleInput) -> ModuleResult`
2. Registering it in the module registry (YAML or Python dict)
3. No other changes needed — the planner auto-discovers registered modules

### 3.4 Data Selection Strategy (Phase 3: `subsample` module)

The subsample module implements intelligent data minimization:

| Method | When to use | How it works |
|--------|-------------|--------------|
| **Stratified random** | Balanced classes, large dataset | Random sample preserving class ratios |
| **Class-balanced** | Imbalanced classes | Equal samples per class (downsample majority) |
| **Diversity sampling** | Text/embedding data | Embed samples, cluster, sample from each cluster |
| **Uncertainty sampling** | Iterative/active learning | Train lightweight model, select samples near decision boundary |
| **Coreset selection** | Large dataset, preserve distribution | Greedy facility location on embeddings |
| **Deduplication-first** | Noisy/scraped data | Remove near-duplicates before any selection |

Selection is **goal-aware**: the planner picks the method based on task type,
dataset size, and class distribution. User can override via explicit parameters.

**Redundancy avoidance:** The `duplicates` module runs before `subsample`.
Near-duplicate detection uses MinHash (for text) or feature hashing (for tabular)
to flag and remove redundant samples before selection.

### 3.5 Synthetic Data Generation (Phase 3: `augment` module)

When data is insufficient or imbalanced:

```
1. Identify deficit: which classes need more samples, and how many
2. Sample seed examples from existing data for the deficit class
3. Generate synthetic samples via LLM (gpt-4.1-mini per cost policy):
   - Prompt: "Generate a {task_type} example for class {label}
              similar in style to these examples: {seeds}"
   - Validate: check generated sample isn't near-duplicate of existing data
4. Merge synthetic samples into dataset with synthetic=True flag
5. Report: how many generated, quality metrics, duplication rate
```

**Cost control:** Uses `gpt-4.1-mini` for generation. Batch generation
(multiple samples per call). Estimated cost: ~$0.01 per 100 synthetic samples.

---

## 4. Target Metric

Primary metric: **analysis completeness score** — percentage of requested
analysis modules that run successfully and produce actionable findings.

Secondary metrics:
- **Column detection accuracy** — % of columns correctly classified (input/label/meta)
- **Subsample quality** — downstream model performance on subsample vs full dataset
- **Synthetic sample quality** — classifier accuracy with vs without synthetic augmentation
- **Time to insight** — wall-clock time from invocation to report

AHVS can optimize these iteratively across cycles.

---

## 5. Constraints

| Constraint | Requirement |
|------------|-------------|
| **Privacy** | Raw data never sent to LLM. Only schema + aggregated stats sent for planning. |
| **Cost** | 2-3 LLM calls per analysis (plan + synthesize + optional augment). Labeling uses gpt-4.1-mini only. |
| **Reproducibility** | Every run produces a timestamped output directory with full manifest. |
| **Local execution** | All modules run locally. No cloud compute for analysis. |
| **ACP mode** | Primary execution via ACP (Claude Code subscription). No external API keys required for orchestration. |
| **Extensibility** | New modules added via registry without touching core code. |

---

## 6. AHVS Integration

### 6.1 Extension Points Used

| AHVS mechanism | How data_analyst uses it |
|----------------|-------------------------|
| **Domain pack** | `ahvs/domain_packs/data_analyst_prompts.yaml` + `data_analyst_skills.yaml` |
| **Skill library** | Registers analysis modules as AHVS skills (EDA, subsample, augment, etc.) |
| **CLI subcommand** | `ahvs data_analyst --data <path> --task <type> --goal "..."` |
| **Claude Code skill** | `/ahvs_data_analyst` slash command via `.claude/skills/` |
| **Memory** | Writes analysis results to `<target_repo>/.ahvs/memory/` |
| **Evolution store** | Logs lessons (e.g., "diversity sampling outperformed random for ABSA") |

### 6.2 Package Structure (new files)

```
ahvs/
├── data_analyst/
│   ├── __init__.py           # Public API: analyze(), DataProfile, AnalysisPlan
│   ├── profiler.py           # Phase 1: data loading, schema inference, column detection,
│   │                         #   feasibility check (adapted from KD data_loader +
│   │                         #   column_matcher + data_intelligence)
│   ├── planner.py            # Phase 2: LLM-assisted goal alignment + plan [NEW]
│   ├── executor.py           # Phase 3: module execution engine [NEW]
│   ├── synthesizer.py        # Phase 4: report + export (adapted from KD reporters)
│   ├── models.py             # Dataclasses: DataProfile, AnalysisPlan, ModuleInput/Result [NEW]
│   ├── registry.py           # Module registry (discover + load modules) [NEW]
│   ├── validators.py         # Consolidated validation framework (adapted from KD validators)
│   └── modules/
│       ├── __init__.py
│       ├── eda.py            # Descriptive stats + data quality + LLM insights
│       │                     #   (adapted from KD DataQualityAnalyzer + LLMInsightsGenerator)
│       ├── class_balance.py  # Label frequency + imbalance metrics
│       │                     #   (adapted from KD ClassDistributionAnalyzer)
│       ├── text_stats.py     # Token stats, vocab, per-class breakdown
│       │                     #   (adapted from KD TextStatisticsAnalyzer)
│       ├── correlation.py    # Feature correlation matrix, VIF [NEW]
│       ├── duplicates.py     # MinHash LSH near-duplicate detection
│       │                     #   (adapted from KD deduplicate_lsh)
│       ├── outliers.py       # IQR, z-score, isolation forest [NEW]
│       ├── subsample.py      # 6 selection strategies: stratified, class-balanced,
│       │                     #   diversity (adapted from KD sample_diverse),
│       │                     #   uncertainty, coreset, dedup-first [NEW strategies]
│       ├── augment.py        # Synthetic text + label generation
│       │                     #   (label gen adapted from KD DataAnnotator) [NEW text gen]
│       ├── split.py          # Train/val/test stratified split [NEW]
│       └── export.py         # CSV, Parquet, JSON, HuggingFace export [NEW]
├── domain_packs/
│   ├── data_analyst_prompts.yaml   # Hypothesis prompts for data analysis
│   └── data_analyst_skills.yaml    # Skill definitions for modules
```

### 6.3 CLI Interface

```bash
# Full automatic analysis
ahvs data_analyst --data absa_test.parquet --task classification --goal "ABSA sentiment"

# Specific modules only
ahvs data_analyst --data data.csv --modules eda,class_balance,subsample

# With synthetic augmentation
ahvs data_analyst --data data.csv --task classification --augment --augment-target 1000

# As part of AHVS cycle (domain pack mode)
ahvs --repo ./my_project --question "Optimize training data for ABSA" --domain data_analyst
```

### 6.4 Python API

```python
from ahvs.data_analyst import analyze, DataProfile

# One-call analysis
result = analyze(
    data_path="absa_test.parquet",
    task="classification",
    goal="aspect-based sentiment analysis",
    modules=None,  # auto-select
    output_dir="analysis_output/"
)

# Granular access
from ahvs.data_analyst.profiler import profile_data
from ahvs.data_analyst.modules.subsample import run as subsample

profile = profile_data("absa_test.parquet")
subset = subsample(ModuleInput(df=profile.df, profile=profile, ...))
```

---

## 7. Risks & Open Questions

| Risk | Mitigation |
|------|------------|
| **Column detection errors** | Always present detected columns for user confirmation before proceeding. Never auto-commit to wrong column roles. |
| **Synthetic data quality** | Validate generated samples: check for near-duplicates, run lightweight classifier sanity check, flag low-confidence generations. |
| **Scope creep** | v1 focuses on tabular + text for classification. CV, regression, NER are v2. |
| **Module explosion** | Start with 10 core modules. New modules require a clear use case + test coverage. |
| **Large files** | Stream-read large files (Parquet chunking, CSV iterator). Profile on sample first, then full data. |

**Open questions:**
1. Should the subsample module support active learning loops (iterative label-then-select)?
2. Should augmentation support image data (diffusion models) or text-only for v1?
3. How should the agent handle multi-file datasets (e.g., train.csv + test.csv)?

---

## 8. Prior Art & Lessons

| Tool | What it does well | Gap for our use case |
|------|-------------------|---------------------|
| **KD Pipeline (ours)** | 16 reusable components: data loading, 3-layer column matching, class distribution, text stats, data quality, LSH dedup, diversity sampling, validators, LLM insights, reporting, feasibility checks | NLP-classification-only, no synthetic text gen, no outlier/correlation analysis, no interactive viz, no goal-directed planning, no multi-format export |
| **pandas-profiling / ydata-profiling** | Excellent auto-EDA | No goal-awareness, no selection, no augmentation |
| **sweetviz** | Good comparison reports | Static, no ML task awareness |
| **AIDE ML (WecoAI)** | Tree-search over code solutions | No data analysis primitives, generates full scripts |
| **agentic-data-scientist (K-Dense)** | Multi-agent orchestration | Welded to Google ADK, no built-in analysis modules |
| **ChatGPT Code Interpreter** | Flexible, natural language | Not reproducible, not integrated, data leaves machine |

**Key differentiator:** `ahvs_data_analyst` is the only tool that combines
goal-aware analysis, intelligent subsampling, synthetic augmentation, and
AHVS-native iterative optimization — all running locally with reproducible output.

---

## 9. KD Pipeline — Reusable Components

The existing Knowledge Distillation pipeline (`/home/ubuntu/vision/hackathon_knowledge_distillation/`)
already implements significant data analysis infrastructure. Rather than rebuilding,
`ahvs_data_analyst` will **import and extend** these proven components.

### 9.1 Components to Reuse Directly

| Component | KD Source File | What It Does | How data_analyst Uses It |
|-----------|---------------|--------------|--------------------------|
| **DataLoader** | `src/data_analyzer/data_loader.py` | CSV/Parquet loading, encoding detection, delimiter inference | Phase 1 profiler — extend with JSON/JSONL/SQL/HuggingFace support |
| **ColumnMatcher** | `src/data_analyzer/column_matcher.py` | 3-layer column detection (fuzzy + heuristic + LLM semantic) | Phase 2 goal alignment — generalize beyond NLP to CV/regression/multi-label |
| **Label detection** | `data_loader.py:find_label_column()` | 4-layer pattern matching for label columns | Phase 1 profiler — as-is, extend patterns for regression targets |
| **ClassDistributionAnalyzer** | `src/data_analyzer/analyzers/class_distribution.py` | Counts, %, imbalance ratio, entropy, missing/unexpected classes | `class_balance` module — wrap with ModuleResult interface |
| **TextStatisticsAnalyzer** | `src/data_analyzer/analyzers/text_statistics.py` | Length/word stats, vocab size, duplicates, empty text detection | `text_stats` module — add per-class breakdown, percentile plots |
| **DataQualityAnalyzer** | `src/data_analyzer/analyzers/data_quality.py` | Nulls, encoding issues, annotation status, composite quality score | `eda` module — integrate into data quality section |
| **deduplicate_lsh()** | `src/data_collector/deduplicator.py` | MinHash + LSH fuzzy duplicate detection | `duplicates` module — as-is |
| **sample_diverse()** | `src/data_collector/sampler.py` | Cluster-based diversity-preserving sampling | `subsample` module — as one strategy alongside stratified/coreset/uncertainty |
| **Validation framework** | `src/data_analyzer/validators/` | 4 validators (sample size, class balance, data quality, task alignment) | Phase 1 quality checks — extend with custom validator support |
| **LLMInsightsGenerator** | `src/data_analyzer/llm/insights.py` | Per-class qualitative analysis + task fit evaluation via LLM | Phase 4 synthesis — integrate into report generation |
| **MarkdownReporter** | `src/data_analyzer/reporters/markdown.py` | Structured markdown report with recommendations | Phase 4 reporting — extend with visualization sections |
| **JSONExporter** | `src/data_analyzer/reporters/json_export.py` | Machine-readable analysis export | Phase 4 reporting — as-is |
| **Recommendation engine** | `src/data_analyzer/agent.py` | Status-based recommendations (NOT_RECOMMENDED → RECOMMENDED) | Phase 4 — extend with per-module recommendations |
| **Feasibility check** | `src/kd_agent/data_intelligence.py` | Pre-flight validation (row count, nulls, API keys, column alignment) | Pre-Phase 1 gate — generalize beyond KD-specific checks |
| **DataAnnotator** | `src/annotator/annotator.py` | LLM-based label generation with structured output | `augment` module — reuse for label generation, build text generation on top |

### 9.2 Components to Build New (KD Gaps)

| Capability | Why KD Doesn't Have It | data_analyst Implementation |
|------------|------------------------|----------------------------|
| **Synthetic text/sample generation** | KD only generates labels, never text | `augment` module: seed examples → LLM generates new samples → dedup validate |
| **Outlier detection** | Not needed for KD's label-then-train flow | `outliers` module: IQR, z-score, isolation forest |
| **Correlation analysis** | KD is text-only, no numeric features | `correlation` module: Pearson/Spearman matrix, VIF, feature importance |
| **Interactive visualizations** | KD uses static markdown only | `eda` module: matplotlib/plotly charts (distributions, box plots, heatmaps) |
| **Advanced subsampling** | KD only has cluster-diverse | `subsample` module: add uncertainty, coreset, stratified, dedup-first strategies |
| **Multi-format export** | KD outputs CSV only | `export` module: CSV, Parquet, JSON, HuggingFace Dataset |
| **Train/val/test split** | KD handles this internally during training | `split` module: stratified split with configurable ratios |
| **Goal-directed planning** | KD has fixed pipeline stages | Phase 2 planner: LLM selects modules based on user goal + data profile |

### 9.3 Integration Strategy — Copy and Adapt (Standalone)

**Decision:** Fork the relevant KD code into `ahvs/data_analyst/`. No runtime
dependency on the KD repo. The data_analyst is a fully self-contained package.

**Rationale:** The KD pipeline is NLP-classification-specific. The data_analyst
needs to generalize to CV, regression, NER, multi-label, etc. The copied code
will diverge quickly, so a dependency would create friction rather than save work.

```
hackathon_knowledge_distillation/src/       BB_AHVS/ahvs/data_analyst/
                                            (copy + generalize)
├── data_analyzer/
│   ├── data_loader.py            ──copy──► profiler.py (loading section)
│   ├── column_matcher.py         ──copy──► profiler.py (column detection section)
│   ├── analyzers/
│   │   ├── class_distribution.py ──copy──► modules/class_balance.py
│   │   ├── text_statistics.py    ──copy──► modules/text_stats.py
│   │   └── data_quality.py       ──copy──► modules/eda.py (quality section)
│   ├── validators/               ──copy──► validators.py (consolidated)
│   ├── reporters/
│   │   ├── markdown.py           ──copy──► synthesizer.py (report section)
│   │   └── json_export.py        ──copy──► synthesizer.py (export section)
│   └── llm/insights.py           ──copy──► modules/eda.py (LLM insights section)
├── data_collector/
│   ├── deduplicator.py           ──copy──► modules/duplicates.py
│   └── sampler.py                ──copy──► modules/subsample.py (one strategy)
└── kd_agent/
    └── data_intelligence.py      ──copy──► profiler.py (feasibility section)
```

**Key changes during adaptation:**
- Remove all KD-specific assumptions (task_id patterns, annotation_status, completion columns)
- Generalize column detection beyond NLP: add numeric feature detection, image path detection, regression target detection
- Replace KD's `ColumnMapping`/`TaskSpec` with data_analyst's `DataProfile`/`AnalysisPlan`
- Convert all outputs to `ModuleResult` interface
- Replace hardcoded `gpt-4.1-nano` with configurable LLM client (respecting cost policy)
- Add type hints and docstrings consistent with AHVS codebase style

---

## 10. Implementation Phases

| Phase | Scope | Modules | Build vs Reuse |
|-------|-------|---------|----------------|
| **v1 — Core** | Profiling + EDA + selection for tabular/text classification. Wire up KD adapters, build module executor + planner. | profiler, eda, class_balance, text_stats, duplicates, subsample, split, export | **70% reuse** (KD adapters for loader, column matcher, 3 analyzers, dedup, sampling, validators, reporters) + **30% new** (planner, executor, module interface, split, export) |
| **v2 — Augmentation + Advanced** | Synthetic text generation, outlier & correlation analysis, advanced selection | augment, correlation, outliers, uncertainty sampling | **90% new** (KD has label gen only, no outlier/correlation/uncertainty) |
| **v3 — Multi-task** | CV support, regression, NER, multi-file datasets | image_stats, regression_stats, entity_stats | **100% new** |
| **v4 — Active Learning** | Iterative label-select loops with AHVS cycles | active_learning module, human-in-the-loop labeling | **100% new** |
