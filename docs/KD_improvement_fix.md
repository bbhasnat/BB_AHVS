# KD Improvement & Fix Plan

**Purpose:** Detailed TODOs for the KD repo (`hackathon_knowledge_distillation`). These fixes were identified during an end-to-end test of the KD agent (SDK mode) on 2026-04-02, running branch `h_agent_2`.

**Context:** The KD agent (claude-agent-sdk powered) ran the full pipeline from data inspection through DistilBERT training. It succeeded overall, but hit 5 bugs that it had to work around. These fixes will make the pipeline reliable without workarounds.

**Branch strategy:** Create `kd_agent_fixes` from `h_agent_2`, implement fixes, test.

**Fix strategy — bottom-up, two rounds:**
1. **Round 1: Fix the pipeline layer FIRST** — fix bugs in `src/` modules (auto_ml, annotator, tml_classifier, data_collector, data_analyzer). Test using **direct CLI** (`python -m src.auto_ml.main ...`) without any agent or MCP involvement. This isolates pipeline bugs from agent bugs.
2. **Round 2: Fix the MCP/agent layer AFTER** — once the pipeline works clean via CLI, update `src/kd_agent/tools/` MCP wrappers to match. Test via the agent SDK.

This ensures you always know which layer is broken when something fails.

**LLM Policy:**
- **Default: ACP (Claude Code / Codex subscription)** — use for all agent orchestration, reasoning, code generation, and any task where Claude is sufficient. This maximizes subscription value and avoids per-call API costs.
- **When external LLM API is needed** (e.g., KD pipeline annotation, prompt building, prompt selection): use **gpt-4.1-mini** (never gpt-4o) or **Gemini 2.5 Flash Lite**. Update all default model configs accordingly.
- **Never use gpt-4o** — it is cost-prohibitive for annotation/labeling workloads.

---

## Bug 1: Data Collector Config Key Mismatch

**Round:** 1 (pipeline layer) + 2 (MCP layer)

**Symptom:** Stage 2 (data collection) fails. The config YAML structure doesn't match what `data_collector/pipeline.py` expects.

**Root cause:** The config structure passed to `run_pipeline()` uses wrong keys.

**Round 1 — fix pipeline layer:**
1. Read `src/data_collector/pipeline.py` and `src/data_collector/config.py` to find the exact expected config schema
2. Read `examples/auto_label/config.yaml` for a working reference config
3. Test data collector directly via CLI:
   ```bash
   python -m src.data_collector.main examples/data_collector/config_local_databricks.yaml \
     -o /tmp/test_collected.csv
   ```
4. If the CLI works with the example config, the pipeline layer is fine — the bug is only in how the agent builds the config.

**Round 2 — fix MCP layer (only after Round 1 passes):**
- `src/kd_agent/config_builder.py` — update `data_collector` section in `build_default_config()` to match the working config schema
- `src/kd_agent/tools/stage_2_data_collector.py` — verify the tool passes the right parameters

**Acceptance:**
- Round 1: `python -m src.auto_ml.main` with a correct config completes stage 2
- Round 2: The `collect_data` MCP tool works without errors

---

## Bug 2: Data Analyzer Import Error

**Round:** 1 (pipeline layer)

**Symptom:** Stage 5 (dataset analysis) fails with an import error.

**Root cause:** The `src/data_analyzer/` module has an import that fails in the `py11` conda environment. Likely a missing dependency or a circular import.

**Round 1 — fix pipeline layer:**
1. Reproduce the import error directly:
   ```bash
   python -c "from src.data_analyzer.agent import DataAnalyticsAgent"
   ```
2. Read the full traceback to identify the missing module or circular import
3. Fix the import (install missing package, fix circular import, or add lazy import)
4. Test via CLI:
   ```bash
   python -m src.data_analyzer.main \
     --spec examples/prompt_builder/example_spec_sentiment.yaml \
     --data project_data_temp/sentiment_run/annotated_data.csv \
     --output /tmp/test_report.md
   ```

**Round 2 — MCP layer:** Should work automatically once the import is fixed, since the MCP tool (`src/kd_agent/tools/stage_5_data_analyzer.py`) just calls `DataAnalyticsAgent`. Verify after Round 1.

**Acceptance:** `python -m src.data_analyzer.main` completes and produces a quality report.

---

## Bug 3: TML Classifier Config Format Mismatch

**Round:** 1 (pipeline layer — verify CLI works) + 2 (MCP layer — fix config generation)

**Symptom:** Stage 6 (train TML model) fails with `Missing required field: project_id`.

**Root cause:** The config YAML built for `ClassifierAgent` doesn't match the schema expected by `src/tml_classifier/spec_parser.py`.

**Round 1 — verify pipeline layer works with correct config:**
1. Read `examples/tml_classifier/classifier_spec_example.yaml` to see the expected YAML structure
2. Read `src/tml_classifier/spec_parser.py` to see all required fields
3. Test directly via CLI with the example config:
   ```bash
   python -m src.tml_classifier.main \
     --spec examples/tml_classifier/classifier_spec_example.yaml --verbose
   ```
4. If this works, the pipeline is fine — the bug is in how the MCP tool builds the config
5. If this fails, fix `spec_parser.py` or the example config first

**Round 2 — fix MCP layer:**
- `src/kd_agent/tools/stage_6_tml_classifier.py` — update the config dict it builds to match the correct schema. Required fields include at minimum:
  - `version` (int), `project_name` (str), `project_id` (str)
  - `data.csv_path`, `data.input_column`, `data.output_column`, `data.train_test_split`
  - `training.model`, `training.hyperparameters.num_train_epochs`, etc.
  - `evaluation.metrics`
  - `output.model_save_path`

**Acceptance:**
- Round 1: `python -m src.tml_classifier.main --spec <correct_config>` trains successfully
- Round 2: `train_tml_model` MCP tool produces a valid config and training completes

---

## Bug 4: Annotation Produces Freeform Labels Instead of Spec Classes

**Round:** 1 (pipeline layer)

**Symptom:** After annotation (Stage 4), the `completion` column contains freeform labels (e.g., "gratitude", "distress", "boredom") instead of the spec-defined classes (e.g., "neutral", "happy", "anger", "excited").

**Root cause:** The annotator prompts the LLM to label text, but the prompt doesn't constrain output to ONLY the classes defined in the task spec. The LLM interprets freely.

**Round 1 — fix pipeline layer:**
1. Read the `selected_prompt.json` from a test run to see the actual prompt template
2. Check if the prompt includes explicit instructions like "You MUST classify into one of these categories: [neutral, happy, anger, excited]"
3. If not, fix at the source:
   - Option A (preferred): Fix in `src/prompt_builder/compiler.py` — ensure all generated prompts include strict class enumeration
   - Option B: Fix in `src/annotator/pipeline.py` — append class constraint to every prompt before sending to LLM
   - Option C (fallback): Fix in `src/annotator/output_handler.py` — post-process completions to map to nearest spec class
4. Test via direct CLI:
   ```bash
   python -m src.annotator.main \
     --data sample_data/train_sentiment_small.csv \
     --prompt project_data_temp/sentiment_run/selected_prompt.json \
     --output /tmp/test_annotated.csv
   # Then check: python -c "import pandas as pd; print(pd.read_csv('/tmp/test_annotated.csv')['completion'].head())"
   ```

**Round 2 — MCP layer:** No MCP changes needed — the MCP tool just calls the annotator pipeline.

**Acceptance:** After annotation, the label column should only contain values from the spec's class list.

---

## Bug 5: CSV Encoding Issue (Latin-1 characters)

**Round:** 1 (pipeline layer)

**Symptom:** `sample_data/train_sentiment_small.csv` contains a `Land Area (Km²)` column with a Latin-1 encoded character (²). The annotator fails with a UTF-8 decoding error.

**Root cause:** The annotator's CSV reader uses default UTF-8 encoding and doesn't handle Latin-1 fallback.

**Round 1 — fix pipeline layer:**
1. Find all `pd.read_csv()` calls in `src/annotator/`
2. Add encoding detection or fallback: try UTF-8 first, catch `UnicodeDecodeError`, retry with `latin-1`
3. Consider also fixing in `src/data_collector/reader.py` and any other CSV readers
4. Test:
   ```bash
   python -m src.annotator.main \
     --data sample_data/train_sentiment_small.csv \
     --prompt examples/prompt_selector/final_selected_prompt_kd_interface.json \
     --output /tmp/test_encoding.csv
   ```

**Round 2 — MCP layer:** No MCP changes needed.

**Acceptance:** Annotation works on CSVs with non-UTF-8 characters without manual encoding conversion.

---

## Improvement 1: Change Default LLM Models (Cost Optimization)

**Round:** 1 (pipeline layer) + 2 (MCP layer)

**Problem:** The KD pipeline defaults to `gpt-4o` for annotation and prompt building. This is unnecessarily expensive for bulk labeling. The test run used gpt-4o for annotating 99 rows — overkill.

**Policy:** Use `gpt-4.1-mini` (or `gemini-2.5-flash-lite` as alternative) for all pipeline LLM calls. Never use `gpt-4o`. Claude (via ACP / subscription) handles all reasoning/orchestration — no per-call cost there.

**Round 1 — fix pipeline layer (example configs + pipeline defaults):**
- `examples/auto_ml/config.yaml` — update `prompt_builder.model` and `annotator.model` → `gpt-4.1-mini`
- `examples/auto_label/config.yaml` — same
- Any hardcoded model defaults in `src/prompt_builder/`, `src/annotator/`, `src/prompt_selector/`

**Round 2 — fix MCP layer (agent defaults):**
- `src/kd_agent/config_builder.py` — change default `annotator.model` and `prompt_builder.model` → `gpt-4.1-mini`
- `src/kd_agent/tools/stage_1_prompt_builder.py` — if it passes a default model, change to `gpt-4.1-mini`
- `src/kd_agent/tools/stage_3_prompt_selector.py` — same
- `src/kd_agent/tools/stage_4_annotator.py` — same (this is the most important one — bulk labeling)
- `src/kd_agent/prompts.py` — if the system prompt mentions default models, update

**How to find all occurrences:**
```bash
grep -rn "gpt-4o\|gpt-4\.1\"" src/ examples/ --include="*.py" --include="*.yaml"
```

**Acceptance:** Running the full pipeline without explicitly specifying a model uses `gpt-4.1-mini` by default. No reference to `gpt-4o` remains in pipeline defaults, agent defaults, or example configs.

---

## Improvement 2: SDK Mode Auto-Proceed Flag

**Round:** 2 (MCP/agent layer only)

**Problem:** In SDK mode (one-shot), the agent asks "Shall I proceed?" and waits for user input that never comes. The first test run stopped here.

**Files to fix:**
- `src/kd_agent/prompts.py` — the system prompt
- `src/kd_agent/main.py` — the SDK mode entry point

**How to fix:**
1. In `main.py:run_sdk_mode()`, append to the prompt: "This is SDK mode — proceed automatically without asking for confirmation."
2. Alternatively, add a `--auto-proceed` flag that injects this instruction
3. Or modify the system prompt to detect SDK mode and skip confirmations

**Acceptance:** `python -m src.kd_agent.main --mode sdk --prompt "run full pipeline" --data data.csv --spec spec.yaml` completes without stopping at confirmation prompts.

---

## Deferred: Merge `h_agent_2` to `main`

**Status:** Deferred until all fixes are tested and verified. Do NOT merge until the user explicitly approves after full end-to-end testing.

---

## Test Plan

### Round 1 Tests (pipeline layer — no agent, direct CLI)

Run these FIRST, after fixing pipeline bugs. All should pass before touching MCP.

```bash
conda activate py11
cd /home/ubuntu/vision/hackathon_knowledge_distillation

# Test 1: Full pipeline via CLI (the most important test)
python -m src.auto_ml.main examples/auto_ml/config.yaml \
  --spec examples/prompt_builder/example_spec_sentiment.yaml \
  --data sample_data/train_sentiment_small.csv

# Expected: All stages complete, model trained, outputs in project_data_temp/

# Test 2: Individual stages that failed
python -m src.data_collector.main examples/data_collector/config_local_databricks.yaml \
  -o /tmp/test_collected.csv

python -c "from src.data_analyzer.agent import DataAnalyticsAgent; print('import OK')"

python -m src.tml_classifier.main \
  --spec examples/tml_classifier/classifier_spec_example.yaml --verbose

# Test 3: Verify default model is gpt-4.1-mini
grep -rn "gpt-4o" src/ examples/ --include="*.py" --include="*.yaml"
# Expected: zero matches (or only in comments/docs)
```

### Round 2 Tests (MCP/agent layer — only after Round 1 passes)

```bash
# Test 4: SDK mode (one-shot, auto-proceed)
CLAUDECODE="" python -m src.kd_agent.main --mode sdk \
  --data sample_data/train_sentiment_small.csv \
  --spec examples/prompt_builder/example_spec_sentiment.yaml \
  --prompt "Run the full pipeline for sentiment classification"

# Expected: All 7 stages complete via agent, no workarounds needed

# Test 5: Interactive mode (manual — chat with the agent)
python -m src.kd_agent.main --data sample_data/train_sentiment_small.csv
# Then type: "run full pipeline with spec examples/prompt_builder/example_spec_sentiment.yaml"
```

---

## Priority Order

### Round 1 (pipeline layer — do these first, test via CLI):
1. **Improvement 1** (default models → gpt-4.1-mini) — cost savings, quick win
2. **Bug 5** (CSV encoding) — easy fix, unblocks annotation on real data
3. **Bug 4** (freeform labels) — produces bad training data
4. **Bug 2** (analyzer import) — stage 5 skipped
5. **Bug 1** (data collector config — pipeline-side, if any)
6. **Bug 3** (TML config — verify CLI works with correct config)

### Round 2 (MCP/agent layer — only after Round 1 passes):
7. **Bug 1** (data collector — MCP config builder fix)
8. **Bug 3** (TML — MCP config builder fix)
9. **Improvement 1** (model defaults in MCP tools)
10. **Improvement 2** (SDK auto-proceed flag)
