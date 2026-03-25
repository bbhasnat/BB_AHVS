---
name: ahvs_onboarding
description: >-
  Onboards any repository for AHVS (Automated Hypothesis Validation System).
  Inspects the codebase, creates a headless CLI eval script if evaluation only
  exists in notebooks/interactive form, writes .ahvs/baseline_metric.json with
  optimization goals and constraints, and verifies the eval command works.
  Use this skill whenever the user says "onboard for AHVS", "prepare for AHVS",
  "set up AHVS", "ahvs onboard", "get this repo ready for AHVS", or mentions
  wanting to run AHVS on a project that doesn't have .ahvs/ yet. Also trigger
  when the user asks to "improve a metric" or "optimize evaluation" and AHVS
  setup is missing. If you see a repo without .ahvs/baseline_metric.json and
  the user wants to run ahvs, this skill should fire first.
---

# AHVS Onboarding

This skill turns any repository into an AHVS-ready target. It handles the full onboarding pipeline: codebase analysis, eval command extraction, baseline configuration, and verification.

The user should never need to manually author JSON or figure out what AHVS needs. This skill inspects the repo, creates whatever is missing, asks only the smallest necessary follow-up questions, and writes verified artifacts.

## What This Skill Does and Does Not Do

**Does:**
1. Inspect the target repo to understand its evaluation pipeline
2. Create a headless CLI eval script when evaluation only exists in notebooks or interactive form
3. Write `.ahvs/baseline_metric.json` with metric, constraints, system levers, and prior experiments
4. Verify the eval command produces parseable metric output
5. Align on optimization goals with the user (which metric, budget, hypothesis diversity)
6. Refuse to advance when onboarding is incomplete or unsafe

**Does not:**
1. Invent baseline values or guess fake eval commands
2. Start an AHVS cycle — that's a separate step after onboarding
3. Hide reduced-trust mode when the target is not in git

## Return Contract

Every onboarding pass ends with one of three statuses:

| Status | Meaning | What happens next |
|--------|---------|-------------------|
| `ready` | All gates passed, artifacts written, eval verified | User can run `ahvs` |
| `needs_user_input` | Promising but missing info | Ask the follow-up, then re-evaluate |
| `blocked` | Cannot proceed safely | Explain the blocker and what must change |

Return `ready` **only** when ALL of these are true:
1. Target path exists and is understood
2. Optimization metric is clearly identified
3. A **headless** evaluation command exists and has been tested
4. Baseline value is known (from existing results or a verified run)
5. `.ahvs/baseline_metric.json` is internally consistent
6. The eval command stdout is parseable (prints `metric_name: value`)

## Workflow

### Phase 1: Discover — Deep Codebase Inspection

Use Glob and Read to scan the target comprehensively. Check these paths first:

```
README*, pyproject.toml, requirements.txt, package.json, Makefile,
.github/workflows/*, .ahvs/*, promptfoo*, eval.py, scripts/eval*,
tests/, benchmarks/, *.yaml configs, *.ipynb notebooks,
checkpoints/, results/, experiments.*
```

From this inspection, identify:
- **Project type** (Python package, Node app, RAG pipeline, ML model, classifier, etc.)
- **Evaluation pipeline** — how metrics are currently computed. This is critical:
  - CLI script? → may be usable directly
  - Jupyter notebook? → need to extract into a CLI script
  - CI pipeline? → extract the eval step
  - No evaluation? → blocked — cannot onboard
- **Candidate metric names** (from eval output, checkpoint files, experiment registries)
- **Existing experiments** (checkpoint files, experiment YAML, prior results)
- **System levers** — what parameters can be tuned (models, prompts, algorithms, configs)
- **Python environment** — which conda/venv has the project's dependencies
- **Data dependencies** — ground truth files, datasets, API keys needed

Use Grep to search for metric patterns:
```
accuracy, relevance, f1, precision, recall, score, bleu, rouge, loss,
eval_command, eval.py, evaluate, benchmark
```

### Phase 2: Extract or Create Eval Command

This is the hardest and most important step. AHVS needs a shell command that:
1. Runs headlessly (no notebook, no interactive input)
2. Prints `metric_name: numeric_value` to stdout
3. Is reproducible (same inputs → same output)

**Case A: CLI eval script already exists**
- Verify it prints the metric to stdout in parseable form
- If it doesn't print to stdout, wrap it or modify output
- Test it

**Case B: Evaluation lives in a Jupyter notebook**
This is the most common case. Extract the eval logic into a standalone Python script:

1. Read the notebook cells to understand the full pipeline:
   - Data loading (parquet, CSV, database, API)
   - Inference loop (LLM calls, model predictions)
   - Metric computation (P/R/F1, accuracy, custom metrics)
2. Create `<package>/run_eval.py` (or similar) that:
   - Loads config from the project's existing config file
   - Loads ground truth data (use pandas, not Spark, for simplicity)
   - Supports `--eval-only` mode (evaluate existing checkpoint, no inference — fast and free)
   - Supports full run mode (inference + evaluation)
   - Prints metrics to stdout: primary metric first, then secondary metrics
   - Sends detailed logs to stderr (so stdout stays clean for AHVS parsing)
   - Supports CLI overrides for key parameters (model, prompt version, etc.)
3. Test the script works in eval-only mode against existing checkpoints

**Case C: Evaluation is in CI or a Makefile**
- Extract the eval command from CI config
- Verify it works locally
- May need env var setup

**Case D: No evaluation exists**
- Return `blocked` — explain that AHVS needs a measurable metric
- Suggest how the user could create one

### Phase 3: Align — Understand Optimization Goals

Ask the user (if not already clear from context):

1. **"What metric should AHVS optimize?"** — The primary metric. Must be numeric.
2. **"Are there metrics that must not regress?"** — Secondary metrics with floors (regression guards).
3. **"What's your model/cost budget?"** — Which models are acceptable, cost-per-row limits.
4. **"Should hypotheses go beyond prompt changes?"** — Algorithmic changes, data strategies, threshold tuning, architecture changes.

Encode all answers into the baseline file. If the user says "improve precision but don't tank F1", that becomes:
```json
{
  "primary_metric": "precision",
  "regression_floor": {"f1_score": 0.62}
}
```

### Phase 4: Gather Prior Experiments

Check for existing experiment results (checkpoint files, experiment registries, reports). Prior experiments are gold for AHVS — they prevent repeating dead ends.

Look for:
- Experiment registry files (YAML, JSON)
- Checkpoint directories with results
- Reports or analysis documents
- Comments in config files about what was tried

Encode findings as `prior_experiments` in the baseline:
```json
{
  "prior_experiments": {
    "best_precision": {"config": "model_x_strategy_y", "precision": 0.75, "f1": 0.57, "problem": "F1 too low"},
    "best_f1": {"config": "model_a_strategy_b", "precision": 0.71, "f1": 0.67, "problem": "Precision not high enough"}
  }
}
```

### Phase 5: Document System Levers

Identify all tunable parameters in the codebase and encode them so AHVS generates diverse hypotheses:

```json
{
  "constraints": {
    "model_budget": "Use model X only. No expensive models.",
    "hypothesis_scope": "Not limited to prompt changes. Must include algorithmic changes."
  },
  "system_levers": {
    "strategies": ["list of available strategies"],
    "scoring_modes": ["list of modes"],
    "algorithmic_areas": [
      "module.py: what can be changed here",
      "other_module.py: what can be tuned"
    ]
  }
}
```

This ensures AHVS hypotheses propose real algorithmic changes (post selection, threshold tuning, scoring logic) not just prompt rewrites.

### Phase 6: Write and Verify Artifacts

Create `.ahvs/` directory, then write:

**Required: `.ahvs/baseline_metric.json`**

The enriched schema (see `references/artifact_contract.md` for required fields):
```json
{
  "primary_metric": "<metric_name>",
  "<metric_name>": <numeric_value>,
  "recorded_at": "<ISO-8601>",
  "commit": "<git SHA>",
  "eval_command": "<reproducible headless command>",
  "optimization_goal": "<what to optimize and constraints in plain English>",
  "regression_floor": {"<secondary_metric>": <minimum_value>},
  "constraints": {"model_budget": "...", "hypothesis_scope": "..."},
  "system_levers": {"...": "..."},
  "prior_experiments": {"...": "..."},
  "notes": "<context for AHVS hypothesis generation>"
}
```

**Required: eval script** (if created in Phase 2)

**Optional:**
- `.ahvs/regression_guard.sh`
- `.ahvs/eval/*`
- `.ahvs/AHVS_preparation.md` — human-readable record of what was done

### Phase 7: Verify — Test Everything

Before returning `ready`, run the eval command and verify:

```bash
# Run eval command, capture stdout only
<eval_command> 2>/dev/null
```

Check that:
1. It exits with code 0
2. stdout contains `<primary_metric>: <numeric_value>`
3. The value matches the baseline (or is close if it's a new measurement)

If verification fails, diagnose and fix before returning `ready`.

### Phase 8: Gate — Return Status

Present a clear status block:

```
## AHVS Onboarding: [STATUS]

**Target:** /path/to/repo (git / non-git)
**Metric:** precision = 0.7128 (optimize ↑)
**Regression floor:** f1_score >= 0.62
**Model budget:** [constraints]
**Eval command:** [command]
**Eval verified:** yes — output matches baseline
**Files written:** .ahvs/baseline_metric.json, src/package/run_eval.py

**Warnings:**
- (any caveats)

**Next step:**
ahvs --repo /path/to/repo --question "..."
```

## Git Mode Policy

Always check `git rev-parse --is-inside-work-tree` on the target.

**Git repo:**
- Record current commit SHA in baseline
- AHVS uses detached worktrees for per-hypothesis execution — higher trust
- This is the recommended mode

**Not a git repo:**
- Do NOT block — AHVS auto-initializes git at Stage 1 if needed
- Inform the user that git will be auto-initialized for worktree isolation
- The `.git` directory persists after the cycle

## Failure Modes

**Return `blocked` when:**
- Target path does not exist
- Metric is too vague to measure (e.g. "make it better")
- No reproducible eval path can be established — and none can be created
- No ground truth or benchmark data exists

**Return `needs_user_input` when:**
- Repo looks promising but metric needs confirmation
- Found a likely eval command but it needs approval
- Baseline value is missing but can likely be measured
- Optimization goal or constraints are unclear

## Reference Files

Read these as needed — they contain the detailed policies:

- `references/artifact_contract.md` — baseline JSON schema, required vs optional fields
- `references/eval_command_policy.md` — how to accept or reject candidate eval commands
- `references/git_mode_policy.md` — git vs non-git execution trust model
