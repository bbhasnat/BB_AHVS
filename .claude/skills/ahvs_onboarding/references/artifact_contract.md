# Artifact Contract

`ahvs_onboarding` should treat `.ahvs/baseline_metric.json` as the required machine-readable onboarding artifact for AHVS.

## Required File

Path:

- `<target>/.ahvs/baseline_metric.json`

## Required Fields

The file must contain:

- `primary_metric` — name of the metric to optimize
- a numeric field whose key matches `primary_metric` — current baseline value
- `recorded_at` — ISO-8601 timestamp
- `eval_command` — headless shell command that prints `metric_name: value` to stdout

Recommended:

- `commit` — git SHA when baseline was recorded

## Enriched Fields

These fields significantly improve AHVS hypothesis quality. Always include them when the information is available:

- `optimization_goal` — plain English description of what to optimize and any constraints
- `regression_floor` — secondary metrics with minimum acceptable values (e.g. `{"f1_score": 0.62}`)
- `constraints` — budget limits, model restrictions, hypothesis scope requirements
- `system_levers` — tunable parameters in the codebase (strategies, modes, algorithmic areas)
- `prior_experiments` — results from past experiments with config details and problems identified
- `notes` — additional context (number of records, key insights, known hard cases)

## Minimal Example

```json
{
  "primary_metric": "answer_relevance",
  "answer_relevance": 0.74,
  "recorded_at": "2026-03-18T10:00:00Z",
  "commit": "abc1234",
  "eval_command": "python scripts/eval.py --dataset data/test.jsonl"
}
```

## Enriched Example

```json
{
  "primary_metric": "precision",
  "precision": 0.7128,
  "f1_score": 0.6699,
  "recall": 0.6319,
  "recorded_at": "2026-03-19T10:00:00Z",
  "commit": "9c41b35",
  "eval_command": "cd /path/to/project && python -m package.run_eval --eval-only",
  "optimization_goal": "Maximize precision while keeping f1_score >= 0.62",
  "regression_floor": {
    "f1_score": 0.62
  },
  "constraints": {
    "model_budget": "Use cheap model only. No expensive models.",
    "hypothesis_scope": "Not limited to prompt changes. Must include algorithmic changes."
  },
  "system_levers": {
    "strategies": ["random", "keyword", "semantic"],
    "scoring_modes": ["five_class", "direct_binary"],
    "algorithmic_areas": [
      "post_selector.py: selection strategy, keyword scoring, similarity thresholds",
      "parsing.py: binary mapping thresholds, fallback defaults",
      "prompts.py: analyst prompt structure, guard wording"
    ]
  },
  "prior_experiments": {
    "best_precision": {"config": "expensive_model_prv2", "precision": 0.76, "f1": 0.57, "problem": "F1 too low"},
    "best_f1": {"config": "cheap_model_kw_prv1", "precision": 0.71, "f1": 0.67, "problem": "Precision not high enough"}
  },
  "notes": "12 cohorts, 1557 records. Keyword selection beats random. Over-labeling guard helps precision but hurts recall."
}
```

## Rules

1. The metric value must be numeric.
2. `recorded_at` should be ISO-8601.
3. `eval_command` must be headless — no notebook, no interactive input.
4. `eval_command` output must be parseable: `metric_name: numeric_value` on stdout.
5. `commit` should be included when the target is a git repo.
6. Do not write placeholder values such as `0`, `TODO`, or `fill me in later` unless the user explicitly asks for a draft-only artifact and understands it is not AHVS-ready.
7. The enriched fields (`constraints`, `system_levers`, `prior_experiments`) are not required but strongly recommended — they make AHVS hypothesis generation significantly better.

## Optional Related Artifacts

The skill may also create:

- A CLI eval script (e.g. `run_eval.py`) when evaluation only exists in notebooks
- `.ahvs/regression_guard.sh` — regression protection script
- `.ahvs/eval/*` — eval configs
- `.ahvs/AHVS_preparation.md` — human-readable record of what onboarding did
