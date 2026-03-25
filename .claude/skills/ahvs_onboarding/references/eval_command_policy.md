# Eval Command Policy

The `eval_command` is the most important onboarding field because AHVS depends on it to measure whether a hypothesis actually improved the target system.

## Acceptable Sources

Good candidates include:

1. an existing repo evaluation script
2. an existing benchmark or CI command that emits the target metric
3. an existing Promptfoo or equivalent evaluation config
4. a newly created helper command, if its purpose is explicit and its output is parseable

## Rejection Rules

Reject a candidate command if it is:

1. only a placeholder
2. unrelated to the target metric
3. not reproducible enough to serve as a baseline measurement path
4. likely to succeed without actually emitting the metric
5. obviously stale or disconnected from the current codebase

## Validation Guidance

Before returning `ready`, try to establish that the command is credible.

Preferred order:

1. detect an existing candidate from the repo
2. explain why it looks like the right evaluation path
3. get confirmation if the command is ambiguous, expensive, or risky
4. sanity-check it when practical
5. store that exact command in `.ahvs/baseline_metric.json`

## Baseline Extraction Rule

A candidate `eval_command` is only useful if the metric value can be obtained from:

1. the command output
2. a produced artifact that clearly records the metric
3. an already existing trusted result artifact tied to that command

If no trustworthy metric extraction path exists, onboarding is not `ready`.

## Common Good Examples

- `promptfoo eval --config .ahvs/eval/baseline.yaml`
- `python scripts/eval.py --dataset data/test.jsonl`
- `make eval`

These examples are only acceptable if they really measure the intended metric for the current project.
