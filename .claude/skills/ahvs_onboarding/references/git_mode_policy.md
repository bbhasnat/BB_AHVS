# Git Mode Policy

AHVS requires a git-backed target repository. Worktree-based isolation is the only supported execution mode.

## Git Repo Mode

If the target is inside a git repository:

1. record the current commit SHA in `.ahvs/baseline_metric.json`
2. explain that AHVS uses detached worktrees for per-hypothesis execution
3. describe this as the standard execution mode

Suggested explanation:

> This project is in git, so AHVS will evaluate hypotheses in detached worktrees tied to a real commit. That gives strong reproducibility and clean patch tracking.

## Non-Git Target

If the target is not a git repo:

1. **block onboarding** — AHVS requires git
2. instruct the user to run `git init` and commit their code first
3. explain why: AHVS creates per-hypothesis worktrees from committed HEAD

Suggested explanation:

> This target is not in git. AHVS requires a git repository because it creates isolated worktrees for each hypothesis. Please run `git init && git add -A && git commit -m "initial"` first.

## Single File or Loose Directory

Single-file and loose-directory projects must be placed under version control before onboarding.

## Blocking Rule

Missing git is a **hard blocker**. Do not proceed with onboarding until the target is a git repository.
