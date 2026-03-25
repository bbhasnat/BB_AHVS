# Git Mode Policy

AHVS uses git worktrees for per-hypothesis isolation. If the target is not already a git repo, AHVS auto-initializes one at Stage 1.

## Git Repo (standard)

If the target is inside a git repository:

1. record the current commit SHA in `.ahvs/baseline_metric.json`
2. explain that AHVS uses detached worktrees for per-hypothesis execution
3. describe this as the standard execution mode

Suggested explanation:

> This project is in git, so AHVS will evaluate hypotheses in detached worktrees tied to a real commit. That gives strong reproducibility and clean patch tracking.

## Non-Git Target (auto-initialized)

If the target is not a git repo:

1. AHVS will auto-initialize git at Stage 1 (`git init && git add -A && git commit`)
2. explain this to the user — it's logged and recorded in the cycle manifest
3. the `.git` directory persists after the cycle (useful for future work)

Suggested explanation:

> This target is not in git. AHVS will auto-initialize a git repo so it can create isolated worktrees for each hypothesis. The `.git` directory will remain after the cycle.

## Blocking Rule

Missing git is handled automatically — it is not a blocker. The only hard failure is if `git init` itself fails (e.g., git is not installed).
