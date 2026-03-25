# Git Mode Policy

AHVS behaves differently depending on whether the target is a git repo.

## Git Repo Mode

If the target is inside a git repository:

1. prefer recording the current commit SHA in `.ahvs/baseline_metric.json`
2. explain that AHVS can use detached worktrees for per-hypothesis execution
3. describe this as the higher-trust and more reproducible mode

Suggested explanation:

> This project is in git, so AHVS can evaluate hypotheses in detached worktrees tied to a real commit. That gives stronger reproducibility and cleaner patch tracking.

## Non-Git Mode

If the target is not a git repo:

1. do not block by default
2. explain that AHVS will fall back to sandbox-only execution
3. explain that this reduces reproducibility, traceability, and patch management confidence

Suggested explanation:

> This target is not in git, so AHVS can still run, but only in sandbox-only mode. That is lower trust than repo-grounded worktrees because there is no commit anchor and patch tracking is weaker.

## Single File or Loose Directory

Single-file and loose-directory projects are allowed, but they should be described as lower-trust unless the user first places them under version control.

## Blocking Rule

Missing git should usually be a warning, not a blocker.

Block only if the surrounding workflow specifically requires git-backed reproducibility and the user does not want the reduced-trust mode.
