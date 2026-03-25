# Failure Classification Rules

This reference defines how the observer classifies hypothesis execution outcomes
and the exact procedure for handling each category.

## Framework Protections (know these before classifying)

The AHVS framework has several built-in protections that fire automatically
during hypothesis execution. These are **features, not bugs** — do not try
to "fix" them or classify them as FRAMEWORK_BUG.

### Forbidden file filter

The framework blocks Claude Code from modifying eval-harness entry points.
When you see `[AHVS] blocked N forbidden file(s)` in the logs, this is
intentional protection working correctly.

| File | Policy | Why |
|---|---|---|
| `run_eval.py` | Hard-blocked | Eval entry point — modifying it breaks the pipeline |
| `evaluation.py` | Hard-blocked | Eval infrastructure |
| `test_*.py`, `*_test.py` | Hard-blocked | Test files — never part of eval |
| `__init__.py` | Warn-only (applied) | May be legitimate, logged for review |
| `main.py` | Warn-only (applied) | May be legitimate, logged for review |

If Claude Code's only meaningful change was in a blocked file, the hypothesis
will produce zero valid files and score `extraction_failed`. This is a
**HYPOTHESIS_MISS** — the hypothesis strategy was incompatible with the
framework's safety constraints.

### Pre-eval import sanity check

After applying Claude Code's files to the worktree, the framework verifies
the eval module can still import. When you see `pre-eval import check FAILED`
in the logs, it means Claude Code broke the module structure (circular imports,
missing dependencies, syntax errors in spliced code).

This is a **HYPOTHESIS_MISS** — Claude Code produced code that broke the
target repo's import chain. Do not classify as FRAMEWORK_BUG.

### Authoritative eval policy

When `eval_command` is configured, it is the **only trusted measurement
source**. Self-reported result.json files are skipped when eval_command is
configured. This prevents fabricated metrics.

When you see `extraction_failed` with a configured `eval_command`, it means
the real eval didn't produce a parseable metric.

### Stale worktree cleanup

The framework automatically removes stale worktrees from previous runs
before creating new ones. This is no longer a failure mode.

---

## The Three Categories

### FRAMEWORK_BUG

A genuine bug in the AHVS framework code that prevented the hypothesis
from getting a fair evaluation. This includes two subcategories:

#### Subcategory A: Direct code bugs

Errors in `ahvs/` code itself.

**Indicators:**
- `ImportError` or `ModuleNotFoundError` in `ahvs/` code (not target repo code)
- Worktree creation itself fails (not just the import check after file apply)
- `EvolutionStore` or `ContextLoader` crashes during Stage 2 or 7
- LLM client connection failure (network error, API timeout) in AHVS orchestration
- Checkpoint write/read corruption preventing stage resume
- AST splice produces invalid Python on valid input (the splicing logic itself is buggy)

#### Subcategory B: Missing guardrails

The framework failed to provide adequate context, filtering, or enforcement
to give the hypothesis a fair chance. The hypothesis idea was valid but the
framework set it up to fail.

**Indicators — these are missing-guardrail issues:**
- Claude Code writes files to repo root (e.g., `parsing.py`) when source code
  lives under a subdirectory (e.g., `src/autoqa/parsing.py`) — the framework
  did not inject the source directory path or enforce correct paths
- `prompt_rewrite` hypothesis generated when `eval_command` uses `--eval-only`
  — structurally unmeasurable, the framework should have filtered this type
  at generation time
- Hypothesis description references a file by basename without full path
  because the hypothesis generation prompt had no repo file tree context
- ≥2 hypotheses in the same cycle fail with the same root cause (e.g., all
  wrote to wrong paths) — systemic pattern, not individual hypothesis error

**How to distinguish from HYPOTHESIS_MISS:**

| Question | If YES → | If NO → |
|---|---|---|
| Did the framework tell Claude Code the correct source directory? | HYPOTHESIS_MISS | FRAMEWORK_BUG (missing guardrail) |
| Did the framework filter out unmeasurable hypothesis types? | HYPOTHESIS_MISS | FRAMEWORK_BUG (missing guardrail) |
| Did the hypothesis gen prompt include the repo file tree? | HYPOTHESIS_MISS | FRAMEWORK_BUG (missing guardrail) |
| Was the Claude Code given adequate context to place files correctly? | HYPOTHESIS_MISS | FRAMEWORK_BUG (missing guardrail) |

**Key insight:** If the framework had a guardrail that SHOULD exist but DOESN'T,
and that missing guardrail caused the failure, it's a FRAMEWORK_BUG — even though
the error manifests in Claude Code output, not in `ahvs/` stack traces.

**NOT a FRAMEWORK_BUG (common misclassifications):**
- `ImportError` in the target repo after Claude Code changes when Claude Code was
  given correct path context → HYPOTHESIS_MISS
- Files blocked by forbidden file filter → HYPOTHESIS_MISS (intentional protection)
- Pre-eval import check failure when Claude Code had correct context → HYPOTHESIS_MISS
- `extraction_failed` when eval_command is configured and Claude Code had adequate
  context → HYPOTHESIS_MISS
- eval_command crashes after Claude Code rewrote a core module with correct paths
  → HYPOTHESIS_MISS

**Action for Subcategory A:** Observer fixes the framework code, runs pytest
gate, reports fix details to team lead. Team lead decides whether to commit.
Reports RERUN_NEEDED.

**Action for Subcategory B:** Observer implements the missing guardrail in the
framework code (e.g., adds source dir detection to executor.py, adds
prompt_rewrite filtering to hypothesis gen, injects repo file tree into prompts),
runs pytest gate, reports fix details to team lead. Team lead decides whether
to commit. Reports RERUN_NEEDED.

### HYPOTHESIS_MISS

The hypothesis was given a fair chance — with adequate context, correct paths,
and a measurable hypothesis type — but didn't improve the metric.

**Indicators:**
- Metric measured but at or below baseline
- Metric regressed (negative delta)
- `extraction_failed` because Claude Code broke imports despite having correct
  path context (pre-eval check failed)
- `extraction_failed` because eval_command crashed on hypothesis-generated code
  that was correctly placed
- All Claude Code files were blocked by forbidden file filter (bad hypothesis strategy)

**Action:** Record the lesson. Report PASS. Move on.

**Key insight:** This is expected and valuable. Failed hypotheses teach the next cycle
what NOT to try. The observer must NOT:
- Lower the baseline to make the hypothesis look successful
- Modify eval thresholds
- Reclassify it as FRAMEWORK_BUG to trigger a re-run
- Edit the hypothesis code to force a pass

### AMBIGUOUS

The logs don't make it clear whether the issue is framework or hypothesis.

**Indicators:**
- Error in a shared dependency that could be either framework or hypothesis
- Worktree exists but eval produces unexpected output format (parsing issue?)
- Metric present but suspiciously identical to baseline with code_change hypothesis
  (could be wrong paths OR genuinely no-impact change)

**Action:** Escalate to the team lead with:
1. The specific log lines that are ambiguous
2. Your best guess (FRAMEWORK_BUG or HYPOTHESIS_MISS)
3. What additional information would resolve the ambiguity

The lead makes the final call.

---

## Commit-After-Fix Procedure

When `COMMIT_FIX` is `true` (default), every framework fix follows this flow:

### 1. Observer fixes and passes pytest gate (see below)

### 2. Observer reports to team lead

The report must include:
- Classification (FRAMEWORK_BUG subcategory A or B)
- Root cause description
- Files changed (with a summary of each change)
- Pytest result: N/N passing
- Recommendation: RERUN_NEEDED

### 3. Team lead reviews and commits

The team lead:
a. Reads the diff: `git diff` on the changed files
b. If satisfied, commits the fix:
```bash
cd {ARC_DIR} && \
git add <changed files> && \
git commit -m "fix: <description of what observer fixed>"
```
c. If not satisfied, asks the observer to revise or reverts

### 4. Team lead tells executor to re-run

Send the executor the re-run command. The executor runs AHVS CLI via Bash,
which spawns a fresh Python process — it will pick up the committed code
changes automatically.

### 5. If executor fails to load new code

If the executor reports errors that suggest it has stale state (e.g., referencing
old behavior, failing on code that was just fixed), the team lead should:
a. Shut down the executor agent
b. Respawn a fresh executor with the same prompt from `agent_prompts.md`
c. Send the re-run command to the new executor

---

## Pytest Gate Procedure

Every framework fix by the observer must pass through this gate. No exceptions.
This applies to both Subcategory A and Subcategory B fixes.

### Before the fix

```bash
cd {ARC_DIR} && \
{PYTHON} -m pytest tests/test_ahvs.py -v \
  2>&1 | tee /tmp/pytest_before_fix.log
```

Record the number of passing tests (currently 209). This is the baseline.

### After the fix

```bash
cd {ARC_DIR} && \
{PYTHON} -m pytest tests/test_ahvs.py -v \
  2>&1 | tee /tmp/pytest_after_fix.log
```

### Gate criteria

| Criterion | Required |
|---|---|
| All previously-passing tests still pass | Yes — zero regressions allowed |
| No new test failures | Yes |
| Full AHVS test suite passes (209 tests) | Yes |
| New test added for the specific fix | Recommended but not blocking |

### If the gate fails

1. **Revert** the fix immediately
2. **Save** both log files
3. **Escalate** to the team lead with both logs and an explanation of what went wrong
4. **Do NOT** retry the fix without lead approval

---

## Memory Write Requirements

Every bug fix and every lesson must be written to memory **immediately** — not at
end of session, not in batch, not "later."

### For FRAMEWORK_BUG fixes (both subcategories), write all three:

1. **Friction log**: `.ahvs/cycles/<cycle_id>/friction_log.md`
   ```markdown
   ## Operator Notes

   ### {timestamp} — {bug description}
   - Classification: FRAMEWORK_BUG (Subcategory {A|B})
   - Root cause: {explanation}
   - Fix: {what was changed}
   - Tests: {pass count before} → {pass count after}
   - Committed: {yes|no} (commit hash if yes)
   - Hypothesis affected: {H_ID}
   ```

2. **Lessons JSONL**: `.ahvs/evolution/lessons.jsonl`
   ```json
   {"type": "framework_bug", "subcategory": "missing_guardrail", "description": "...", "fix": "...", "file": "...", "committed": true, "commit_hash": "...", "timestamp": "..."}
   ```

3. **AHVS memory**: `<target_repo>/.ahvs/memory/`
   Write a new memory file and update `<target_repo>/.ahvs/memory/INDEX.md`.

### For HYPOTHESIS_MISS, write:

1. **Lessons JSONL** only:
   ```json
   {"type": "hypothesis_miss", "hypothesis_id": "H1", "metric": 0.74, "baseline": 0.75, "description": "...", "timestamp": "..."}
   ```

This data feeds the next cycle's context loader (Stage 2), preventing repeated dead ends.
