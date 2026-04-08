# AHVS Multi-Agent Execution — Supervised Hypothesis Cycles

Multi-agent execution is the fourth and final stage of the AHVS pipeline. It runs the full 8-stage AHVS optimization cycle using three coordinated Claude Code agents: a **team lead** (your session), an **executor**, and an **observer**.

```
brainstorm → genesis → onboarding → ahvs cycles
                                        ▲ you are here
```

---

## Table of Contents

1. [What Multi-Agent Does](#1-what-multi-agent-does)
2. [The Three Agents](#2-the-three-agents)
3. [Prerequisites](#3-prerequisites)
4. [Configuration](#4-configuration)
5. [The 5-Phase Flow](#5-the-5-phase-flow)
6. [Failure Classification](#6-failure-classification)
7. [Memory Discipline](#7-memory-discipline)
8. [Error Handling](#8-error-handling)
9. [When to Use Multi-Agent vs CLI](#9-when-to-use-multi-agent-vs-cli)

---

## 1. What Multi-Agent Does

1. Generates hypotheses (AHVS Stages 1-3)
2. Presents them in a browser GUI for human selection (or auto-approves)
3. Creates an agent team with executor and observer
4. Runs each selected hypothesis one at a time — executor runs it, observer verifies
5. Re-runs hypotheses that failed due to framework bugs (after observer fixes)
6. Archives results (Stages 7-8) and shuts down the team

**The human's only manual step** is selecting hypotheses in the browser GUI. Even that can be skipped with auto-approve.

**Multi-agent does NOT:**
- Onboard a repo for AHVS — use `/ahvs_onboarding` first
- Force every hypothesis to pass — genuine misses are expected and recorded as lessons
- Run hypotheses in parallel — they run sequentially to avoid GPU contention

## 2. The Three Agents

| Agent | Model | Role |
|-------|-------|------|
| **Team Lead** | Your session | Orchestrates the cycle, commits fixes, reports progress |
| **Executor** | Sonnet | Runs each hypothesis via the AHVS CLI |
| **Observer** | Opus | Verifies results, classifies failures, fixes framework bugs |

The team lead is you (your Claude Code session). The executor and observer are spawned as subagents via Claude Code's Agent Teams.

## 3. Prerequisites

Before running, the skill verifies:

| Check | How |
|-------|-----|
| AHVS is onboarded | `.ahvs/baseline_metric.json` exists |
| Correct branch | `git branch --show-current` |
| Clean working tree | `git status` shows no uncommitted changes |
| Env vars loadable | `.env` file exists in target repo |
| Python environment | Correct conda/venv has all dependencies |

If any check fails, the skill stops and tells you what's missing.

## 4. Configuration

| Parameter | Source | Example |
|---|---|---|
| `AHVS_DIR` | BB_AHVS install path | `/home/ubuntu/vision/BB_AHVS` |
| `REPO_PATH` | Target repo | `/home/ubuntu/vision/my_project` |
| `ENV_FILE` | API keys, env vars | `/home/ubuntu/vision/my_project/.env` |
| `PYTHON` | Python binary | `/home/ubuntu/miniconda3/envs/my_env/bin/python` |
| `MAX_HYPS` | Hypotheses (1-5) | `3` |
| `PROVIDER` | LLM provider | `acp` |
| `MODEL` | LLM model | `anthropic/claude-opus-4-6` |
| `API_KEY_ENV` | Env var name for API key | `OPENROUTER_API_KEY` |
| `AUTO_APPROVE` | Skip GUI? | `false` (default) |
| `COMMIT_FIX` | Commit observer fixes? | `true` (default) |

## 5. The 5-Phase Flow

### Phase 1 — Generate Hypotheses

Runs AHVS Stages 1-3 only (`--until-stage AHVS_HYPOTHESIS_GEN`). Produces `hypotheses.md` in the cycle directory with N hypothesis proposals.

### Phase 2 — Human Selection (or Auto-Approve)

**Default (AUTO_APPROVE=false):** Launches a browser GUI at `http://localhost:8765/`. You see hypothesis cards with checkboxes. Select which to run and click Submit. The skill writes `selection.json` and proceeds.

**Auto-approve (AUTO_APPROVE=true):** Skips the GUI. All hypotheses are selected.

### Phase 3 — Team Setup

Creates the agent team and spawns executor + observer with role-specific prompts. The executor gets the AHVS CLI commands to run. The observer gets the failure classification rules and fix procedures.

### Phase 4 — Per-Hypothesis Loop

For each selected hypothesis, **sequentially**:

**Step A: Executor runs the hypothesis**
- Executes AHVS from `AHVS_HUMAN_SELECTION` through `AHVS_EXECUTION`
- Reports: exit code, result path, any errors

**Step B: Observer verifies**
- Reads the result directory and execution log
- Classifies the outcome (see [Failure Classification](#6-failure-classification))

**Step C: Team lead decides**

| Observer says | Action |
|---|---|
| **PASS** (hypothesis miss or metric improved) | Move to next hypothesis |
| **RERUN_NEEDED** (framework bug was fixed) | Lead reviews fix, commits if correct, re-runs |
| **AMBIGUOUS** | Lead reads evidence, decides PASS or RERUN |

If a hypothesis needs re-running 3+ times, the lead escalates to the human.

**Step C1: Review and commit observer's fix**

When the observer fixes a framework bug:
1. Lead reviews the diff
2. If correct: commits the fix, re-runs the hypothesis
3. If wrong: reverts, asks observer for a different approach
4. If executor has stale state: lead shuts down and respawns a fresh executor

**Step D: Report progress**

After each hypothesis: `"{H_ID} complete — {outcome}. {N_remaining} remaining."`

### Phase 5 — Archive and Shutdown

1. Runs AHVS Stages 7-8 (`--from-stage AHVS_REPORT_MEMORY`)
2. Shuts down executor and observer agents
3. Deletes the team
4. Reports final summary:

```
Done. 2/3 hypotheses improved metric.
1 framework bug found and fixed (1 committed).
Report: /path/to/.ahvs/cycles/<cycle_id>/report.md
```

## 6. Failure Classification

| Type | Meaning | Who fixes | Re-run? |
|---|---|---|---|
| **FRAMEWORK_BUG (A)** | Direct code bug in `ahvs/` | Observer fixes, lead commits | Yes |
| **FRAMEWORK_BUG (B)** | Missing guardrail set hypothesis up to fail | Observer fixes, lead commits | Yes |
| **HYPOTHESIS_MISS** | Hypothesis had a fair chance, metric didn't improve | Nobody | No |
| **AMBIGUOUS** | Can't tell from logs | Lead decides | Maybe |

The observer uses the rules in `references/failure_classification.md` to classify. The lead reviews and can override.

## 7. Memory Discipline

Follows the project's core principle: **lessons survive sessions**.

The team lead must:
- Write a memory file for every framework bug to `{REPO_PATH}/.ahvs/memory/`
- Update `{REPO_PATH}/.ahvs/memory/INDEX.md` after each write
- Record lessons to `{REPO_PATH}/.ahvs/evolution/lessons.jsonl`
- **Never defer memory writes to end of session**

The observer also writes memory and lessons independently. The lead verifies this by checking the files after each fix.

## 8. Error Handling

| Situation | Action |
|---|---|
| Phase 1 fails (generation error) | Show error, do not proceed |
| GUI times out or user closes browser | Ask: retry or auto-approve? |
| Executor crashes mid-hypothesis | Observer classifies, then re-run or skip |
| Observer fix breaks tests | Observer reverts, escalates to lead |
| Observer fix looks wrong to lead | Lead reverts, asks observer to try differently |
| Same hypothesis fails 3+ times | Stop loop, escalate to human |
| Executor has stale state after fix | Lead shuts down and respawns executor |
| Archive stage fails | Show error, but hypothesis results are already saved |

## 9. When to Use Multi-Agent vs CLI

| Scenario | Use |
|----------|-----|
| First AHVS cycle on a new project | **Multi-agent** — observer catches framework issues |
| Known-stable project, quick iteration | **CLI** — faster, less overhead |
| Long-running campaign (5+ hypotheses) | **Multi-agent** — supervision prevents wasted cycles |
| CI/CD integration | **CLI** with `--auto-approve` |
| Debugging framework bugs | **Multi-agent** — observer diagnoses and fixes in real time |

---

## Usage

In Claude Code (terminal):
```
/ahvs_multiagent
```

In Claude Code (browser form):
```
/ahvs_multiagent:gui
```

**Tip:** Use tmux so the session survives terminal disconnects:
```bash
tmux new -s ahvs
# then run /ahvs_multiagent inside tmux
# detach with Ctrl+B, D — reattach with tmux attach -t ahvs
```
