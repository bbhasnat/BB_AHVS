# BB_AHVS — Claude Instructions

## What is AHVS?

AHVS (Adaptive Hypothesis Validation System) is a standalone 8-stage cyclic pipeline that autonomously improves LLM/RAG systems. Given a target repo and a question like "How can we improve answer_relevance by 5%?", it generates hypotheses, tests them in isolated git worktrees, measures results against a baseline metric, and archives lessons for future cycles.

**Package:** `ahvs` (installed via `pip install -e .` from this repo)
**CLI:** `ahvs --repo <path> --question "..." --provider <provider> --model <model> --api-key-env <KEY>`
**Python API:** `from ahvs import AHVSConfig, execute_ahvs_cycle, AHVSStage, HypothesisResult`

## AHVS Memory Discipline (CRITICAL)

This repo is built on the principle that **lessons survive sessions**. The entire AHVS loop depends on memory being written in real time — not as a wrap-up step.

### When you find a bug, failure, or lesson — write it NOW:

1. **AHVS memory** (`<target_repo>/.ahvs/memory/`) — write or update a memory file in the target repo immediately
2. **AHVS friction log** (`<target_repo>/.ahvs/cycles/<cycle_id>/friction_log.md`) — add operator notes under `## Operator Notes`
3. **AHVS lessons** (`<target_repo>/.ahvs/evolution/lessons.jsonl`) — append a JSON line for cross-cycle lessons

All AHVS project-specific memory lives in the **target repo**, not in Claude's machine-local memory or this framework repo. This ensures memory is portable across machines and stays with the project it describes.

### What triggers an immediate memory write:
- Any bug found (framework, config, or repo-level)
- Any hypothesis that measured at baseline despite a real code change
- Any eval/measurement failure with a diagnosed root cause
- Any "it works now but here's why it was broken" insight
- Any infrastructure constraint (missing tool, wrong path, model mismatch)

### What NOT to do:
- Do NOT defer memory writes to end of session
- Do NOT rely on conversation output as memory — it disappears
- Do NOT mark a task complete without writing its lessons to memory first

## Key Behaviors

- **Never auto-approve hypotheses in multi-agent mode.** Always show the browser GUI for hypothesis selection unless the user explicitly says "auto-approve" or "skip selection".
- **Write bugs/lessons in real time**, not as a batch at session end. The conversation output is not memory — it disappears. The files are memory.
- **Claude Code is the code agent** for hypothesis execution (via `claude` CLI).

## Package Structure

```
ahvs/
├── cli.py               # Standalone CLI entry point
├── config.py             # AHVSConfig dataclass
├── executor.py           # 8 stage handlers (largest file, ~2500 lines)
├── runner.py             # execute_ahvs_cycle() orchestration loop
├── stages.py             # AHVSStage enum + transition state machine
├── contracts.py          # Stage I/O contracts
├── context_loader.py     # Baseline + EvolutionStore → context_bundle.json
├── health.py             # Pre-flight checks
├── prompts.py            # AHVSPromptManager + stage prompts
├── result.py             # HypothesisResult contract
├── skills.py             # Skill library
├── worktree.py           # Git worktree lifecycle
├── hypothesis_selector.py # Browser GUI for hypothesis selection
├── evolution.py          # Cross-cycle lesson storage (EvolutionStore)
├── domain_packs/         # Domain-specific prompt + skill overrides
│   ├── ml_prompts.yaml   # Traditional ML hypothesis prompts
│   └── ml_skills.yaml    # ML skill templates (sklearn, optuna, etc.)
├── llm/                  # LLM client factory
│   ├── __init__.py       # create_llm_client() factory
│   ├── client.py         # OpenAI-compatible LLMClient
│   ├── anthropic_adapter.py  # Anthropic Messages API adapter
│   └── acp_client.py     # ACP agent client
└── utils/
    └── thinking_tags.py  # Strip <think> tags from LLM output
```

## Skills

- `.claude/skills/ahvs_onboarding/` — Onboards a new repo for AHVS (creates `.ahvs/baseline_metric.json`)
- `.claude/skills/ahvs_multiagent/` — Runs AHVS cycles with multi-agent supervision (team lead + executor + observer)

## Tests

```bash
pytest tests/test_ahvs.py -v   # 229 tests
```
