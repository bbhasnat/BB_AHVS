# BB_AHVS — Claude Instructions

## What is AHVS?

AHVS (Adaptive Hypothesis Validation System) is a standalone 8-stage cyclic pipeline that autonomously improves LLM/RAG systems. Given a target repo and a question like "How can we improve answer_relevance by 5%?", it generates hypotheses, tests them in isolated git worktrees, measures results against a baseline metric, and archives lessons for future cycles.

**Package:** `ahvs` — install via `./install.sh` (or `pip install -e . && ahvs install`)
**CLI:** `ahvs --repo <path-or-name> --question "..." --provider <provider> --model <model> --api-key-env <KEY>`
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
├── cli.py               # CLI entry point (ahvs, ahvs genesis, ahvs install/update/uninstall)
├── installer.py          # Global installer (skills → ~/.claude/skills/, commands → ~/.claude/commands/)
├── registry.py           # Repo registry (~/.ahvs/registry.json)
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
├── gui.py                # Schema-driven browser form server
├── gui_schemas.py        # Pre-built GUI schemas (genesis, multiagent, onboarding)
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

## Skills & Commands

Skills (`.claude/skills/`) define the full workflow; commands (`.claude/commands/`) register subcommands.

- `/ahvs_brainstorm` — Pre-genesis design exploration (what to build and why)
- `/ahvs_onboarding` — Onboards a new repo for AHVS (creates `.ahvs/baseline_metric.json`)
- `/ahvs_onboarding:gui` — Browser form for onboarding
- `/ahvs_multiagent` — Runs AHVS cycles with multi-agent supervision (team lead + executor + observer)
- `/ahvs_multiagent:gui` — Browser form for multi-agent cycle
- `/ahvs_genesis` — Creates a new AHVS project from raw data
- `/ahvs_genesis:gui` — Browser form for genesis
- `/ahvs_data_analyst` — Goal-directed ML data analysis (profiles, EDA, class balance, text stats, duplicates [lexical/semantic/hybrid], embedding clustering, diversity subsample, split). Auto-escalates to hybrid dedup + clustering for large datasets. GPU required for semantic features; graceful fallback without. Supports `--provider acp` and `--dedup-mode`.
- `/ahvs_data_analyst:gui` — Browser-based report viewer for data analyst results

## Tests

```bash
pytest tests/test_ahvs.py -v   # 284 tests
```

## Project Memory Pattern

For any long-running project (spans multiple sessions), build a `.memory/` directory in the project root — NOT in this framework repo. Mirrors AHVS memory discipline so the same principles apply broadly.

### Structure

```
<project>/.memory/
├── MEMORY.md               # index — one-line hook per entry
├── state/
│   └── project_state.md    # current truth snapshot; update as project evolves
├── decisions/
│   └── NNN-topic.md        # one file per significant choice, with WHY
├── conversations/
│   └── NNN-topic.md        # user direction shifts with DIRECT QUOTES
└── lessons/
    └── topic.md            # cross-session gotchas, bugs, infrastructure quirks
```

### When to write

| Trigger | Write into |
|---|---|
| Tool / model / scope decision finalized | `decisions/` |
| User gives direction that shifts the plan | `conversations/` |
| Discover a non-obvious bug, gotcha, or constraint | `lessons/` |
| End of a major phase (data prep done, model trained, etc.) | `state/project_state.md` |

### Principles

1. **Write in real time, not at session end.** Conversations evaporate; files persist.
2. **Every decision must record its why.** Not just "we chose X" but "we chose X because Y."
3. **Include direct quotes in conversations.** They're ground truth.
4. **Project memory lives with the project.** Portable across machines; stays when people rotate.

Reference exemplar: `/home/ubuntu/vision/AEBSA/.memory/`.

## Reports Pattern

Every analysis, evaluation, or summary output from Claude or AHVS must be saved to disk as **both HTML and Markdown**, not served only from a running process.

### Why

A live-only report disappears when the server stops, the port is reused, or the browser cache refuses to reload. Files on disk survive.

### How

- Default directory: `<project>/reports/`
- Include a `reports/README.md` index listing every artifact + description
- Reusable template: `ahvs.templates.decomposed_analysis_gui` (handles json_array completions)
  - `save_reports(csv, output_dir, name=...)` → writes both HTML and MD
  - `serve_analysis(csv, port=8765)` → optional live GUI on top

### Rules

- Always tell the user the **filename** AND the live URL (not just the URL)
- If a live GUI is shown, the equivalent static file must already be on disk
- Markdown copies go into git; HTML copies render in browsers

## Demo Notebook (required deliverable)

Every completed project (model trained, system built, algorithm shipped) must ship a `demo/` directory showing concrete usage.

### Structure

```
<project>/demo/
├── README.md                  # prerequisites, examples covered, how to run
├── demo_<project>.ipynb       # Jupyter — cell-by-cell
└── demo_<project>.py          # Script — same examples, runnable end-to-end
```

### Required examples

Cover at minimum:
- The happy path (primary use case)
- Edge cases (empty output, absence, ambiguity)
- Generalization if the system was trained to generalize
- Batch / parallel inference pattern
- Loading your own data

### Rules

- **Verify it runs.** Actually execute the demo script before declaring the project done.
- Both files share helper functions (`analyze`, `pretty`, etc.) — dedup with imports or copy-paste
- Script version runs end-to-end in under a minute
- Self-contained — load API keys safely (per-line .env), load system prompts from committed artifacts

Reference exemplar: `/home/ubuntu/vision/AEBSA/demo/`

## Credential Hygiene (mandatory)

Secrets must never appear in conversation logs, terminal output, or temp files.

### Never do

```bash
# WRONG — prints key to stdout, captured in logs
env | grep OPENAI_API_KEY
cat .env
echo $API_KEY

# WRONG — multi-line capture if .env has multiple vars
KEY=$(grep OPENAI .env | cut -d= -f2)
```

### Do instead

```python
# Existence check without revealing the value
import os
print("OPENAI_API_KEY:", "SET" if os.environ.get("OPENAI_API_KEY") else "UNSET")

# Per-line .env parsing
for line in open(".env"):
    line = line.strip()
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
        break

# Or just use python-dotenv
from dotenv import load_dotenv
load_dotenv()
```

### If a secret leaks

1. Flag it **immediately** in the conversation so the user can rotate
2. Sweep `/tmp/`, task output dirs, and log files for the leaked value
3. Note the incident in project `.memory/lessons/credential-hygiene.md`

### Why

Accidents happen — tool outputs are captured in conversation history that may be retained and indexed. The cost of a leak is high (revocation + audit); the cost of defensive habits is zero.
