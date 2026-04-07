---
name: ahvs_multiagent:gui
description: Browser-based form for multi-agent AHVS cycle — collects repo path, question, provider, model, and options via localhost GUI.
argument-hint: ""
---

EXECUTE IMMEDIATELY — launch the browser form, then proceed with the multi-agent cycle.

## Execution

1. Tell the user: "Launching multi-agent form at **http://localhost:8765/** — please fill it out and click Submit."

2. Run this Python snippet to serve the form (blocks until user submits):

```python
from ahvs.gui import run_form
from ahvs.gui_schemas import MULTIAGENT_SCHEMA
inputs = run_form(MULTIAGENT_SCHEMA)
```

3. After form returns, map the collected inputs:
   - `inputs["repo_path"]` → Target repository path
   - `inputs["question"]` → Cycle question
   - `inputs["provider"]` → LLM provider
   - `inputs["model"]` → LLM model
   - `inputs["max_hypotheses"]` → Max hypotheses to generate
   - `inputs["auto_approve"]` → Boolean — skip hypothesis selection if True
   - `inputs["domain"]` → Domain pack (llm or ml)

4. Proceed directly to **Phase 1: Generate Hypotheses** from the `ahvs_multiagent` skill with the collected values.

## Rules

- Always tell the user the URL (http://localhost:8765/) and that they need to open it
- The form handles validation — trust its output
- Default provider is ACP (local Claude agent) per user preference
- If `auto_approve` is True, skip the hypothesis selection GUI entirely
