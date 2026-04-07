---
name: ahvs_onboarding:gui
description: Browser-based form for AHVS onboarding — collects repo path, metric name, eval command, and notes via localhost GUI.
argument-hint: ""
---

EXECUTE IMMEDIATELY — launch the browser form, then proceed with onboarding.

## Execution

1. Tell the user: "Launching onboarding form at **http://localhost:8765/** — please fill it out and click Submit."

2. Run this Python snippet to serve the form (blocks until user submits):

```python
from ahvs.gui import run_form
from ahvs.gui_schemas import ONBOARDING_SCHEMA
inputs = run_form(ONBOARDING_SCHEMA)
```

3. After form returns, map the collected inputs:
   - `inputs["repo_path"]` → Repository path to onboard
   - `inputs["metric_name"]` → Primary metric to optimize
   - `inputs["eval_command"]` → Eval command (may be empty — auto-discover)
   - `inputs["notes"]` → Additional context (may be empty)

4. Proceed with the standard onboarding workflow from the `ahvs_onboarding` skill — skip conversational input gathering since you already have all values.

## Rules

- Always tell the user the URL (http://localhost:8765/) and that they need to open it
- The form handles validation — trust its output
- If `eval_command` is empty, auto-discover it as the main skill would
- If `notes` is provided, use it as additional context during codebase inspection
