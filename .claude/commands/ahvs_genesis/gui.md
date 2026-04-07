---
name: ahvs_genesis:gui
description: Browser-based form for Genesis project creation — collects problem, data path, metric, output dir, and execution mode via localhost GUI.
argument-hint: ""
---

EXECUTE IMMEDIATELY — launch the browser form, then proceed with Genesis.

## Execution

1. Tell the user: "Launching Genesis form at **http://localhost:8765/** — please fill it out and click Submit."

2. Run this Python snippet to serve the form (blocks until user submits):

```python
from ahvs.gui import run_form
from ahvs.gui_schemas import GENESIS_SCHEMA
inputs = run_form(GENESIS_SCHEMA)
```

3. After form returns, map the collected inputs:
   - `inputs["problem"]` → Problem description
   - `inputs["data_path"]` → Data file path
   - `inputs["target_metric"]` → Target metric
   - `inputs["output_dir"]` → Output directory
   - `inputs["mode"]` → Execution mode (pipeline or agent)
   - `inputs["classes"]` → Classes (may be empty)
   - `inputs["input_column"]` → Input column name

4. Show the user a confirmation summary, then proceed with the standard Genesis workflow from the `ahvs_genesis` skill (Step 2: Confirm and Run).

## Rules

- Always tell the user the URL (http://localhost:8765/) and that they need to open it
- The form handles validation — trust its output
- Output directory is ALWAYS required — the form enforces this
- Use `gpt-4.1-mini` for annotation, never `gpt-4o`
