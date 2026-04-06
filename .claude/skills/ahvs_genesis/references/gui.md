# GUI Mode — /ahvs_genesis:gui

Launches a browser-based form to collect all Genesis inputs interactively, instead of asking text questions in the terminal.

## Trigger

- User invokes `/ahvs_genesis:gui`
- User says "genesis with gui", "genesis form", "genesis in browser"

## Workflow

### Step 1: Launch the browser form

Run this Python snippet to serve the form and block until the user submits:

```python
from ahvs.gui import run_form
from ahvs.gui_schemas import GENESIS_SCHEMA
inputs = run_form(GENESIS_SCHEMA)
```

**IMPORTANT:** Tell the user the URL printed to stdout and that they need to fill out the form in their browser. The call blocks until submission.

### Step 2: Use the collected inputs

The returned `inputs` dict maps directly to the Genesis workflow:

| Key | Maps to |
|-----|---------|
| `inputs["problem"]` | Problem description |
| `inputs["data_path"]` | Data file path |
| `inputs["target_metric"]` | Target metric |
| `inputs["output_dir"]` | Output directory |
| `inputs["mode"]` | Execution mode (`pipeline` or `agent`) |
| `inputs["classes"]` | Classes (may be empty string) |
| `inputs["input_column"]` | Input column name |

### Step 3: Proceed with Genesis

After collecting inputs via GUI, follow the standard Genesis workflow from
**Step 2 (Confirm and Run)** in the main SKILL.md — skip the text-based
input gathering since you already have all values.

Show the user a summary of what was collected and ask for confirmation before
running the genesis command.

## Rules

1. Always tell the user the URL and that they need to open it
2. The form handles validation (required fields, path existence) — trust its output
3. After form submission, proceed exactly as if the user had typed the same values in chat
4. The LLM cost policy still applies: use `gpt-4.1-mini` for annotation, never `gpt-4o`
5. Output directory is ALWAYS required — the form enforces this
