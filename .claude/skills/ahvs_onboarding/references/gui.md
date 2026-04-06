# GUI Mode — /ahvs_onboarding:gui

Launches a browser-based form to collect all onboarding inputs interactively, instead of asking text questions in the terminal.

## Trigger

- User invokes `/ahvs_onboarding:gui`
- User says "onboard with gui", "onboarding form", "onboard in browser"

## Workflow

### Step 1: Launch the browser form

Run this Python snippet to serve the form and block until the user submits:

```python
from ahvs.gui import run_form
from ahvs.gui_schemas import ONBOARDING_SCHEMA
inputs = run_form(ONBOARDING_SCHEMA)
```

**IMPORTANT:** Tell the user the URL printed to stdout and that they need to fill out the form in their browser. The call blocks until submission.

### Step 2: Use the collected inputs

The returned `inputs` dict maps directly to the onboarding workflow:

| Key | Maps to |
|-----|---------|
| `inputs["repo_path"]` | Repository path to onboard |
| `inputs["metric_name"]` | Primary metric to optimize |
| `inputs["eval_command"]` | Eval command (may be empty — auto-discover) |
| `inputs["notes"]` | Additional context (may be empty) |

### Step 3: Proceed with onboarding

After collecting inputs via GUI, follow the standard onboarding workflow
from the main SKILL.md — skip any conversational input gathering since
you already have all values.

Proceed directly to codebase inspection with the collected values.

## Rules

1. Always tell the user the URL and that they need to open it
2. The form handles validation (required fields, path existence) — trust its output
3. After form submission, proceed exactly as if the user had typed the same values in chat
4. If `eval_command` is empty, auto-discover it as the main skill would
5. If `notes` is provided, use it as additional context during codebase inspection
