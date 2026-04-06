# GUI Mode — /ahvs_multiagent:gui

Launches a browser-based form to collect all multi-agent cycle inputs interactively, instead of asking text questions in the terminal.

## Trigger

- User invokes `/ahvs_multiagent:gui`
- User says "multi-agent with gui", "ahvs gui", "run cycle in browser"

## Workflow

### Step 1: Launch the browser form

Run this Python snippet to serve the form and block until the user submits:

```python
from ahvs.gui import run_form
from ahvs.gui_schemas import MULTIAGENT_SCHEMA
inputs = run_form(MULTIAGENT_SCHEMA)
```

**IMPORTANT:** Tell the user the URL printed to stdout and that they need to fill out the form in their browser. The call blocks until submission.

### Step 2: Use the collected inputs

The returned `inputs` dict maps directly to the multi-agent workflow:

| Key | Maps to |
|-----|---------|
| `inputs["repo_path"]` | Target repository path |
| `inputs["question"]` | Cycle question |
| `inputs["provider"]` | LLM provider |
| `inputs["model"]` | LLM model |
| `inputs["max_hypotheses"]` | Max hypotheses to generate |
| `inputs["auto_approve"]` | Boolean — skip hypothesis selection if True |
| `inputs["domain"]` | Domain pack (`llm` or `ml`) |

### Step 3: Proceed with multi-agent cycle

After collecting inputs via GUI, follow the standard multi-agent workflow
from the main SKILL.md — skip any conversational input gathering since
you already have all values.

Proceed directly to **Phase 1: Generate Hypotheses** with the collected values.

## Rules

1. Always tell the user the URL and that they need to open it
2. The form handles validation (required fields, path existence) — trust its output
3. After form submission, proceed exactly as if the user had typed the same values in chat
4. Default provider is ACP (local Claude agent) per user preference
5. If `auto_approve` is True, skip the hypothesis selection GUI entirely
