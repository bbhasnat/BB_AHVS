---
name: ahvs_data_analyst:gui
description: Browser-based report viewer — renders data analyst analysis results as a styled HTML page with embedded figures.
argument-hint: "[path to analysis_report.md or output directory]"
---

EXECUTE IMMEDIATELY — collect the report path, then launch the viewer.

## Execution

1. If the user provided a path argument, use it directly. Otherwise, ask:
   "What is the path to the analysis report (analysis_report.md) or the analysis output directory?"

2. Tell the user: "Opening analysis report at **http://localhost:8765/** — view it in your browser."

3. Run this to serve the report:

```bash
ahvs data_analyst --view <report_path>
```

Or via Python if you need non-blocking mode:

```python
from ahvs.report_viewer import serve_report
serve_report("<report_path>", port=8765)
```

4. The server blocks until Ctrl+C. Tell the user: "Report is being served at http://localhost:8765/ — press Ctrl+C in the terminal to stop."

## Rules

- Always serve on port 8765 (per user preference for AHVS GUI forms)
- Always tell the user the URL (http://localhost:8765/) and that they need to open it
- If given a directory path, the viewer automatically looks for `analysis_report.md` inside
- If the report has PNG figures in subdirectories (eda/, class_balance/, etc.), they are embedded automatically
- The viewer uses the same dark theme as other AHVS GUIs
