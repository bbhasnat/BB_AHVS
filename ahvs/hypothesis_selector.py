"""AHVS Hypothesis Selector — browser-based GUI for human hypothesis selection.

Reads hypotheses.md from a cycle directory, serves a checkbox form on localhost,
waits for the human to submit their selection, then writes selection.json.

Usage (called by team lead between stage gen and stage exec):
    python -m ahvs.hypothesis_selector <cycle_dir>

Or programmatically:
    from ahvs.hypothesis_selector import run_selector
    selected_ids = run_selector(cycle_dir)  # blocks until human submits
"""

from __future__ import annotations

import html
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse


# ---------------------------------------------------------------------------
# Hypothesis parser (mirrors executor._parse_hypotheses, no import needed)
# ---------------------------------------------------------------------------


def _parse_hypotheses(text: str) -> list[dict]:
    hypotheses = []
    blocks = re.split(r"^##\s+(H\d+)\s*$", text, flags=re.MULTILINE)
    it = iter(blocks)
    next(it, None)  # skip preamble
    for hyp_id, body in zip(it, it):
        hyp: dict = {"id": hyp_id.strip()}
        for field, pattern in [
            ("type", r"\*\*Type:\*\*\s*(.+)"),
            ("description", r"\*\*Description:\*\*\s*(.+)"),
            ("rationale", r"\*\*Rationale:\*\*\s*(.+)"),
            ("estimated_cost", r"\*\*Estimated Cost:\*\*\s*(.+)"),
        ]:
            m = re.search(pattern, body, re.IGNORECASE)
            hyp[field] = m.group(1).strip() if m else ""
        hypotheses.append(hyp)
    return hypotheses


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AHVS — Hypothesis Selection</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 2rem;
    }}
    .header {{
      max-width: 860px;
      margin: 0 auto 2rem;
    }}
    h1 {{
      font-size: 1.5rem;
      font-weight: 700;
      color: #a78bfa;
      margin-bottom: 0.4rem;
    }}
    .question {{
      font-size: 0.95rem;
      color: #94a3b8;
      margin-bottom: 0.25rem;
    }}
    .cycle-dir {{
      font-size: 0.8rem;
      color: #475569;
      font-family: monospace;
    }}
    .cards {{
      max-width: 860px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .card {{
      background: #1e2130;
      border: 1px solid #2d3448;
      border-radius: 10px;
      padding: 1.2rem 1.4rem;
      cursor: pointer;
      transition: border-color 0.15s, background 0.15s;
      display: flex;
      gap: 1rem;
      align-items: flex-start;
    }}
    .card:hover {{ border-color: #6366f1; background: #232840; }}
    .card.selected {{ border-color: #a78bfa; background: #1a1d35; }}
    .card input[type=checkbox] {{
      width: 18px;
      height: 18px;
      margin-top: 3px;
      flex-shrink: 0;
      accent-color: #a78bfa;
      cursor: pointer;
    }}
    .card-body {{ flex: 1; }}
    .card-title {{
      font-size: 1rem;
      font-weight: 600;
      color: #c4b5fd;
      margin-bottom: 0.3rem;
    }}
    .badge {{
      display: inline-block;
      font-size: 0.7rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
      background: #312e81;
      color: #a5b4fc;
      margin-left: 0.5rem;
      vertical-align: middle;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .card-desc {{
      font-size: 0.9rem;
      color: #cbd5e1;
      margin-bottom: 0.5rem;
      line-height: 1.5;
    }}
    .card-rationale {{
      font-size: 0.82rem;
      color: #64748b;
      line-height: 1.45;
    }}
    .card-rationale strong {{ color: #94a3b8; }}
    .cost {{
      font-size: 0.78rem;
      color: #475569;
      margin-top: 0.4rem;
    }}
    .actions {{
      max-width: 860px;
      margin: 2rem auto 0;
      display: flex;
      gap: 1rem;
      align-items: center;
    }}
    .btn {{
      padding: 0.65rem 1.5rem;
      border-radius: 7px;
      font-size: 0.9rem;
      font-weight: 600;
      border: none;
      cursor: pointer;
      transition: opacity 0.15s;
    }}
    .btn:hover {{ opacity: 0.88; }}
    .btn-primary {{
      background: #7c3aed;
      color: #fff;
    }}
    .btn-secondary {{
      background: #1e2130;
      color: #94a3b8;
      border: 1px solid #2d3448;
    }}
    .counter {{
      font-size: 0.85rem;
      color: #64748b;
      margin-left: auto;
    }}
    .counter span {{ color: #a78bfa; font-weight: 700; }}
    .success-banner {{
      display: none;
      max-width: 860px;
      margin: 1.5rem auto;
      background: #14532d;
      border: 1px solid #16a34a;
      border-radius: 8px;
      padding: 1rem 1.4rem;
      color: #86efac;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>AHVS — Hypothesis Selection</h1>
    <p class="question">Cycle question: <strong>{question}</strong></p>
    <p class="cycle-dir">{cycle_dir}</p>
  </div>

  <div class="success-banner" id="banner">
    Selection submitted. You can close this tab.
  </div>

  <form method="POST" action="/submit" id="form">
    <div class="cards" id="cards">
{cards_html}
    </div>
    <div class="actions">
      <button type="button" class="btn btn-secondary" onclick="selectAll()">Select All</button>
      <button type="button" class="btn btn-secondary" onclick="selectNone()">Clear</button>
      <span class="counter"><span id="count">0</span> selected</span>
      <button type="submit" class="btn btn-primary">Submit Selection →</button>
    </div>
  </form>

  <script>
    function updateCount() {{
      var n = document.querySelectorAll('input[type=checkbox]:checked').length;
      document.getElementById('count').textContent = n;
      document.querySelectorAll('.card').forEach(function(card) {{
        var cb = card.querySelector('input[type=checkbox]');
        card.classList.toggle('selected', cb.checked);
      }});
    }}
    function selectAll() {{
      document.querySelectorAll('input[type=checkbox]').forEach(function(cb) {{ cb.checked = true; }});
      updateCount();
    }}
    function selectNone() {{
      document.querySelectorAll('input[type=checkbox]').forEach(function(cb) {{ cb.checked = false; }});
      updateCount();
    }}
    document.querySelectorAll('.card').forEach(function(card) {{
      card.addEventListener('click', function(e) {{
        if (e.target.tagName !== 'INPUT') {{
          var cb = card.querySelector('input[type=checkbox]');
          cb.checked = !cb.checked;
          updateCount();
        }}
      }});
    }});
    document.querySelectorAll('input[type=checkbox]').forEach(function(cb) {{
      cb.addEventListener('change', updateCount);
    }});
    document.getElementById('form').addEventListener('submit', function(e) {{
      e.preventDefault();
      var selected = Array.from(document.querySelectorAll('input[type=checkbox]:checked'))
                         .map(function(cb) {{ return cb.value; }});
      if (selected.length === 0) {{
        alert('Please select at least one hypothesis.');
        return;
      }}
      fetch('/submit', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{selected: selected}})
      }}).then(function(r) {{
        if (r.ok) {{
          document.getElementById('banner').style.display = 'block';
          document.getElementById('form').style.opacity = '0.4';
          document.getElementById('form').style.pointerEvents = 'none';
          window.scrollTo(0, 0);
        }}
      }});
    }});
    updateCount();
  </script>
</body>
</html>
"""

_CARD_TEMPLATE = """\
      <label class="card" for="cb_{hyp_id}">
        <input type="checkbox" id="cb_{hyp_id}" name="hypothesis" value="{hyp_id}" checked>
        <div class="card-body">
          <div class="card-title">{hyp_id}<span class="badge">{hyp_type}</span></div>
          <p class="card-desc">{description}</p>
          <p class="card-rationale"><strong>Rationale:</strong> {rationale}</p>
          {cost_html}
        </div>
      </label>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _SelectionState:
    def __init__(self) -> None:
        self.selected: list[str] = []
        self.done = threading.Event()


def _build_html(hypotheses: list[dict], cycle_dir: Path, question: str) -> str:
    cards = []
    for h in hypotheses:
        cost = h.get("estimated_cost", "").strip()
        cost_html = f'<p class="cost">Cost: {html.escape(cost)}</p>' if cost else ""
        cards.append(_CARD_TEMPLATE.format(
            hyp_id=html.escape(h["id"]),
            hyp_type=html.escape(h.get("type", "unknown")),
            description=html.escape(h.get("description", "")),
            rationale=html.escape(h.get("rationale", "")),
            cost_html=cost_html,
        ))
    return _HTML_TEMPLATE.format(
        question=html.escape(question),
        cycle_dir=html.escape(str(cycle_dir)),
        cards_html="\n".join(cards),
    )


def _make_handler(html_content: str, state: _SelectionState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # silence access log
            pass

        def do_GET(self) -> None:
            body = html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/submit":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
                state.selected = [str(s).upper() for s in data.get("selected", [])]
            except (json.JSONDecodeError, TypeError):
                state.selected = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            state.done.set()

    return Handler


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_selector(
    cycle_dir: Path,
    *,
    question: str = "",
    port: int = 8765,
    open_browser: bool = True,
) -> list[str]:
    """Serve the hypothesis selection GUI and return the chosen IDs.

    Blocks until the human submits their selection.

    Args:
        cycle_dir: Path to the AHVS cycle directory containing hypotheses.md.
        question: Cycle question string (displayed in the UI).
        port: Port to listen on. Default 8765. 0 = OS assigns a free port.
        open_browser: If True, open the default browser automatically.

    Returns:
        List of selected hypothesis IDs (e.g. ['H1', 'H3']).

    Raises:
        FileNotFoundError: If hypotheses.md is not found in cycle_dir.
        ValueError: If hypotheses.md contains no parseable hypotheses.
    """
    hyp_path = cycle_dir / "hypotheses.md"
    if not hyp_path.exists():
        raise FileNotFoundError(
            f"hypotheses.md not found in {cycle_dir}. "
            "Run AHVS stages 1-3 first: --until-stage AHVS_HYPOTHESIS_GEN"
        )

    hypotheses = _parse_hypotheses(hyp_path.read_text(encoding="utf-8"))
    if not hypotheses:
        raise ValueError(f"No parseable hypotheses found in {hyp_path}")

    # Try to read question from context_bundle.json if not provided
    if not question:
        bundle_path = cycle_dir / "context_bundle.json"
        if bundle_path.exists():
            try:
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                question = bundle.get("question", "")
            except (json.JSONDecodeError, KeyError):
                pass

    state = _SelectionState()
    html_content = _build_html(hypotheses, cycle_dir, question)
    handler_class = _make_handler(html_content, state)

    server = HTTPServer(("0.0.0.0", port), handler_class)
    actual_port = server.server_address[1]

    url = f"http://127.0.0.1:{actual_port}/"
    print(f"\n{'='*60}")
    print(f"  HYPOTHESIS SELECTOR — http://localhost:{actual_port}/")
    print(f"{'='*60}")
    print(f"  {len(hypotheses)} hypotheses: {[h['id'] for h in hypotheses]}")
    print(f"  Waiting for your selection... (submit in browser)")
    print(f"{'='*60}\n")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    if open_browser:
        webbrowser.open(url)

    state.done.wait()
    server.shutdown()

    # Write selection.json into the cycle dir
    selection_data = {
        "selected": state.selected,
        "rationale": "Human selection via hypothesis_selector GUI",
        "approved_by": "human",
    }
    sel_path = cycle_dir / "selection.json"
    sel_path.write_text(json.dumps(selection_data, indent=2), encoding="utf-8")

    print(f"[selector] Human selected: {state.selected}")
    print(f"[selector] selection.json written to {sel_path}")
    return state.selected


# ---------------------------------------------------------------------------
# CLI entry point: python -m researchclaw.ahvs.hypothesis_selector <cycle_dir>
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m researchclaw.ahvs.hypothesis_selector <cycle_dir> [question]",
              file=sys.stderr)
        return 1

    cycle_dir = Path(args[0]).resolve()
    question = args[1] if len(args) > 1 else ""

    try:
        selected = run_selector(cycle_dir, question=question)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not selected:
        print("[selector] No hypotheses selected — cycle aborted.", file=sys.stderr)
        return 1

    print(f"[selector] Done. Selected: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
