"""AHVS Hypothesis Selector — browser-based GUI for human hypothesis selection.

Reads hypotheses.md from a cycle directory, serves a checkbox form on localhost,
waits for the human to submit their selection, then writes selection.json.

Supports inline add, edit, reorder, and delete of hypotheses before selection.

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
# Valid hypothesis types (canonical source: hypothesis_ops.VALID_TYPES)
# ---------------------------------------------------------------------------

from ahvs.hypothesis_ops import VALID_TYPES as _VALID_TYPES_SET
HYPOTHESIS_TYPES = sorted(_VALID_TYPES_SET)


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
      transition: border-color 0.15s, background 0.15s;
    }}
    .card:hover {{ border-color: #6366f1; background: #232840; }}
    .card.selected {{ border-color: #a78bfa; background: #1a1d35; }}
    .card.editing {{ border-color: #f59e0b; background: #1a1d2a; }}
    .card-top {{
      display: flex;
      gap: 1rem;
      align-items: flex-start;
      cursor: pointer;
    }}
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
    .badge.operator {{
      background: #713f12;
      color: #fbbf24;
    }}
    .badge.edited {{
      background: #4c1d95;
      color: #c084fc;
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
    .card-actions {{
      display: flex;
      gap: 0.5rem;
      margin-top: 0.7rem;
      flex-wrap: wrap;
    }}
    .card-btn {{
      padding: 0.3rem 0.7rem;
      border-radius: 5px;
      font-size: 0.75rem;
      font-weight: 600;
      border: 1px solid #2d3448;
      background: #161827;
      color: #94a3b8;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .card-btn:hover {{ border-color: #6366f1; color: #e2e8f0; }}
    .card-btn.danger:hover {{ border-color: #ef4444; color: #fca5a5; }}
    /* Edit form inside card */
    .edit-form {{
      display: none;
      margin-top: 0.8rem;
      padding-top: 0.8rem;
      border-top: 1px solid #2d3448;
    }}
    .card.editing .edit-form {{ display: block; }}
    .card.editing .card-top {{ opacity: 0.5; pointer-events: none; }}
    .form-row {{
      display: flex;
      gap: 0.75rem;
      margin-bottom: 0.6rem;
      align-items: center;
    }}
    .form-row label {{
      font-size: 0.8rem;
      color: #94a3b8;
      min-width: 90px;
      text-align: right;
    }}
    .form-row input, .form-row textarea, .form-row select {{
      flex: 1;
      background: #0f1117;
      border: 1px solid #2d3448;
      border-radius: 5px;
      color: #e2e8f0;
      padding: 0.4rem 0.6rem;
      font-size: 0.85rem;
      font-family: inherit;
    }}
    .form-row textarea {{
      resize: vertical;
      min-height: 60px;
    }}
    .form-row input:focus, .form-row textarea:focus, .form-row select:focus {{
      outline: none;
      border-color: #6366f1;
    }}
    .edit-actions {{
      display: flex;
      gap: 0.5rem;
      justify-content: flex-end;
      margin-top: 0.5rem;
    }}
    .actions {{
      max-width: 860px;
      margin: 2rem auto 0;
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
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
    .btn-add {{
      background: #065f46;
      color: #6ee7b7;
      border: 1px solid #059669;
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
    /* Add hypothesis modal */
    .modal-overlay {{
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.7);
      z-index: 100;
      align-items: center;
      justify-content: center;
    }}
    .modal-overlay.visible {{ display: flex; }}
    .modal {{
      background: #1e2130;
      border: 1px solid #2d3448;
      border-radius: 12px;
      padding: 2rem;
      width: 90%;
      max-width: 600px;
      max-height: 80vh;
      overflow-y: auto;
    }}
    .modal h2 {{
      font-size: 1.2rem;
      color: #a78bfa;
      margin-bottom: 1.2rem;
    }}
    .modal .form-row label {{ min-width: 110px; }}
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

  <form id="form" onsubmit="return false;">
    <div class="cards" id="cards">
{cards_html}
    </div>
    <div class="actions">
      <button type="button" class="btn btn-add" onclick="showAddModal()">+ Add Hypothesis</button>
      <button type="button" class="btn btn-secondary" onclick="selectAll()">Select All</button>
      <button type="button" class="btn btn-secondary" onclick="selectNone()">Clear</button>
      <span class="counter"><span id="count">0</span> selected</span>
      <button type="button" class="btn btn-primary" onclick="submitSelection()">Submit Selection &rarr;</button>
    </div>
  </form>

  <!-- Add Hypothesis Modal -->
  <div class="modal-overlay" id="addModal">
    <div class="modal">
      <h2>Add Custom Hypothesis</h2>
      <div class="form-row">
        <label for="new-type">Type</label>
        <select id="new-type">
{type_options}
        </select>
      </div>
      <div class="form-row">
        <label for="new-desc">Description</label>
        <textarea id="new-desc" placeholder="What to change and how..."></textarea>
      </div>
      <div class="form-row">
        <label for="new-rationale">Rationale</label>
        <textarea id="new-rationale" placeholder="Why this might improve the metric..."></textarea>
      </div>
      <div class="form-row">
        <label for="new-cost">Est. Cost</label>
        <select id="new-cost">
          <option value="">—</option>
          <option value="low">Low</option>
          <option value="medium" selected>Medium</option>
          <option value="high">High</option>
        </select>
      </div>
      <div class="edit-actions">
        <button type="button" class="btn btn-secondary" onclick="hideAddModal()">Cancel</button>
        <button type="button" class="btn btn-add" onclick="addHypothesis()">Add</button>
      </div>
    </div>
  </div>

  <script>
    // ── State ──
    var hypotheses = {hypotheses_json};
    var nextNum = hypotheses.length + 1;
    var structuralEdit = false;  // track whether any add/edit/reorder/remove happened

    function render() {{
      var container = document.getElementById('cards');
      container.innerHTML = '';
      hypotheses.forEach(function(h, idx) {{
        var card = document.createElement('div');
        card.className = 'card' + (h._selected ? ' selected' : '');
        card.dataset.id = h.id;
        card.dataset.idx = idx;

        var sourceBadge = '';
        if (h._source === 'operator') sourceBadge = '<span class="badge operator">custom</span>';
        else if (h._edited) sourceBadge = '<span class="badge edited">edited</span>';

        var costHtml = h.estimated_cost ? '<p class="cost">Cost: ' + esc(h.estimated_cost) + '</p>' : '';

        card.innerHTML =
          '<div class="card-top" onclick="toggleSelect(' + idx + ', event)">' +
            '<input type="checkbox"' + (h._selected ? ' checked' : '') + ' onclick="toggleSelect(' + idx + ', event)">' +
            '<div class="card-body">' +
              '<div class="card-title">' + esc(h.id) + '<span class="badge">' + esc(h.type || 'code_change') + '</span>' + sourceBadge + '</div>' +
              '<p class="card-desc">' + esc(h.description) + '</p>' +
              '<p class="card-rationale"><strong>Rationale:</strong> ' + esc(h.rationale) + '</p>' +
              costHtml +
            '</div>' +
          '</div>' +
          '<div class="card-actions">' +
            '<button class="card-btn" onclick="startEdit(' + idx + ')">Edit</button>' +
            (idx > 0 ? '<button class="card-btn" onclick="moveUp(' + idx + ')">Move Up</button>' : '') +
            (idx < hypotheses.length - 1 ? '<button class="card-btn" onclick="moveDown(' + idx + ')">Move Down</button>' : '') +
            '<button class="card-btn danger" onclick="removeHyp(' + idx + ')">Remove</button>' +
          '</div>' +
          '<div class="edit-form" id="editform-' + idx + '">' +
            '<div class="form-row">' +
              '<label>Type</label>' +
              '<select id="edit-type-' + idx + '">' + typeOptionsHtml(h.type) + '</select>' +
            '</div>' +
            '<div class="form-row">' +
              '<label>Description</label>' +
              '<textarea id="edit-desc-' + idx + '">' + esc(h.description) + '</textarea>' +
            '</div>' +
            '<div class="form-row">' +
              '<label>Rationale</label>' +
              '<textarea id="edit-rationale-' + idx + '">' + esc(h.rationale) + '</textarea>' +
            '</div>' +
            '<div class="form-row">' +
              '<label>Est. Cost</label>' +
              '<select id="edit-cost-' + idx + '">' + costOptionsHtml(h.estimated_cost) + '</select>' +
            '</div>' +
            '<div class="edit-actions">' +
              '<button class="btn btn-secondary" onclick="cancelEdit(' + idx + ')">Cancel</button>' +
              '<button class="btn btn-add" onclick="saveEdit(' + idx + ')">Save</button>' +
            '</div>' +
          '</div>';

        container.appendChild(card);
      }});
      updateCount();
    }}

    function esc(s) {{ return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

    var validTypes = {types_json};
    function typeOptionsHtml(current) {{
      return validTypes.map(function(t) {{
        return '<option value="' + t + '"' + (t === current ? ' selected' : '') + '>' + t + '</option>';
      }}).join('');
    }}
    function costOptionsHtml(current) {{
      var opts = ['', 'low', 'medium', 'high'];
      var labels = ['—', 'Low', 'Medium', 'High'];
      return opts.map(function(v, i) {{
        return '<option value="' + v + '"' + (v === (current || '').toLowerCase() ? ' selected' : '') + '>' + labels[i] + '</option>';
      }}).join('');
    }}

    // ── Selection ──
    function toggleSelect(idx, e) {{
      e.stopPropagation();
      hypotheses[idx]._selected = !hypotheses[idx]._selected;
      render();
    }}
    function selectAll() {{
      hypotheses.forEach(function(h) {{ h._selected = true; }});
      render();
    }}
    function selectNone() {{
      hypotheses.forEach(function(h) {{ h._selected = false; }});
      render();
    }}
    function updateCount() {{
      var n = hypotheses.filter(function(h) {{ return h._selected; }}).length;
      document.getElementById('count').textContent = n;
    }}

    // ── Edit ──
    function startEdit(idx) {{
      var card = document.querySelector('[data-idx="' + idx + '"]');
      card.classList.add('editing');
    }}
    function cancelEdit(idx) {{
      var card = document.querySelector('[data-idx="' + idx + '"]');
      card.classList.remove('editing');
    }}
    function saveEdit(idx) {{
      var h = hypotheses[idx];
      h.type = document.getElementById('edit-type-' + idx).value;
      h.description = document.getElementById('edit-desc-' + idx).value;
      h.rationale = document.getElementById('edit-rationale-' + idx).value;
      h.estimated_cost = document.getElementById('edit-cost-' + idx).value;
      h._edited = true;
      structuralEdit = true;
      render();
    }}

    // ── Reorder ──
    function moveUp(idx) {{
      if (idx <= 0) return;
      structuralEdit = true;
      var tmp = hypotheses[idx - 1];
      hypotheses[idx - 1] = hypotheses[idx];
      hypotheses[idx] = tmp;
      renumber();
      render();
    }}
    function moveDown(idx) {{
      if (idx >= hypotheses.length - 1) return;
      structuralEdit = true;
      var tmp = hypotheses[idx + 1];
      hypotheses[idx + 1] = hypotheses[idx];
      hypotheses[idx] = tmp;
      renumber();
      render();
    }}
    function renumber() {{
      hypotheses.forEach(function(h, i) {{ h.id = 'H' + (i + 1); }});
    }}

    // ── Add ──
    function showAddModal() {{ document.getElementById('addModal').classList.add('visible'); }}
    function hideAddModal() {{ document.getElementById('addModal').classList.remove('visible'); }}
    function addHypothesis() {{
      var desc = document.getElementById('new-desc').value.trim();
      if (!desc) {{ alert('Description is required.'); return; }}
      structuralEdit = true;
      hypotheses.push({{
        id: 'H' + nextNum++,
        type: document.getElementById('new-type').value,
        description: desc,
        rationale: document.getElementById('new-rationale').value.trim() || 'Operator-provided hypothesis',
        estimated_cost: document.getElementById('new-cost').value,
        _source: 'operator',
        _selected: true
      }});
      renumber();
      hideAddModal();
      // Clear form
      document.getElementById('new-desc').value = '';
      document.getElementById('new-rationale').value = '';
      render();
    }}

    // ── Remove ──
    function removeHyp(idx) {{
      if (!confirm('Remove ' + hypotheses[idx].id + '?')) return;
      structuralEdit = true;
      hypotheses.splice(idx, 1);
      renumber();
      render();
    }}

    // ── Submit ──
    function submitSelection() {{
      var selected = hypotheses.filter(function(h) {{ return h._selected; }})
                               .map(function(h) {{ return h.id; }});
      if (selected.length === 0) {{
        alert('Please select at least one hypothesis.');
        return;
      }}
      // Clean internal flags before sending — preserve all fields
      var cleaned = hypotheses.map(function(h) {{
        var c = {{}};
        c.id = h.id;
        c.type = h.type || 'code_change';
        c.description = h.description || '';
        c.rationale = h.rationale || '';
        c.estimated_cost = h.estimated_cost || '';
        c.required_tools = h.required_tools || [];
        if (h._source) c._source = h._source;
        if (h._edited) c._edited = true;
        return c;
      }});
      fetch('/submit', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{selected: selected, hypotheses: cleaned, modified: structuralEdit}})
      }}).then(function(r) {{
        if (r.ok) {{
          document.getElementById('banner').style.display = 'block';
          document.getElementById('form').style.opacity = '0.4';
          document.getElementById('form').style.pointerEvents = 'none';
          document.getElementById('addModal').classList.remove('visible');
          window.scrollTo(0, 0);
        }}
      }});
    }}

    // ── Init ──
    hypotheses.forEach(function(h) {{ h._selected = true; }});
    render();
  </script>
</body>
</html>
"""

_CARD_TEMPLATE = ""  # Cards are now rendered client-side via JS


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _SelectionState:
    def __init__(self) -> None:
        self.selected: list[str] = []
        self.hypotheses: list[dict] | None = None  # modified list from GUI
        self.done = threading.Event()


def _safe_json_for_script(obj: object) -> str:
    """Serialize to JSON and escape characters that break <script> blocks.

    Prevents XSS from hypothesis text containing </script> or similar.
    """
    raw = json.dumps(obj)
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _build_html(hypotheses: list[dict], cycle_dir: Path, question: str) -> str:
    # Build type <option> tags for the add modal
    type_options = "\n".join(
        f'          <option value="{t}"'
        + (' selected' if t == 'code_change' else '')
        + f'>{t}</option>'
        for t in HYPOTHESIS_TYPES
    )
    # Prepare JSON-safe hypothesis data (preserve all fields for round-trip)
    hyp_json = _safe_json_for_script([
        {
            "id": h["id"],
            "type": h.get("type", "code_change"),
            "description": h.get("description", ""),
            "rationale": h.get("rationale", ""),
            "estimated_cost": h.get("estimated_cost", ""),
            "required_tools": h.get("required_tools", []),
            "_source": h.get("_source", "llm"),
            "_edited": h.get("_edited", False),
        }
        for h in hypotheses
    ])
    return _HTML_TEMPLATE.format(
        question=html.escape(question),
        cycle_dir=html.escape(str(cycle_dir)),
        cards_html="      <!-- rendered client-side -->",
        type_options=type_options,
        hypotheses_json=hyp_json,
        types_json=_safe_json_for_script(HYPOTHESIS_TYPES),
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
                # Only capture modified hypothesis list when GUI had structural edits
                if data.get("modified") and isinstance(data.get("hypotheses"), list):
                    state.hypotheses = data["hypotheses"]
            except (json.JSONDecodeError, TypeError):
                state.selected = []
                state.hypotheses = None
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

    Blocks until the human submits their selection. If the human adds, edits,
    reorders, or removes hypotheses via the GUI, hypotheses.md is rewritten
    before returning.

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
    print(f"  You can add, edit, reorder, or remove hypotheses in the GUI.")
    print(f"  Waiting for your selection... (submit in browser)")
    print(f"{'='*60}\n")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    if open_browser:
        webbrowser.open(url)

    state.done.wait()
    server.shutdown()

    # If the GUI sent back a modified hypothesis list, rewrite hypotheses.md
    if state.hypotheses is not None:
        from ahvs.hypothesis_ops import hypotheses_to_markdown
        hyp_path.write_text(
            hypotheses_to_markdown(state.hypotheses), encoding="utf-8"
        )
        print(f"[selector] Rewrote hypotheses.md with {len(state.hypotheses)} hypotheses (GUI modifications)")

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
# CLI entry point: python -m ahvs.hypothesis_selector <cycle_dir>
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m ahvs.hypothesis_selector <cycle_dir> [question]",
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
