"""AHVS Generic Form GUI — browser-based input collection for any skill.

Serves a schema-driven HTML form on localhost, blocks until the user submits,
and returns the form data as a plain dict.

Usage:
    from ahvs.gui import run_form

    schema = {
        "title": "My Form",
        "fields": [
            {"name": "question", "type": "text", "label": "Question", "required": True},
            {"name": "mode", "type": "radio", "label": "Mode",
             "options": [{"value": "fast", "label": "Fast"}, {"value": "slow", "label": "Slow"}]},
        ],
    }
    result = run_form(schema)  # blocks until browser submit
    print(result)  # {"question": "...", "mode": "fast"}
"""

from __future__ import annotations

import html
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_FIELD_TYPES = {"text", "textarea", "select", "radio", "checkbox", "info"}


def _validate_schema(schema: dict) -> None:
    """Raise ValueError if the schema is malformed."""
    if "title" not in schema:
        raise ValueError("Schema must have a 'title' key")
    fields = schema.get("fields")
    if not fields or not isinstance(fields, list):
        raise ValueError("Schema must have a non-empty 'fields' list")
    names: set[str] = set()
    for i, f in enumerate(fields):
        if "name" not in f:
            raise ValueError(f"Field {i} missing 'name'")
        if "type" not in f:
            raise ValueError(f"Field {i} ({f['name']}) missing 'type'")
        if f["type"] not in _FIELD_TYPES:
            raise ValueError(
                f"Field {i} ({f['name']}) has unknown type '{f['type']}'. "
                f"Allowed: {_FIELD_TYPES}"
            )
        if f["name"] in names:
            raise ValueError(f"Duplicate field name: '{f['name']}'")
        names.add(f["name"])
        if "label" not in f and f["type"] != "info":
            raise ValueError(f"Field {i} ({f['name']}) missing 'label'")
        if f["type"] in ("select", "radio") and not f.get("options"):
            raise ValueError(
                f"Field {i} ({f['name']}) is type '{f['type']}' but has no 'options'"
            )


# ---------------------------------------------------------------------------
# CSS theme (matches hypothesis_selector dark theme)
# ---------------------------------------------------------------------------

_BASE_CSS = """\
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 2rem;
    }
    .container { max-width: 860px; margin: 0 auto; }
    h1 {
      font-size: 1.5rem; font-weight: 700; color: #a78bfa;
      margin-bottom: 0.4rem;
    }
    .subtitle {
      font-size: 0.95rem; color: #94a3b8; margin-bottom: 2rem;
    }
    .field-group {
      margin-bottom: 1.5rem;
      transition: opacity 0.2s, max-height 0.3s;
      overflow: hidden;
    }
    .field-group.hidden {
      opacity: 0; max-height: 0; margin-bottom: 0; pointer-events: none;
    }
    .field-label {
      display: block; font-size: 0.9rem; font-weight: 600;
      color: #c4b5fd; margin-bottom: 0.4rem;
    }
    .field-label .req { color: #f87171; margin-left: 2px; }
    .field-help {
      font-size: 0.78rem; color: #64748b; margin-top: 0.25rem;
    }
    .field-error {
      font-size: 0.78rem; color: #f87171; margin-top: 0.25rem; display: none;
    }
    input[type=text], textarea, select {
      width: 100%; padding: 0.65rem 0.85rem;
      background: #1e2130; border: 1px solid #2d3448;
      border-radius: 7px; color: #e2e8f0;
      font-size: 0.9rem; font-family: inherit;
      transition: border-color 0.15s;
    }
    input[type=text]:focus, textarea:focus, select:focus {
      outline: none; border-color: #6366f1;
    }
    input[type=text]::placeholder, textarea::placeholder {
      color: #475569;
    }
    textarea { min-height: 100px; resize: vertical; }
    select { cursor: pointer; }
    select option { background: #1e2130; color: #e2e8f0; }

    /* Radio group */
    .radio-group { display: flex; flex-direction: column; gap: 0.6rem; }
    .radio-option {
      background: #1e2130; border: 1px solid #2d3448;
      border-radius: 8px; padding: 0.8rem 1rem;
      cursor: pointer; display: flex; align-items: center; gap: 0.8rem;
      transition: border-color 0.15s, background 0.15s;
    }
    .radio-option:hover { border-color: #6366f1; background: #232840; }
    .radio-option.selected { border-color: #a78bfa; background: #1a1d35; }
    .radio-option input[type=radio] {
      accent-color: #a78bfa; width: 16px; height: 16px; cursor: pointer;
    }
    .radio-label { font-size: 0.9rem; color: #cbd5e1; }

    /* Checkbox toggle */
    .toggle-wrap {
      display: flex; align-items: center; gap: 0.8rem;
      background: #1e2130; border: 1px solid #2d3448;
      border-radius: 8px; padding: 0.8rem 1rem; cursor: pointer;
      transition: border-color 0.15s;
    }
    .toggle-wrap:hover { border-color: #6366f1; }
    .toggle-wrap input[type=checkbox] {
      accent-color: #a78bfa; width: 18px; height: 18px; cursor: pointer;
    }
    .toggle-label { font-size: 0.9rem; color: #cbd5e1; }

    /* Info block */
    .info-block {
      background: #1a1d35; border: 1px solid #312e81;
      border-radius: 8px; padding: 0.8rem 1rem;
      font-size: 0.88rem; color: #a5b4fc; line-height: 1.5;
    }

    /* Path indicator */
    .path-status {
      font-size: 0.75rem; margin-top: 0.2rem; min-height: 1em;
    }
    .path-status.valid { color: #4ade80; }
    .path-status.invalid { color: #f87171; }

    /* Actions */
    .actions {
      margin-top: 2rem; display: flex; gap: 1rem; align-items: center;
    }
    .btn {
      padding: 0.7rem 2rem; border-radius: 7px;
      font-size: 0.95rem; font-weight: 600;
      border: none; cursor: pointer; transition: opacity 0.15s;
    }
    .btn:hover { opacity: 0.88; }
    .btn-primary { background: #7c3aed; color: #fff; }
    .btn-secondary {
      background: #1e2130; color: #94a3b8; border: 1px solid #2d3448;
    }
    .success-banner {
      display: none; background: #14532d; border: 1px solid #16a34a;
      border-radius: 8px; padding: 1rem 1.4rem;
      color: #86efac; font-weight: 600; margin-bottom: 1.5rem;
    }
    .server-errors {
      display: none; background: #450a0a; border: 1px solid #dc2626;
      border-radius: 8px; padding: 1rem 1.4rem;
      color: #fca5a5; font-size: 0.88rem; margin-bottom: 1rem;
    }
"""


# ---------------------------------------------------------------------------
# Field renderers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value else ""


def _render_field(field: dict) -> str:
    """Return HTML for a single form field."""
    ftype = field["type"]
    name = _esc(field["name"])
    label = _esc(field.get("label", ""))
    required = field.get("required", False)
    default = field.get("default", "")
    placeholder = _esc(field.get("placeholder", ""))
    help_text = _esc(field.get("help", ""))
    validate_path = field.get("validate_path", False)

    # show_when attribute
    show_when = field.get("show_when")
    sw_attr = ""
    if show_when:
        sw_attr = f' data-show-when=\'{json.dumps(show_when)}\''

    req_mark = '<span class="req">*</span>' if required else ""
    req_attr = ' data-required="true"' if required else ""
    help_html = f'<div class="field-help">{help_text}</div>' if help_text else ""
    error_html = f'<div class="field-error" id="err_{name}"></div>'
    path_attr = ' data-validate-path="true"' if validate_path else ""
    path_html = f'<div class="path-status" id="path_{name}"></div>' if validate_path else ""

    if ftype == "text":
        inner = (
            f'<input type="text" name="{name}" id="f_{name}" '
            f'value="{_esc(default)}" placeholder="{placeholder}"{req_attr}{path_attr}>'
            f'{path_html}'
        )
    elif ftype == "textarea":
        inner = (
            f'<textarea name="{name}" id="f_{name}" '
            f'placeholder="{placeholder}"{req_attr}>{_esc(default)}</textarea>'
        )
    elif ftype == "select":
        opts_html = ""
        for opt in field.get("options", []):
            if isinstance(opt, dict):
                val, lbl = _esc(opt["value"]), _esc(opt["label"])
            else:
                val = lbl = _esc(opt)
            sel = " selected" if str(opt if isinstance(opt, str) else opt.get("value", "")) == str(default) else ""
            opts_html += f'<option value="{val}"{sel}>{lbl}</option>'
        inner = f'<select name="{name}" id="f_{name}"{req_attr}>{opts_html}</select>'
    elif ftype == "radio":
        radios = ""
        for opt in field.get("options", []):
            if isinstance(opt, dict):
                val, lbl = _esc(opt["value"]), _esc(opt["label"])
            else:
                val = lbl = _esc(opt)
            checked = " checked" if str(opt if isinstance(opt, str) else opt.get("value", "")) == str(default) else ""
            radios += (
                f'<label class="radio-option{" selected" if checked else ""}">'
                f'<input type="radio" name="{name}" value="{val}"{checked}>'
                f'<span class="radio-label">{lbl}</span></label>'
            )
        inner = f'<div class="radio-group">{radios}</div>'
    elif ftype == "checkbox":
        checked = " checked" if default else ""
        inner = (
            f'<label class="toggle-wrap">'
            f'<input type="checkbox" name="{name}" id="f_{name}"{checked}>'
            f'<span class="toggle-label">{label}</span></label>'
        )
        # For checkbox, label is inside the toggle, skip outer label
        return (
            f'<div class="field-group"{sw_attr}>'
            f'{inner}{help_html}{error_html}</div>'
        )
    elif ftype == "info":
        content = _esc(field.get("content", field.get("default", "")))
        return (
            f'<div class="field-group"{sw_attr}>'
            f'<div class="info-block">{content}</div></div>'
        )
    else:
        return ""

    return (
        f'<div class="field-group"{sw_attr}>'
        f'<label class="field-label" for="f_{name}">{label}{req_mark}</label>'
        f'{inner}{help_html}{error_html}</div>'
    )


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

_JS = """\
(function() {
  // Conditional visibility
  function evalShowWhen() {
    document.querySelectorAll('[data-show-when]').forEach(function(el) {
      var cond = JSON.parse(el.getAttribute('data-show-when'));
      var visible = true;
      for (var key in cond) {
        var input = document.querySelector('[name="' + key + '"]:checked')
                 || document.querySelector('[name="' + key + '"]');
        var val = input ? input.value : '';
        if (val !== cond[key]) { visible = false; break; }
      }
      el.classList.toggle('hidden', !visible);
    });
  }

  // Radio styling
  function updateRadioStyles() {
    document.querySelectorAll('.radio-group').forEach(function(g) {
      g.querySelectorAll('.radio-option').forEach(function(opt) {
        var rb = opt.querySelector('input[type=radio]');
        opt.classList.toggle('selected', rb.checked);
      });
    });
  }

  // Attach change listeners
  document.querySelectorAll('input, textarea, select').forEach(function(el) {
    el.addEventListener('change', function() { evalShowWhen(); updateRadioStyles(); });
    el.addEventListener('input', function() { evalShowWhen(); });
  });

  // Radio option click
  document.querySelectorAll('.radio-option').forEach(function(opt) {
    opt.addEventListener('click', function(e) {
      if (e.target.tagName !== 'INPUT') {
        var rb = opt.querySelector('input[type=radio]');
        rb.checked = true;
        rb.dispatchEvent(new Event('change', {bubbles: true}));
      }
    });
  });

  // Path validation on blur
  document.querySelectorAll('[data-validate-path]').forEach(function(el) {
    el.addEventListener('blur', function() {
      var v = el.value.trim();
      var ind = document.getElementById('path_' + el.name);
      if (!ind || !v) { if (ind) ind.textContent = ''; return; }
      fetch('/validate-path', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path: v})
      }).then(function(r) { return r.json(); }).then(function(d) {
        ind.textContent = d.exists ? '\\u2713 Path exists' : '\\u2717 Path not found';
        ind.className = 'path-status ' + (d.exists ? 'valid' : 'invalid');
      }).catch(function() { ind.textContent = ''; });
    });
  });

  // Form submission
  document.getElementById('ahvs-form').addEventListener('submit', function(e) {
    e.preventDefault();
    // Clear errors
    document.querySelectorAll('.field-error').forEach(function(el) { el.style.display = 'none'; });
    document.querySelector('.server-errors').style.display = 'none';

    // Client-side required check (only visible fields)
    var missing = false;
    document.querySelectorAll('[data-required]').forEach(function(el) {
      var group = el.closest('.field-group');
      if (group && group.classList.contains('hidden')) return;
      var val = el.value ? el.value.trim() : '';
      if (!val) {
        var err = document.getElementById('err_' + el.name);
        if (err) { err.textContent = 'This field is required'; err.style.display = 'block'; }
        missing = true;
      }
    });
    // Check required radio groups
    document.querySelectorAll('.radio-group').forEach(function(g) {
      var first = g.querySelector('input[type=radio]');
      if (!first) return;
      var group = g.closest('.field-group');
      if (group && group.classList.contains('hidden')) return;
      if (group && group.querySelector('[data-required]')) return; // handled above
      // Radio groups with required parent
    });
    if (missing) return;

    // Gather data
    var data = {};
    document.querySelectorAll('input[type=text], textarea, select').forEach(function(el) {
      data[el.name] = el.value;
    });
    document.querySelectorAll('input[type=radio]:checked').forEach(function(el) {
      data[el.name] = el.value;
    });
    document.querySelectorAll('input[type=checkbox]').forEach(function(el) {
      data[el.name] = el.checked;
    });

    fetch('/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    }).then(function(r) { return r.json(); }).then(function(d) {
      if (d.ok) {
        document.getElementById('banner').style.display = 'block';
        document.getElementById('ahvs-form').style.opacity = '0.4';
        document.getElementById('ahvs-form').style.pointerEvents = 'none';
        window.scrollTo(0, 0);
      } else if (d.errors) {
        var box = document.querySelector('.server-errors');
        var msgs = [];
        for (var k in d.errors) {
          msgs.push('<strong>' + k + ':</strong> ' + d.errors[k]);
          var err = document.getElementById('err_' + k);
          if (err) { err.textContent = d.errors[k]; err.style.display = 'block'; }
        }
        box.innerHTML = msgs.join('<br>');
        box.style.display = 'block';
      }
    }).catch(function(err) {
      var box = document.querySelector('.server-errors');
      box.textContent = 'Submission failed: ' + err;
      box.style.display = 'block';
    });
  });

  // Init
  evalShowWhen();
  updateRadioStyles();
})();
"""


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------


def _build_html(schema: dict) -> str:
    title = _esc(schema["title"])
    subtitle = _esc(schema.get("subtitle", ""))
    submit_label = _esc(schema.get("submit_label", "Submit"))

    fields_html = "\n".join(_render_field(f) for f in schema["fields"])

    subtitle_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{_BASE_CSS}</style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    {subtitle_html}

    <div class="success-banner" id="banner">
      Submitted successfully. You can close this tab.
    </div>
    <div class="server-errors"></div>

    <form id="ahvs-form">
      {fields_html}
      <div class="actions">
        <button type="button" class="btn btn-secondary" onclick="document.getElementById('ahvs-form').reset(); document.querySelectorAll('.field-error').forEach(function(e){{e.style.display='none'}}); document.querySelector('.server-errors').style.display='none';">Reset</button>
        <button type="submit" class="btn btn-primary">{submit_label} &rarr;</button>
      </div>
    </form>
  </div>
  <script>{_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _FormState:
    def __init__(self) -> None:
        self.result: dict = {}
        self.done = threading.Event()


def _make_handler(
    html_content: str,
    state: _FormState,
    fields: list[dict],
):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            pass  # silence access log

        def do_GET(self) -> None:
            body = html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)

            if self.path == "/validate-path":
                self._handle_validate_path(raw)
                return

            if self.path != "/submit":
                self.send_response(404)
                self.end_headers()
                return

            self._handle_submit(raw)

        def _handle_validate_path(self, raw: bytes) -> None:
            try:
                data = json.loads(raw)
                p = data.get("path", "")
                exists = os.path.exists(p)
            except (json.JSONDecodeError, TypeError):
                exists = False
            resp = json.dumps({"exists": exists}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def _handle_submit(self, raw: bytes) -> None:
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data = {}

            # Server-side validation
            errors: dict[str, str] = {}
            field_map = {f["name"]: f for f in fields}
            for f in fields:
                name = f["name"]
                val = data.get(name, "")

                # Required check (skip hidden conditional fields)
                if f.get("required") and not str(val).strip():
                    # Check if field is conditionally hidden
                    show_when = f.get("show_when")
                    if show_when:
                        visible = all(
                            str(data.get(k, "")) == str(v)
                            for k, v in show_when.items()
                        )
                        if not visible:
                            continue
                    errors[name] = "This field is required"

                # Path validation
                if f.get("validate_path") and str(val).strip():
                    if not os.path.exists(str(val).strip()):
                        errors[name] = f"Path not found: {val}"

            if errors:
                resp = json.dumps({"ok": False, "errors": errors}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return

            state.result = data
            resp = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            state.done.set()

    return Handler


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_DEFAULT_PORT = 8765


def run_form(
    schema: dict,
    *,
    port: int = _DEFAULT_PORT,
    open_browser: bool = True,
) -> dict:
    """Serve a browser form and block until the user submits.

    Args:
        schema: Form definition dict with keys:
            - title (str): Page title
            - subtitle (str, optional): Subtitle text
            - fields (list[dict]): Field definitions
            - submit_label (str, optional): Submit button text
        port: Port to listen on. 0 = OS assigns a free port.
        open_browser: If True, open the default browser automatically.

    Returns:
        Dict mapping field names to submitted values.

    Raises:
        ValueError: If schema is invalid.
    """
    _validate_schema(schema)

    state = _FormState()
    html_content = _build_html(schema)
    handler_class = _make_handler(html_content, state, schema["fields"])

    server = HTTPServer(("0.0.0.0", port), handler_class)
    actual_port = server.server_address[1]

    url = f"http://127.0.0.1:{actual_port}/"
    title = schema.get("title", "AHVS Form")
    print(f"\n{'=' * 60}")
    print(f"  {title} — http://localhost:{actual_port}/")
    print(f"{'=' * 60}")
    print(f"  Open the URL above in your browser to fill out the form.")
    print(f"  Waiting for submission...")
    print(f"{'=' * 60}\n")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    if open_browser:
        webbrowser.open(url)

    state.done.wait()
    server.shutdown()

    print(f"[gui] Form submitted: {list(state.result.keys())}")
    return state.result


# ---------------------------------------------------------------------------
# CLI entry point: python -m ahvs.gui '{"title": "Test", "fields": [...]}'
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "Usage: python -m ahvs.gui '<schema_json>'\n"
            "   or: python -m ahvs.gui --schema-file <path.json>",
            file=sys.stderr,
        )
        return 1

    if args[0] == "--schema-file":
        if len(args) < 2:
            print("Error: --schema-file requires a path argument", file=sys.stderr)
            return 1
        schema = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    else:
        schema = json.loads(args[0])

    try:
        result = run_form(schema)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
