"""Decomposed Annotation Analysis GUI — reusable template.

Serves a dark-themed HTML dashboard that decomposes complex structured
annotations (json_array, multi-task completions) into sub-field distributions
with side-by-side input/output sample cards.

Usage:
    python decomposed_analysis_gui.py <annotated_csv> [--port 8765] [--task-fields entity,sentiment,confidence,perspective]

Or as a library:
    from ahvs.templates.decomposed_analysis_gui import serve_analysis
    serve_analysis("path/to/annotated_data.csv", port=8765)

The annotated CSV must have columns: text, completion (JSON string), and
optionally: target_entities, prompt, entity_group.

Style approved by user on 2026-04-14 — "excellent, comprehensive and exceptional."
"""

from __future__ import annotations

import argparse
import json
import html as html_mod
from collections import Counter, OrderedDict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pandas as pd


# ── HTML/CSS template ──────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #0f1117; color: #e2e8f0; padding: 2rem; line-height: 1.6; }
.container { max-width: 1100px; margin: 0 auto; }
h1 { color: #fff; margin-bottom: .3rem; }
h2 { color: #90caf9; margin: 2rem 0 1rem; border-bottom: 1px solid #333; padding-bottom: .5rem; }
h3 { color: #b0bec5; margin-bottom: .5rem; }
h4 { color: #fff; margin-bottom: .5rem; }
h5 { color: #90caf9; margin-bottom: .5rem; font-size: .85rem; text-transform: uppercase; }
.subtitle { color: #78909c; margin-bottom: 2rem; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
.stat { background: #1a1d2e; border-radius: 8px; padding: 1rem; text-align: center; }
.stat-value { font-size: 2rem; font-weight: bold; color: #4fc3f7; }
.stat-label { font-size: .85rem; color: #78909c; }
.card { background: #1a1d2e; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: .5rem; color: #78909c; font-size: .85rem; border-bottom: 1px solid #333; }
td { padding: .4rem .5rem; border-bottom: 1px solid #222; }
.badge { background: #333; color: #90caf9; padding: 2px 8px; border-radius: 12px; font-size: .8rem; }
.tag { display: inline-block; background: #263238; padding: 2px 8px; border-radius: 4px; font-size: .85rem; margin: 2px; }
.tag-pos { background: #1b5e20; color: #a5d6a7; }
.tag-neg { background: #b71c1c; color: #ef9a9a; }
.tag-neu { background: #37474f; color: #b0bec5; }
.tag-mix { background: #4a148c; color: #ce93d8; }
.tag-absent { background: #333; color: #666; }
.dist-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 800px) { .dist-grid { grid-template-columns: 1fr; } }
.sample-card { background: #1a1d2e; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; border-left: 3px solid #4fc3f7; }
.io-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: .5rem; }
@media (max-width: 800px) { .io-grid { grid-template-columns: 1fr; } }
.input-box { background: #111827; border-radius: 6px; padding: 1rem; }
.output-box { background: #0d1117; border-radius: 6px; padding: 1rem; }
.text-content { font-size: .9rem; color: #b0bec5; max-height: 120px; overflow-y: auto; }
.record { margin: .5rem 0; padding: .5rem; background: #111; border-radius: 4px; }
em { color: #78909c; font-size: .85rem; }
pre { background: #0a0a0a; padding: .5rem; border-radius: 4px; }
"""


def _esc(text: Any, max_len: int = 0) -> str:
    s = html_mod.escape(str(text))
    return s[:max_len] if max_len else s


def _dist_table(data: Counter | dict, title: str) -> str:
    if isinstance(data, Counter):
        items = data.most_common()
    elif isinstance(data, dict):
        items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    else:
        items = list(data)
    total = sum(v for _, v in items)
    rows = ""
    for val, cnt in items:
        pct = cnt / total * 100 if total else 0
        bar_w = max(1, int(pct * 3))
        rows += (
            f"<tr><td>{_esc(val)}</td><td>{cnt}</td><td>{pct:.1f}%</td>"
            f"<td><div style='background:#4fc3f7;height:14px;width:{bar_w}px;"
            f"border-radius:3px'></div></td></tr>\n"
        )
    return (
        f'<div class="card"><h3>{title} <span class="badge">{total} total</span></h3>'
        f"<table><thead><tr><th>Value</th><th>Count</th><th>%</th><th></th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _build_sample_card(
    idx: int,
    row: pd.Series,
    comp: dict,
) -> str:
    es = comp.get("entity_sentiments", [])
    input_text = _esc(row["text"], 500)
    target_ents = _esc(row.get("target_entities", ""))
    prompt_snip = _esc(row.get("prompt", ""), 600)

    output_parts = []
    if not es:
        output_parts.append('<span class="tag tag-absent">entity_sentiments: []</span>')
    else:
        for r in es:
            sc = {
                "positive": "tag-pos",
                "negative": "tag-neg",
                "neutral": "tag-neu",
                "mixed": "tag-mix",
            }.get(r.get("sentiment", ""), "")
            output_parts.append(
                f"<div class='record'><span class='tag'>{_esc(r.get('entity', ''))}</span> "
                f"<span class='tag {sc}'>{r.get('sentiment', '')}</span> "
                f"<span class='tag'>conf: {r.get('confidence', '')}</span> "
                f"<span class='tag'>persp: {_esc(r.get('perspective', ''))}</span><br>"
                f"<em>evidence: \"{_esc(r.get('evidence', ''), 120)}\"</em></div>"
            )

    def _field(key):
        """Extract a field that may be either a dict with 'label'/'value' or a string."""
        v = comp.get(key, "")
        if isinstance(v, dict):
            return v.get("label") or v.get("value") or ""
        return str(v) if v else ""

    overall = _field("overall_sentiment") or "?"
    presence = _field("entity_presence") or "?"
    rationale = _esc(_field("sentiment_rationale"), 250)

    return f"""<div class="sample-card">
      <h4>Sample #{idx}</h4>
      <div class="io-grid">
        <div class="input-box"><h5>INPUT (Prompt)</h5>
          <p><strong>Target entities:</strong> {target_ents}</p>
          <p class="text-content">{input_text}</p>
          <details><summary style="color:#78909c;cursor:pointer;margin-top:8px">Full prompt snippet</summary>
          <pre style="font-size:.75rem;color:#666;white-space:pre-wrap;max-height:200px;overflow:auto">{prompt_snip}</pre></details>
        </div>
        <div class="output-box"><h5>OUTPUT (Completion)</h5>
          <p><strong>entity_sentiments:</strong></p>{''.join(output_parts)}
          <p style="margin-top:8px"><strong>overall_sentiment:</strong> <span class="tag">{overall}</span>
          &nbsp;<strong>entity_presence:</strong> <span class="tag">{presence}</span></p>
          <p><strong>rationale:</strong> <em>{rationale}</em></p>
        </div>
      </div></div>"""


def build_analysis_html(
    df: pd.DataFrame,
    title: str = "Annotation Quality Analysis",
    subtitle: str = "",
    n_samples: int = 10,
) -> str:
    """Build the full HTML page from an annotated DataFrame."""

    all_entities: list[str] = []
    all_sentiments: list[str] = []
    all_confidences: list[str] = []
    all_perspectives: list[str] = []
    all_overall: list[str] = []
    all_presence: list[str] = []
    records_per_doc: list[int] = []
    parse_errors = 0

    def _unwrap(comp: dict) -> dict:
        """Unwrap 'results' wrapper if present."""
        if "results" in comp and isinstance(comp["results"], dict):
            return comp["results"]
        return comp

    for _, row in df.iterrows():
        try:
            comp = _unwrap(json.loads(row["completion"]))
            es = comp.get("entity_sentiments", [])
            records_per_doc.append(len(es))
            for r in es:
                all_entities.append(r.get("entity", "MISSING"))
                all_sentiments.append(r.get("sentiment", "MISSING"))
                all_confidences.append(r.get("confidence", "MISSING"))
                all_perspectives.append(r.get("perspective", "MISSING"))
            def _label(key):
                v = comp.get(key)
                if isinstance(v, dict):
                    return v.get("label") or v.get("value") or "MISSING"
                return str(v) if v else "MISSING"
            all_overall.append(_label("overall_sentiment"))
            all_presence.append(_label("entity_presence"))
        except (json.JSONDecodeError, TypeError):
            parse_errors += 1
            records_per_doc.append(0)
            all_overall.append("PARSE_ERROR")
            all_presence.append("PARSE_ERROR")

    top15_ents = OrderedDict(Counter(all_entities).most_common(15))

    # Sample cards — evenly spaced
    step = max(1, len(df) // n_samples)
    sample_indices = list(range(0, len(df), step))[:n_samples]
    sample_cards = ""
    for i in sample_indices:
        row = df.iloc[i]
        try:
            comp = _unwrap(json.loads(row["completion"]))
        except (json.JSONDecodeError, TypeError):
            continue
        sample_cards += _build_sample_card(i, row, comp)

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class="container">
<h1>{_esc(title)}</h1>
<p class="subtitle">{_esc(subtitle) or f"Decomposed analysis of {len(df)} annotated samples"}</p>

<h2>Overview</h2>
<div class="stats-grid">
  <div class="stat"><div class="stat-value">{len(df)}</div><div class="stat-label">Docs annotated</div></div>
  <div class="stat"><div class="stat-value">{sum(records_per_doc)}</div><div class="stat-label">Entity-sentiment records</div></div>
  <div class="stat"><div class="stat-value">{sum(records_per_doc)/max(len(records_per_doc),1):.1f}</div><div class="stat-label">Avg records/doc</div></div>
  <div class="stat"><div class="stat-value">{records_per_doc.count(0)}</div><div class="stat-label">Empty outputs ([])</div></div>
  <div class="stat"><div class="stat-value">{parse_errors}</div><div class="stat-label">Parse errors</div></div>
  <div class="stat"><div class="stat-value">{len(set(all_entities))}</div><div class="stat-label">Unique entities</div></div>
</div>

<h2>Sub-task Distributions</h2>
<div class="dist-grid">
  {_dist_table(Counter(all_sentiments), "Sentiment (entity_sentiments)")}
  {_dist_table(Counter(all_confidences), "Confidence (entity_sentiments)")}
  {_dist_table(Counter(all_perspectives), "Perspective (entity_sentiments)")}
  {_dist_table(Counter(all_overall), "Overall Sentiment (per doc)")}
  {_dist_table(Counter(all_presence), "Entity Presence (per doc)")}
  {_dist_table(top15_ents, "Top 15 Entities (entity_sentiments)")}
</div>

<h2>Records Per Document</h2>
{_dist_table(Counter(records_per_doc), "Number of entity-sentiment records per document")}

<h2>Sample Annotations &mdash; Input + Output</h2>
<p class="subtitle">{n_samples} samples showing prompt (input) and completion (output) side by side</p>
{sample_cards}
</div></body></html>"""


def build_analysis_markdown(
    df: pd.DataFrame,
    title: str = "Annotation Quality Analysis",
    subtitle: str = "",
    n_samples: int = 10,
) -> str:
    """Build a markdown version of the analysis report (mirror of HTML content)."""

    all_entities: list[str] = []
    all_sentiments: list[str] = []
    all_confidences: list[str] = []
    all_perspectives: list[str] = []
    all_overall: list[str] = []
    all_presence: list[str] = []
    records_per_doc: list[int] = []
    parse_errors = 0

    def _unwrap(comp: dict) -> dict:
        if "results" in comp and isinstance(comp["results"], dict):
            return comp["results"]
        return comp

    def _label(comp, key):
        v = comp.get(key)
        if isinstance(v, dict):
            return v.get("label") or v.get("value") or "MISSING"
        return str(v) if v else "MISSING"

    for _, row in df.iterrows():
        try:
            comp = _unwrap(json.loads(row["completion"]))
            es = comp.get("entity_sentiments", [])
            records_per_doc.append(len(es))
            for r in es:
                all_entities.append(r.get("entity", "MISSING"))
                all_sentiments.append(r.get("sentiment", "MISSING"))
                all_confidences.append(r.get("confidence", "MISSING"))
                all_perspectives.append(r.get("perspective", "MISSING"))
            all_overall.append(_label(comp, "overall_sentiment"))
            all_presence.append(_label(comp, "entity_presence"))
        except (json.JSONDecodeError, TypeError):
            parse_errors += 1
            records_per_doc.append(0)

    def _dist_md(data, heading):
        if isinstance(data, Counter):
            items = data.most_common()
        else:
            items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in items)
        lines = [f"\n### {heading} ({total} total)\n", "| Value | Count | % |", "|---|---|---|"]
        for val, cnt in items:
            pct = cnt / total * 100 if total else 0
            lines.append(f"| {val} | {cnt} | {pct:.1f}% |")
        return "\n".join(lines) + "\n"

    top15_ents = OrderedDict(Counter(all_entities).most_common(15))

    out = [f"# {title}\n"]
    if subtitle:
        out.append(f"_{subtitle}_\n")

    out.append("\n## Overview\n")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Docs annotated | {len(df)} |")
    out.append(f"| Entity-sentiment records | {sum(records_per_doc)} |")
    out.append(f"| Avg records/doc | {sum(records_per_doc)/max(len(records_per_doc),1):.1f} |")
    out.append(f"| Empty outputs ([]) | {records_per_doc.count(0)} |")
    out.append(f"| Parse errors | {parse_errors} |")
    out.append(f"| Unique entities | {len(set(all_entities))} |")

    out.append("\n## Sub-task Distributions")
    out.append(_dist_md(Counter(all_sentiments), "Sentiment (entity_sentiments)"))
    out.append(_dist_md(Counter(all_confidences), "Confidence (entity_sentiments)"))
    out.append(_dist_md(Counter(all_perspectives), "Perspective (entity_sentiments)"))
    out.append(_dist_md(Counter(all_overall), "Overall Sentiment (per doc)"))
    out.append(_dist_md(Counter(all_presence), "Entity Presence (per doc)"))
    out.append(_dist_md(top15_ents, "Top 15 Entities"))
    out.append(_dist_md(Counter(records_per_doc), "Records per Document"))

    # Sample annotations
    out.append("\n## Sample Annotations (Input + Output)\n")
    step = max(1, len(df) // n_samples)
    sample_indices = list(range(0, len(df), step))[:n_samples]
    for i in sample_indices:
        row = df.iloc[i]
        try:
            comp = _unwrap(json.loads(row["completion"]))
        except (json.JSONDecodeError, TypeError):
            continue
        es = comp.get("entity_sentiments", [])
        out.append(f"\n### Sample #{i}\n")
        out.append(f"**Input:** `{str(row.get('target_entities', ''))[:80]}`\n")
        text = str(row['text'])[:400].replace("\n", " ")
        out.append(f"> {text}\n")
        out.append("\n**Output:**\n")
        if not es:
            out.append("- `entity_sentiments: []` (empty)\n")
        else:
            for r in es:
                out.append(
                    f"- **{r.get('entity','')}**: {r.get('sentiment','')} "
                    f"(conf={r.get('confidence','')}, persp={r.get('perspective','')})"
                )
                ev = str(r.get('evidence',''))[:120]
                out.append(f'  - evidence: "{ev}"')
        out.append(f"- overall_sentiment: `{_label(comp, 'overall_sentiment')}`")
        out.append(f"- entity_presence: `{_label(comp, 'entity_presence')}`")

    return "\n".join(out)


def save_reports(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    name: str = "analysis",
    title: str = "Annotation Quality Analysis",
    subtitle: str = "",
    n_samples: int = 10,
) -> tuple[Path, Path]:
    """Save both HTML and markdown versions of the report to disk.

    Returns (html_path, md_path).
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html = build_analysis_html(df, title, subtitle, n_samples)
    md = build_analysis_markdown(df, title, subtitle, n_samples)

    html_path = output_dir / f"{name}.html"
    md_path = output_dir / f"{name}.md"
    html_path.write_text(html)
    md_path.write_text(md)

    return html_path, md_path


def serve_analysis(
    csv_path: str | Path,
    *,
    port: int = 8765,
    title: str = "Annotation Quality Analysis",
    subtitle: str = "",
    n_samples: int = 10,
    blocking: bool = True,
) -> HTTPServer | None:
    """Load annotated CSV and serve the analysis dashboard."""
    df = pd.read_csv(csv_path)
    html_bytes = build_analysis_html(df, title, subtitle, n_samples).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

    server = HTTPServer(("0.0.0.0", port), Handler)
    url = f"http://localhost:{port}/"
    print(f"\n{'=' * 60}")
    print(f"  Decomposed Analysis GUI")
    print(f"  {url}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'=' * 60}\n")

    if blocking:
        server.serve_forever()
        return None
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve decomposed annotation analysis GUI")
    parser.add_argument("csv_path", help="Path to annotated CSV (must have text + completion columns)")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765)")
    parser.add_argument("--title", default="Annotation Quality Analysis")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of sample cards to show")
    args = parser.parse_args()
    serve_analysis(args.csv_path, port=args.port, title=args.title, subtitle=args.subtitle, n_samples=args.n_samples)
