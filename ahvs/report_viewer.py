"""AHVS Data Analyst Report Viewer — renders analysis reports in the browser.

Reads an ``analysis_report.md`` file, converts it to styled HTML with
embedded PNG figures (base64), and serves it on localhost. The page uses
the same dark theme as the AHVS GUI forms.

Usage:
    from ahvs.report_viewer import serve_report
    serve_report("analysis_20260408/analysis_report.md")   # blocks until Ctrl-C

CLI:
    ahvs data_analyst --view analysis_20260408/analysis_report.md
"""

from __future__ import annotations

import base64
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Markdown → HTML conversion (stdlib only, no dependencies)
# ---------------------------------------------------------------------------


def _md_to_html(md: str) -> str:
    """Convert markdown to HTML using basic regex transformations.

    Handles: headings, tables, code blocks, inline code, bold, italic,
    links, images, horizontal rules, lists, and paragraphs.
    """
    lines = md.split("\n")
    html_parts: list[str] = []
    in_code_block = False
    in_table = False
    in_list = False
    list_type = ""  # "ul" or "ol"

    i = 0
    while i < len(lines):
        line = lines[i]

        # Fenced code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                html_parts.append("</code></pre>")
                in_code_block = False
            else:
                lang = line.strip()[3:].strip()
                cls = f' class="language-{lang}"' if lang else ""
                html_parts.append(f"<pre><code{cls}>")
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            html_parts.append(_escape(line))
            i += 1
            continue

        # Close table if we're leaving table context
        if in_table and not line.strip().startswith("|"):
            html_parts.append("</tbody></table></div>")
            in_table = False

        # Close list if we're leaving list context
        if in_list and not re.match(r"^\s*[-*+]\s|^\s*\d+\.\s", line) and line.strip():
            html_parts.append(f"</{list_type}>")
            in_list = False

        stripped = line.strip()

        # Horizontal rule
        if re.match(r"^(-{3,}|_{3,}|\*{3,})\s*$", stripped):
            html_parts.append("<hr>")
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            level = len(m.group(1))
            text = _inline(m.group(2))
            slug = re.sub(r"[^\w-]", "", m.group(2).lower().replace(" ", "-"))
            html_parts.append(
                f'<h{level} id="{slug}">{text}</h{level}>'
            )
            i += 1
            continue

        # Table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]

            # Separator row (|---|---|)
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                i += 1
                continue

            if not in_table:
                # First row = header
                html_parts.append(
                    '<div class="table-wrap"><table><thead><tr>'
                )
                for cell in cells:
                    html_parts.append(f"<th>{_inline(cell)}</th>")
                html_parts.append("</tr></thead><tbody>")
                in_table = True
            else:
                html_parts.append("<tr>")
                for cell in cells:
                    html_parts.append(f"<td>{_inline(cell)}</td>")
                html_parts.append("</tr>")
            i += 1
            continue

        # Unordered list
        m = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if m:
            if not in_list or list_type != "ul":
                if in_list:
                    html_parts.append(f"</{list_type}>")
                html_parts.append("<ul>")
                in_list = True
                list_type = "ul"
            html_parts.append(f"<li>{_inline(m.group(2))}</li>")
            i += 1
            continue

        # Ordered list
        m = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if m:
            if not in_list or list_type != "ol":
                if in_list:
                    html_parts.append(f"</{list_type}>")
                html_parts.append("<ol>")
                in_list = True
                list_type = "ol"
            html_parts.append(f"<li>{_inline(m.group(2))}</li>")
            i += 1
            continue

        # Empty line
        if not stripped:
            i += 1
            continue

        # Paragraph
        html_parts.append(f"<p>{_inline(stripped)}</p>")
        i += 1

    # Close open blocks
    if in_code_block:
        html_parts.append("</code></pre>")
    if in_table:
        html_parts.append("</tbody></table></div>")
    if in_list:
        html_parts.append(f"</{list_type}>")

    return "\n".join(html_parts)


def _escape(text: str) -> str:
    """HTML-escape text."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline(text: str) -> str:
    """Process inline markdown: bold, italic, code, links, images."""
    # Images: ![alt](src)
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        r'<img src="\2" alt="\1" class="report-img">',
        text,
    )
    # Links: [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2">\1</a>',
        text,
    )
    # Bold+italic: ***text***
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    # Bold: **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic: *text*
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code: `text`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


# ---------------------------------------------------------------------------
# Image embedding — replace relative image paths with base64 data URIs
# ---------------------------------------------------------------------------


def _embed_images(html: str, report_dir: Path) -> str:
    """Replace <img src="relative/path.png"> with base64 data URIs."""

    def _replace_img(match: re.Match) -> str:
        src = match.group(1)
        # Skip already-embedded or absolute URLs
        if src.startswith("data:") or src.startswith("http"):
            return match.group(0)

        img_path = (report_dir / src).resolve()
        if not img_path.is_file():
            # Try without leading ./
            img_path = (report_dir / src.lstrip("./")).resolve()
        if not img_path.is_file():
            return match.group(0)  # leave as-is

        mime, _ = mimetypes.guess_type(str(img_path))
        if not mime:
            mime = "image/png"

        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{data}" alt="{match.group(2)}" class="report-img">'

    return re.sub(
        r'<img\s+src="([^"]+)"\s+alt="([^"]*)"\s+class="report-img">',
        _replace_img,
        html,
    )


# ---------------------------------------------------------------------------
# HTML template with dark theme
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #0f1117;
        color: #e2e8f0;
        min-height: 100vh;
        padding: 2rem 1rem;
        line-height: 1.7;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}

    /* Header */
    .header {{
        border-bottom: 1px solid #2d3448;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }}
    .header h1 {{
        font-size: 1.6rem; font-weight: 700; color: #a78bfa;
        margin-bottom: 0.3rem;
    }}
    .header .meta {{
        font-size: 0.85rem; color: #64748b;
    }}

    /* Headings */
    h1 {{ font-size: 1.5rem; color: #a78bfa; margin: 2rem 0 0.8rem; }}
    h2 {{ font-size: 1.3rem; color: #c4b5fd; margin: 1.8rem 0 0.7rem;
          border-bottom: 1px solid #1e2130; padding-bottom: 0.4rem; }}
    h3 {{ font-size: 1.1rem; color: #ddd6fe; margin: 1.4rem 0 0.5rem; }}
    h4, h5, h6 {{ font-size: 1rem; color: #e2e8f0; margin: 1.2rem 0 0.4rem; }}

    /* Paragraphs */
    p {{ margin: 0.6rem 0; color: #cbd5e1; }}

    /* Links */
    a {{ color: #818cf8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    /* Bold / Italic */
    strong {{ color: #f1f5f9; }}

    /* Inline code */
    code {{
        background: #1e2130; color: #a5b4fc;
        padding: 0.15rem 0.4rem; border-radius: 4px;
        font-size: 0.88em;
    }}

    /* Code blocks */
    pre {{
        background: #1a1d2e; border: 1px solid #2d3448;
        border-radius: 8px; padding: 1rem 1.2rem;
        overflow-x: auto; margin: 1rem 0;
    }}
    pre code {{
        background: none; padding: 0; color: #c4b5fd;
        font-size: 0.85rem; line-height: 1.6;
    }}

    /* Tables */
    .table-wrap {{
        overflow-x: auto; margin: 1rem 0;
        border-radius: 8px; border: 1px solid #2d3448;
    }}
    table {{
        width: 100%; border-collapse: collapse;
        font-size: 0.88rem;
    }}
    thead {{ background: #1a1d35; }}
    th {{
        text-align: left; padding: 0.7rem 1rem;
        color: #c4b5fd; font-weight: 600;
        border-bottom: 2px solid #312e81;
    }}
    td {{
        padding: 0.6rem 1rem; color: #cbd5e1;
        border-bottom: 1px solid #1e2130;
    }}
    tbody tr:hover {{ background: #161928; }}

    /* Lists */
    ul, ol {{
        margin: 0.6rem 0 0.6rem 1.8rem; color: #cbd5e1;
    }}
    li {{ margin: 0.3rem 0; }}

    /* Images */
    .report-img {{
        max-width: 100%; height: auto;
        border-radius: 8px; border: 1px solid #2d3448;
        margin: 1rem 0; display: block;
    }}

    /* Horizontal rule */
    hr {{
        border: none; border-top: 1px solid #2d3448;
        margin: 2rem 0;
    }}

    /* Footer */
    .footer {{
        margin-top: 3rem; padding-top: 1.5rem;
        border-top: 1px solid #2d3448;
        text-align: center; font-size: 0.8rem; color: #475569;
    }}

    /* Print */
    @media print {{
        body {{ background: #fff; color: #111; padding: 0; }}
        .container {{ max-width: 100%; }}
        h1, h2, h3 {{ color: #111; }}
        th {{ color: #333; }}
        td, p, li {{ color: #333; }}
        code {{ background: #f0f0f0; color: #333; }}
        pre {{ background: #f5f5f5; border-color: #ddd; }}
        pre code {{ color: #333; }}
        .report-img {{ border-color: #ddd; }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{title}</h1>
        <div class="meta">{meta}</div>
    </div>
    {body}
    <div class="footer">
        Generated by AHVS Data Analyst
    </div>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Build full HTML page from markdown file
# ---------------------------------------------------------------------------


def build_report_html(report_path: str | Path) -> str:
    """Read a markdown report and return a complete HTML page.

    Images referenced with relative paths are embedded as base64 data URIs
    so the page is fully self-contained.
    """
    report_path = Path(report_path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")

    md_text = report_path.read_text(encoding="utf-8")
    report_dir = report_path.parent

    # Extract title from first heading or filename
    title_match = re.match(r"^#\s+(.+)", md_text, re.MULTILINE)
    title = title_match.group(1) if title_match else report_path.stem

    # Convert markdown to HTML
    body_html = _md_to_html(md_text)

    # Embed images
    body_html = _embed_images(body_html, report_dir)

    meta = f"Source: {report_path.name} &mdash; {report_dir}"

    return _PAGE_TEMPLATE.format(
        title=_escape(title),
        meta=meta,
        body=body_html,
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 8765


def serve_report(
    report_path: str | Path,
    *,
    port: int = _DEFAULT_PORT,
    open_browser: bool = True,
    blocking: bool = True,
) -> Optional[HTTPServer]:
    """Serve an analysis report as a styled HTML page in the browser.

    Args:
        report_path: Path to ``analysis_report.md`` or the analysis directory
            (will look for ``analysis_report.md`` inside).
        port: Port to listen on (default: 8765).
        open_browser: Open the default browser automatically.
        blocking: If True, block until interrupted. If False, return the
            server instance (caller must shut it down).

    Returns:
        HTTPServer if non-blocking, else None.
    """
    rp = Path(report_path).resolve()

    # If a directory was given, look for the report inside
    if rp.is_dir():
        candidate = rp / "analysis_report.md"
        if candidate.is_file():
            rp = candidate
        else:
            raise FileNotFoundError(
                f"No analysis_report.md found in {rp}"
            )

    html_content = build_report_html(rp)
    html_bytes = html_content.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            pass  # silence access log

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

    server = HTTPServer(("0.0.0.0", port), Handler)
    actual_port = server.server_address[1]
    url = f"http://localhost:{actual_port}/"

    print(f"\n{'=' * 60}")
    print(f"  AHVS Data Analyst Report Viewer")
    print(f"  {url}")
    print(f"{'=' * 60}")
    print(f"  Report: {rp.name}")
    print(f"  Directory: {rp.parent}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"{'=' * 60}\n")

    if open_browser:
        webbrowser.open(url)

    if blocking:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return None
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
