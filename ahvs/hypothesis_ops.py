"""Hypothesis list operations — add, edit, insert, renumber, serialize.

Used by both the CLI (--add/--edit/--insert-hypothesis) and the browser
GUI to modify the hypothesis list between generation and selection.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Valid hypothesis types (from AHVS prompts)
VALID_TYPES = {
    "prompt_rewrite",
    "model_comparison",
    "config_change",
    "dspy_optimize",
    "code_change",
    "architecture_change",
    "multi_llm_judge",
}


def _next_id(hypotheses: list[dict]) -> str:
    """Return the next available H-id (e.g. 'H4' if H1-H3 exist)."""
    existing = set()
    for h in hypotheses:
        m = re.match(r"H(\d+)", h.get("id", ""), re.IGNORECASE)
        if m:
            existing.add(int(m.group(1)))
    n = 1
    while n in existing:
        n += 1
    return f"H{n}"


def _renumber(hypotheses: list[dict]) -> list[dict]:
    """Renumber hypotheses sequentially as H1, H2, ..."""
    for i, h in enumerate(hypotheses, 1):
        h["id"] = f"H{i}"
    return hypotheses


def apply_ops(hypotheses: list[dict], ops: list[dict]) -> list[dict]:
    """Apply a sequence of add/edit/insert operations to a hypothesis list.

    Args:
        hypotheses: Mutable list of hypothesis dicts (modified in place).
        ops: List of operation dicts with "op" key.

    Returns:
        The modified hypothesis list (same reference as input).
    """
    for op in ops:
        kind = op.get("op")
        if kind == "add":
            hypotheses = _apply_add(hypotheses, op)
        elif kind == "edit":
            hypotheses = _apply_edit(hypotheses, op)
        elif kind == "insert":
            hypotheses = _apply_insert(hypotheses, op)
        else:
            logger.warning("Unknown hypothesis op %r — skipping", kind)
    return hypotheses


def _apply_add(hypotheses: list[dict], op: dict) -> list[dict]:
    """Append a new hypothesis with the next available ID."""
    new_id = _next_id(hypotheses)
    hyp = {
        "id": new_id,
        "type": op.get("type", "code_change"),
        "description": op.get("description", ""),
        "rationale": op.get("rationale", "Operator-provided hypothesis"),
        "estimated_cost": op.get("estimated_cost", ""),
        "required_tools": op.get("required_tools", []),
        "_source": "operator",
    }
    hypotheses.append(hyp)
    logger.info("Added hypothesis %s (operator): %s", new_id, hyp["description"][:80])
    return hypotheses


def _apply_edit(hypotheses: list[dict], op: dict) -> list[dict]:
    """Edit fields of an existing hypothesis by ID."""
    target_id = op.get("id", "").upper()
    fields = op.get("fields", {})
    for h in hypotheses:
        if h["id"].upper() == target_id:
            h.update(fields)
            h.setdefault("_source", "llm")
            h["_edited"] = True
            logger.info("Edited hypothesis %s: updated %s", target_id, list(fields.keys()))
            return hypotheses
    logger.warning("Edit target %s not found — skipping", target_id)
    return hypotheses


def _apply_insert(hypotheses: list[dict], op: dict) -> list[dict]:
    """Insert a new hypothesis at a 1-indexed position and renumber."""
    pos = max(1, min(op.get("position", 1), len(hypotheses) + 1))
    hyp = {
        "id": "",  # will be set by renumber
        "type": op.get("type", "code_change"),
        "description": op.get("description", ""),
        "rationale": op.get("rationale", "Operator-provided hypothesis"),
        "estimated_cost": op.get("estimated_cost", ""),
        "required_tools": op.get("required_tools", []),
        "_source": "operator",
    }
    hypotheses.insert(pos - 1, hyp)
    _renumber(hypotheses)
    logger.info(
        "Inserted hypothesis at position %d (now %s): %s",
        pos, hyp["id"], hyp["description"][:80],
    )
    return hypotheses


def hypotheses_to_markdown(hypotheses: list[dict]) -> str:
    """Serialize a hypothesis list back to the hypotheses.md format.

    Produces the canonical markdown format that _parse_hypotheses() can read.
    """
    blocks: list[str] = []
    for h in hypotheses:
        lines = [f"## {h['id']}"]
        lines.append(f"**Type:** {h.get('type', 'code_change')}")
        lines.append(f"**Description:** {h.get('description', '')}")
        lines.append(f"**Rationale:** {h.get('rationale', '')}")
        if h.get("estimated_cost"):
            lines.append(f"**Estimated Cost:** {h['estimated_cost']}")
        tools = h.get("required_tools", [])
        if tools:
            if isinstance(tools, list):
                tools = ", ".join(tools)
            lines.append(f"**Required Tools:** {tools}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def apply_ops_and_rewrite(
    hypotheses: list[dict],
    ops: list[dict],
    hyp_path: Path,
) -> list[dict]:
    """Apply ops, rewrite hypotheses.md, return updated list.

    Convenience wrapper that applies operations and persists the result.
    """
    if not ops:
        return hypotheses
    hypotheses = apply_ops(hypotheses, ops)
    hyp_path.write_text(hypotheses_to_markdown(hypotheses), encoding="utf-8")
    logger.info(
        "Rewrote %s with %d hypotheses after %d ops",
        hyp_path, len(hypotheses), len(ops),
    )
    return hypotheses
