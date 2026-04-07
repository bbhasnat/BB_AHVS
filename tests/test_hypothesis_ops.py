"""Tests for ahvs.hypothesis_ops — add, edit, insert, renumber, serialize."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ahvs.hypothesis_ops import (
    VALID_TYPES,
    apply_ops,
    apply_ops_and_rewrite,
    hypotheses_to_markdown,
    _next_id,
    _renumber,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hypotheses(n: int = 3) -> list[dict]:
    """Build a list of N sample hypotheses."""
    return [
        {
            "id": f"H{i}",
            "type": "code_change",
            "description": f"Description for H{i}",
            "rationale": f"Rationale for H{i}",
            "estimated_cost": "low",
            "required_tools": [],
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# _next_id
# ---------------------------------------------------------------------------

class TestNextId:
    def test_sequential(self):
        hyps = _make_hypotheses(3)
        assert _next_id(hyps) == "H4"

    def test_gap(self):
        hyps = [{"id": "H1"}, {"id": "H3"}]
        assert _next_id(hyps) == "H2"

    def test_empty(self):
        assert _next_id([]) == "H1"


# ---------------------------------------------------------------------------
# _renumber
# ---------------------------------------------------------------------------

class TestRenumber:
    def test_sequential(self):
        hyps = [{"id": "H5"}, {"id": "H2"}, {"id": "H9"}]
        _renumber(hyps)
        assert [h["id"] for h in hyps] == ["H1", "H2", "H3"]


# ---------------------------------------------------------------------------
# apply_ops — add
# ---------------------------------------------------------------------------

class TestApplyAdd:
    def test_add_appends(self):
        hyps = _make_hypotheses(2)
        ops = [{"op": "add", "type": "prompt_rewrite", "description": "Custom hypothesis"}]
        result = apply_ops(hyps, ops)
        assert len(result) == 3
        assert result[2]["id"] == "H3"
        assert result[2]["type"] == "prompt_rewrite"
        assert result[2]["description"] == "Custom hypothesis"
        assert result[2]["_source"] == "operator"

    def test_add_defaults_type(self):
        hyps = _make_hypotheses(1)
        ops = [{"op": "add", "description": "No type specified"}]
        result = apply_ops(hyps, ops)
        assert result[1]["type"] == "code_change"

    def test_add_multiple(self):
        hyps = _make_hypotheses(1)
        ops = [
            {"op": "add", "description": "First added"},
            {"op": "add", "description": "Second added"},
        ]
        result = apply_ops(hyps, ops)
        assert len(result) == 3
        assert result[1]["id"] == "H2"
        assert result[2]["id"] == "H3"


# ---------------------------------------------------------------------------
# apply_ops — edit
# ---------------------------------------------------------------------------

class TestApplyEdit:
    def test_edit_updates_fields(self):
        hyps = _make_hypotheses(3)
        ops = [{"op": "edit", "id": "H2", "fields": {"description": "Updated desc"}}]
        result = apply_ops(hyps, ops)
        assert result[1]["description"] == "Updated desc"
        assert result[1]["_edited"] is True
        # Other fields unchanged
        assert result[1]["rationale"] == "Rationale for H2"

    def test_edit_case_insensitive(self):
        hyps = _make_hypotheses(2)
        ops = [{"op": "edit", "id": "h1", "fields": {"type": "config_change"}}]
        result = apply_ops(hyps, ops)
        assert result[0]["type"] == "config_change"

    def test_edit_nonexistent_skips(self):
        hyps = _make_hypotheses(2)
        ops = [{"op": "edit", "id": "H99", "fields": {"description": "nope"}}]
        result = apply_ops(hyps, ops)
        assert len(result) == 2
        assert result[0]["description"] == "Description for H1"

    def test_edit_multiple_fields(self):
        hyps = _make_hypotheses(2)
        ops = [{"op": "edit", "id": "H1", "fields": {
            "description": "New desc",
            "rationale": "New rationale",
            "estimated_cost": "high",
        }}]
        result = apply_ops(hyps, ops)
        assert result[0]["description"] == "New desc"
        assert result[0]["rationale"] == "New rationale"
        assert result[0]["estimated_cost"] == "high"


# ---------------------------------------------------------------------------
# apply_ops — insert
# ---------------------------------------------------------------------------

class TestApplyInsert:
    def test_insert_at_position_1(self):
        hyps = _make_hypotheses(2)
        ops = [{"op": "insert", "position": 1, "description": "Inserted first"}]
        result = apply_ops(hyps, ops)
        assert len(result) == 3
        assert result[0]["description"] == "Inserted first"
        assert result[0]["id"] == "H1"
        assert result[1]["id"] == "H2"
        assert result[2]["id"] == "H3"

    def test_insert_at_end(self):
        hyps = _make_hypotheses(2)
        ops = [{"op": "insert", "position": 3, "description": "Inserted at end"}]
        result = apply_ops(hyps, ops)
        assert len(result) == 3
        assert result[2]["description"] == "Inserted at end"

    def test_insert_clamps_position(self):
        hyps = _make_hypotheses(2)
        # Position 999 clamps to end
        ops = [{"op": "insert", "position": 999, "description": "Clamped"}]
        result = apply_ops(hyps, ops)
        assert result[2]["description"] == "Clamped"

    def test_insert_renumbers(self):
        hyps = _make_hypotheses(3)
        ops = [{"op": "insert", "position": 2, "description": "Inserted at 2"}]
        result = apply_ops(hyps, ops)
        ids = [h["id"] for h in result]
        assert ids == ["H1", "H2", "H3", "H4"]
        assert result[1]["description"] == "Inserted at 2"


# ---------------------------------------------------------------------------
# Mixed operations
# ---------------------------------------------------------------------------

class TestMixedOps:
    def test_add_then_edit(self):
        hyps = _make_hypotheses(1)
        ops = [
            {"op": "add", "description": "Added"},
            {"op": "edit", "id": "H2", "fields": {"description": "Edited after add"}},
        ]
        result = apply_ops(hyps, ops)
        assert result[1]["description"] == "Edited after add"

    def test_insert_then_add(self):
        hyps = _make_hypotheses(2)
        ops = [
            {"op": "insert", "position": 1, "description": "Inserted"},
            {"op": "add", "description": "Added"},
        ]
        result = apply_ops(hyps, ops)
        assert len(result) == 4
        assert result[0]["description"] == "Inserted"
        assert result[3]["description"] == "Added"

    def test_empty_ops(self):
        hyps = _make_hypotheses(3)
        result = apply_ops(hyps, [])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# hypotheses_to_markdown
# ---------------------------------------------------------------------------

class TestHypothesesToMarkdown:
    def test_round_trip(self):
        hyps = _make_hypotheses(2)
        md = hypotheses_to_markdown(hyps)
        assert "## H1" in md
        assert "## H2" in md
        assert "**Type:** code_change" in md
        assert "**Description:** Description for H1" in md
        assert "**Rationale:** Rationale for H2" in md

    def test_includes_cost_and_tools(self):
        hyps = [{
            "id": "H1",
            "type": "config_change",
            "description": "Desc",
            "rationale": "Rat",
            "estimated_cost": "high",
            "required_tools": ["docker", "promptfoo"],
        }]
        md = hypotheses_to_markdown(hyps)
        assert "**Estimated Cost:** high" in md
        assert "**Required Tools:** docker, promptfoo" in md

    def test_omits_empty_cost(self):
        hyps = [{"id": "H1", "type": "code_change", "description": "D", "rationale": "R", "estimated_cost": ""}]
        md = hypotheses_to_markdown(hyps)
        assert "Estimated Cost" not in md

    def test_parseable_by_parse_hypotheses(self):
        """The markdown output should be parseable by _parse_hypotheses."""
        from ahvs.hypothesis_selector import _parse_hypotheses
        hyps = _make_hypotheses(3)
        md = hypotheses_to_markdown(hyps)
        parsed = _parse_hypotheses(md)
        assert len(parsed) == 3
        assert parsed[0]["id"] == "H1"
        assert parsed[2]["description"] == "Description for H3"


# ---------------------------------------------------------------------------
# apply_ops_and_rewrite
# ---------------------------------------------------------------------------

class TestApplyOpsAndRewrite:
    def test_writes_file(self, tmp_path: Path):
        hyp_path = tmp_path / "hypotheses.md"
        hyps = _make_hypotheses(2)
        hyp_path.write_text(hypotheses_to_markdown(hyps))
        ops = [{"op": "add", "description": "New one"}]
        result = apply_ops_and_rewrite(hyps, ops, hyp_path)
        assert len(result) == 3
        content = hyp_path.read_text()
        assert "## H3" in content
        assert "New one" in content

    def test_noop_with_empty_ops(self, tmp_path: Path):
        hyp_path = tmp_path / "hypotheses.md"
        hyps = _make_hypotheses(2)
        original_md = hypotheses_to_markdown(hyps)
        hyp_path.write_text(original_md)
        result = apply_ops_and_rewrite(hyps, [], hyp_path)
        assert len(result) == 2
        # File should not have been touched (early return)
        assert hyp_path.read_text() == original_md


# ---------------------------------------------------------------------------
# CLI flag parsing (integration with cli._parse_hypothesis_ops)
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_parse_add_flag(self):
        from ahvs.cli import _parse_hypothesis_ops
        import argparse
        ns = argparse.Namespace(
            add_hypotheses=['{"type":"config_change","description":"Test add"}'],
            edit_hypotheses=[],
            insert_hypotheses=[],
        )
        ops = _parse_hypothesis_ops(ns)
        assert len(ops) == 1
        assert ops[0]["op"] == "add"
        assert ops[0]["type"] == "config_change"
        assert ops[0]["description"] == "Test add"

    def test_parse_edit_flag(self):
        from ahvs.cli import _parse_hypothesis_ops
        import argparse
        ns = argparse.Namespace(
            add_hypotheses=[],
            edit_hypotheses=['H2:{"description":"edited"}'],
            insert_hypotheses=[],
        )
        ops = _parse_hypothesis_ops(ns)
        assert len(ops) == 1
        assert ops[0]["op"] == "edit"
        assert ops[0]["id"] == "H2"
        assert ops[0]["fields"]["description"] == "edited"

    def test_parse_insert_flag(self):
        from ahvs.cli import _parse_hypothesis_ops
        import argparse
        ns = argparse.Namespace(
            add_hypotheses=[],
            edit_hypotheses=[],
            insert_hypotheses=['2:{"type":"code_change","description":"inserted"}'],
        )
        ops = _parse_hypothesis_ops(ns)
        assert len(ops) == 1
        assert ops[0]["op"] == "insert"
        assert ops[0]["position"] == 2
        assert ops[0]["description"] == "inserted"

    def test_parse_multiple_flags(self):
        from ahvs.cli import _parse_hypothesis_ops
        import argparse
        ns = argparse.Namespace(
            add_hypotheses=['{"description":"a1"}', '{"description":"a2"}'],
            edit_hypotheses=['H1:{"rationale":"new"}'],
            insert_hypotheses=[],
        )
        ops = _parse_hypothesis_ops(ns)
        assert len(ops) == 3
        assert ops[0]["op"] == "add"
        assert ops[1]["op"] == "add"
        assert ops[2]["op"] == "edit"

    def test_parse_empty(self):
        from ahvs.cli import _parse_hypothesis_ops
        import argparse
        ns = argparse.Namespace(
            add_hypotheses=[],
            edit_hypotheses=[],
            insert_hypotheses=[],
        )
        ops = _parse_hypothesis_ops(ns)
        assert ops == []
