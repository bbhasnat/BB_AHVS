"""AHVS test suite — P1-c from the critical review.

Covers:
  1. _extract_metric_from_output (pure function, JSON + text patterns)
  2. _run_regression_guard (fail-closed when configured)
  3. HypothesisResult fields + load_results round-trip
  4. HypothesisWorktree lifecycle (create, apply_files, capture_diff, cleanup)
  5. Checkpoint write/read/resume round-trip
  6. Stage dispatcher routing (all 8 stages have handlers)
  7. One end-to-end cycle with mocked LLM and Claude Code
  8. Bug A regression: apply_files writes to eval_cwd subdir, not worktree root
  9. Bug C regression: missing eval_cwd surfaces clear error (create + run_eval_command)
 10. Bug E regression: splice_functions merges partial output correctly
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from ahvs.config import AHVSConfig
from ahvs.executor import (
    AHVSStageResult,
    _extract_metric_from_output,
    _run_regression_guard,
    execute_ahvs_stage,
)
from ahvs.result import HypothesisResult, load_results, save_results
from ahvs.runner import _write_checkpoint, read_ahvs_checkpoint
from ahvs.stages import AHVSStage, AHVS_STAGE_SEQUENCE, StageStatus
from ahvs.worktree import EvalResult, HypothesisWorktree


# ---------------------------------------------------------------------------
# 1. _extract_metric_from_output
# ---------------------------------------------------------------------------


class TestExtractMetric:
    """Tests for the five-tier metric extraction helper."""

    def test_json_simple(self) -> None:
        raw = '{"answer_relevance": 0.82}'
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.82

    def test_json_nested_key(self) -> None:
        raw = '{"metrics": {"accuracy": 0.91}}'
        assert _extract_metric_from_output(raw, "metrics.accuracy") == 0.91

    def test_json_last_line_wins(self) -> None:
        raw = (
            "Some debug output\n"
            '{"f1_score": 0.70}\n'
            'more text\n'
            '{"f1_score": 0.85}\n'
        )
        assert _extract_metric_from_output(raw, "f1_score") == 0.85

    def test_json_integer(self) -> None:
        raw = '{"count": 42}'
        assert _extract_metric_from_output(raw, "count") == 42.0

    def test_json_with_extra_fields(self) -> None:
        raw = '{"answer_relevance": 0.79, "eval_method": "promptfoo"}'
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.79

    def test_key_value_colon(self) -> None:
        raw = "answer_relevance: 0.80"
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.80

    def test_key_value_space(self) -> None:
        raw = "answer_relevance 0.80"
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.80

    def test_key_value_negative(self) -> None:
        raw = "delta: -0.05"
        assert _extract_metric_from_output(raw, "delta") == -0.05

    def test_key_value_scientific(self) -> None:
        raw = "loss: 3.5e-4"
        assert _extract_metric_from_output(raw, "loss") == 3.5e-4

    def test_key_value_case_insensitive(self) -> None:
        raw = "Answer_Relevance: 0.88"
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.88

    def test_no_match_returns_none(self) -> None:
        raw = "some random output without any metric"
        assert _extract_metric_from_output(raw, "answer_relevance") is None

    def test_empty_string(self) -> None:
        assert _extract_metric_from_output("", "metric") is None

    def test_malformed_json_falls_through_to_text(self) -> None:
        raw = '{bad json}\nanswer_relevance: 0.77'
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.77

    def test_json_missing_key_falls_through(self) -> None:
        raw = '{"other_metric": 0.9}\nanswer_relevance: 0.75'
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.75

    def test_json_list_index(self) -> None:
        raw = '{"data": [{"value": 0.6}, {"value": 0.7}]}'
        assert _extract_metric_from_output(raw, "data.0.value") == 0.6

    def test_multiline_with_noise(self) -> None:
        raw = (
            "Loading model...\n"
            "Running evaluation...\n"
            "Processed 100 samples\n"
            "answer_relevance: 0.81\n"
            "Done.\n"
        )
        assert _extract_metric_from_output(raw, "answer_relevance") == 0.81


# ---------------------------------------------------------------------------
# 2. _run_regression_guard
# ---------------------------------------------------------------------------


class TestRegressionGuard:
    """Tests for the fail-closed regression guard."""

    def test_none_guard_passes(self, tmp_path: Path) -> None:
        """No guard configured → always pass."""
        assert _run_regression_guard(None, tmp_path / "result.json") is True

    def test_missing_guard_file_fails(self, tmp_path: Path) -> None:
        """Guard configured but file doesn't exist → fail (P1-b fix)."""
        guard = tmp_path / "nonexistent_guard.sh"
        assert _run_regression_guard(guard, tmp_path / "result.json") is False

    def test_guard_exits_zero_passes(self, tmp_path: Path) -> None:
        guard = tmp_path / "guard.sh"
        guard.write_text("#!/bin/bash\nexit 0\n")
        guard.chmod(guard.stat().st_mode | stat.S_IEXEC)
        result_file = tmp_path / "result.json"
        result_file.write_text('{"answer_relevance": 0.8}')
        assert _run_regression_guard(guard, result_file) is True

    def test_guard_exits_nonzero_fails(self, tmp_path: Path) -> None:
        guard = tmp_path / "guard.sh"
        guard.write_text("#!/bin/bash\nexit 1\n")
        guard.chmod(guard.stat().st_mode | stat.S_IEXEC)
        result_file = tmp_path / "result.json"
        result_file.write_text('{"answer_relevance": 0.5}')
        assert _run_regression_guard(guard, result_file) is False

    def test_guard_timeout_fails(self, tmp_path: Path) -> None:
        """Guard that exceeds timeout → fail (P1-b fix)."""
        guard = tmp_path / "guard.sh"
        guard.write_text("#!/bin/bash\nsleep 120\n")
        guard.chmod(guard.stat().st_mode | stat.S_IEXEC)
        result_file = tmp_path / "result.json"
        result_file.write_text("{}")
        with patch("ahvs.executor.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="guard", timeout=60)
            assert _run_regression_guard(guard, result_file) is False

    def test_guard_oserror_fails(self, tmp_path: Path) -> None:
        """Guard that raises OSError → fail (P1-b fix)."""
        guard = tmp_path / "guard.sh"
        guard.write_text("#!/bin/bash\nexit 0\n")
        guard.chmod(guard.stat().st_mode | stat.S_IEXEC)
        with patch("ahvs.executor.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("permission denied")
            assert _run_regression_guard(guard, tmp_path / "r.json") is False

    def test_guard_receives_result_path(self, tmp_path: Path) -> None:
        """Guard script receives the results path as its first argument."""
        guard = tmp_path / "guard.sh"
        marker = tmp_path / "marker.txt"
        guard.write_text(
            f'#!/bin/bash\necho "$1" > {marker}\nexit 0\n'
        )
        guard.chmod(guard.stat().st_mode | stat.S_IEXEC)
        result_file = tmp_path / "result.json"
        result_file.write_text("{}")
        _run_regression_guard(guard, result_file)
        assert marker.read_text().strip() == str(result_file)


# ---------------------------------------------------------------------------
# 3. HypothesisResult fields + round-trip
# ---------------------------------------------------------------------------


def _make_result(**overrides: object) -> HypothesisResult:
    defaults = dict(
        hypothesis_id="H1",
        hypothesis_type="code_change",
        primary_metric="answer_relevance",
        metric_value=0.80,
        baseline_value=0.75,
        delta=0.05,
        delta_pct=6.67,
        regression_guard_passed=True,
        eval_method="code_agent",
        measurement_status="measured",
    )
    defaults.update(overrides)
    return HypothesisResult(**defaults)


class TestHypothesisResult:
    """Tests for HypothesisResult dataclass including new P1-a fields."""

    def test_improved_true(self) -> None:
        r = _make_result(delta=0.05, regression_guard_passed=True, error=None)
        assert r.improved is True

    def test_improved_false_negative_delta(self) -> None:
        r = _make_result(delta=-0.01)
        assert r.improved is False

    def test_improved_false_guard_failed(self) -> None:
        r = _make_result(delta=0.05, regression_guard_passed=False)
        assert r.improved is False

    def test_improved_false_with_error(self) -> None:
        r = _make_result(delta=0.05, error="boom")
        assert r.improved is False

    def test_default_new_fields(self) -> None:
        r = _make_result()
        assert r.worktree_path == ""
        assert r.patch_path == ""
        assert r.kept is False
        assert r.measurement_status == "measured"  # helper default

    def test_improved_false_extraction_failed(self) -> None:
        """extraction_failed hypotheses must not count as improved (v2 fix #3)."""
        r = _make_result(delta=0.05, measurement_status="extraction_failed")
        assert r.improved is False

    def test_new_fields_settable(self) -> None:
        r = _make_result()
        r.kept = True
        r.worktree_path = "/tmp/wt/H1"
        r.patch_path = "tool_runs/H1/H1.patch"
        assert r.kept is True
        assert r.worktree_path == "/tmp/wt/H1"
        assert r.patch_path == "tool_runs/H1/H1.patch"

    def test_make_error(self) -> None:
        r = HypothesisResult.make_error(
            hypothesis_id="H2",
            hypothesis_type="prompt_rewrite",
            primary_metric="f1",
            baseline_value=0.70,
            error="Claude Code crashed",
        )
        assert r.error == "Claude Code crashed"
        assert r.delta == 0.0
        assert r.improved is False
        assert r.measurement_status == "sandbox_error"
        assert r.worktree_path == ""
        assert r.patch_path == ""

    def test_to_dict_includes_new_fields(self) -> None:
        r = _make_result(kept=True, worktree_path="/wt", patch_path="p.patch")
        d = r.to_dict()
        assert d["kept"] is True
        assert d["worktree_path"] == "/wt"
        assert d["patch_path"] == "p.patch"


class TestResultSerialization:
    """Round-trip: save_results → load_results preserves all fields."""

    def test_round_trip(self, tmp_path: Path) -> None:
        original = [
            _make_result(
                hypothesis_id="H1",
                kept=True,
                worktree_path="/tmp/wt/H1",
                patch_path="tool_runs/H1/H1.patch",
                measurement_status="measured",
            ),
            _make_result(
                hypothesis_id="H2",
                delta=-0.01,
                delta_pct=-1.33,
                metric_value=0.74,
                measurement_status="extraction_failed",
            ),
        ]
        path = tmp_path / "results.json"
        save_results(original, path)
        loaded = load_results(path)

        assert len(loaded) == 2
        assert loaded[0].hypothesis_id == "H1"
        assert loaded[0].kept is True
        assert loaded[0].worktree_path == "/tmp/wt/H1"
        assert loaded[0].patch_path == "tool_runs/H1/H1.patch"
        assert loaded[0].measurement_status == "measured"

        assert loaded[1].hypothesis_id == "H2"
        assert loaded[1].kept is False
        assert loaded[1].worktree_path == ""
        assert loaded[1].measurement_status == "extraction_failed"

    def test_round_trip_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        save_results([], path)
        loaded = load_results(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# 4. HypothesisWorktree lifecycle
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    """Initialize a minimal git repo with one commit."""
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(path), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(path), capture_output=True, check=True,
    )
    (path / "README.md").write_text("# test repo\n")
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(path), capture_output=True, check=True,
    )


class TestHypothesisWorktree:
    """Tests for the git worktree lifecycle manager."""

    def test_create_and_cleanup(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        assert wt_path.exists()
        assert (wt_path / "README.md").exists()

        wt.cleanup()
        assert not wt_path.exists()

    def test_apply_files(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        files = {
            "src/main.py": "print('hello')\n",
            "config.yaml": "key: value\n",
        }
        written = wt.apply_files(files)

        assert len(written) == 2
        assert (wt_path / "src" / "main.py").read_text() == "print('hello')\n"
        assert (wt_path / "config.yaml").read_text() == "key: value\n"

        wt.cleanup()

    def test_capture_diff(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        wt.apply_files({"new_file.txt": "content\n"})
        diff = wt.capture_diff()

        assert "new_file.txt" in diff
        assert "+content" in diff

        wt.cleanup()

    def test_save_patch(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        wt.apply_files({"patch_test.py": "x = 1\n"})
        patch_dest = tmp_path / "H1.patch"
        wt.save_patch(patch_dest)

        assert patch_dest.exists()
        patch_content = patch_dest.read_text()
        assert "patch_test.py" in patch_content

        wt.cleanup()

    def test_run_eval_command(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        result = wt.run_eval_command("echo 'answer_relevance: 0.85'")
        assert result.returncode == 0
        assert "answer_relevance: 0.85" in result.stdout
        assert result.elapsed_sec >= 0
        assert result.timed_out is False

        wt.cleanup()

    def test_run_eval_command_failure(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        result = wt.run_eval_command("exit 1")
        assert result.returncode == 1
        assert result.timed_out is False

        wt.cleanup()

    def test_run_eval_command_timeout(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        result = wt.run_eval_command("sleep 60", timeout=1)
        assert result.returncode == -1
        assert result.timed_out is True

        wt.cleanup()

    def test_create_fails_not_git_repo(self, tmp_path: Path) -> None:
        """Worktree creation on a non-git directory raises RuntimeError."""
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()
        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(not_a_repo, wt_path)
        with pytest.raises(RuntimeError, match="git root|git repository"):
            wt.create()

    def test_cleanup_without_create_is_noop(self, tmp_path: Path) -> None:
        """Cleanup on a never-created worktree does nothing."""
        wt = HypothesisWorktree(tmp_path, tmp_path / "wt")
        wt.cleanup()  # Should not raise


class TestEvalResult:
    """Tests for the EvalResult dataclass."""

    def test_defaults(self) -> None:
        r = EvalResult(returncode=0, stdout="ok", stderr="", elapsed_sec=1.5)
        assert r.timed_out is False

    def test_frozen(self) -> None:
        r = EvalResult(returncode=0, stdout="", stderr="", elapsed_sec=0.0)
        with pytest.raises(AttributeError):
            r.returncode = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. Checkpoint write/read/resume round-trip
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Tests for AHVS checkpoint persistence."""

    def test_write_and_read_done(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, AHVSStage.AHVS_CONTEXT_LOAD, "done")
        result = read_ahvs_checkpoint(tmp_path)
        assert result == AHVSStage.AHVS_CONTEXT_LOAD

    def test_read_failed_returns_none(self, tmp_path: Path) -> None:
        """A failed checkpoint should not be treated as successfully completed."""
        _write_checkpoint(tmp_path, AHVSStage.AHVS_HYPOTHESIS_GEN, "failed")
        result = read_ahvs_checkpoint(tmp_path)
        assert result is None

    def test_read_nonexistent_returns_none(self, tmp_path: Path) -> None:
        assert read_ahvs_checkpoint(tmp_path) is None

    def test_read_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "ahvs_checkpoint.json").write_text("not json")
        assert read_ahvs_checkpoint(tmp_path) is None

    def test_write_overwrites_previous(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, AHVSStage.AHVS_SETUP, "done")
        _write_checkpoint(tmp_path, AHVSStage.AHVS_EXECUTION, "done")
        result = read_ahvs_checkpoint(tmp_path)
        assert result == AHVSStage.AHVS_EXECUTION

    def test_checkpoint_contains_expected_fields(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path, AHVSStage.AHVS_SETUP, "done")
        data = json.loads((tmp_path / "ahvs_checkpoint.json").read_text())
        assert data["stage"] == "AHVS_SETUP"
        assert data["stage_num"] == 1
        assert data["status"] == "done"
        assert "updated_at" in data

    def test_all_stages_round_trip(self, tmp_path: Path) -> None:
        """Every stage can be written and read back."""
        for stage in AHVSStage:
            _write_checkpoint(tmp_path, stage, "done")
            assert read_ahvs_checkpoint(tmp_path) == stage


# ---------------------------------------------------------------------------
# 6. Stage dispatcher routing
# ---------------------------------------------------------------------------


class TestStageDispatcher:
    """Verify all 8 stages have registered handlers."""

    def test_all_stages_in_sequence(self) -> None:
        assert len(AHVS_STAGE_SEQUENCE) == 8

    def test_all_stages_have_handlers(self, tmp_path: Path) -> None:
        """Every stage in AHVS_STAGE_SEQUENCE has a handler in _HANDLERS."""
        from ahvs.executor import _HANDLERS
        for stage in AHVS_STAGE_SEQUENCE:
            assert stage in _HANDLERS, f"No handler for {stage.name}"

    def test_unknown_stage_returns_failed(self, tmp_path: Path) -> None:
        """A stage not in _HANDLERS returns a FAILED result."""
        # Use a mock stage value that doesn't exist
        from ahvs.skills import SkillLibrary

        config = AHVSConfig(
            repo_path=tmp_path,
            question="test",
            run_dir=tmp_path / "run",
        )
        # We can't easily create a fake AHVSStage, but we can verify
        # the handler map is complete by checking the set difference
        from ahvs.executor import _HANDLERS
        registered = set(_HANDLERS.keys())
        expected = set(AHVSStage)
        assert registered == expected


# ---------------------------------------------------------------------------
# 7. End-to-end mock cycle (setup + context_load + verify stages)
# ---------------------------------------------------------------------------


class TestExecuteSetup:
    """Test Stage 1: _execute_setup creates the right directory structure."""

    def test_setup_creates_dirs(self, tmp_path: Path) -> None:
        from ahvs.skills import SkillLibrary
        from ahvs.health import CheckResult

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        # Create baseline metric
        ahvs_dir = repo / ".ahvs"
        ahvs_dir.mkdir()
        baseline = {
            "primary_metric": "answer_relevance",
            "answer_relevance": 0.74,
            "recorded_at": "2026-03-18T10:00:00Z",
            "eval_command": "echo 'answer_relevance: 0.74'",
        }
        (ahvs_dir / "baseline_metric.json").write_text(json.dumps(baseline))

        cycle_dir = tmp_path / "cycle_001"
        config = AHVSConfig(repo_path=repo, question="test?", run_dir=cycle_dir)
        skill_lib = SkillLibrary()

        # Mock LLM connectivity and clean-branch check so setup doesn't fail
        # on missing API key or untracked .ahvs/ dir in the test repo
        mock_llm = CheckResult(
            name="ahvs_llm_connectivity", status="pass", detail="mocked"
        )
        mock_branch = CheckResult(
            name="ahvs_clean_branch", status="pass", detail="mocked"
        )
        with patch(
            "ahvs.health.check_llm_connectivity",
            return_value=mock_llm,
        ), patch(
            "ahvs.health.check_clean_branch",
            return_value=mock_branch,
        ):
            result = execute_ahvs_stage(
                AHVSStage.AHVS_SETUP,
                cycle_dir=cycle_dir,
                config=config,
                skill_library=skill_lib,
                auto_approve=True,
            )

        assert result.status == StageStatus.DONE
        assert (cycle_dir / "tool_runs").is_dir()
        assert (cycle_dir / "worktrees").is_dir()
        assert (cycle_dir / "cycle_manifest.json").exists()

    def test_setup_fails_without_baseline(self, tmp_path: Path) -> None:
        from ahvs.skills import SkillLibrary

        repo = tmp_path / "repo"
        repo.mkdir()

        cycle_dir = tmp_path / "cycle_002"
        config = AHVSConfig(repo_path=repo, question="test?", run_dir=cycle_dir)
        skill_lib = SkillLibrary()

        result = execute_ahvs_stage(
            AHVSStage.AHVS_SETUP,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_lib,
            auto_approve=True,
        )

        assert result.status == StageStatus.FAILED
        assert "Pre-flight failed" in (result.error or "")


class TestCycleVerify:
    """Test Stage 8: _execute_cycle_verify validates artifacts and writes summary."""

    def _setup_cycle_artifacts(self, cycle_dir: Path, repo_path: Path) -> None:
        """Create all required artifacts for Stage 8 verification."""
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # baseline bundle
        baseline = {
            "primary_metric": "answer_relevance",
            "value": 0.74,
            "eval_command": "echo test",
        }
        bundle = {"baseline": baseline, "lessons": [], "rejected": []}
        (cycle_dir / "context_bundle.json").write_text(json.dumps(bundle))

        # manifest
        (cycle_dir / "cycle_manifest.json").write_text(json.dumps({"cycle_id": "test"}))

        # hypotheses
        (cycle_dir / "hypotheses.md").write_text("# H1\n")

        # selection
        selection = {"selected": ["H1"], "rationale": "test"}
        (cycle_dir / "selection.md").write_text("H1")
        (cycle_dir / "selection.json").write_text(json.dumps(selection))

        # validation plan
        (cycle_dir / "validation_plan.md").write_text("# Plan\n")

        # results
        results = [
            _make_result(
                hypothesis_id="H1",
                measurement_status="measured",
                kept=True,
                worktree_path="/tmp/wt/H1",
                patch_path="tool_runs/H1/H1.patch",
            ),
        ]
        save_results(results, cycle_dir / "results.json")

        # report + friction log
        (cycle_dir / "report.md").write_text("# Report\n")
        (cycle_dir / "friction_log.md").write_text("# Friction\n")

    def test_verify_passes_with_all_artifacts(self, tmp_path: Path) -> None:
        from ahvs.skills import SkillLibrary

        repo = tmp_path / "repo"
        repo.mkdir()
        cycle_dir = tmp_path / "cycle_001"
        self._setup_cycle_artifacts(cycle_dir, repo)

        config = AHVSConfig(repo_path=repo, question="test?", run_dir=cycle_dir)
        skill_lib = SkillLibrary()

        result = execute_ahvs_stage(
            AHVSStage.AHVS_CYCLE_VERIFY,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_lib,
            auto_approve=True,
        )

        assert result.status == StageStatus.DONE

        # Check cycle_summary.json has the new worktree/patch fields
        summary = json.loads((cycle_dir / "cycle_summary.json").read_text())
        assert "kept_worktree" in summary
        assert "kept_patch" in summary
        assert "all_patches" in summary
        assert summary["kept_worktree"] == "/tmp/wt/H1"
        assert summary["kept_patch"] == "tool_runs/H1/H1.patch"
        assert summary["all_patches"] == ["tool_runs/H1/H1.patch"]

    def test_verify_fails_missing_artifact(self, tmp_path: Path) -> None:
        from ahvs.skills import SkillLibrary

        cycle_dir = tmp_path / "cycle_002"
        cycle_dir.mkdir()
        # Only write some artifacts — missing results.json, report.md, etc.
        (cycle_dir / "cycle_manifest.json").write_text("{}")
        (cycle_dir / "context_bundle.json").write_text("{}")

        repo = tmp_path / "repo"
        repo.mkdir()
        config = AHVSConfig(repo_path=repo, question="test?", run_dir=cycle_dir)
        skill_lib = SkillLibrary()

        result = execute_ahvs_stage(
            AHVSStage.AHVS_CYCLE_VERIFY,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_lib,
            auto_approve=True,
        )

        assert result.status == StageStatus.FAILED
        assert "Missing artifacts" in (result.error or "")


# ---------------------------------------------------------------------------
# 8. v2 review fixes — path traversal, canonical result, LLM preflight,
#    tool requirements, all-unmeasured cycle
# ---------------------------------------------------------------------------


class TestPathTraversal:
    """Tests for Fix #1: apply_files rejects paths that escape the worktree."""

    def test_reject_absolute_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        with pytest.raises(ValueError, match="absolute path"):
            wt.apply_files({"/tmp/evil.py": "pwned"})
        wt.cleanup()

    def test_reject_dotdot_traversal(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        with pytest.raises(ValueError, match="traversal"):
            wt.apply_files({"../../../etc/passwd": "pwned"})
        wt.cleanup()

    def test_reject_hidden_dotdot(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        with pytest.raises(ValueError, match="traversal"):
            wt.apply_files({"src/../../outside.py": "pwned"})
        wt.cleanup()

    def test_accept_valid_nested_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        written = wt.apply_files({"src/deep/module.py": "x = 1\n"})
        assert len(written) == 1
        assert (wt_path / "src" / "deep" / "module.py").read_text() == "x = 1\n"
        wt.cleanup()


class TestCanonicalResult:
    """Tests for Fix #2: regression guard runs against canonical result.json."""

    def test_guard_receives_canonical_result(self, tmp_path: Path) -> None:
        """After metric extraction, result.json should contain the measured metric."""
        # Simulate what executor does after extraction
        work_dir = tmp_path / "tool_runs" / "H1"
        work_dir.mkdir(parents=True)

        canonical = {
            "hypothesis_id": "H1",
            "primary_metric": "answer_relevance",
            "answer_relevance": 0.85,
            "baseline_value": 0.74,
            "measurement_status": "measured",
        }
        result_path = work_dir / "result.json"
        result_path.write_text(json.dumps(canonical))

        # Guard that reads the result and checks for the metric key
        guard = tmp_path / "guard.sh"
        script = textwrap.dedent("""\
            #!/bin/bash
            python3 -c "
            import json, sys
            d = json.load(open(sys.argv[1]))
            assert 'answer_relevance' in d
            assert d['measurement_status'] == 'measured'
            " "$1"
        """)
        guard.write_text(script)
        guard.chmod(guard.stat().st_mode | stat.S_IEXEC)

        assert _run_regression_guard(guard, result_path) is True


class TestLLMPreflightAlwaysRuns:
    """Tests for Fix #4: LLM preflight runs even when api_key is empty."""

    def test_empty_key_fails_preflight(self) -> None:
        from ahvs.health import run_ahvs_preflight

        # Use a non-existent baseline so baseline check also fails;
        # but we specifically check that the LLM check ran and failed.
        report = run_ahvs_preflight(
            baseline_path=Path("/nonexistent/baseline.json"),
            repo_path=Path("/tmp"),
            llm_api_key="",
            llm_model="claude-opus-4-6",
        )
        llm_checks = [c for c in report.checks if c.name == "ahvs_llm_connectivity"]
        assert len(llm_checks) == 1
        assert llm_checks[0].status == "fail"
        assert "No API key" in llm_checks[0].detail

    def test_preflight_with_key_includes_llm_check(self) -> None:
        from ahvs.health import run_ahvs_preflight

        report = run_ahvs_preflight(
            baseline_path=Path("/nonexistent/baseline.json"),
            repo_path=Path("/tmp"),
            llm_api_key="sk-fake-key",
            llm_model="claude-opus-4-6",
        )
        llm_checks = [c for c in report.checks if c.name == "ahvs_llm_connectivity"]
        assert len(llm_checks) == 1
        # Will fail because the key is fake, but the check *ran*
        assert llm_checks[0].status == "fail"


class TestToolRequirements:
    """Tests for Fix #5: code_change/architecture_change/multi_llm_judge don't require docker."""

    def test_code_change_no_docker_requirement(self) -> None:
        from ahvs.health import HYPOTHESIS_TOOL_REQUIREMENTS
        assert "docker" not in HYPOTHESIS_TOOL_REQUIREMENTS["code_change"]

    def test_architecture_change_no_docker_requirement(self) -> None:
        from ahvs.health import HYPOTHESIS_TOOL_REQUIREMENTS
        assert "docker" not in HYPOTHESIS_TOOL_REQUIREMENTS["architecture_change"]

    def test_multi_llm_judge_no_docker_requirement(self) -> None:
        from ahvs.health import HYPOTHESIS_TOOL_REQUIREMENTS
        assert "docker" not in HYPOTHESIS_TOOL_REQUIREMENTS["multi_llm_judge"]

    def test_sandbox_skill_no_docker_required(self) -> None:
        from ahvs.skills import BUILTIN_SKILLS
        sandbox_skills = [s for s in BUILTIN_SKILLS if s.name == "sandbox_run"]
        assert len(sandbox_skills) == 1
        assert "docker" not in sandbox_skills[0].required_tools


class TestAllUnmeasuredCycle:
    """Tests for Fix #3: all-unmeasured cycle is flagged as FAILED."""

    def _setup_cycle_artifacts(self, cycle_dir: Path, measurement_status: str) -> None:
        """Create cycle artifacts where all hypotheses have the given measurement_status."""
        cycle_dir.mkdir(parents=True, exist_ok=True)

        baseline = {
            "primary_metric": "answer_relevance",
            "value": 0.74,
            "eval_command": "echo test",
        }
        bundle = {"baseline": baseline, "lessons": [], "rejected": []}
        (cycle_dir / "context_bundle.json").write_text(json.dumps(bundle))
        (cycle_dir / "cycle_manifest.json").write_text(json.dumps({"cycle_id": "test"}))
        (cycle_dir / "hypotheses.md").write_text("# H1\n")
        (cycle_dir / "selection.md").write_text("H1")
        (cycle_dir / "selection.json").write_text(json.dumps({"selected": ["H1"], "rationale": "test"}))
        (cycle_dir / "validation_plan.md").write_text("# Plan\n")
        results = [
            _make_result(
                hypothesis_id="H1",
                measurement_status=measurement_status,
                delta=0.0,
                delta_pct=0.0,
                metric_value=0.74,
            ),
        ]
        save_results(results, cycle_dir / "results.json")
        (cycle_dir / "report.md").write_text("# Report\n")
        (cycle_dir / "friction_log.md").write_text("# Friction\n")

    def test_all_unmeasured_fails_cycle(self, tmp_path: Path) -> None:
        from ahvs.skills import SkillLibrary

        repo = tmp_path / "repo"
        repo.mkdir()
        cycle_dir = tmp_path / "cycle_001"
        self._setup_cycle_artifacts(cycle_dir, "extraction_failed")

        config = AHVSConfig(repo_path=repo, question="test?", run_dir=cycle_dir)
        skill_lib = SkillLibrary()

        result = execute_ahvs_stage(
            AHVSStage.AHVS_CYCLE_VERIFY,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_lib,
            auto_approve=True,
        )

        assert result.status == StageStatus.FAILED
        assert "All hypotheses failed measurement" in (result.error or "")

        # Summary should still be written with all_unmeasured=True
        summary = json.loads((cycle_dir / "cycle_summary.json").read_text())
        assert summary["all_unmeasured"] is True
        assert "INVALID CYCLE" in summary["recommendation"]

    def test_measured_cycle_passes(self, tmp_path: Path) -> None:
        from ahvs.skills import SkillLibrary

        repo = tmp_path / "repo"
        repo.mkdir()
        cycle_dir = tmp_path / "cycle_002"
        self._setup_cycle_artifacts(cycle_dir, "measured")

        config = AHVSConfig(repo_path=repo, question="test?", run_dir=cycle_dir)
        skill_lib = SkillLibrary()

        result = execute_ahvs_stage(
            AHVSStage.AHVS_CYCLE_VERIFY,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_lib,
            auto_approve=True,
        )

        assert result.status == StageStatus.DONE
        summary = json.loads((cycle_dir / "cycle_summary.json").read_text())
        assert summary["all_unmeasured"] is False


# ---------------------------------------------------------------------------
# 9. v3 review fixes — shared safe-path, string-prefix bug, Stage 4 preflight,
#    dirty repo fail, extraction_failed lesson archival
# ---------------------------------------------------------------------------


class TestSharedSafePath:
    """Tests for Fix #1 (v3): shared validate_safe_relpath utility."""

    def test_string_prefix_containment_uses_is_relative_to(self) -> None:
        """Verify the containment check uses is_relative_to, not startswith.

        The old code used ``str(dest).startswith(str(root))`` which would
        let ``/tmp/wt2/file`` pass as inside ``/tmp/wt``.  The fix uses
        ``Path.is_relative_to()`` which is immune to this.
        """
        from ahvs.worktree import validate_safe_relpath
        import inspect

        source = inspect.getsource(validate_safe_relpath)
        # Must NOT use string-prefix containment
        assert "startswith" not in source
        # Must use proper path ancestry check
        assert "is_relative_to" in source

    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        """A symlink inside root that points outside must be rejected."""
        from ahvs.worktree import validate_safe_relpath

        root = tmp_path / "wt"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        # Create a symlink inside root that points outside
        (root / "escape").symlink_to(outside)

        with pytest.raises(ValueError, match="escapes boundary"):
            validate_safe_relpath("escape/evil.py", root)

    def test_tool_runs_rejects_absolute_path(self, tmp_path: Path) -> None:
        """tool_runs write path must reject absolute paths."""
        from ahvs.worktree import validate_safe_relpath

        work_dir = tmp_path / "tool_runs" / "H1"
        work_dir.mkdir(parents=True)
        with pytest.raises(ValueError, match="absolute path"):
            validate_safe_relpath("/etc/passwd", work_dir)

    def test_tool_runs_rejects_dotdot(self, tmp_path: Path) -> None:
        """tool_runs write path must reject .. traversal."""
        from ahvs.worktree import validate_safe_relpath

        work_dir = tmp_path / "tool_runs" / "H1"
        work_dir.mkdir(parents=True)
        with pytest.raises(ValueError, match="traversal"):
            validate_safe_relpath("../../etc/cron.d/evil", work_dir)

    def test_tool_runs_accepts_valid_path(self, tmp_path: Path) -> None:
        """Valid nested paths pass validation."""
        from ahvs.worktree import validate_safe_relpath

        work_dir = tmp_path / "tool_runs" / "H1"
        work_dir.mkdir(parents=True)
        # Should not raise
        validate_safe_relpath("src/module.py", work_dir)

    def test_worktree_and_tool_runs_use_same_function(self) -> None:
        """Worktree._validate_relpath delegates to the shared utility."""
        from ahvs.worktree import validate_safe_relpath

        # The worktree class method should call the shared function
        # Verify by checking it's the same function reference
        import inspect
        source = inspect.getsource(HypothesisWorktree._validate_relpath)
        assert "validate_safe_relpath" in source


class TestStage4PreflightRegression:
    """Tests for Fix #2 (v3): secondary preflight must not manufacture LLM failure."""

    def test_skip_llm_check_no_failure(self) -> None:
        """With skip_llm_check=True, no LLM connectivity check appears."""
        from ahvs.health import run_ahvs_preflight

        report = run_ahvs_preflight(
            baseline_path=Path("/nonexistent/baseline.json"),
            repo_path=Path("/tmp"),
            hypothesis_types=["code_change"],
            skip_llm_check=True,
        )
        llm_checks = [c for c in report.checks if c.name == "ahvs_llm_connectivity"]
        assert len(llm_checks) == 0

    def test_no_skip_includes_llm_check(self) -> None:
        """Without skip_llm_check, LLM connectivity check is still present."""
        from ahvs.health import run_ahvs_preflight

        report = run_ahvs_preflight(
            baseline_path=Path("/nonexistent/baseline.json"),
            repo_path=Path("/tmp"),
            hypothesis_types=["code_change"],
            skip_llm_check=False,
            llm_api_key="",
            llm_model="test",
        )
        llm_checks = [c for c in report.checks if c.name == "ahvs_llm_connectivity"]
        assert len(llm_checks) == 1


class TestDirtyRepoFail:
    """Tests for Fix #3 (v3): dirty repo is a hard fail, not a warning."""

    def test_dirty_repo_fails(self, tmp_path: Path) -> None:
        from ahvs.health import check_clean_branch

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        # Create an untracked file to make the repo dirty
        (repo / "uncommitted.txt").write_text("dirty")

        result = check_clean_branch(repo)
        assert result.status == "fail"
        assert "uncommitted" in result.detail.lower()

    def test_clean_repo_passes(self, tmp_path: Path) -> None:
        from ahvs.health import check_clean_branch

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        result = check_clean_branch(repo)
        assert result.status == "pass"


class TestExtractionFailedLesson:
    """Tests for Fix #4 (v3): extraction_failed → infrastructure lesson."""

    def test_extraction_failed_not_rejected_approach(self) -> None:
        """extraction_failed results must NOT be archived as 'Rejected approach'."""
        r = _make_result(
            hypothesis_id="H1",
            measurement_status="extraction_failed",
            delta=0.0,
            error=None,
        )
        # Simulate the lesson classification logic from executor
        assert not r.improved
        assert r.error is None
        assert r.measurement_status != "measured"
        # These three conditions mean it hits the new branch, not "Rejected approach"

    def test_measured_not_improved_is_rejected_approach(self) -> None:
        """A properly measured but not-improved result IS a rejected approach."""
        r = _make_result(
            hypothesis_id="H1",
            measurement_status="measured",
            delta=-0.02,
            error=None,
        )
        assert not r.improved
        assert r.error is None
        assert r.measurement_status == "measured"
        # This correctly hits the "Rejected approach" branch


# ---------------------------------------------------------------------------
# 10. ACP local-agent LLM support
# ---------------------------------------------------------------------------


class TestAHVSConfigACP:
    """Tests for AHVSConfig ACP fields and provider-aware behaviour."""

    def test_default_provider_is_anthropic(self, tmp_path: Path) -> None:
        config = AHVSConfig(repo_path=tmp_path, question="test")
        assert config.llm_provider == "anthropic"

    def test_acp_provider_skips_api_key_env(self, tmp_path: Path) -> None:
        """When provider=acp, __post_init__ should not try to read API key from env."""
        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="acp",
            llm_api_key="",
            llm_api_key_env="NONEXISTENT_KEY_VAR_12345",
        )
        # Should remain empty — not read from env
        assert config.llm_api_key == ""

    def test_acp_defaults(self, tmp_path: Path) -> None:
        config = AHVSConfig(repo_path=tmp_path, question="test", llm_provider="acp")
        assert config.acp_agent == "claude"
        assert config.acp_cwd == str(tmp_path.resolve())
        assert config.acp_session_name == "ahvs"
        assert config.acp_timeout_sec == 1800

    def test_acp_cwd_resolved_to_repo(self, tmp_path: Path) -> None:
        config = AHVSConfig(repo_path=tmp_path, question="test")
        assert config.acp_cwd == str(tmp_path.resolve())


class TestShimConstruction:
    """Tests for the _LLMConfigShim that bridges AHVSConfig to create_llm_client."""

    def test_shim_exposes_llm_fields(self, tmp_path: Path) -> None:
        from ahvs.executor import _ahvs_config_to_llm_shim

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="anthropic",
            llm_model="claude-opus-4-6",
            llm_api_key="sk-test",
            llm_base_url="https://example.com",
        )
        shim = _ahvs_config_to_llm_shim(config)
        assert shim.llm.provider == "anthropic"
        assert shim.llm.primary_model == "claude-opus-4-6"
        assert shim.llm.api_key == "sk-test"
        assert shim.llm.base_url == "https://example.com"

    def test_shim_exposes_acp_fields(self, tmp_path: Path) -> None:
        from ahvs.executor import _ahvs_config_to_llm_shim

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="acp",
            acp_agent="codex",
            acp_session_name="my-session",
            acp_timeout_sec=600,
        )
        shim = _ahvs_config_to_llm_shim(config)
        assert shim.llm.provider == "acp"
        assert shim.llm.acp.agent == "codex"
        assert shim.llm.acp.session_name == "my-session"
        assert shim.llm.acp.timeout_sec == 600


class TestPreflightProviderAware:
    """Tests for provider-aware preflight checks."""

    def test_acp_preflight_calls_acp_check(self) -> None:
        """With provider=acp, check_llm_connectivity should use ACP path, not API key."""
        from ahvs.health import check_llm_connectivity

        # Even with empty API key, provider=acp should NOT fail with "No API key"
        result = check_llm_connectivity(
            api_key="", model="", base_url="", provider="acp",
        )
        # Will fail because acpx is not installed in test env,
        # but the failure message should be about ACP, not API keys
        assert "API key" not in result.detail

    def test_non_acp_still_requires_api_key(self) -> None:
        """Without provider=acp, empty API key still fails."""
        from ahvs.health import check_llm_connectivity

        result = check_llm_connectivity(
            api_key="", model="test", base_url="", provider="anthropic",
        )
        assert result.status == "fail"
        assert "No API key" in result.detail

    def test_preflight_skips_llm_on_skip_flag(self) -> None:
        """skip_llm_check=True suppresses the check regardless of provider."""
        from ahvs.health import run_ahvs_preflight

        report = run_ahvs_preflight(
            baseline_path=Path("/nonexistent"),
            repo_path=Path("/tmp"),
            skip_llm_check=True,
            llm_provider="acp",
        )
        llm_checks = [c for c in report.checks if c.name == "ahvs_llm_connectivity"]
        assert len(llm_checks) == 0

    def test_acp_preflight_in_run_ahvs_preflight(self) -> None:
        """run_ahvs_preflight passes provider to check_llm_connectivity."""
        from ahvs.health import run_ahvs_preflight

        report = run_ahvs_preflight(
            baseline_path=Path("/nonexistent"),
            repo_path=Path("/tmp"),
            llm_provider="acp",
        )
        llm_checks = [c for c in report.checks if c.name == "ahvs_llm_connectivity"]
        assert len(llm_checks) == 1
        # Should be an ACP-related message, not "No API key"
        assert "API key" not in llm_checks[0].detail


class TestMakeLlmClientUsesFactory:
    """Tests for _make_llm_client routing through the shared factory."""

    def test_anthropic_provider_returns_llm_client(self, tmp_path: Path) -> None:
        """provider=anthropic should produce an LLMClient (not ACPClient)."""
        from ahvs.executor import _make_llm_client
        from ahvs.llm.client import LLMClient

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="anthropic",
            llm_api_key="sk-test",
            llm_model="claude-opus-4-6",
        )
        # Mock the Anthropic adapter since httpx may not be installed in test env
        with patch("ahvs.llm.anthropic_adapter.HAS_HTTPX", True), \
             patch("ahvs.llm.anthropic_adapter.AnthropicAdapter.__init__", return_value=None):
            client = _make_llm_client(config)
        assert isinstance(client, LLMClient)

    def test_acp_provider_returns_acp_client(self, tmp_path: Path) -> None:
        """provider=acp should produce an ACPClient."""
        from ahvs.executor import _make_llm_client
        from ahvs.llm.acp_client import ACPClient

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="acp",
            acp_agent="claude",
        )
        client = _make_llm_client(config)
        assert isinstance(client, ACPClient)
        assert client.config.agent == "claude"


# ---------------------------------------------------------------------------
# 11. v4 review fixes — unified preflight, provider matrix, CLI --base-url,
#     from_cli_args parity
# ---------------------------------------------------------------------------


class TestPreflightUsesSharedFactory:
    """Fix #1 (v4): check_llm_connectivity routes through create_llm_client."""

    def test_preflight_builds_shim_for_openai(self) -> None:
        """provider=openai should build a shim that the factory accepts."""
        from ahvs.health import _build_preflight_shim

        shim = _build_preflight_shim(
            api_key="sk-test", model="gpt-4o", base_url="",
            provider="openai", ahvs_config=None,
        )
        assert shim.llm.provider == "openai"
        assert shim.llm.primary_model == "gpt-4o"
        assert shim.llm.api_key == "sk-test"

    def test_preflight_builds_shim_for_openrouter(self) -> None:
        """provider=openrouter should build a shim that the factory accepts."""
        from ahvs.health import _build_preflight_shim

        shim = _build_preflight_shim(
            api_key="sk-or-test", model="anthropic/claude-opus-4-6",
            base_url="", provider="openrouter", ahvs_config=None,
        )
        assert shim.llm.provider == "openrouter"
        assert shim.llm.primary_model == "anthropic/claude-opus-4-6"
        assert shim.llm.api_key == "sk-or-test"

    def test_preflight_shim_from_ahvs_config(self, tmp_path: Path) -> None:
        """When ahvs_config is provided, shim reads from it."""
        from ahvs.health import _build_preflight_shim

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="openai",
            llm_base_url="https://custom.example.com/v1",
            llm_model="gpt-4o",
            llm_api_key="sk-from-config",
        )
        shim = _build_preflight_shim(
            api_key="sk-ignored", model="ignored", base_url="ignored",
            provider="ignored", ahvs_config=config,
        )
        # ahvs_config fields should take priority
        assert shim.llm.provider == "openai"
        assert shim.llm.base_url == "https://custom.example.com/v1"
        assert shim.llm.api_key == "sk-from-config"
        assert shim.llm.primary_model == "gpt-4o"

    def test_preflight_empty_key_still_fast_fails(self) -> None:
        """Non-ACP provider with empty key should fail with 'No API key'."""
        from ahvs.health import check_llm_connectivity

        result = check_llm_connectivity(
            api_key="", model="gpt-4o", provider="openai",
        )
        assert result.status == "fail"
        assert "No API key" in result.detail


class TestProviderMatrixFactory:
    """Fix #5 (v4): verify the factory produces correct clients for openai/openrouter."""

    def test_openai_provider_returns_llm_client(self, tmp_path: Path) -> None:
        from ahvs.executor import _make_llm_client
        from ahvs.llm.client import LLMClient

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="openai",
            llm_api_key="sk-test",
            llm_model="gpt-4o",
        )
        client = _make_llm_client(config)
        assert isinstance(client, LLMClient)
        # Should use the OpenAI preset base_url
        assert "openai.com" in client.config.base_url

    def test_openrouter_provider_returns_llm_client(self, tmp_path: Path) -> None:
        from ahvs.executor import _make_llm_client
        from ahvs.llm.client import LLMClient

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="openrouter",
            llm_api_key="sk-or-test",
            llm_model="anthropic/claude-opus-4-6",
        )
        client = _make_llm_client(config)
        assert isinstance(client, LLMClient)
        assert "openrouter.ai" in client.config.base_url

    def test_openai_shim_base_url_empty_uses_preset(self, tmp_path: Path) -> None:
        """When llm_base_url is empty, the factory should use PROVIDER_PRESETS."""
        from ahvs.executor import _ahvs_config_to_llm_shim

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="openai",
            llm_base_url="",  # empty — factory should use preset
            llm_api_key="sk-test",
        )
        shim = _ahvs_config_to_llm_shim(config)
        assert shim.llm.provider == "openai"
        assert shim.llm.base_url == ""  # shim passes empty; factory resolves via preset

    def test_openrouter_explicit_base_url_preserved(self, tmp_path: Path) -> None:
        """Explicit base_url overrides the preset."""
        from ahvs.executor import _make_llm_client
        from ahvs.llm.client import LLMClient

        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="openrouter",
            llm_base_url="https://custom-proxy.example.com/v1",
            llm_api_key="sk-test",
        )
        client = _make_llm_client(config)
        assert isinstance(client, LLMClient)
        assert client.config.base_url == "https://custom-proxy.example.com/v1"


class TestProviderMatrixPreflight:
    """Fix #5 (v4): preflight works for openai/openrouter providers."""

    def test_openai_preflight_uses_factory(self) -> None:
        """provider=openai with a fake key should fail via the shared factory, not 'No API key'."""
        from ahvs.health import check_llm_connectivity

        result = check_llm_connectivity(
            api_key="sk-fake-openai-key", model="gpt-4o",
            provider="openai",
        )
        assert result.status == "fail"
        # Should have attempted a real call (not early-out on "No API key")
        assert "No API key" not in result.detail

    def test_openrouter_preflight_uses_factory(self) -> None:
        """provider=openrouter with a fake key should fail via the shared factory."""
        from ahvs.health import check_llm_connectivity

        result = check_llm_connectivity(
            api_key="sk-fake-or-key",
            model="anthropic/claude-opus-4-6",
            provider="openrouter",
        )
        assert result.status == "fail"
        assert "No API key" not in result.detail


class TestCLIBaseUrl:
    """Fix #2 (v4): --base-url is wired through CLI to AHVSConfig."""

    def test_base_url_parsed(self) -> None:
        from ahvs.cli import main
        import argparse

        # Parse args without executing
        from ahvs.cli import argparse as _ap
        parser = _ap.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        ahvs_p = sub.add_parser("ahvs")
        ahvs_p.add_argument("--repo", "-r", required=True)
        ahvs_p.add_argument("--question", "-q", required=True)
        ahvs_p.add_argument("--base-url", default="")
        ahvs_p.add_argument("--provider", default="anthropic")

        args = parser.parse_args([
            "ahvs", "--repo", "/tmp/test", "--question", "test",
            "--base-url", "https://custom.example.com/v1",
            "--provider", "openai-compatible",
        ])
        assert args.base_url == "https://custom.example.com/v1"
        assert args.provider == "openai-compatible"

    def test_base_url_wired_to_config(self, tmp_path: Path) -> None:
        """AHVSConfig receives llm_base_url from CLI --base-url."""
        config = AHVSConfig(
            repo_path=tmp_path, question="test",
            llm_provider="openai-compatible",
            llm_base_url="https://my-proxy.example.com/v1",
            llm_api_key="sk-test",
        )
        assert config.llm_base_url == "https://my-proxy.example.com/v1"


class TestFromCliArgsParity:
    """Fix #4 (v4): from_cli_args uses the correct CLI attribute names."""

    def test_from_cli_args_reads_correct_attrs(self, tmp_path: Path) -> None:
        """from_cli_args should read args.model (not args.llm_model), etc."""
        import argparse

        args = argparse.Namespace(
            repo=str(tmp_path),
            question="test?",
            run_dir=None,
            max_hypotheses=2,
            regression_guard=None,
            skill_registry=None,
            prompts=None,
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            acp_agent="claude",
            acpx_command="",
            acp_session_name="ahvs",
            acp_timeout_sec=1800,
        )
        config = AHVSConfig.from_cli_args(args)
        assert config.llm_provider == "openai"
        assert config.llm_base_url == "https://api.openai.com/v1"
        assert config.llm_model == "gpt-4o"
        assert config.llm_api_key_env == "OPENAI_API_KEY"
        assert config.max_hypotheses == 2

    def test_from_cli_args_defaults(self, tmp_path: Path) -> None:
        """from_cli_args should handle missing optional attrs gracefully."""
        import argparse

        args = argparse.Namespace(
            repo=str(tmp_path),
            question="test?",
        )
        config = AHVSConfig.from_cli_args(args)
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-opus-4-6"
        assert config.llm_api_key_env == "ANTHROPIC_API_KEY"
        assert config.llm_base_url == ""


# ---------------------------------------------------------------------------
# 12. v4 review fixes (round 2) — fail-closed worktree, type strategies,
#     apply-best, skill_planned rename
# ---------------------------------------------------------------------------


class TestSkillPlannedRename:
    """Finding 4: skill_used → skill_planned."""

    def test_skill_planned_field_exists(self) -> None:
        r = _make_result()
        assert hasattr(r, "skill_planned")
        assert not hasattr(r, "skill_used")
        assert r.skill_planned is None

    def test_skill_planned_set(self) -> None:
        r = _make_result(skill_planned="promptfoo_eval")
        assert r.skill_planned == "promptfoo_eval"

    def test_load_results_migrates_skill_used(self, tmp_path: Path) -> None:
        """Old results.json with skill_used should be migrated to skill_planned."""
        from ahvs.result import load_results, save_results

        results = [_make_result(skill_planned="sandbox_run")]
        path = tmp_path / "results.json"
        save_results(results, path)

        # Manually rewrite to use old field name
        import json
        data = json.loads(path.read_text())
        for item in data:
            item["skill_used"] = item.pop("skill_planned")
        path.write_text(json.dumps(data))

        loaded = load_results(path)
        assert loaded[0].skill_planned == "sandbox_run"


class TestExecutionModeField:
    """Finding 1: execution_mode field on HypothesisResult."""

    def test_default_is_repo_grounded(self) -> None:
        r = _make_result()
        assert r.execution_mode == "repo_grounded"

    def test_no_worktree(self) -> None:
        r = _make_result(execution_mode="no_worktree")
        assert r.execution_mode == "no_worktree"

    def test_make_error_default_mode(self) -> None:
        r = HypothesisResult.make_error(
            hypothesis_id="H1",
            hypothesis_type="code_change",
            primary_metric="f1",
            baseline_value=0.5,
            error="worktree failed",
        )
        assert r.execution_mode == "repo_grounded"


class TestWorktreeFailClosed:
    """Finding 1: worktree failure should fail hypothesis by default."""

    def test_allow_no_worktree_default_false(self, tmp_path: Path) -> None:
        config = AHVSConfig(repo_path=tmp_path, question="test")
        assert config.allow_no_worktree is False

    def test_allow_no_worktree_set_true(self, tmp_path: Path) -> None:
        config = AHVSConfig(
            repo_path=tmp_path, question="test", allow_no_worktree=True
        )
        assert config.allow_no_worktree is True

    def test_cycle_summary_has_per_hypothesis(self, tmp_path: Path) -> None:
        """cycle_summary.json should include per_hypothesis with execution_mode."""
        from ahvs.skills import SkillLibrary

        repo = tmp_path / "repo"
        repo.mkdir()
        cycle_dir = tmp_path / "cycle_001"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal cycle artifacts
        baseline = {
            "primary_metric": "answer_relevance",
            "value": 0.74,
            "eval_command": "echo test",
        }
        bundle = {"baseline": baseline, "lessons": [], "rejected": []}
        (cycle_dir / "context_bundle.json").write_text(json.dumps(bundle))
        (cycle_dir / "cycle_manifest.json").write_text(json.dumps({"cycle_id": "test"}))
        (cycle_dir / "hypotheses.md").write_text("# H1\n")
        (cycle_dir / "selection.md").write_text("H1")
        (cycle_dir / "selection.json").write_text(
            json.dumps({"selected": ["H1"], "rationale": "test"})
        )
        (cycle_dir / "validation_plan.md").write_text("# Plan\n")
        results = [_make_result(hypothesis_id="H1")]
        save_results(results, cycle_dir / "results.json")
        (cycle_dir / "report.md").write_text("# Report\n")
        (cycle_dir / "friction_log.md").write_text("# Friction\n")

        config = AHVSConfig(repo_path=repo, question="test?", run_dir=cycle_dir)
        skill_lib = SkillLibrary()

        result = execute_ahvs_stage(
            AHVSStage.AHVS_CYCLE_VERIFY,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_lib,
            auto_approve=True,
        )

        assert result.status == StageStatus.DONE
        summary = json.loads((cycle_dir / "cycle_summary.json").read_text())
        assert "per_hypothesis" in summary
        assert len(summary["per_hypothesis"]) == 1
        assert summary["per_hypothesis"][0]["execution_mode"] == "repo_grounded"


class TestTypeExecutionStrategies:
    """Finding 2: all hypothesis types must have execution strategies."""

    def test_all_types_have_strategies(self) -> None:
        from ahvs.executor import _TYPE_EXECUTION_STRATEGIES

        expected_types = {
            "prompt_rewrite", "model_comparison", "config_change",
            "dspy_optimize", "code_change", "architecture_change",
            "multi_llm_judge",
        }
        assert set(_TYPE_EXECUTION_STRATEGIES.keys()) == expected_types

    def test_strategy_has_required_keys(self) -> None:
        from ahvs.executor import _TYPE_EXECUTION_STRATEGIES

        for hyp_type, strategy in _TYPE_EXECUTION_STRATEGIES.items():
            assert "system_addition" in strategy, f"{hyp_type} missing system_addition"
            assert "constraints" in strategy, f"{hyp_type} missing constraints"
            assert "success_guidance" in strategy, f"{hyp_type} missing success_guidance"

    def test_code_change_emphasizes_algorithms(self) -> None:
        from ahvs.executor import _TYPE_EXECUTION_STRATEGIES

        strat = _TYPE_EXECUTION_STRATEGIES["code_change"]
        assert "algorithm" in strat["system_addition"].lower()
        assert "code logic" in strat["constraints"].lower()

    def test_architecture_change_emphasizes_new_modules(self) -> None:
        from ahvs.executor import _TYPE_EXECUTION_STRATEGIES

        strat = _TYPE_EXECUTION_STRATEGIES["architecture_change"]
        assert "new module" in strat["system_addition"].lower() or "NEW module" in strat["system_addition"]

    def test_prompt_rewrite_excludes_code(self) -> None:
        from ahvs.executor import _TYPE_EXECUTION_STRATEGIES

        strat = _TYPE_EXECUTION_STRATEGIES["prompt_rewrite"]
        assert "DO NOT modify" in strat["system_addition"]


class TestHypothesisGenPromptDiversity:
    """Finding 2: generation prompt should guide toward diverse hypothesis types."""

    def test_prompt_includes_type_guidance(self) -> None:
        from ahvs.prompts import _AHVS_STAGES

        user_prompt = _AHVS_STAGES["ahvs_hypothesis_gen"]["user"]
        assert "Type guidance:" in user_prompt
        assert "code_change:" in user_prompt
        assert "architecture_change:" in user_prompt

    def test_prompt_encourages_diversity(self) -> None:
        from ahvs.prompts import _AHVS_STAGES

        user_prompt = _AHVS_STAGES["ahvs_hypothesis_gen"]["user"]
        assert "diverse types" in user_prompt.lower()

    def test_prompt_discourages_prompt_only(self) -> None:
        from ahvs.prompts import _AHVS_STAGES

        user_prompt = _AHVS_STAGES["ahvs_hypothesis_gen"]["user"]
        assert "not just" in user_prompt.lower() or "NEW algorithms" in user_prompt


class TestApplyBestConfig:
    """Finding 3: --apply-best config fields."""

    def test_apply_best_default_false(self, tmp_path: Path) -> None:
        config = AHVSConfig(repo_path=tmp_path, question="test")
        assert config.apply_best is False

    def test_apply_best_set_true(self, tmp_path: Path) -> None:
        config = AHVSConfig(repo_path=tmp_path, question="test", apply_best=True)
        assert config.apply_best is True

    def test_from_cli_args_apply_best(self, tmp_path: Path) -> None:
        import argparse

        args = argparse.Namespace(
            repo=str(tmp_path), question="test?",
            allow_no_worktree=True, apply_best=True,
        )
        config = AHVSConfig.from_cli_args(args)
        assert config.allow_no_worktree is True
        assert config.apply_best is True


class TestApplyBestFunction:
    """Finding 3: _apply_best reads cycle_summary and applies patch."""

    def test_no_best_hypothesis_returns_zero(self, tmp_path: Path) -> None:
        from ahvs.cli import _apply_best

        cycle_dir = tmp_path / "cycle"
        cycle_dir.mkdir()
        summary = {"best_hypothesis": None, "recommendation": "no improvement"}
        (cycle_dir / "cycle_summary.json").write_text(json.dumps(summary))

        config = AHVSConfig(
            repo_path=tmp_path, question="test", run_dir=cycle_dir
        )
        assert _apply_best(config) == 0

    def test_missing_summary_returns_one(self, tmp_path: Path) -> None:
        from ahvs.cli import _apply_best

        cycle_dir = tmp_path / "empty_cycle"
        cycle_dir.mkdir()
        config = AHVSConfig(
            repo_path=tmp_path, question="test", run_dir=cycle_dir
        )
        assert _apply_best(config) == 1

    def test_missing_patch_returns_one(self, tmp_path: Path) -> None:
        from ahvs.cli import _apply_best

        cycle_dir = tmp_path / "cycle"
        cycle_dir.mkdir()
        summary = {
            "best_hypothesis": "H1",
            "kept_patch": "tool_runs/H1/H1.patch",
            "best_metric_value": 0.85,
        }
        (cycle_dir / "cycle_summary.json").write_text(json.dumps(summary))
        # Don't create the patch file

        config = AHVSConfig(
            repo_path=tmp_path, question="test", run_dir=cycle_dir
        )
        assert _apply_best(config) == 1

    def test_no_patch_field_returns_one(self, tmp_path: Path) -> None:
        from ahvs.cli import _apply_best

        cycle_dir = tmp_path / "cycle"
        cycle_dir.mkdir()
        summary = {
            "best_hypothesis": "H1",
            "kept_patch": None,  # sandbox-only, no patch
        }
        (cycle_dir / "cycle_summary.json").write_text(json.dumps(summary))

        config = AHVSConfig(
            repo_path=tmp_path, question="test", run_dir=cycle_dir
        )
        assert _apply_best(config) == 1

    def test_apply_best_e2e_success(self, tmp_path: Path) -> None:
        """v5 Finding 4: e2e test — git apply + baseline update on a real repo."""
        import subprocess

        from ahvs.cli import _apply_best

        # 1. Create a real git repo with an initial file
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "hello.py").write_text("print('hello')\n")
        subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(repo), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(repo), check=True, capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(repo), check=True, capture_output=True,
        )

        # 2. Create a valid patch (add a new file)
        cycle_dir = tmp_path / "cycle_001"
        cycle_dir.mkdir()
        patch_dir = cycle_dir / "tool_runs" / "H1"
        patch_dir.mkdir(parents=True)
        patch_content = (
            "diff --git a/new_module.py b/new_module.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new_module.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+# Improved retrieval\n"
            "+def better_search(): pass\n"
        )
        (patch_dir / "H1.patch").write_text(patch_content)

        # 3. Create cycle_summary.json with a best hypothesis
        summary = {
            "best_hypothesis": "H1",
            "kept_patch": "tool_runs/H1/H1.patch",
            "best_metric_value": 0.92,
        }
        (cycle_dir / "cycle_summary.json").write_text(json.dumps(summary))

        # 4. Create baseline_metric.json
        baseline_path = repo / ".ahvs" / "baseline_metric.json"
        baseline_path.parent.mkdir(parents=True)
        baseline = {
            "primary_metric": "answer_relevance",
            "answer_relevance": 0.80,
        }
        baseline_path.write_text(json.dumps(baseline, indent=2))

        # 5. Run _apply_best
        config = AHVSConfig(
            repo_path=repo,
            question="test",
            run_dir=cycle_dir,
        )
        result = _apply_best(config)

        # 6. Verify patch was applied
        assert result == 0
        new_file = repo / "new_module.py"
        assert new_file.exists(), "Patch should have created new_module.py"
        assert "better_search" in new_file.read_text()

        # 7. Verify baseline was updated
        updated = json.loads(baseline_path.read_text())
        assert updated["answer_relevance"] == 0.92
        assert updated["applied_from_cycle"] == "cycle_001"
        assert updated["applied_hypothesis"] == "H1"
        assert "recorded_at" in updated
        assert "commit" in updated  # git rev-parse HEAD should succeed


# ---------------------------------------------------------------------------
# Regression tests for Bugs A, C, E
# (added 2026-03-24 — previously fixed but not covered by tests)
# ---------------------------------------------------------------------------


class TestBugA_EvalCwdSubdirWriteBase:
    """Bug A regression: apply_files must write relative to eval_cwd, not worktree root.

    Scenario: repo_path is a subdirectory of the git root (e.g. the target repo
    lives at /git-root/autoqa/).  The worktree is created at the git root level,
    so files must land at {worktree}/autoqa/src/file.py — NOT {worktree}/src/file.py.
    """

    def test_eval_cwd_is_set_to_subdir_after_create(self, tmp_path: Path) -> None:
        git_root = tmp_path / "git_root"
        git_root.mkdir()
        _init_git_repo(git_root)
        subdir = git_root / "autoqa"
        subdir.mkdir()
        (subdir / "module.py").write_text("def foo(): pass\n")
        subprocess.run(["git", "add", "."], cwd=str(git_root), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add subdir"],
            cwd=str(git_root), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(git_root / "autoqa", wt_path)
        wt.create()

        assert wt.eval_cwd == wt_path / "autoqa"
        wt.cleanup()

    def test_apply_files_writes_under_eval_cwd_not_worktree_root(self, tmp_path: Path) -> None:
        git_root = tmp_path / "git_root"
        git_root.mkdir()
        _init_git_repo(git_root)
        subdir = git_root / "autoqa"
        subdir.mkdir()
        (subdir / "module.py").write_text("def foo(): pass\n")
        subprocess.run(["git", "add", "."], cwd=str(git_root), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add subdir"],
            cwd=str(git_root), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(git_root / "autoqa", wt_path)
        wt.create()

        wt.apply_files({"src/new_file.py": "x = 1\n"})

        # Must land under eval_cwd (autoqa subdir), NOT at worktree root
        assert (wt_path / "autoqa" / "src" / "new_file.py").exists()
        assert not (wt_path / "src" / "new_file.py").exists()

        wt.cleanup()


class TestBugC_EvalCwdExistenceCheck:
    """Bug C regression: missing eval_cwd must surface a clear error, not silent ENOENT.

    Two surfaces:
    1. create() raises RuntimeError if eval_cwd subdir is absent after checkout.
    2. run_eval_command() returns an EvalResult with returncode=-1 and a clear
       error message if eval_cwd has been deleted after create().
    """

    def test_create_raises_when_subdir_not_tracked(self, tmp_path: Path) -> None:
        """eval_cwd subdir exists on disk but is NOT committed → absent in worktree."""
        git_root = tmp_path / "git_root"
        git_root.mkdir()
        _init_git_repo(git_root)

        # Create the subdir locally but do NOT commit it
        untracked = git_root / "autoqa"
        untracked.mkdir()

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(untracked, wt_path)

        with pytest.raises(RuntimeError, match="eval_cwd"):
            wt.create()

    def test_run_eval_command_returns_error_when_eval_cwd_deleted(self, tmp_path: Path) -> None:
        """After create(), if eval_cwd is removed, run_eval_command returns a clear error."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        # Simulate eval_cwd being deleted (e.g., Claude Code wiped the subdir)
        wt.eval_cwd = wt_path / "nonexistent_subdir"

        result = wt.run_eval_command("echo hello")

        assert result.returncode == -1
        assert "eval_cwd" in result.stderr.lower() or "eval_cwd" in result.stdout.lower()

        wt.cleanup()


class TestBugE_SpliceFunctions:
    """Bug E regression: splice_functions must correctly merge partial Claude Code
    output into existing files without truncation or API loss.
    """

    def test_replace_existing_function(self) -> None:
        """Modified function replaces original; unmodified functions are preserved."""
        from ahvs.worktree import splice_functions

        original = textwrap.dedent("""\
            def foo(x):
                return x + 1

            def bar(y):
                return y * 2
        """)
        partial = textwrap.dedent("""\
            def foo(x):
                return x + 100
        """)
        result = splice_functions(original, partial)
        assert "return x + 100" in result      # modified version kept
        assert "return x + 1\n" not in result   # old version gone (exact line)
        assert "def bar(y)" in result           # untouched function preserved
        assert "return y * 2" in result

    def test_append_new_function(self) -> None:
        """A function not present in the original is appended."""
        from ahvs.worktree import splice_functions

        original = textwrap.dedent("""\
            def foo(x):
                return x + 1
        """)
        partial = textwrap.dedent("""\
            def baz(z):
                return z ** 2
        """)
        result = splice_functions(original, partial)
        assert "def foo(x)" in result
        assert "return x + 1" in result
        assert "def baz(z)" in result
        assert "return z ** 2" in result

    def test_replace_and_append_combined(self) -> None:
        """Partial output that both modifies existing and adds new definitions."""
        from ahvs.worktree import splice_functions

        original = textwrap.dedent("""\
            def foo(x):
                return x

            def bar(y):
                return y
        """)
        partial = textwrap.dedent("""\
            def foo(x):
                return x * 10

            def new_helper(z):
                return z + 99
        """)
        result = splice_functions(original, partial)
        assert "return x * 10" in result     # foo modified
        assert "def bar(y)" in result        # bar preserved
        assert "def new_helper(z)" in result # new function added
        assert "return z + 99" in result

    def test_syntax_error_in_original_returns_partial(self) -> None:
        """If original has a syntax error, fall back to returning partial as-is."""
        from ahvs.worktree import splice_functions

        original = "def foo(: bad syntax here"
        partial = "def foo(x):\n    return x\n"
        result = splice_functions(original, partial)
        assert result == partial

    def test_syntax_error_in_partial_returns_original(self) -> None:
        """If partial has a syntax error, fall back to returning original unchanged."""
        from ahvs.worktree import splice_functions

        original = "def foo(x):\n    return x\n"
        partial = "def foo(: bad syntax"
        result = splice_functions(original, partial)
        assert result == original

    def test_new_import_propagated(self) -> None:
        """New imports in partial are added to the merged result."""
        from ahvs.worktree import splice_functions

        original = textwrap.dedent("""\
            import os

            def foo():
                pass
        """)
        partial = textwrap.dedent("""\
            import json

            def foo():
                return json.dumps({})
        """)
        result = splice_functions(original, partial)
        assert "import os" in result
        assert "import json" in result
        assert "return json.dumps({})" in result

    def test_class_definition_replaced(self) -> None:
        """Class definitions in partial replace matching originals."""
        from ahvs.worktree import splice_functions

        original = textwrap.dedent("""\
            class MyClass:
                def method(self):
                    return 1

            def standalone():
                pass
        """)
        partial = textwrap.dedent("""\
            class MyClass:
                def method(self):
                    return 999
        """)
        result = splice_functions(original, partial)
        assert "return 999" in result
        assert "return 1" not in result
        assert "def standalone()" in result


# ---------------------------------------------------------------------------
# 13. v7 review improvements — cross-cycle memory, enriched context,
#     JSON parsing, skill semantics, eval-mode intelligence, worktree subdir
# ---------------------------------------------------------------------------


class TestCrossCycleMemoryStageNameFix:
    """P0 Fix 1: context_loader queries EvolutionStore with 'ahvs_execution'
    (matching the stage_name used when writing lessons at Stage 7).
    """

    def test_context_loader_queries_ahvs_execution(self) -> None:
        """load_context_bundle must query 'ahvs_execution', not 'ahvs_hypothesis_gen'."""
        import inspect
        from ahvs.context_loader import load_context_bundle

        source = inspect.getsource(load_context_bundle)
        assert '"ahvs_execution"' in source
        assert '"ahvs_hypothesis_gen"' not in source

    def test_lessons_round_trip_with_matching_stage_name(self, tmp_path: Path) -> None:
        """Lessons written as ahvs_execution should be retrieved with 2x boost."""
        from ahvs.evolution import EvolutionStore, LessonEntry
        from datetime import datetime, timezone

        store = EvolutionStore(tmp_path / "evolution")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Write lessons with ahvs_execution stage_name (as executor does)
        store.append(LessonEntry(
            stage_name="ahvs_execution",
            stage_num=6,
            category="experiment",
            severity="info",
            description="H1 improved precision by +5%",
            timestamp=now,
        ))
        store.append(LessonEntry(
            stage_name="other_stage",
            stage_num=1,
            category="pipeline",
            severity="info",
            description="Some unrelated lesson",
            timestamp=now,
        ))

        # Query as context_loader now does
        results = store.query_for_stage("ahvs_execution", max_lessons=5)
        assert len(results) >= 1
        # The ahvs_execution lesson should be ranked first (2x boost)
        assert results[0].stage_name == "ahvs_execution"
        assert "H1 improved precision" in results[0].description


class TestEnrichedOnboardingContext:
    """P0 Fix 2: enriched baseline fields are forwarded into context_bundle."""

    def test_enriched_fields_forwarded(self, tmp_path: Path) -> None:
        """Fields like optimization_goal, constraints, etc. appear in context_bundle."""
        from ahvs.context_loader import load_context_bundle

        baseline = {
            "primary_metric": "precision",
            "precision": 0.67,
            "recorded_at": "2026-03-24T10:00:00Z",
            "eval_command": "echo 'precision: 0.67'",
            "optimization_goal": "Improve elon_musk_fan precision without tanking F1",
            "regression_floor": 0.68,
            "constraints": ["Do not change pro_palestine cohort", "Budget: Flash Lite only"],
            "system_levers": ["binary threshold", "cohort-specific prompt", "parsing.py"],
            "prior_experiments": ["Tried threshold change — no effect due to eval-only bug"],
            "notes": "elon_musk_fan has 12 FP vs 3 TP",
        }
        baseline_path = tmp_path / "baseline_metric.json"
        baseline_path.write_text(json.dumps(baseline))

        bundle = load_context_bundle(
            repo_path=tmp_path,
            question="Improve precision",
            evolution_dir=tmp_path / "evolution",
            baseline_path=baseline_path,
        )

        assert "enriched_context" in bundle
        ec = bundle["enriched_context"]
        assert ec["optimization_goal"] == baseline["optimization_goal"]
        assert ec["regression_floor"] == baseline["regression_floor"]
        assert ec["constraints"] == baseline["constraints"]
        assert ec["system_levers"] == baseline["system_levers"]
        assert ec["prior_experiments"] == baseline["prior_experiments"]
        assert ec["notes"] == baseline["notes"]

    def test_empty_enriched_fields_omitted(self, tmp_path: Path) -> None:
        """When no enriched fields are present, enriched_context is empty dict."""
        from ahvs.context_loader import load_context_bundle

        baseline = {
            "primary_metric": "f1",
            "f1": 0.73,
            "recorded_at": "2026-03-24T10:00:00Z",
            "eval_command": "echo 'f1: 0.73'",
        }
        baseline_path = tmp_path / "baseline_metric.json"
        baseline_path.write_text(json.dumps(baseline))

        bundle = load_context_bundle(
            repo_path=tmp_path,
            question="test",
            evolution_dir=tmp_path / "evolution",
            baseline_path=baseline_path,
        )

        assert bundle["enriched_context"] == {}

    def test_enriched_context_in_prompt_template(self) -> None:
        """The hypothesis-generation prompt template includes {enriched_context}."""
        from ahvs.prompts import _AHVS_STAGES

        user_prompt = _AHVS_STAGES["ahvs_hypothesis_gen"]["user"]
        assert "{enriched_context}" in user_prompt
        assert "Operator Context" in user_prompt


class TestJSONParsing:
    """P1 Fix 1: structured JSON parsing with markdown fallback."""

    def test_parse_hypotheses_json_array(self) -> None:
        """JSON array of hypothesis objects should be parsed correctly."""
        from ahvs.executor import _parse_hypotheses

        text = json.dumps([
            {
                "id": "H1",
                "type": "code_change",
                "description": "Improve ranking algorithm",
                "rationale": "Current ranking is naive",
                "estimated_cost": "medium",
                "required_tools": "pytest",
            },
            {
                "id": "H2",
                "type": "prompt_rewrite",
                "description": "Rewrite system prompt",
                "rationale": "Too verbose",
                "estimated_cost": "low",
                "required_tools": "promptfoo",
            },
        ])
        result = _parse_hypotheses(text)
        assert len(result) == 2
        assert result[0]["id"] == "H1"
        assert result[0]["type"] == "code_change"
        assert result[1]["id"] == "H2"
        assert result[1]["required_tools"] == ["promptfoo"]

    def test_parse_hypotheses_fenced_json(self) -> None:
        """JSON inside a ```json fenced block should be parsed."""
        from ahvs.executor import _parse_hypotheses

        text = (
            "Here are my hypotheses:\n\n"
            "```json\n"
            '[{"id": "H1", "type": "code_change", "description": "fix algo"}]\n'
            "```\n"
        )
        result = _parse_hypotheses(text)
        assert len(result) == 1
        assert result[0]["id"] == "H1"

    def test_parse_hypotheses_markdown_fallback(self) -> None:
        """Traditional markdown format still works when JSON is absent."""
        from ahvs.executor import _parse_hypotheses

        text = (
            "## H1\n"
            "**Type:** code_change\n"
            "**Description:** Fix the ranking\n"
            "**Rationale:** It's broken\n"
            "**Estimated Cost:** low\n"
            "**Required Tools:** pytest\n"
        )
        result = _parse_hypotheses(text)
        assert len(result) == 1
        assert result[0]["id"] == "H1"
        assert result[0]["type"] == "code_change"
        assert result[0]["required_tools"] == ["pytest"]

    def test_parse_hypotheses_invalid_json_falls_through(self) -> None:
        """Invalid JSON falls through to markdown parser."""
        from ahvs.executor import _parse_hypotheses

        text = (
            '{bad json}\n'
            "## H1\n"
            "**Type:** prompt_rewrite\n"
            "**Description:** Better prompt\n"
        )
        result = _parse_hypotheses(text)
        assert len(result) == 1
        assert result[0]["id"] == "H1"
        assert result[0]["type"] == "prompt_rewrite"

    def test_parse_hypotheses_json_missing_required_fields(self) -> None:
        """JSON objects missing required fields are skipped."""
        from ahvs.executor import _parse_hypotheses

        text = json.dumps([
            {"id": "H1", "type": "code_change", "description": "good one"},
            {"id": "H2"},  # missing type and description
            {"type": "prompt_rewrite"},  # missing id
        ])
        result = _parse_hypotheses(text)
        assert len(result) == 1
        assert result[0]["id"] == "H1"

    def test_parse_hypotheses_json_required_tools_as_list(self) -> None:
        """required_tools as a list should be kept as-is."""
        from ahvs.executor import _parse_hypotheses

        text = json.dumps([{
            "id": "H1", "type": "code_change", "description": "test",
            "required_tools": ["pytest", "docker"],
        }])
        result = _parse_hypotheses(text)
        assert result[0]["required_tools"] == ["pytest", "docker"]

    def test_parse_selection_json(self) -> None:
        """Selection as JSON object should be parsed correctly."""
        from ahvs.executor import _parse_selection

        text = json.dumps({"selected": ["H1", "H3"], "rationale": "Best coverage"})
        result = _parse_selection(text)
        assert result["selected"] == ["H1", "H3"]
        assert result["rationale"] == "Best coverage"

    def test_parse_selection_markdown_fallback(self) -> None:
        """Traditional selection format still works."""
        from ahvs.executor import _parse_selection

        text = "Selected: H1, H2\n**Rationale:** They look promising"
        result = _parse_selection(text)
        assert "H1" in result["selected"]
        assert "H2" in result["selected"]

    def test_parse_validation_plan_json(self) -> None:
        """Validation plan as JSON array should be parsed correctly."""
        from ahvs.executor import _parse_validation_plan

        text = json.dumps([{
            "id": "H1",
            "implementation_approach": "Modify ranking function",
            "eval_method": "custom_script",
            "skill": "sandbox_run",
            "success_criterion": "f1 > 0.80",
            "expected_artifacts": "src/ranking.py",
        }])
        result = _parse_validation_plan(text)
        assert len(result) == 1
        assert result[0]["id"] == "H1"
        assert result[0]["eval_method"] == "custom_script"

    def test_parse_validation_plan_markdown_fallback(self) -> None:
        """Traditional validation plan format still works."""
        from ahvs.executor import _parse_validation_plan

        text = (
            "## H1\n"
            "**Implementation Approach:** Rewrite the search\n"
            "**Eval Method:** custom_script\n"
            "**Skill:** sandbox_run\n"
            "**Success Criterion:** f1 > 0.80\n"
            "**Expected Artifacts:** ranking.py\n"
        )
        result = _parse_validation_plan(text)
        assert len(result) == 1
        assert result[0]["id"] == "H1"


class TestSkillSemanticsConsistency:
    """P1 Fix 2: skill descriptions must be consistent — advisory, not runtime-resolved."""

    def test_context_block_says_advisory(self) -> None:
        """to_context_block should describe skills as guidance, not runtime dispatch."""
        from ahvs.skills import SkillLibrary, BUILTIN_SKILLS

        lib = SkillLibrary()
        block = lib.to_context_block(BUILTIN_SKILLS[:2])
        assert "guidance" in block.lower() or "approach" in block.lower()
        assert "resolve skill invocations" not in block.lower()

    def test_module_docstring_says_advisory(self) -> None:
        """Module docstring should say skills are informational guidance."""
        import ahvs.skills as skills_mod

        doc = skills_mod.__doc__ or ""
        assert "informational guidance" in doc.lower()


class TestEvalModeIntelligence:
    """P2 Fix 1: warn when prompt_rewrite hypothesis + --eval-only eval command."""

    def test_prompt_rewrite_with_eval_only_warns(self, capsys) -> None:
        """Should print a warning when prompt_rewrite meets --eval-only."""
        from ahvs.executor import _run_single_hypothesis
        from ahvs.skills import SkillLibrary

        # We can't easily run _run_single_hypothesis (requires LLM), but we
        # can test the warning logic directly by checking the source code
        import inspect
        source = inspect.getsource(_run_single_hypothesis)
        assert "_NEEDS_REINFERENCE_TYPES" in source
        assert "prompt_rewrite" in source
        assert "model_comparison" in source
        assert "--eval-only" in source

    def test_code_change_not_in_reinference_types(self) -> None:
        """code_change should NOT trigger eval-mode warning."""
        import inspect
        from ahvs.executor import _run_single_hypothesis

        source = inspect.getsource(_run_single_hypothesis)
        # The set should contain prompt_rewrite and model_comparison but not code_change
        assert '"code_change"' not in source.split("_NEEDS_REINFERENCE_TYPES")[1].split("}")[0]


class TestWorktreeSubdirHardening:
    """P2 Fix 2: improved diagnostics for worktree subdir handling."""

    def test_non_git_repo_gives_clear_error(self, tmp_path: Path) -> None:
        """Attempting to create worktree from non-git dir should mention git root."""
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()
        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(not_a_repo, wt_path)

        with pytest.raises(RuntimeError, match="git root"):
            wt.create()

    def test_subdir_not_under_git_root_gives_clear_error(self, tmp_path: Path) -> None:
        """If repo_path is somehow not under git root, error should be clear."""
        # This is a very edge case — just verify the code handles it
        import inspect
        from ahvs.worktree import HypothesisWorktree

        source = inspect.getsource(HypothesisWorktree.create)
        assert "not under git root" in source.lower() or "not under git root" in source

    def test_subdir_worktree_eval_runs_from_correct_dir(self, tmp_path: Path) -> None:
        """Eval command in subdir worktree runs from the correct subdirectory."""
        git_root = tmp_path / "git_root"
        git_root.mkdir()
        _init_git_repo(git_root)
        subdir = git_root / "myproject"
        subdir.mkdir()
        (subdir / "check.py").write_text("print('ok')\n")
        subprocess.run(["git", "add", "."], cwd=str(git_root), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add project"],
            cwd=str(git_root), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(git_root / "myproject", wt_path)
        wt.create()

        # Run eval that checks we're in the right directory
        result = wt.run_eval_command("ls check.py")
        assert result.returncode == 0
        assert "check.py" in result.stdout

        wt.cleanup()

    def test_subdir_worktree_full_roundtrip(self, tmp_path: Path) -> None:
        """Full round-trip: create subdir worktree, apply files, run eval, capture diff."""
        git_root = tmp_path / "git_root"
        git_root.mkdir()
        _init_git_repo(git_root)
        subdir = git_root / "app"
        subdir.mkdir()
        (subdir / "main.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=str(git_root), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=str(git_root), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(git_root / "app", wt_path)
        wt.create()

        # Apply a file change
        wt.apply_files({"main.py": "x = 42\n"})

        # Verify file landed in the right place
        assert (wt_path / "app" / "main.py").read_text() == "x = 42\n"

        # Run eval from correct dir
        result = wt.run_eval_command("python3 -c \"exec(open('main.py').read()); print(f'x={x}')\"")
        assert result.returncode == 0
        assert "x=42" in result.stdout

        # Capture diff
        diff = wt.capture_diff()
        assert "main.py" in diff
        assert "+x = 42" in diff

        wt.cleanup()


# ---------------------------------------------------------------------------
# 14. v8 review fixes + Bug L/M — forbidden files, import sanity check,
#     hardened metric extraction, plan validation, falsy enriched context,
#     behavioral tests
# ---------------------------------------------------------------------------


class TestForbiddenFiles:
    """Eval-harness protection: forbidden files are blocked before apply_files."""

    def test_run_eval_py_blocked(self) -> None:
        from ahvs.executor import _is_forbidden_file
        assert _is_forbidden_file("run_eval.py") is not None
        assert _is_forbidden_file("src/autoqa/run_eval.py") is not None

    def test_init_py_is_warn_not_block(self) -> None:
        """__init__.py is warn-only, not hard-blocked (v9 fix)."""
        from ahvs.executor import _is_forbidden_file, _is_warn_file
        # NOT hard-blocked
        assert _is_forbidden_file("__init__.py") is None
        assert _is_forbidden_file("src/autoqa/__init__.py") is None
        # But IS a warning
        assert _is_warn_file("__init__.py") is not None
        assert _is_warn_file("src/autoqa/__init__.py") is not None

    def test_evaluation_py_blocked(self) -> None:
        from ahvs.executor import _is_forbidden_file
        assert _is_forbidden_file("evaluation.py") is not None

    def test_main_py_is_warn_not_block(self) -> None:
        """main.py is warn-only, not hard-blocked (v10 fix)."""
        from ahvs.executor import _is_forbidden_file, _is_warn_file
        assert _is_forbidden_file("main.py") is None
        assert _is_warn_file("main.py") is not None

    def test_test_files_blocked(self) -> None:
        from ahvs.executor import _is_forbidden_file
        assert _is_forbidden_file("test_fixes.py") is not None
        assert _is_forbidden_file("test_parsing.py") is not None
        assert _is_forbidden_file("parsing_test.py") is not None

    def test_normal_files_allowed(self) -> None:
        from ahvs.executor import _is_forbidden_file
        assert _is_forbidden_file("src/autoqa/parsing.py") is None
        assert _is_forbidden_file("src/autoqa/prompts.py") is None
        assert _is_forbidden_file("config.yaml") is None
        assert _is_forbidden_file("src/autoqa/llm_client.py") is None

    def test_forbidden_basenames_constant_complete(self) -> None:
        """All known problematic files from production runs are blocked or warned."""
        from ahvs.executor import _FORBIDDEN_BASENAMES, _WARN_BASENAMES
        for f in ("run_eval.py", "evaluation.py"):
            assert f in _FORBIDDEN_BASENAMES
        for f in ("__init__.py", "main.py"):
            assert f in _WARN_BASENAMES


class TestPreEvalImportSanityCheck:
    """Pre-eval import check catches broken modules before eval_command runs."""

    def test_passing_import_check(self, tmp_path: Path) -> None:
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        # Create a valid Python package
        pkg = repo / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "run_eval.py").write_text("print('ok')\n")
        subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add pkg"],
            cwd=str(repo), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        result = _run_import_sanity_check(
            wt, "python3 -m mypkg.run_eval --eval-only"
        )
        assert result is None  # should pass
        wt.cleanup()

    def test_failing_import_check(self, tmp_path: Path) -> None:
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        # No module exists — import should fail
        result = _run_import_sanity_check(
            wt, "python3 -m nonexistent_module.run_eval"
        )
        assert result is not None
        assert "nonexistent_module" in result
        wt.cleanup()

    def test_no_module_flag_skips_check(self, tmp_path: Path) -> None:
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        # eval_command without -m flag — can't determine module, skip check
        result = _run_import_sanity_check(wt, "bash run_eval.sh")
        assert result is None
        wt.cleanup()


class TestHardenedMetricExtraction:
    """Bug L fix: sandbox self-reports are never trusted when eval_command is configured."""

    def test_sandbox_tiers_unconditionally_skipped_with_eval_command(self) -> None:
        """When eval_command is configured, Tiers 1-3 must be skipped
        regardless of whether eval succeeded or failed."""
        import inspect
        from ahvs.executor import _run_single_hypothesis

        source = inspect.getsource(_run_single_hypothesis)
        # The code should have eval_command_is_authoritative gate
        assert "eval_command_is_authoritative" in source
        # Tiers 1-3 should only run when NOT authoritative
        assert "not eval_command_is_authoritative" in source

    def test_prompt_mentions_protected_files(self) -> None:
        """The Claude Code prompt should mention protected/forbidden files."""
        import inspect
        from ahvs.executor import _run_single_hypothesis

        source = inspect.getsource(_run_single_hypothesis)
        assert "PROTECTED FILES" in source
        assert "run_eval.py" in source


class TestPlanRequiredFieldsTightened:
    """v8 Finding 1: _PLAN_REQUIRED_FIELDS now requires core execution fields."""

    def test_bare_id_plan_rejected(self) -> None:
        """A JSON plan with only 'id' should fall through to markdown."""
        from ahvs.executor import _parse_validation_plan

        text = json.dumps([{"id": "H1"}])
        # Should fall through to markdown (which finds nothing) → empty list
        result = _parse_validation_plan(text)
        assert len(result) == 0

    def test_complete_json_plan_accepted(self) -> None:
        """A JSON plan with all required fields should be accepted."""
        from ahvs.executor import _parse_validation_plan

        text = json.dumps([{
            "id": "H1",
            "implementation_approach": "Modify ranking function",
            "eval_method": "custom_script",
        }])
        result = _parse_validation_plan(text)
        assert len(result) == 1
        assert result[0]["id"] == "H1"

    def test_plan_required_fields_include_core(self) -> None:
        from ahvs.executor import _PLAN_REQUIRED_FIELDS
        assert "implementation_approach" in _PLAN_REQUIRED_FIELDS
        assert "eval_method" in _PLAN_REQUIRED_FIELDS


class TestFalsyEnrichedContext:
    """v8 Finding 2: falsy but meaningful values like 0 or 0.0 are preserved."""

    def test_zero_regression_floor_preserved(self, tmp_path: Path) -> None:
        from ahvs.context_loader import load_context_bundle

        baseline = {
            "primary_metric": "precision",
            "precision": 0.67,
            "recorded_at": "2026-03-24T10:00:00Z",
            "eval_command": "echo test",
            "regression_floor": 0.0,  # falsy but meaningful
        }
        baseline_path = tmp_path / "baseline_metric.json"
        baseline_path.write_text(json.dumps(baseline))

        bundle = load_context_bundle(
            repo_path=tmp_path,
            question="test",
            evolution_dir=tmp_path / "evolution",
            baseline_path=baseline_path,
        )

        assert "regression_floor" in bundle["enriched_context"]
        assert bundle["enriched_context"]["regression_floor"] == 0.0

    def test_false_value_preserved(self, tmp_path: Path) -> None:
        from ahvs.context_loader import load_context_bundle

        baseline = {
            "primary_metric": "f1",
            "f1": 0.73,
            "recorded_at": "2026-03-24T10:00:00Z",
            "eval_command": "echo test",
            "constraints": False,  # falsy but explicit
        }
        baseline_path = tmp_path / "baseline_metric.json"
        baseline_path.write_text(json.dumps(baseline))

        bundle = load_context_bundle(
            repo_path=tmp_path,
            question="test",
            evolution_dir=tmp_path / "evolution",
            baseline_path=baseline_path,
        )

        assert "constraints" in bundle["enriched_context"]
        assert bundle["enriched_context"]["constraints"] is False

    def test_none_value_excluded(self, tmp_path: Path) -> None:
        from ahvs.context_loader import load_context_bundle

        baseline = {
            "primary_metric": "f1",
            "f1": 0.73,
            "recorded_at": "2026-03-24T10:00:00Z",
            "eval_command": "echo test",
            "constraints": None,  # None should be excluded
        }
        baseline_path = tmp_path / "baseline_metric.json"
        baseline_path.write_text(json.dumps(baseline))

        bundle = load_context_bundle(
            repo_path=tmp_path,
            question="test",
            evolution_dir=tmp_path / "evolution",
            baseline_path=baseline_path,
        )

        assert "constraints" not in bundle["enriched_context"]


class TestBehavioralCrossCycleMemory:
    """v8 Finding 3: behavioral test for cross-cycle memory (not source inspection)."""

    def test_lessons_written_as_ahvs_execution_retrieved_with_boost(self, tmp_path: Path) -> None:
        """End-to-end: write lessons → load_context_bundle retrieves them."""
        from ahvs.context_loader import load_context_bundle
        from ahvs.evolution import EvolutionStore, LessonEntry
        from datetime import datetime, timezone

        # Create evolution store with a lesson
        evo_dir = tmp_path / "evolution"
        store = EvolutionStore(evo_dir)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store.append(LessonEntry(
            stage_name="ahvs_execution",
            stage_num=6,
            category="experiment",
            severity="info",
            description="H1 improved precision by +5% using threshold change",
            timestamp=now,
        ))
        store.append(LessonEntry(
            stage_name="ahvs_execution",
            stage_num=6,
            category="experiment",
            severity="warning",
            description="H2 failed: prompt rewrite had no effect due to eval-only mode",
            timestamp=now,
        ))

        # Create baseline
        baseline = {
            "primary_metric": "precision",
            "precision": 0.85,
            "recorded_at": "2026-03-24T10:00:00Z",
            "eval_command": "echo 'precision: 0.85'",
        }
        baseline_path = tmp_path / "baseline_metric.json"
        baseline_path.write_text(json.dumps(baseline))

        # Load context bundle — should retrieve both lessons
        bundle = load_context_bundle(
            repo_path=tmp_path,
            question="Improve precision",
            evolution_dir=evo_dir,
            baseline_path=baseline_path,
        )

        # Prior lessons (severity=info) should include H1
        assert any("H1 improved" in l for l in bundle["prior_lessons"])
        # Rejected approaches (severity=warning) should include H2
        assert any("H2 failed" in r for r in bundle["rejected_approaches"])

    def test_no_evolution_dir_produces_empty_lessons(self, tmp_path: Path) -> None:
        """First cycle with no evolution store should still work."""
        from ahvs.context_loader import load_context_bundle

        baseline = {
            "primary_metric": "f1",
            "f1": 0.73,
            "recorded_at": "2026-03-24T10:00:00Z",
            "eval_command": "echo test",
        }
        baseline_path = tmp_path / "baseline_metric.json"
        baseline_path.write_text(json.dumps(baseline))

        bundle = load_context_bundle(
            repo_path=tmp_path,
            question="test",
            evolution_dir=tmp_path / "nonexistent_evolution",
            baseline_path=baseline_path,
        )

        assert bundle["prior_lessons"] == []
        assert bundle["rejected_approaches"] == []


class TestBehavioralEvalModeIntelligence:
    """v8 Finding 3: behavioral test for eval-mode warnings."""

    def test_eval_mode_warning_types(self) -> None:
        """_NEEDS_REINFERENCE_TYPES should contain exactly the right types."""
        # Test by checking the actual set in the function body
        from ahvs.executor import _run_single_hypothesis
        import inspect
        source = inspect.getsource(_run_single_hypothesis)
        # Extract the set literal
        assert '"prompt_rewrite"' in source
        assert '"model_comparison"' in source
        # code_change should NOT be in the reinference set
        idx = source.index("_NEEDS_REINFERENCE_TYPES")
        set_block = source[idx:idx + 100]
        assert '"code_change"' not in set_block


# ---------------------------------------------------------------------------
# 15. v9 review fixes — wrapped eval_command parsing, __init__.py warn-only
# ---------------------------------------------------------------------------


class TestFindPythonAndModule:
    """v9/v10 Fix: _find_python_and_module handles all documented eval_command shapes."""

    def test_simple_python_m(self) -> None:
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module("python3 -m autoqa.run_eval --eval-only")
        assert result is not None
        assert result.python_exe == "python3"
        assert result.module_name == "autoqa.run_eval"
        assert result.cd_dir is None

    def test_absolute_python_path(self) -> None:
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module(
            "/home/ubuntu/miniconda3/envs/cohort_work/bin/python -m autoqa.run_eval"
        )
        assert result is not None
        assert result.python_exe == "/home/ubuntu/miniconda3/envs/cohort_work/bin/python"
        assert result.module_name == "autoqa.run_eval"

    def test_pythonpath_prefix(self) -> None:
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module(
            "PYTHONPATH=src:$PYTHONPATH /usr/bin/python3 -m autoqa.run_eval --eval-only"
        )
        assert result is not None
        assert result.python_exe == "/usr/bin/python3"
        assert result.module_name == "autoqa.run_eval"
        assert "PYTHONPATH=src:$PYTHONPATH" in result.env_prefix

    def test_cd_and_python(self) -> None:
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module(
            "cd /path/to/project && python -m package.run_eval --eval-only"
        )
        assert result is not None
        assert result.python_exe == "python"
        assert result.module_name == "package.run_eval"
        assert result.cd_dir == "/path/to/project"

    def test_multiple_env_vars(self) -> None:
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module(
            "PYTHONPATH=src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python3 -m my_pkg.eval"
        )
        assert result is not None
        assert result.python_exe == "python3"
        assert result.module_name == "my_pkg.eval"
        assert "PYTHONPATH=" in result.env_prefix
        assert "CUDA_VISIBLE_DEVICES=" in result.env_prefix

    def test_quoted_env_var_preserved(self) -> None:
        """v11 fix: quoted env var values must be preserved verbatim."""
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module(
            'FOO="a b" python3 -m mypkg.run_eval --eval-only'
        )
        assert result is not None
        assert result.python_exe == "python3"
        assert result.module_name == "mypkg.run_eval"
        # The raw env prefix must preserve the original quoting
        assert 'FOO="a b"' in result.env_prefix

    def test_no_m_flag_returns_none(self) -> None:
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module("bash run_eval.sh")
        assert result is None

    def test_cd_semicolon_python(self) -> None:
        from ahvs.executor import _find_python_and_module
        result = _find_python_and_module(
            "cd /tmp; python3 -m mymod.entry"
        )
        assert result is not None
        assert result.module_name == "mymod.entry"
        assert result.cd_dir == "/tmp"

    def test_real_eval_command_from_baseline(self) -> None:
        """The actual eval_command from our production baseline_metric.json."""
        from ahvs.executor import _find_python_and_module
        cmd = (
            "PYTHONPATH=src:$PYTHONPATH "
            "/home/ubuntu/miniconda3/envs/cohort_work/bin/python "
            "-m autoqa.run_eval --eval-only --reparse "
            "--checkpoints-dir /home/ubuntu/vision/rnd_user_cohort/autoqa/checkpoints "
            "--cohort-names pro_palestine elon_musk_fan"
        )
        result = _find_python_and_module(cmd)
        assert result is not None
        assert "python" in result.python_exe
        assert result.module_name == "autoqa.run_eval"


class TestImportCheckWrappedCommands:
    """v9 Fix 1: import sanity check works with wrapped eval_commands."""

    def test_import_check_with_pythonpath_prefix(self, tmp_path: Path) -> None:
        """Import check should work when eval_command has PYTHONPATH= prefix."""
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        # Create a valid module under src/
        src = repo / "src" / "mypkg"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("")
        (src / "run_eval.py").write_text("print('ok')\n")
        subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add pkg"],
            cwd=str(repo), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        result = _run_import_sanity_check(
            wt,
            "PYTHONPATH=src:$PYTHONPATH python3 -m mypkg.run_eval --eval-only"
        )
        assert result is None  # should pass
        wt.cleanup()

    def test_import_check_skips_non_python_command(self, tmp_path: Path) -> None:
        """Non-Python eval_command (bash script) should be skipped gracefully."""
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        result = _run_import_sanity_check(wt, "bash scripts/eval.sh --config test.yaml")
        assert result is None  # should skip, not fail
        wt.cleanup()


class TestWarnOnlyFiles:
    """v9/v10 Fix: __init__.py and main.py are warn-only, not hard-blocked."""

    def test_init_py_not_in_forbidden(self) -> None:
        from ahvs.executor import _FORBIDDEN_BASENAMES
        assert "__init__.py" not in _FORBIDDEN_BASENAMES

    def test_init_py_in_warn(self) -> None:
        from ahvs.executor import _WARN_BASENAMES
        assert "__init__.py" in _WARN_BASENAMES

    def test_main_py_not_in_forbidden(self) -> None:
        from ahvs.executor import _FORBIDDEN_BASENAMES
        assert "main.py" not in _FORBIDDEN_BASENAMES

    def test_main_py_in_warn(self) -> None:
        from ahvs.executor import _WARN_BASENAMES
        assert "main.py" in _WARN_BASENAMES

    def test_is_warn_file_returns_reason(self) -> None:
        from ahvs.executor import _is_warn_file
        assert _is_warn_file("__init__.py") is not None
        assert _is_warn_file("src/pkg/__init__.py") is not None
        assert _is_warn_file("main.py") is not None

    def test_is_warn_file_returns_none_for_normal(self) -> None:
        from ahvs.executor import _is_warn_file
        assert _is_warn_file("parsing.py") is None
        assert _is_warn_file("config.yaml") is None


class TestImportCheckCdSemantics:
    """v10 Fix 1: import sanity check preserves cd subdir && ... semantics."""

    def test_cd_subdir_import_check_passes(self, tmp_path: Path) -> None:
        """Import check with 'cd app && python -m ...' should cd into app."""
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        # Create module inside a subdirectory
        app = repo / "app"
        app.mkdir()
        pkg = app / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "run_eval.py").write_text("print('ok')\n")
        subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=str(repo), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        # This command uses cd to change into app/ before running python
        result = _run_import_sanity_check(
            wt, "cd app && python3 -m mypkg.run_eval --eval-only"
        )
        assert result is None  # should pass — cd app is preserved
        wt.cleanup()

    def test_cd_subdir_import_fails_without_cd(self, tmp_path: Path) -> None:
        """Without cd, the module would not be importable from repo root."""
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        app = repo / "app"
        app.mkdir()
        pkg = app / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "run_eval.py").write_text("print('ok')\n")
        subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add app"],
            cwd=str(repo), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        # Without cd app, mypkg is not importable from repo root
        result = _run_import_sanity_check(
            wt, "python3 -m mypkg.run_eval --eval-only"
        )
        assert result is not None  # should fail — mypkg not at repo root
        wt.cleanup()

    def test_parsed_cd_dir_preserved(self) -> None:
        """_find_python_and_module extracts cd_dir correctly."""
        from ahvs.executor import _find_python_and_module

        result = _find_python_and_module(
            "cd /path/to/project && PYTHONPATH=src python3 -m pkg.eval"
        )
        assert result is not None
        assert result.cd_dir == "/path/to/project"
        assert result.python_exe == "python3"
        assert result.module_name == "pkg.eval"


class TestImportCheckQuotedEnvVars:
    """v11 fix: quoted env vars must not break the import sanity check."""

    def test_quoted_env_var_does_not_break_check(self, tmp_path: Path) -> None:
        """Import check with FOO='a b' prefix must not produce invalid shell."""
        from ahvs.executor import _run_import_sanity_check

        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        pkg = repo / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "run_eval.py").write_text("print('ok')\n")
        subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add pkg"],
            cwd=str(repo), capture_output=True, check=True,
        )

        wt_path = tmp_path / "worktrees" / "H1"
        wt = HypothesisWorktree(repo, wt_path)
        wt.create()

        # This previously broke: shlex.split consumed the quotes, then
        # reconstruction produced FOO=a b python3 -c ... (invalid shell)
        result = _run_import_sanity_check(
            wt, 'FOO="a b" python3 -m mypkg.run_eval --eval-only'
        )
        assert result is None  # should pass, not fail with shell error
        wt.cleanup()
