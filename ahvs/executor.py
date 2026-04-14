"""AHVS stage executor — 8 handler functions, one per AHVS stage.

Each handler receives (cycle_dir, config, skill_library, auto_approve)
and returns an AHVSStageResult.

Stage dispatch is performed by execute_ahvs_stage().
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ahvs.config import AHVSConfig
from ahvs.contracts import AHVS_CONTRACTS
from ahvs.context_loader import load_context_bundle
from ahvs.health import run_ahvs_preflight
from ahvs.prompts import AHVSPromptManager
from ahvs.result import HypothesisResult, save_results
from ahvs.skills import SkillLibrary
from ahvs.stages import AHVSStage, StageStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AHVSStageResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AHVSStageResult:
    """Outcome of executing a single AHVS stage."""

    stage: AHVSStage
    status: StageStatus
    artifacts: tuple[str, ...]
    error: str | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_historical_digest(digest: dict) -> str:
    """Format a historical digest dict into compact prompt-friendly text."""
    if not digest or not digest.get("by_hypothesis_type"):
        return "No historical data available yet."

    lines = [
        f"Aggregated from {digest.get('total_lessons', 0)} lessons "
        f"across {digest.get('total_cycles', 0)} older cycles:"
    ]
    for htype, stats in digest["by_hypothesis_type"].items():
        improved = stats.get("improved", 0)
        total = stats.get("total", 0)
        parts = [f"**{htype}**: {total} attempts"]
        if "improved" in stats:
            parts.append(f"{improved}/{total} improved")
        if "avg_delta" in stats:
            parts.append(f"avg Δ{stats['avg_delta']:+.4f}")
        if "best_delta" in stats:
            parts.append(f"best Δ{stats['best_delta']:+.4f}")
        if "kept_count" in stats:
            parts.append(f"{stats['kept_count']} kept")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------


def _make_llm_client(config: AHVSConfig) -> Any:
    """Build an LLM client from AHVSConfig via the AHVS LLM factory.

    Routes through ``ahvs.llm.create_llm_client()`` so that provider
    selection (Anthropic, OpenAI, ACP, etc.) is handled in one place.
    A lightweight shim object bridges AHVSConfig fields to the shape
    expected by the factory.

    When caching is enabled (default), wraps the client in a
    ``CachedClientWrapper`` that deduplicates identical LLM calls.
    ACP clients are NOT cached (they maintain stateful sessions).
    """
    from ahvs.llm import create_llm_client

    shim = _ahvs_config_to_llm_shim(config)
    client = create_llm_client(shim)

    # Wrap with cache for API-based providers (not ACP — stateful sessions)
    if config.cache_enabled and config.llm_provider != "acp":
        from ahvs.llm.cache import LLMCache, CachedClientWrapper, _is_cache_enabled

        if _is_cache_enabled():
            cache_dir = config.repo_path / ".ahvs" / ".llm_cache"
            cache = LLMCache(cache_dir)
            client = CachedClientWrapper(client, cache)
            logger.info("LLM cache enabled: %s", cache_dir)

    return client


class _AcpShim:
    """Mimics ``AcpConfig`` for the shared LLM factory."""

    __slots__ = ("agent", "cwd", "acpx_command", "session_name", "timeout_sec")

    def __init__(self, config: AHVSConfig) -> None:
        self.agent = config.acp_agent
        self.cwd = config.acp_cwd
        self.acpx_command = config.acpx_command
        self.session_name = config.acp_session_name
        self.timeout_sec = config.acp_timeout_sec


class _LlmShim:
    """Mimics ``LlmConfig`` for the shared LLM factory."""

    __slots__ = (
        "provider", "base_url", "api_key", "api_key_env",
        "primary_model", "fallback_models", "acp",
    )

    def __init__(self, config: AHVSConfig) -> None:
        self.provider = config.llm_provider
        self.base_url = config.llm_base_url
        self.api_key = config.llm_api_key
        self.api_key_env = config.llm_api_key_env
        self.primary_model = config.llm_model
        self.fallback_models: tuple[str, ...] = ()
        self.acp = _AcpShim(config)


class _LLMConfigShim:
    """Minimal shim bridging AHVSConfig to the shape ``create_llm_client`` expects."""

    __slots__ = ("llm",)

    def __init__(self, config: AHVSConfig) -> None:
        self.llm = _LlmShim(config)


def _ahvs_config_to_llm_shim(config: AHVSConfig) -> _LLMConfigShim:
    """Convert an AHVSConfig into a shim compatible with ``create_llm_client``."""
    return _LLMConfigShim(config)


# ---------------------------------------------------------------------------
# Type-specific execution strategies (Finding 2, v4)
# ---------------------------------------------------------------------------

_TYPE_EXECUTION_STRATEGIES: dict[str, dict[str, str]] = {
    "prompt_rewrite": {
        "system_addition": (
            "You are implementing a PROMPT REWRITE hypothesis. Focus EXCLUSIVELY on:\n"
            "- System prompt improvements (clarity, specificity, role definition)\n"
            "- Few-shot example selection and formatting\n"
            "- Instruction structure and ordering\n"
            "- Output format specifications\n"
            "DO NOT modify any code logic, algorithms, or configuration parameters.\n"
            "Only modify prompt template files, YAML prompt configs, or string literals "
            "containing prompts."
        ),
        "constraints": "Scope: prompt template files only. No algorithm or config changes.",
        "success_guidance": "Measure output quality improvement from prompt changes alone.",
    },
    "model_comparison": {
        "system_addition": (
            "You are implementing a MODEL COMPARISON hypothesis. Focus on:\n"
            "- Swapping model identifiers (e.g. gpt-4o → claude-sonnet-4-6)\n"
            "- Adjusting API call parameters for the new model\n"
            "- Ensuring output parsing is compatible with the new model's format\n"
            "Minimal code changes — primarily config/model ID swaps."
        ),
        "constraints": "Scope: model IDs and API parameters only. No algorithm redesign.",
        "success_guidance": "Compare output quality and cost across models.",
    },
    "config_change": {
        "system_addition": (
            "You are implementing a CONFIG CHANGE hypothesis. Focus on:\n"
            "- Hyperparameter tuning: temperature, top_p, top_k, max_tokens\n"
            "- Retrieval parameters: chunk_size, chunk_overlap, top_k results\n"
            "- Embedding dimensions, batch sizes, similarity thresholds\n"
            "Only modify configuration files or parameter values. No new algorithms."
        ),
        "constraints": "Scope: config files and numeric parameters only.",
        "success_guidance": "Measure metric sensitivity to parameter changes.",
    },
    "dspy_optimize": {
        "system_addition": (
            "You are implementing a DSPy OPTIMIZATION hypothesis. Focus on:\n"
            "- Creating DSPy modules (Signatures, ChainOfThought, ReAct)\n"
            "- Defining DSPy metrics for optimization\n"
            "- Using DSPy optimizers (BootstrapFewShot, MIPRO, BayesianSignatureOptimizer)\n"
            "- Compiling and evaluating optimized prompts programmatically\n"
            "This requires creating DSPy module code, not just editing prompts."
        ),
        "constraints": "Must use DSPy framework. Create proper DSPy modules.",
        "success_guidance": "Use DSPy's built-in evaluation to measure improvement.",
    },
    "code_change": {
        "system_addition": (
            "You are implementing a CODE CHANGE hypothesis. Focus on REAL algorithmic changes:\n"
            "- New or modified data processing pipelines\n"
            "- Improved retrieval strategies (BM25, hybrid search, query expansion)\n"
            "- Better ranking/reranking functions\n"
            "- Enhanced text chunking or preprocessing algorithms\n"
            "- New post-processing or answer synthesis logic\n"
            "DO NOT just change prompts or config values. Write actual code that implements "
            "a different algorithm or data flow."
        ),
        "constraints": "Must modify actual code logic, not just prompts or configs.",
        "success_guidance": "Demonstrate algorithmic improvement via metric change.",
    },
    "architecture_change": {
        "system_addition": (
            "You are implementing an ARCHITECTURE CHANGE hypothesis. Focus on structural redesign:\n"
            "- New retrieval pipeline components (re-rankers, query routers, caching layers)\n"
            "- Hybrid search architectures (combining vector + keyword retrieval)\n"
            "- Multi-stage processing pipelines\n"
            "- New modules for context compression, document filtering, or answer verification\n"
            "- Agent-based architectures with tool use\n"
            "Create NEW modules/classes, not just modify existing ones. Think about data flow "
            "and component interfaces."
        ),
        "constraints": "Must introduce new modules or restructure component boundaries.",
        "success_guidance": "Demonstrate architectural improvement via metric and design clarity.",
    },
    "multi_llm_judge": {
        "system_addition": (
            "You are implementing a MULTI-LLM JUDGE hypothesis. Focus on:\n"
            "- Multi-model consensus: query N models, aggregate answers\n"
            "- Judge-based evaluation: use one LLM to evaluate another's output\n"
            "- Debate/critique patterns: models critique each other's answers\n"
            "- Confidence-weighted selection across model outputs\n"
            "Write evaluation/judging code that coordinates multiple LLM calls."
        ),
        "constraints": "Must implement multi-model coordination, not single-model changes.",
        "success_guidance": "Measure quality improvement from multi-model approach.",
    },
}


# ---------------------------------------------------------------------------
# Forbidden file patterns — eval-harness protection
# ---------------------------------------------------------------------------

# Basenames that Claude Code is NOT allowed to write to the worktree.
# These are eval-harness infrastructure files.  Modifying them consistently
# breaks the eval pipeline (circular imports, broken argparse, missing entry
# points).  The prompt tells Claude Code not to touch them; this filter
# enforces it at the framework level.
#
# Hard-blocked: always rejected.  Only files that are definitively
# eval-harness entry points — modifying these has caused eval pipeline
# crashes in every observed case.
_FORBIDDEN_BASENAMES: frozenset[str] = frozenset({
    "run_eval.py",
    "evaluation.py",
})

# Warn-only: logged as a warning but still applied.  These files CAN
# be legitimately modified by some hypotheses, but have historically
# been a source of eval-harness corruption.  The warning helps operators
# diagnose failures without blocking valid changes.
_WARN_BASENAMES: frozenset[str] = frozenset({
    "__init__.py",
    "main.py",
})

# Patterns for files that are always standalone scripts / test files
# and should never be applied to the worktree.
_FORBIDDEN_PREFIXES: tuple[str, ...] = ("test_", "tests_")
_FORBIDDEN_SUFFIXES: tuple[str, ...] = ("_test.py",)


def _is_forbidden_file(relpath: str, *, repo_has_src: bool = False) -> str | None:
    """Return a reason string if *relpath* should be hard-blocked, else None.

    Files in ``_WARN_BASENAMES`` are NOT blocked — use
    ``_is_warn_file()`` to check those separately.

    When *repo_has_src* is True, root-level ``.py`` files (no directory
    component) are blocked because Claude Code should be modifying files
    inside the ``src/`` subtree, not creating standalone scripts at the
    repo root.
    """
    from pathlib import PurePosixPath

    p = PurePosixPath(relpath)
    basename = p.name

    if basename in _FORBIDDEN_BASENAMES:
        return f"forbidden eval-harness file: {basename}"

    if any(basename.startswith(pfx) for pfx in _FORBIDDEN_PREFIXES):
        return f"test file not allowed in worktree: {basename}"

    if any(basename.endswith(s) for s in _FORBIDDEN_SUFFIXES):
        return f"test file not allowed in worktree: {basename}"

    # Block root-level .py files when the repo has a src/ directory.
    # Claude Code frequently writes to e.g. "parsing.py" at the repo root
    # instead of "src/autoqa/parsing.py".  These root-level files are
    # never executed by the eval pipeline and waste the hypothesis.
    if repo_has_src and basename.endswith(".py") and p.parent == PurePosixPath("."):
        return (
            f"root-level .py file rejected — repo has src/ directory, "
            f"write to the correct path (e.g. src/.../{basename})"
        )

    return None


def _remap_root_file(basename: str, repo_path: "Path") -> str | None:
    """Try to find a unique ``src/`` file matching *basename*.

    When Claude Code writes ``parsing.py`` at the repo root but
    ``src/autoqa/parsing.py`` exists, return the correct relative path
    so the framework can auto-remap instead of blocking.

    Returns the remapped relative path (e.g. ``src/autoqa/parsing.py``)
    if exactly one match is found under ``src/``, else ``None``.
    """
    src_dir = repo_path / "src"
    if not src_dir.is_dir():
        return None
    matches = [
        p for p in src_dir.rglob(basename)
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".git" not in p.parts
    ]
    if len(matches) == 1:
        return str(matches[0].relative_to(repo_path))
    return None


# ---------------------------------------------------------------------------
# Secret-scan preflight
# ---------------------------------------------------------------------------

# File patterns that may contain secrets — warn before granting agent access
_SECRET_FILE_PATTERNS: tuple[str, ...] = (
    ".env", ".env.*", "*.secret", "*.pem", "*.key",
    "credentials.json", "service_account.json", "secrets.yaml",
    "secrets.yml", ".netrc", ".npmrc", ".pypirc",
)


def _scan_for_secrets(target_dir: Path) -> list[str]:
    """Quick scan for files that may contain secrets.

    Returns a list of warning strings (empty if clean).
    """
    warnings: list[str] = []
    for pattern in _SECRET_FILE_PATTERNS:
        for match in target_dir.glob(pattern):
            if match.is_file() and ".git" not in match.parts:
                warnings.append(str(match.relative_to(target_dir)))
    # Also check for common secret patterns in tracked .env* files
    for envfile in target_dir.glob(".env*"):
        if envfile.is_file() and envfile.stat().st_size > 0:
            warnings.append(str(envfile.relative_to(target_dir)))
    return sorted(set(warnings))


def _generate_files_with_claude_code(
    hyp: dict[str, Any],
    hyp_id: str,
    config: "AHVSConfig",
    baseline: dict[str, Any],
    work_dir: Path,
    worktree_path: Path | None = None,
) -> dict[str, str]:
    """Use Claude Code CLI to make targeted file edits for a hypothesis.

    When *worktree_path* is provided, Claude Code runs inside the isolated
    worktree so the live target repo is never modified.  Changes are captured
    via ``git diff`` in the worktree, then reverted — the safety pipeline
    (forbidden-file filter, syntax validation, AST splice) applies them back.

    When *worktree_path* is ``None`` (no-worktree fallback), Claude Code runs
    against the live repo and changes are captured/reverted there.

    All modified file types are captured (not just ``*.py``), so prompt
    templates, config files, and other non-Python edits are included.

    Returns a dict mapping relative file paths → modified file content.
    """
    import json as _json
    import shutil
    import subprocess

    claude_bin = shutil.which("claude")
    if claude_bin is None:
        raise RuntimeError(
            "claude CLI not found — install Claude Code"
        )

    # Determine the directory Claude Code will operate in.
    # Prefer the worktree (isolated); fall back to live repo.
    target_dir = worktree_path if worktree_path is not None else config.repo_path

    # Secret-scan preflight: warn if target dir contains potential secrets
    secret_warnings = _scan_for_secrets(target_dir)
    if secret_warnings:
        logger.warning(
            "%s: potential secret files detected in target directory: %s. "
            "Claude Code will have read access to these files.",
            hyp_id, secret_warnings,
        )
        print(
            f"[AHVS] WARNING: {hyp_id}: potential secret files in target: "
            f"{', '.join(secret_warnings)}"
        )

    description = hyp.get("description", "")
    hyp_type = hyp.get("type", "code_change")
    metric_name = baseline["primary_metric"]
    baseline_value = baseline["value"]
    eval_command = baseline.get("eval_command", "")

    # Collect source files for context (scan the target dir)
    src_files = sorted(
        str(p.relative_to(target_dir))
        for p in target_dir.rglob("*.py")
        if ".ahvs" not in p.parts
        and "__pycache__" not in p.parts
        and ".git" not in p.parts
    )
    # Include the eval entry point even if it lives under .ahvs/ —
    # without this, Claude Code can't see the eval script in the
    # file listing and may search outside the worktree to find it.
    eval_entry = _find_eval_entry_script(eval_command)
    if eval_entry and eval_entry not in src_files:
        eval_path = target_dir / eval_entry
        if eval_path.is_file():
            src_files.append(eval_entry)
    file_listing = "\n".join(f"- {f}" for f in src_files[:30])

    prompt = (
        f"You are making a targeted code change to improve the metric "
        f"'{metric_name}' (current baseline: {baseline_value}).\n\n"
        f"## Hypothesis {hyp_id}\n"
        f"**Type:** {hyp_type}\n"
        f"**Description:** {description}\n\n"
        f"## Target Repository\n"
        f"Path: {target_dir}\n"
        f"**IMPORTANT: All file reads and edits MUST use paths within this "
        f"directory. Do NOT access files outside it.**\n\n"
        f"## Existing source files:\n{file_listing}\n\n"
        f"## Eval command (how the metric is measured):\n"
        f"```\n{eval_command}\n```\n\n"
        f"## Instructions\n"
        f"1. Read the target file(s) mentioned in the hypothesis description.\n"
        f"2. Make ONLY the specific change described. Do NOT rewrite the entire "
        f"file. Do NOT add unrelated code.\n"
        f"3. Preserve all existing functions, classes, constants, and imports "
        f"that you are not explicitly changing.\n"
        f"4. Do NOT modify run_eval.py, evaluation.py, or test files.\n"
        f"5. Do NOT create standalone scripts (main.py, config.py, etc.).\n"
        f"6. After making edits, output a JSON summary of what you changed.\n\n"
        f"Make the change now. Read the file first, then edit it."
    )

    system_prompt = (
        "You are a senior developer making a surgical code edit to an existing "
        "codebase. You use the Read tool to examine files, then the Edit tool "
        "to make minimal, targeted changes. You NEVER rewrite entire files. "
        "You NEVER create new standalone scripts. You modify only what the "
        "hypothesis requires and nothing else."
    )

    logger.info(
        "%s: invoking Claude Code CLI for targeted edit (dir=%s)",
        hyp_id, target_dir,
    )
    print(f"[AHVS] {hyp_id}: using Claude Code for targeted file edit")

    # Ensure the worktree is clean before Claude Code runs — stale edits
    # from a previous run's apply_files() would confuse Claude Code and
    # contaminate the git-status file capture.
    try:
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=str(target_dir), capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=str(target_dir), capture_output=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        pass

    try:
        result = subprocess.run(
            [
                claude_bin, "-p",
                "--model", "opus",
                "--output-format", "json",
                "--system-prompt", system_prompt,
                "--allowedTools", "Read", "Edit", "Glob", "Grep", "Bash(git diff:*)",
                "--dangerously-skip-permissions",
                "--no-session-persistence",
                "--max-budget-usd", "0.50",
                "--add-dir", str(target_dir),
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(target_dir),
        )
    except subprocess.TimeoutExpired:
        logger.error("%s: Claude Code CLI timed out after 300s", hyp_id)
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.error("%s: Claude Code CLI failed: %s", hyp_id, exc)
        return {}

    if result.returncode != 0:
        logger.warning(
            "%s: Claude Code CLI exited %d — stderr: %s",
            hyp_id, result.returncode, result.stderr[:500],
        )

    # Collect ALL modified files by checking git status in the target dir.
    # Claude Code edits files in-place; we capture changes then revert so
    # the safety pipeline (forbidden-file filter, syntax check, splice)
    # can re-apply them with validation.
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True,
            cwd=str(target_dir),
        )
        changed_paths: list[str] = []
        for line in status_result.stdout.strip().splitlines():
            if not line or len(line) < 4:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            status_code = parts[0]
            rel_path = parts[1].strip()
            # Capture modified and added files of ANY type
            if any(c in status_code for c in "MA"):
                changed_paths.append(rel_path)
    except Exception:  # noqa: BLE001
        changed_paths = []

    files: dict[str, str] = {}
    for rel_path in changed_paths:
        full_path = target_dir / rel_path
        if full_path.is_file():
            try:
                files[rel_path] = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(
                    "%s: skipping binary file %s", hyp_id, rel_path,
                )

    # Revert the target dir to clean state — the safety pipeline will
    # re-apply validated files via worktree.apply_files()
    if changed_paths:
        try:
            subprocess.run(
                ["git", "checkout", "--"] + changed_paths,
                cwd=str(target_dir),
                capture_output=True,
            )
            # Also clean up any untracked files Claude Code may have created
            subprocess.run(
                ["git", "clean", "-fd", "--"],
                cwd=str(target_dir),
                capture_output=True,
            )
        except Exception:  # noqa: BLE001
            pass

    logger.info(
        "%s: Claude Code produced %d file(s): %s",
        hyp_id, len(files), list(files.keys()),
    )
    print(f"[AHVS] {hyp_id}: Claude Code edited {len(files)} file(s): {list(files.keys())}")

    # Save Claude Code output for debugging
    log_path = work_dir / "claude_code_output.json"
    log_path.write_text(result.stdout[:50000], encoding="utf-8")

    return files


def _is_warn_file(relpath: str) -> str | None:
    """Return a warning string if *relpath* is a sensitive file, else None.

    These files are applied but a warning is logged so operators can
    diagnose unexpected eval failures.
    """
    from pathlib import PurePosixPath

    basename = PurePosixPath(relpath).name
    if basename in _WARN_BASENAMES:
        return f"sensitive file modified: {basename} (may affect package structure)"
    return None


# ---------------------------------------------------------------------------
# Pre-eval import sanity check
# ---------------------------------------------------------------------------


class _ParsedEvalCommand:
    """Result of parsing an eval_command into its components."""

    __slots__ = ("python_exe", "module_name", "cd_dir", "env_prefix")

    def __init__(
        self,
        python_exe: str,
        module_name: str,
        cd_dir: str | None = None,
        env_prefix: str = "",
    ) -> None:
        self.python_exe = python_exe
        self.module_name = module_name
        self.cd_dir = cd_dir
        # Raw env prefix extracted verbatim from the command string —
        # never reconstructed from parsed tokens, so shell quoting is
        # preserved exactly as the operator wrote it.
        self.env_prefix = env_prefix


def _find_python_and_module(eval_command: str) -> _ParsedEvalCommand | None:
    """Extract the Python executable, module name, and wrapper context.

    Supports common eval_command shapes:
      - ``python -m pkg.mod ...``
      - ``/path/to/python -m pkg.mod ...``
      - ``PYTHONPATH=src:$PYTHONPATH python -m pkg.mod ...``
      - ``cd /path && python -m pkg.mod ...``
      - ``FOO="a b" python -m pkg.mod ...``

    Returns a ``_ParsedEvalCommand`` with the python executable, module
    name, optional ``cd`` target directory, and the raw env prefix string
    (preserved verbatim to avoid shell-quoting issues).
    Returns None if the pattern is not recognizable.
    """
    # Extract cd target from chained commands before splitting
    cd_dir: str | None = None
    python_segment = eval_command

    for sep in ("&&", ";"):
        if sep in eval_command:
            segments = eval_command.split(sep)
            for segment in segments:
                stripped = segment.strip()
                if stripped.startswith("cd "):
                    cd_parts = stripped.split(None, 1)
                    if len(cd_parts) == 2:
                        cd_dir = cd_parts[1].strip()
            for segment in reversed(segments):
                if "-m" in segment:
                    python_segment = segment.strip()
                    break
            break

    # Tokenize to find -m and the python executable
    import shlex
    try:
        parts = shlex.split(python_segment)
    except ValueError:
        parts = python_segment.split()

    # Find -m and the module name
    module_name = None
    m_index = None
    for i, part in enumerate(parts):
        if part == "-m" and i + 1 < len(parts):
            module_name = parts[i + 1]
            m_index = i
            break

    if module_name is None or m_index is None:
        return None

    # Find the Python executable (token before -m, skipping env vars)
    python_exe = None
    if m_index > 0:
        candidate = parts[m_index - 1]
        if "=" not in candidate:
            python_exe = candidate

    if python_exe is None:
        for j in range(m_index - 1, -1, -1):
            tok = parts[j]
            if "=" in tok:
                continue
            if "python" in tok or "/" in tok:
                python_exe = tok
                break

    if python_exe is None:
        return None

    # Extract env prefix VERBATIM from the raw command string.
    # Find where the python executable starts in the raw string and
    # take everything before it.  This preserves shell quoting exactly
    # as the operator wrote it (e.g. FOO="a b" stays as FOO="a b").
    env_prefix = ""
    exe_pos = python_segment.find(python_exe)
    if exe_pos > 0:
        env_prefix = python_segment[:exe_pos]

    return _ParsedEvalCommand(
        python_exe=python_exe,
        module_name=module_name,
        cd_dir=cd_dir,
        env_prefix=env_prefix,
    )


def _find_eval_entry_script(eval_command: str) -> str | None:
    """Extract the Python script name from an eval_command.

    Handles shapes like:
      - ``python run_eval.py --flag``
      - ``cd /path && python run_eval.py --flag``
      - ``python -m pkg.mod`` (returns None — use _find_python_and_module)

    Returns the script filename (e.g. ``run_eval.py``) or None.
    """
    import shlex

    # Strip leading cd
    cmd = eval_command
    for sep in ("&&", ";"):
        if sep in cmd:
            segments = cmd.split(sep)
            for seg in reversed(segments):
                stripped = seg.strip()
                if "python" in stripped.lower():
                    cmd = stripped
                    break
            break

    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()

    # Skip env vars, find python executable, then look for a .py arg
    python_found = False
    for part in parts:
        if not python_found:
            if "python" in part.lower():
                python_found = True
            continue
        # Skip flags
        if part.startswith("-"):
            # -m means module mode — not a script
            if part == "-m":
                return None
            continue
        # First positional arg after python — should be the script
        if part.endswith(".py"):
            return part
        break
    return None


def _analyze_eval_dependencies(
    eval_command: str, repo_path: Path,
) -> tuple[str | None, list[str], list[str]]:
    """Analyze which repo files the eval entry point imports from.

    Returns (entry_point_name, imported_repo_files, all_repo_py_files).
    ``imported_repo_files`` lists repo-relative paths of Python files that
    the eval entry point directly imports.  If the entry point cannot be
    found or parsed, returns (None, [], all_files).
    """
    import ast as _ast

    script_name = _find_eval_entry_script(eval_command)
    if script_name is None:
        return None, [], []

    entry_path = repo_path / script_name
    if not entry_path.is_file():
        return script_name, [], []

    try:
        source = entry_path.read_text(encoding="utf-8")
        tree = _ast.parse(source, filename=script_name)
    except (SyntaxError, OSError):
        return script_name, [], []

    # Collect all top-level module names from import statements
    imported_modules: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    # Check which imported module names correspond to repo-local .py files
    # or packages (directories with __init__.py)
    imported_repo_files: list[str] = []
    for mod_name in sorted(imported_modules):
        # Check for module_name.py
        candidate = repo_path / f"{mod_name}.py"
        if candidate.is_file():
            imported_repo_files.append(f"{mod_name}.py")
            continue
        # Check for package (dir with __init__.py)
        pkg_dir = repo_path / mod_name
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
            imported_repo_files.append(f"{mod_name}/")

    # Gather all repo Python files for comparison
    all_repo_py: list[str] = []
    try:
        all_repo_py = sorted(
            str(p.relative_to(repo_path))
            for p in repo_path.rglob("*.py")
            if ".ahvs" not in p.parts
            and "__pycache__" not in p.parts
            and ".git" not in p.parts
        )
    except Exception:  # noqa: BLE001
        pass

    return script_name, imported_repo_files, all_repo_py


def _run_import_sanity_check(
    worktree: "HypothesisWorktree",
    eval_command: str,
) -> str | None:
    """Verify key modules can still import after Claude Code changes.

    Parses the eval_command to find the Python module being invoked
    (e.g. ``python -m autoqa.run_eval`` → ``autoqa.run_eval``) and
    attempts to import it in the worktree.

    Preserves wrapper semantics:
    - ``cd subdir && python -m ...`` → import check runs from ``subdir``
    - ``PYTHONPATH=src:$PYTHONPATH ...`` → env prefix preserved verbatim
    - ``FOO="a b" python -m ...`` → quoted env vars preserved as-is
    - Non-Python commands (bash scripts) → check is skipped gracefully

    Returns None if the check passes or is skipped, or an error string if
    it fails.
    """
    parsed = _find_python_and_module(eval_command)
    if parsed is None:
        logger.debug(
            "Pre-eval import check: could not parse eval_command — skipping. "
            "cmd=%s", eval_command,
        )
        return None

    # Use the raw env prefix from the original command — no reconstruction
    # from parsed tokens, so shell quoting is preserved exactly.
    env_prefix = parsed.env_prefix
    # Ensure PYTHONPATH=src is present if not already in the prefix
    if "PYTHONPATH=" not in env_prefix:
        env_prefix = f"PYTHONPATH=src:$PYTHONPATH {env_prefix}"

    import_stmt = f"import {parsed.module_name}"
    python_part = f'{env_prefix}{parsed.python_exe} -c "{import_stmt}"'

    # Preserve cd semantics from the original command
    if parsed.cd_dir:
        check_cmd = f"cd {parsed.cd_dir} && {python_part}"
    else:
        check_cmd = python_part

    result = worktree.run_eval_command(check_cmd, timeout=30)
    if result.returncode != 0:
        return (
            f"import {parsed.module_name} failed in worktree after applying "
            f"Claude Code changes. stderr: {result.stderr[:300]}"
        )
    return None


# ---------------------------------------------------------------------------
# Parsing helpers — structured JSON primary, markdown/regex fallback
# ---------------------------------------------------------------------------

# Required fields for each hypothesis dict
_HYPOTHESIS_REQUIRED_FIELDS = {"id", "type", "description"}
_HYPOTHESIS_ALL_FIELDS = {
    "id", "type", "description", "rationale", "estimated_cost", "required_tools",
}

# Required fields for each validation plan dict — require core execution
# fields so a bare {"id": "H1"} doesn't suppress a valid markdown fallback
_PLAN_REQUIRED_FIELDS = {"id", "implementation_approach", "eval_method"}
_PLAN_ALL_FIELDS = {
    "id", "implementation_approach", "eval_method", "skill",
    "success_criterion", "expected_artifacts",
}


def _try_parse_json_block(text: str) -> list[dict] | dict | None:
    """Try to extract and parse a JSON array or object from *text*.

    Looks for ```json fenced blocks first, then bare [...] or {...} blocks.
    Returns None if no valid JSON is found.
    """
    # Try fenced code blocks first
    fenced = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Try to parse the first top-level JSON structure.
    # We look for the first [ or { and try to parse from there.
    # This correctly handles both `[{...}]` arrays and `{...}` objects.
    stripped = text.strip()
    for i, ch in enumerate(stripped):
        if ch in ("[", "{"):
            try:
                return json.loads(stripped[i:])
            except json.JSONDecodeError:
                # Try to find matching bracket by working backwards
                close = "]" if ch == "[" else "}"
                for j in range(len(stripped) - 1, i, -1):
                    if stripped[j] == close:
                        try:
                            return json.loads(stripped[i:j + 1])
                        except json.JSONDecodeError:
                            continue
                break  # Only try the first JSON-starting character
    return None


def _validate_hypothesis(hyp: dict) -> dict | None:
    """Validate and normalise a single hypothesis dict from JSON.

    Returns the normalised dict, or None if required fields are missing.
    """
    if not isinstance(hyp, dict):
        return None
    if not all(hyp.get(f) for f in _HYPOTHESIS_REQUIRED_FIELDS):
        return None
    # Normalise required_tools to list
    rt = hyp.get("required_tools", [])
    if isinstance(rt, str):
        hyp["required_tools"] = [t.strip() for t in rt.split(",") if t.strip()]
    elif not isinstance(rt, list):
        hyp["required_tools"] = []
    # Ensure all expected fields exist with defaults
    for f in _HYPOTHESIS_ALL_FIELDS:
        hyp.setdefault(f, "" if f != "required_tools" else [])
    return hyp


def _validate_plan(plan: dict) -> dict | None:
    """Validate and normalise a single validation plan dict from JSON."""
    if not isinstance(plan, dict):
        return None
    if not all(plan.get(f) for f in _PLAN_REQUIRED_FIELDS):
        return None
    for f in _PLAN_ALL_FIELDS:
        plan.setdefault(f, "")
    return plan


def _parse_hypotheses(text: str) -> list[dict]:
    """Parse hypotheses from LLM output.

    Tries structured JSON first (array of hypothesis objects), then falls
    back to the markdown/regex parser for backward compatibility.
    """
    # --- JSON path ---
    parsed = _try_parse_json_block(text)
    if isinstance(parsed, list) and parsed:
        results = []
        for item in parsed:
            validated = _validate_hypothesis(item)
            if validated is not None:
                results.append(validated)
        if results:
            logger.debug("Parsed %d hypotheses via JSON path", len(results))
            return results

    # --- Markdown/regex fallback ---
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
            ("required_tools", r"\*\*Required Tools:\*\*\s*(.+)"),
        ]:
            m = re.search(pattern, body, re.IGNORECASE)
            hyp[field] = m.group(1).strip() if m else ""
        hyp["required_tools"] = [
            t.strip() for t in hyp.get("required_tools", "").split(",") if t.strip()
        ]
        hypotheses.append(hyp)
    return hypotheses


def _parse_selection(text: str) -> dict:
    """Parse selection from LLM output.

    Tries JSON first, then markdown/regex fallback.
    """
    # --- JSON path ---
    parsed = _try_parse_json_block(text)
    if isinstance(parsed, dict) and "selected" in parsed:
        selected = parsed["selected"]
        if isinstance(selected, list):
            return {
                "selected": list(dict.fromkeys(s for s in selected if isinstance(s, str))),
                "rationale": str(parsed.get("rationale", "")),
            }

    # --- Markdown/regex fallback ---
    selected_ids: list[str] = re.findall(r"\bH\d+\b", text)
    rationale_m = re.search(r"\*\*Rationale:\*\*\s*(.+)", text, re.IGNORECASE)
    return {
        "selected": list(dict.fromkeys(selected_ids)),
        "rationale": rationale_m.group(1).strip() if rationale_m else "",
    }


def _parse_validation_plan(text: str) -> list[dict]:
    """Parse validation plan from LLM output.

    Tries JSON first, then markdown/regex fallback.
    """
    # --- JSON path ---
    parsed = _try_parse_json_block(text)
    if isinstance(parsed, list) and parsed:
        results = []
        for item in parsed:
            validated = _validate_plan(item)
            if validated is not None:
                results.append(validated)
        if results:
            logger.debug("Parsed %d plans via JSON path", len(results))
            return results

    # --- Markdown/regex fallback ---
    plans = []
    blocks = re.split(r"^##\s+(H\d+)\s*$", text, flags=re.MULTILINE)
    it = iter(blocks)
    next(it, None)
    for hyp_id, body in zip(it, it):
        plan: dict = {"id": hyp_id.strip()}
        for field, pattern in [
            ("implementation_approach", r"\*\*Implementation Approach:\*\*\s*(.+)"),
            ("eval_method", r"\*\*Eval Method:\*\*\s*(.+)"),
            ("skill", r"\*\*Skill:\*\*\s*(.+)"),
            ("success_criterion", r"\*\*Success Criterion:\*\*\s*(.+)"),
            ("expected_artifacts", r"\*\*Expected Artifacts:\*\*\s*(.+)"),
        ]:
            m = re.search(pattern, body, re.IGNORECASE)
            plan[field] = m.group(1).strip() if m else ""
        plans.append(plan)
    return plans


def _extract_metric_from_output(raw_output: str, metric_key: str) -> float | None:
    """Try to extract a float metric value from JSON or key:value text output."""

    def _navigate(data: Any, key: str) -> float | None:
        """Navigate a dot-separated key path in a parsed JSON object."""
        parts = key.split(".")
        val = data
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            elif isinstance(val, list) and p.isdigit():
                idx = int(p)
                val = val[idx] if idx < len(val) else None
            else:
                val = None
            if val is None:
                break
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _try_json(data: dict) -> float | None:
        """Try metric_key, then AHVS-standard 'metric_value' fallback."""
        val = _navigate(data, metric_key)
        if val is not None:
            return val
        # AHVS eval scripts emit {"metric_value": <float>} — use as fallback
        return _navigate(data, "metric_value")

    # 1. Try parsing the entire output as a single JSON object (handles
    #    multi-line / indented JSON from eval scripts).
    stripped = raw_output.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            result = _try_json(data)
            if result is not None:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Line-by-line reverse scan for embedded single-line JSON objects
    #    (handles mixed output where JSON is printed among other text).
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                result = _try_json(data)
                if result is not None:
                    return result
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

    # 3. Try "metric_key: 0.82" pattern
    pattern = re.compile(
        rf"{re.escape(metric_key)}\s*[:\s]\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    m = pattern.search(raw_output)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _run_regression_guard(guard_path: Path | None, results_path: Path) -> bool:
    """Run regression guard script. Returns True if passed or not configured."""
    if guard_path is None:
        return True
    if not guard_path.exists():
        logger.error("Regression guard not found: %s", guard_path)
        return False
    try:
        r = subprocess.run(
            [str(guard_path), str(results_path)],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if r.returncode != 0:
            logger.warning(
                "Regression guard FAILED (exit %d): %s",
                r.returncode,
                r.stderr.decode(errors="replace")[:200],
            )
            return False
        return True
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.error("Regression guard failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------


def _ensure_git_repo(repo_path: Path) -> bool:
    """Ensure the target directory is a git repository.

    If the target is not git-backed, initializes a new git repo with all
    existing files committed.  This enables worktree-based isolation for
    non-git codebases without requiring the user to set up git manually.

    Returns True if git was auto-initialized, False if it was already a repo.
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
            cwd=str(repo_path), timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip() == "true":
            return False  # already a git repo
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Not a git repo — initialize one
    logger.info("Target %s is not a git repo — auto-initializing", repo_path)
    print(f"[AHVS] Target is not a git repo — initializing git for worktree isolation")

    subprocess.run(
        ["git", "init"], cwd=str(repo_path),
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "add", "-A"], cwd=str(repo_path),
        capture_output=True, check=True,
    )
    # Commit only if there are staged files (empty dirs have nothing to commit)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(repo_path),
    )
    if status.stdout.strip():
        subprocess.run(
            ["git", "commit", "-m", "ahvs: initial baseline (auto-initialized)"],
            cwd=str(repo_path),
            capture_output=True, check=True,
        )
    else:
        # Create an empty initial commit so HEAD exists for worktrees
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "ahvs: empty initial commit"],
            cwd=str(repo_path),
            capture_output=True, check=True,
        )

    logger.info("Auto-initialized git repo at %s", repo_path)
    print(f"[AHVS] Git initialized with all files committed at HEAD")
    return True


def _execute_setup(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 1: Pre-flight validation and cycle directory setup."""
    cycle_dir.mkdir(parents=True, exist_ok=True)
    (cycle_dir / "tool_runs").mkdir(exist_ok=True)
    (cycle_dir / "worktrees").mkdir(exist_ok=True)

    # Ensure the target is git-backed (required for worktree isolation).
    # If not, auto-initialize a git repo so the full pipeline works.
    try:
        git_was_initialized = _ensure_git_repo(config.repo_path)
    except subprocess.CalledProcessError as exc:
        return AHVSStageResult(
            stage=AHVSStage.AHVS_SETUP,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Failed to initialize git in target directory: {exc.stderr}",
        )

    import os as _os

    api_key = config.llm_api_key or _os.environ.get(config.llm_api_key_env, "")
    report = run_ahvs_preflight(
        baseline_path=config.baseline_path,
        repo_path=config.repo_path,
        regression_guard_path=config.regression_guard_path,
        hypothesis_types=None,  # minimal check at setup time
        llm_api_key=api_key,
        llm_model=config.llm_model,
        llm_base_url=config.llm_base_url,
        llm_provider=config.llm_provider,
        ahvs_config=config,
    )

    if report.overall == "fail":
        failures = [c.detail for c in report.checks if c.status == "fail"]
        return AHVSStageResult(
            stage=AHVSStage.AHVS_SETUP,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Pre-flight failed: {'; '.join(failures)}",
        )

    # Write cycle manifest
    manifest = {
        "cycle_id": cycle_dir.name,
        "question": config.question,
        "repo_path": str(config.repo_path),
        "started_at": _utcnow_iso(),
        "max_hypotheses": config.max_hypotheses,
        "git_auto_initialized": git_was_initialized,
        "regression_guard": str(config.regression_guard_path) if config.regression_guard_path else None,
        "preflight_checks": [
            {"name": c.name, "status": c.status, "detail": c.detail}
            for c in report.checks
        ],
    }
    manifest_path = cycle_dir / "cycle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AHVSStageResult(
        stage=AHVSStage.AHVS_SETUP,
        status=StageStatus.DONE,
        artifacts=("cycle_manifest.json",),
    )


def _execute_context_load(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 2: Load baseline + EvolutionStore → context_bundle.json."""
    try:
        bundle = load_context_bundle(
            repo_path=config.repo_path,
            question=config.question,
            evolution_dir=config.evolution_dir,
            baseline_path=config.baseline_path,
            max_lesson_cycles=config.max_lesson_cycles,
            global_evolution_dir=config.global_evolution_dir,
            enable_cross_project=config.enable_cross_project,
        )
    except (FileNotFoundError, ValueError) as exc:
        return AHVSStageResult(
            stage=AHVSStage.AHVS_CONTEXT_LOAD,
            status=StageStatus.FAILED,
            artifacts=(),
            error=str(exc),
        )

    bundle_path = cycle_dir / "context_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    return AHVSStageResult(
        stage=AHVSStage.AHVS_CONTEXT_LOAD,
        status=StageStatus.DONE,
        artifacts=("context_bundle.json",),
    )


def _execute_hypothesis_gen(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 3: Generate typed hypotheses via LLM."""
    bundle_path = cycle_dir / "context_bundle.json"
    if not bundle_path.exists():
        return AHVSStageResult(
            stage=AHVSStage.AHVS_HYPOTHESIS_GEN,
            status=StageStatus.FAILED,
            artifacts=(),
            error="context_bundle.json not found — run AHVS_CONTEXT_LOAD first",
        )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    baseline = bundle["baseline"]

    prior_lessons_text = (
        "\n".join(f"- {l}" for l in bundle.get("prior_lessons", []))
        or "None recorded yet."
    )
    rejected_text = (
        "\n".join(f"- {r}" for r in bundle.get("rejected_approaches", []))
        or "None recorded yet."
    )
    historical_digest_text = _format_historical_digest(
        bundle.get("historical_digest", {})
    )
    domain_tags_text = ", ".join(bundle.get("domain_tags", ["general"]))

    # Format enriched onboarding context for the prompt
    enriched = bundle.get("enriched_context", {})
    if enriched:
        enriched_lines = []
        for key, val in enriched.items():
            label = key.replace("_", " ").title()
            if isinstance(val, list):
                enriched_lines.append(f"- **{label}:** {', '.join(str(v) for v in val)}")
            elif isinstance(val, dict):
                enriched_lines.append(f"- **{label}:** {json.dumps(val)}")
            else:
                enriched_lines.append(f"- **{label}:** {val}")
        enriched_context_text = "\n".join(enriched_lines)
    else:
        enriched_context_text = "No additional operator context provided."

    pm = AHVSPromptManager(config.prompts_override_path)
    prompt = pm.for_stage(
        "ahvs_hypothesis_gen",
        question=config.question,
        metric_name=baseline["primary_metric"],
        baseline_value=str(baseline["value"]),
        eval_command=baseline.get("eval_command", ""),
        domain_tags=domain_tags_text,
        enriched_context=enriched_context_text,
        prior_lessons=prior_lessons_text,
        rejected_approaches=rejected_text,
        historical_digest=historical_digest_text,
        max_hypotheses=str(config.max_hypotheses),
    )

    try:
        llm = _make_llm_client(config)
        response = llm.chat(
            [{"role": "user", "content": prompt.user}],
            system=prompt.system,
            max_tokens=prompt.max_tokens,
        )
        raw_text = response.content
    except Exception as exc:  # noqa: BLE001
        return AHVSStageResult(
            stage=AHVSStage.AHVS_HYPOTHESIS_GEN,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"LLM call failed: {exc}",
        )

    # Validate and cap
    hypotheses = _parse_hypotheses(raw_text)
    if not hypotheses:
        return AHVSStageResult(
            stage=AHVSStage.AHVS_HYPOTHESIS_GEN,
            status=StageStatus.FAILED,
            artifacts=(),
            error="LLM did not produce any parseable hypotheses",
        )

    # ── Post-filter: remove unmeasurable types under --eval-only ───
    # prompt_rewrite and model_comparison change LLM behaviour, but
    # --eval-only reads frozen checkpoint data so these changes are
    # structurally invisible.  Filter them at generation time instead
    # of wasting a cycle slot.
    _eval_cmd = baseline.get("eval_command", "")
    _UNMEASURABLE_EVAL_ONLY_TYPES = {"prompt_rewrite", "model_comparison"}
    if "--eval-only" in _eval_cmd:
        before_count = len(hypotheses)
        hypotheses = [
            h for h in hypotheses
            if h.get("type", "code_change") not in _UNMEASURABLE_EVAL_ONLY_TYPES
        ]
        dropped = before_count - len(hypotheses)
        if dropped:
            logger.info(
                "Filtered %d unmeasurable hypothesis(es) (type in %s "
                "incompatible with --eval-only eval_command)",
                dropped, _UNMEASURABLE_EVAL_ONLY_TYPES,
            )
            # Rebuild raw_text to exclude filtered hypotheses
            kept_ids = {h["id"] for h in hypotheses}
            raw_text_lines = raw_text.splitlines()
            keep = True
            out_lines_filtered: list[str] = []
            for line in raw_text_lines:
                m = re.match(r"^##\s+(H\d+)\s*$", line)
                if m:
                    keep = m.group(1) in kept_ids
                if keep:
                    out_lines_filtered.append(line)
            raw_text = "\n".join(out_lines_filtered)
        if not hypotheses:
            return AHVSStageResult(
                stage=AHVSStage.AHVS_HYPOTHESIS_GEN,
                status=StageStatus.FAILED,
                artifacts=(),
                error=(
                    "All generated hypotheses were unmeasurable under "
                    "--eval-only mode. Re-run without --eval-only or "
                    "request code_change type hypotheses."
                ),
            )

    if len(hypotheses) > 5:
        logger.warning(
            "LLM generated %d hypotheses — truncating to 5 (hard cap)",
            len(hypotheses),
        )
        raw_text_lines = raw_text.splitlines()
        # Truncate raw_text to the first 5 H-blocks
        kept_ids = {h["id"] for h in hypotheses[:5]}
        keep = True
        out_lines: list[str] = []
        current_id = None
        for line in raw_text_lines:
            m = re.match(r"^##\s+(H\d+)\s*$", line)
            if m:
                current_id = m.group(1)
                keep = current_id in kept_ids
            if keep:
                out_lines.append(line)
        raw_text = "\n".join(out_lines)
        hypotheses = hypotheses[:5]

    hyp_path = cycle_dir / "hypotheses.md"
    hyp_path.write_text(raw_text, encoding="utf-8")

    logger.info(
        "Generated %d hypotheses: %s",
        len(hypotheses),
        [h["id"] for h in hypotheses],
    )

    return AHVSStageResult(
        stage=AHVSStage.AHVS_HYPOTHESIS_GEN,
        status=StageStatus.DONE,
        artifacts=("hypotheses.md",),
    )


def _execute_human_selection(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 4 (GATE): Display hypotheses to operator; record selection.

    Supports three selection modes:

    1. **Pre-specified** (``selection.json`` already exists in ``cycle_dir``):
       Used by conversational/agent-driven callers (e.g. Claude Code) that
       collect the user's choice *before* invoking the executor.  The file
       must contain ``{"selected": ["H1", ...], "rationale": "..."}``.
       When this file is found, the gate honours it and skips all prompts.

    2. **Auto-approve** (``auto_approve=True``):
       Selects every hypothesis.  Used for CI / scripted runs.

    3. **Interactive** (default):
       Prompts the operator on stdin.
    """
    hyp_path = cycle_dir / "hypotheses.md"
    if not hyp_path.exists():
        return AHVSStageResult(
            stage=AHVSStage.AHVS_HUMAN_SELECTION,
            status=StageStatus.FAILED,
            artifacts=(),
            error="hypotheses.md not found",
        )

    hypotheses_text = hyp_path.read_text(encoding="utf-8")
    hypotheses = _parse_hypotheses(hypotheses_text)

    # ── Apply CLI hypothesis modifications (--add/--edit/--insert) ────
    # Track original ID→description mapping so we can remap selection.json
    # after renumbering ops (insert shifts IDs).
    _old_id_to_desc = {h["id"]: h.get("description", "") for h in hypotheses}

    if config.hypothesis_ops:
        from ahvs.hypothesis_ops import apply_ops_and_rewrite
        hypotheses = apply_ops_and_rewrite(
            hypotheses, config.hypothesis_ops, hyp_path,
        )
        hypotheses_text = hyp_path.read_text(encoding="utf-8")
        print(
            f"[AHVS] Applied {len(config.hypothesis_ops)} hypothesis modification(s): "
            f"now {len(hypotheses)} hypotheses"
        )

    # Build old→new ID remap for any renumbering that occurred
    _desc_to_new_id = {h.get("description", ""): h["id"] for h in hypotheses}
    _id_remap = {}
    for old_id, desc in _old_id_to_desc.items():
        new_id = _desc_to_new_id.get(desc)
        if new_id and new_id != old_id:
            _id_remap[old_id] = new_id

    all_ids = [h["id"] for h in hypotheses]

    # ── Mode 1: Pre-specified selection.json ──────────────────────────
    pre_sel_path = cycle_dir / "selection.json"
    if pre_sel_path.exists():
        try:
            pre_sel = json.loads(pre_sel_path.read_text(encoding="utf-8"))
            # Remap IDs if renumbering ops shifted them
            raw_selected = [s.upper() for s in pre_sel.get("selected", [])]
            pre_selected = [
                _id_remap.get(s, s)
                for s in raw_selected
                if _id_remap.get(s, s) in all_ids
            ]
        except (json.JSONDecodeError, TypeError):
            pre_selected = []

        if pre_selected:
            rationale = pre_sel.get("rationale", "caller-provided selection")
            approved_by = pre_sel.get("approved_by", "caller")
            selected = pre_selected
            print(
                f"\n[AHVS] Pre-specified selection: {', '.join(selected)} "
                f"(approved_by={approved_by})"
            )
            return _finalize_selection(
                cycle_dir, config, hypotheses, selected, rationale,
                approved_by, auto_approve,
            )

    # ── Mode 2: Auto-approve ──────────────────────────────────────────
    if auto_approve:
        selected = all_ids
        rationale = "auto-approve: all hypotheses selected"
        print(
            f"\n[AHVS] Auto-approve: selecting all {len(selected)} hypothesis/hypotheses: "
            f"{', '.join(selected)}"
        )
        return _finalize_selection(
            cycle_dir, config, hypotheses, selected, rationale,
            "auto", auto_approve,
        )

    # ── Mode 3: Interactive gate ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("AHVS HUMAN SELECTION — Cycle Gate")
    print("=" * 70)
    print(f"\nQuestion: {config.question}\n")
    print(hypotheses_text)
    print("=" * 70)
    print(f"Available hypotheses: {', '.join(all_ids)}")
    print("Enter IDs to run (e.g. 'H1 H2', 'all', or 'none' to abort):")

    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = "none"

    if raw.lower() == "none" or not raw:
        return AHVSStageResult(
            stage=AHVSStage.AHVS_HUMAN_SELECTION,
            status=StageStatus.FAILED,
            artifacts=(),
            error="Operator aborted cycle at hypothesis selection",
        )
    if raw.lower() == "all":
        selected = all_ids
    else:
        selected = [s.strip() for s in re.findall(r"H\d+", raw, re.IGNORECASE)]
        selected = [s.upper() for s in selected if s.upper() in all_ids]
        if not selected:
            return AHVSStageResult(
                stage=AHVSStage.AHVS_HUMAN_SELECTION,
                status=StageStatus.FAILED,
                artifacts=(),
                error=f"No valid hypothesis IDs found in input: '{raw}'",
            )

    print(f"\nRationale for selection (optional, press Enter to skip):")
    try:
        rationale = input("> ").strip() or "Operator selection"
    except (EOFError, KeyboardInterrupt):
        rationale = "Operator selection"

    print(f"\n[AHVS] Selected: {', '.join(selected)}")

    return _finalize_selection(
        cycle_dir, config, hypotheses, selected, rationale,
        "operator", auto_approve,
    )


def _finalize_selection(
    cycle_dir: Path,
    config: AHVSConfig,
    hypotheses: list[dict],
    selected: list[str],
    rationale: str,
    approved_by: str,
    auto_approve: bool,
) -> AHVSStageResult:
    """Run pre-flight checks, write selection artifacts, return stage result."""
    # Run secondary pre-flight for the selected hypothesis types
    selected_types = [
        h["type"] for h in hypotheses if h["id"] in selected
    ]
    if selected_types:
        tool_report = run_ahvs_preflight(
            baseline_path=config.baseline_path,
            repo_path=config.repo_path,
            regression_guard_path=config.regression_guard_path,
            hypothesis_types=selected_types,
            skip_llm_check=True,  # LLM connectivity already verified at Stage 1
        )
        failed_checks = [c for c in tool_report.checks if c.status == "fail"]
        if failed_checks:
            details = "; ".join(c.detail for c in failed_checks)
            print(f"\n[AHVS] WARNING: Tool pre-flight issues: {details}")
            if auto_approve or approved_by == "caller":
                logger.warning(
                    "%s mode: continuing despite tool pre-flight failures",
                    approved_by,
                )
            else:
                print("[AHVS] Some hypotheses may fail at execution. Continue anyway? (y/N)")
                try:
                    confirm = input("> ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = "n"
                if confirm not in ("y", "yes"):
                    return AHVSStageResult(
                        stage=AHVSStage.AHVS_HUMAN_SELECTION,
                        status=StageStatus.FAILED,
                        artifacts=(),
                        error=f"Operator aborted: tool pre-flight failed: {details}",
                    )

    selection = {
        "selected": selected,
        "rationale": rationale,
        "approved_by": approved_by,
        "timestamp": _utcnow_iso(),
    }
    sel_path = cycle_dir / "selection.md"
    sel_path.write_text(
        f"# Selected Hypotheses\n\n"
        f"**Selected:** {', '.join(selected)}\n\n"
        f"**Rationale:** {rationale}\n\n"
        f"**Approved By:** {selection['approved_by']}\n\n"
        f"**Timestamp:** {selection['timestamp']}\n",
        encoding="utf-8",
    )
    # Also write JSON for easy machine parsing
    (cycle_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    return AHVSStageResult(
        stage=AHVSStage.AHVS_HUMAN_SELECTION,
        status=StageStatus.DONE,
        artifacts=("selection.md",),
    )


def _execute_validation_plan(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 5: Generate per-hypothesis implementation + eval plan via LLM."""
    for required in ("context_bundle.json", "selection.json", "hypotheses.md"):
        if not (cycle_dir / required).exists():
            return AHVSStageResult(
                stage=AHVSStage.AHVS_VALIDATION_PLAN,
                status=StageStatus.FAILED,
                artifacts=(),
                error=f"Missing required artifact: {required}",
            )

    bundle = json.loads((cycle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    selection = json.loads((cycle_dir / "selection.json").read_text(encoding="utf-8"))
    hypotheses_text = (cycle_dir / "hypotheses.md").read_text(encoding="utf-8")
    hypotheses = _parse_hypotheses(hypotheses_text)

    selected_ids = set(selection["selected"])
    selected_hyps = [h for h in hypotheses if h["id"] in selected_ids]

    # Build selected hypotheses text for prompt
    selected_text = "\n\n".join(
        f"### {h['id']}\n"
        f"**Type:** {h['type']}\n"
        f"**Description:** {h['description']}\n"
        f"**Rationale:** {h['rationale']}"
        for h in selected_hyps
    )

    # Determine available skills for context block
    available_tools = skill_library.detect_available_tools()
    all_applicable_skills: list = []
    for h in selected_hyps:
        skills = skill_library.for_hypothesis_type(h["type"], available_tools)
        for s in skills:
            if s not in all_applicable_skills:
                all_applicable_skills.append(s)
    skills_block = skill_library.to_context_block(all_applicable_skills) or "No skills available."

    baseline = bundle["baseline"]

    # ── Eval dependency analysis for validation plan grounding ─────────
    # Without this, the LLM generates plans that modify files the eval
    # never imports (e.g. Train_model.py when run_eval.py is
    # self-contained), dooming the hypothesis before execution begins.
    eval_dep_context = ""
    eval_command = baseline.get("eval_command", "")
    if eval_command:
        entry_name, eval_deps, _all_py = _analyze_eval_dependencies(
            eval_command, config.repo_path,
        )
        if entry_name is not None:
            if eval_deps:
                deps_list = ", ".join(f"`{d}`" for d in eval_deps)
                eval_dep_context = (
                    f"## Eval Dependency Graph — CRITICAL\n"
                    f"The eval entry point `{entry_name}` imports from "
                    f"these repo-local files: {deps_list}.\n"
                    f"**Only changes to these files (or files they "
                    f"transitively import) will affect the measured "
                    f"metric.** Plans that modify other files will have "
                    f"no effect.\n\n"
                )
            else:
                eval_dep_context = (
                    f"## ⚠ SELF-CONTAINED EVAL — PLANS MUST TARGET `{entry_name}` ⚠\n"
                    f"The eval entry point `{entry_name}` does NOT "
                    f"import from any other repo source files. It is "
                    f"**self-contained** with its own training loop, "
                    f"model setup, and evaluation logic.\n\n"
                    f"**Changes to other Python files (e.g. "
                    f"`Train_model.py`) will have ZERO effect on the "
                    f"metric.**\n\n"
                    f"The framework will safely apply changes to "
                    f"`{entry_name}` via AST splice (merging only "
                    f"changed functions/classes). Plans MUST instruct "
                    f"the code agent to output modified functions from "
                    f"`{entry_name}` directly — NOT to modify other "
                    f"files like `Train_model.py`.\n\n"
                )

    pm = AHVSPromptManager(config.prompts_override_path)
    prompt = pm.for_stage(
        "ahvs_validation_plan",
        question=config.question,
        metric_name=baseline["primary_metric"],
        baseline_value=str(baseline["value"]),
        eval_command=eval_command,
        selected_hypotheses_text=selected_text,
        available_skills_block=skills_block,
        eval_dependency_context=eval_dep_context,
    )

    try:
        llm = _make_llm_client(config)
        response = llm.chat(
            [{"role": "user", "content": prompt.user}],
            system=prompt.system,
            max_tokens=prompt.max_tokens,
        )
        plan_text = response.content
    except Exception as exc:  # noqa: BLE001
        return AHVSStageResult(
            stage=AHVSStage.AHVS_VALIDATION_PLAN,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"LLM call failed: {exc}",
        )

    plan_path = cycle_dir / "validation_plan.md"
    plan_path.write_text(plan_text, encoding="utf-8")

    return AHVSStageResult(
        stage=AHVSStage.AHVS_VALIDATION_PLAN,
        status=StageStatus.DONE,
        artifacts=("validation_plan.md",),
    )


def _write_eager_lesson(
    result: HypothesisResult,
    config: AHVSConfig,
    cycle_id: str,
) -> None:
    """Write a lesson for a single hypothesis result immediately after measurement.

    This ensures lessons survive even if the cycle crashes before Stage 7.
    Stage 7 skips re-writing lessons that already exist (dedup by run_id +
    description prefix happens at query time via compact()).
    """
    from ahvs.evolution import EvolutionStore, LessonEntry, LessonCategory

    try:
        store = EvolutionStore(config.evolution_dir)
        now = _utcnow_iso()
        metric_name = result.primary_metric

        # Common structured fields for all branches
        _structured = dict(
            hypothesis_id=result.hypothesis_id,
            hypothesis_type=result.hypothesis_type,
            metric_name=metric_name,
            metric_baseline=result.baseline_value,
            metric_after=result.metric_value,
            metric_delta=result.delta,
            eval_method=result.eval_method,
        )

        if result.improved:
            lesson = LessonEntry(
                stage_name="ahvs_execution",
                stage_num=6,
                category=LessonCategory.EXPERIMENT,
                severity="info",
                description=(
                    f"[{cycle_id}] {result.hypothesis_id} ({result.hypothesis_type}) improved "
                    f"{metric_name} by {result.delta:+.4f} ({result.delta_pct:+.1f}%). "
                    f"Eval: {result.eval_method}."
                ),
                timestamp=now,
                run_id=cycle_id,
                cycle_status="partial",
                **_structured,
            )
        elif result.error:
            lesson = LessonEntry(
                stage_name="ahvs_execution",
                stage_num=6,
                category=LessonCategory.EXPERIMENT,
                severity="warning",
                description=(
                    f"[{cycle_id}] {result.hypothesis_id} ({result.hypothesis_type}) "
                    f"FAILED: {result.error[:150]}"
                ),
                timestamp=now,
                run_id=cycle_id,
                cycle_status="partial",
                **_structured,
            )
        elif result.measurement_status != "measured":
            lesson = LessonEntry(
                stage_name="ahvs_execution",
                stage_num=6,
                category=LessonCategory.EXPERIMENT,
                severity="warning",
                description=(
                    f"[{cycle_id}] {result.hypothesis_id} ({result.hypothesis_type}) "
                    f"measurement failed ({result.measurement_status}). "
                    f"Infrastructure issue — hypothesis was not tested."
                ),
                timestamp=now,
                run_id=cycle_id,
                cycle_status="partial",
                **_structured,
            )
        else:
            lesson = LessonEntry(
                stage_name="ahvs_execution",
                stage_num=6,
                category=LessonCategory.EXPERIMENT,
                severity="info",
                description=(
                    f"[{cycle_id}] {result.hypothesis_id} ({result.hypothesis_type}) did not improve "
                    f"{metric_name} (Δ{result.delta:+.4f}). Rejected approach."
                ),
                timestamp=now,
                run_id=cycle_id,
                cycle_status="partial",
                **_structured,
            )

        store.append(lesson)
        logger.info("Eager lesson written for %s", result.hypothesis_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Eager lesson write failed (non-fatal): %s", exc)


def _execute_hypotheses(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 6: Invoke Claude Code for each selected hypothesis; collect results."""
    for required in ("validation_plan.md", "selection.json", "context_bundle.json"):
        if not (cycle_dir / required).exists():
            return AHVSStageResult(
                stage=AHVSStage.AHVS_EXECUTION,
                status=StageStatus.FAILED,
                artifacts=(),
                error=f"Missing required artifact: {required}",
            )

    selection = json.loads((cycle_dir / "selection.json").read_text(encoding="utf-8"))
    bundle = json.loads((cycle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    plan_text = (cycle_dir / "validation_plan.md").read_text(encoding="utf-8")
    hypotheses_text = (cycle_dir / "hypotheses.md").read_text(encoding="utf-8")

    plans = _parse_validation_plan(plan_text)
    hypotheses = _parse_hypotheses(hypotheses_text)
    hyp_by_id = {h["id"]: h for h in hypotheses}
    plan_by_id = {p["id"]: p for p in plans}

    baseline = bundle["baseline"]
    available_tools = skill_library.detect_available_tools()

    from ahvs.worktree import HypothesisWorktree

    results_and_worktrees: list[tuple[HypothesisResult, HypothesisWorktree | None]] = []

    for hyp_id in selection["selected"]:
        hyp = hyp_by_id.get(hyp_id)
        plan = plan_by_id.get(hyp_id)

        if not hyp or not plan:
            results_and_worktrees.append((
                HypothesisResult.make_error(
                    hypothesis_id=hyp_id,
                    hypothesis_type="unknown",
                    primary_metric=baseline["primary_metric"],
                    baseline_value=float(baseline["value"]),
                    error=f"No hypothesis or plan found for {hyp_id}",
                ),
                None,
            ))
            continue

        result, worktree = _run_single_hypothesis(
            hyp_id=hyp_id,
            hyp=hyp,
            plan=plan,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_library,
            baseline=baseline,
            available_tools=available_tools,
        )
        results_and_worktrees.append((result, worktree))

        # Eager lesson write — survives even if cycle crashes before Stage 7
        _write_eager_lesson(result, config, cycle_dir.name)

        measurement_tag = ""
        if result.measurement_status == "extraction_failed":
            measurement_tag = " [MEASUREMENT FAILED]"
        elif result.measurement_status == "sandbox_error":
            measurement_tag = " [SANDBOX ERROR]"
        print(
            f"[AHVS] {hyp_id}: {baseline['primary_metric']}="
            f"{result.metric_value:.4f} (Δ{result.delta:+.4f}) "
            f"{'✓ improved' if result.improved else '✗ no improvement'}"
            + measurement_tag
            + (f" | ERROR: {result.error}" if result.error else "")
        )

    # ── Keep/revert: save patches, keep best, clean up losers ────────
    results = [r for r, _ in results_and_worktrees]
    improved = [(r, wt) for r, wt in results_and_worktrees if r.improved and wt]
    best_result = (
        max((r for r, _ in improved), key=lambda r: r.delta) if improved else None
    )

    for result, worktree in results_and_worktrees:
        if worktree is None:
            continue
        # Save patch for every hypothesis (audit trail)
        work_dir = cycle_dir / "tool_runs" / result.hypothesis_id
        patch_path = work_dir / f"{result.hypothesis_id}.patch"
        worktree.save_patch(patch_path)
        result.patch_path = str(patch_path.relative_to(cycle_dir))

        if result is best_result:
            result.kept = True
            result.worktree_path = str(worktree.worktree_path)
        else:
            worktree.cleanup()

    save_results(results, cycle_dir / "results.json")

    return AHVSStageResult(
        stage=AHVSStage.AHVS_EXECUTION,
        status=StageStatus.DONE,
        artifacts=("results.json", "tool_runs/"),
    )


def _run_single_hypothesis(
    hyp_id: str,
    hyp: dict,
    plan: dict,
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    baseline: dict,
    available_tools: set[str],
) -> tuple[HypothesisResult, "HypothesisWorktree | None"]:
    """Execute one hypothesis via Claude Code CLI and return (result, worktree).

    The worktree object is returned so the caller can manage keep/revert.
    """
    from ahvs.worktree import HypothesisWorktree

    work_dir = cycle_dir / "tool_runs" / hyp_id
    work_dir.mkdir(parents=True, exist_ok=True)

    hyp_type = hyp.get("type", "code_change")
    metric_name = baseline["primary_metric"]
    baseline_value = float(baseline["value"])

    # Build skill context for this hypothesis type
    skills = skill_library.for_hypothesis_type(hyp_type, available_tools)
    skill_context = skill_library.to_context_block(skills)

    # Build Claude Code problem statement
    eval_command = baseline.get("eval_command", "")

    # Collect repo source files for the Claude Code prompt
    repo_files_listing = ""
    try:
        src_files = sorted(
            str(p.relative_to(config.repo_path))
            for p in config.repo_path.rglob("*.py")
            if ".ahvs" not in p.parts
            and "__pycache__" not in p.parts
            and ".git" not in p.parts
        )
        if src_files:
            repo_files_listing = (
                "### Existing source files (modify these, do NOT create new standalone scripts):\n"
                + "\n".join(f"- `{f}`" for f in src_files[:30])
                + "\n\n"
            )
    except Exception:  # noqa: BLE001
        pass

    # Extract public API signatures from key source files so the Claude Code
    # knows which functions/classes/constants MUST be preserved when rewriting
    # a module.  Without this, Claude Code may drop functions that other
    # modules import, causing ImportError at eval time.
    api_signatures = ""
    try:
        import ast as _ast

        api_lines: list[str] = []
        for src_rel in src_files[:15]:
            src_path = config.repo_path / src_rel
            if not src_path.is_file():
                continue
            try:
                tree = _ast.parse(src_path.read_text(encoding="utf-8"), filename=src_rel)
            except SyntaxError:
                continue
            exports: list[str] = []
            for node in _ast.iter_child_nodes(tree):
                if isinstance(node, _ast.FunctionDef | _ast.AsyncFunctionDef):
                    args = ", ".join(a.arg for a in node.args.args)
                    exports.append(f"  def {node.name}({args})")
                elif isinstance(node, _ast.ClassDef):
                    exports.append(f"  class {node.name}")
                elif isinstance(node, _ast.Assign):
                    for target in node.targets:
                        if isinstance(target, _ast.Name) and target.id.isupper():
                            exports.append(f"  {target.id} = ...")
            if exports:
                api_lines.append(f"#### `{src_rel}`")
                api_lines.extend(exports)
        if api_lines:
            api_signatures = (
                "### Public API of existing modules (MUST preserve these names when modifying):\n"
                "When you rewrite a file, you MUST keep all functions, classes, and\n"
                "constants listed below. Other modules import them — removing or\n"
                "renaming them will cause ImportError at eval time.\n\n"
                + "\n".join(api_lines)
                + "\n\n"
            )
    except Exception:  # noqa: BLE001
        pass

    repo_grounding = (
        f"## Repo-Grounded Execution — CRITICAL INSTRUCTIONS\n"
        f"Your generated files will be applied to a git worktree of the target\n"
        f"repository. Use paths **relative to the repo root**.\n"
        f"Repository root: {config.repo_path}\n\n"
        f"**IMPORTANT: You MUST modify EXISTING repo source files in-place.**\n"
        f"Do NOT create standalone scripts (main.py, config.py, evaluation.py, etc.).\n"
        f"The eval_command runs the EXISTING pipeline — only changes to existing\n"
        f"source files will be picked up. New standalone files are NEVER executed.\n\n"
        f"## PROTECTED FILES — READ CAREFULLY\n"
        f"The following files are part of the eval harness infrastructure.\n"
        f"The framework enforces protection rules on these files:\n\n"
        f"**BLOCKED (changes will be rejected by the framework):**\n"
        f"- `run_eval.py` or any file matching `**/run_eval.py` (eval entry point)\n"
        f"- `evaluation.py` or any file matching `**/evaluation.py`\n"
        f"- Any file matching `test_*.py` or `*_test.py` (test files)\n\n"
        f"**SENSITIVE (changes are applied but flagged for review):**\n"
        f"- `__init__.py` — modify only if your hypothesis requires package structure changes\n"
        f"- `main.py` — modify only if it is part of the actual eval pipeline\n\n"
        f"If you need to change how evaluation works, modify the **processing logic**\n"
        f"(parsing, prompts, config) — NOT the eval entry points.\n\n"
        f"When generating files, use the EXACT relative path of the existing file\n"
        f"you want to modify (e.g. `src/autoqa/parsing.py`, not `parsing.py` or\n"
        f"`my_parsing_fix.py`).\n\n"
        f"## PARTIAL OUTPUT MODE — CRITICAL\n"
        f"Do NOT rewrite the entire file. Output ONLY the functions, methods,\n"
        f"classes, or constants you are modifying or adding.\n"
        f"The framework will splice your changes into the existing file at the\n"
        f"correct locations (matched by function/class name).\n\n"
        f"Rules for partial output:\n"
        f"- Output ONLY the complete definition of each function/method/class you\n"
        f"  are changing. Do NOT include unchanged functions.\n"
        f"- If you modify a method inside a class, output the ENTIRE class\n"
        f"  definition (with all its methods, including unchanged ones).\n"
        f"- Include any NEW import statements your changes require at the top.\n"
        f"  Do NOT repeat existing imports.\n"
        f"- For new top-level constants or variables, include the assignment.\n"
        f"- Do NOT include the rest of the file (unchanged functions, module\n"
        f"  docstrings, etc.) — only your changes.\n\n"
        f"Example — if you only need to change the `classify_user` function in\n"
        f"`src/autoqa/parsing.py`:\n"
        f"```\n"
        f"# New imports your change needs (if any)\n"
        f"from collections import Counter\n\n"
        f"def classify_user(user_data, threshold=0.7):\n"
        f"    # Your complete modified implementation\n"
        f"    ...\n"
        f"```\n\n"
        f"{repo_files_listing}"
        f"{api_signatures}"
    )
    eval_section = ""
    if eval_command:
        eval_section = (
            f"## Eval Command\n"
            f"After your files are applied, this command will be run in the\n"
            f"worktree to measure the result:\n"
            f"```\n{eval_command}\n```\n"
            f"**This command runs the existing pipeline.** Only modifications to\n"
            f"existing source files (listed above) will affect the result.\n"
            f"Standalone scripts you create will NOT be executed.\n\n"
        )

        # Analyze eval entry point dependencies to guide Claude Code
        entry_name, eval_deps, _all_py = _analyze_eval_dependencies(
            eval_command, config.repo_path,
        )
        _eval_is_self_contained = False
        if entry_name is not None:
            if eval_deps:
                deps_list = ", ".join(f"`{d}`" for d in eval_deps)
                eval_section += (
                    f"## Eval Dependency Graph — CRITICAL\n"
                    f"The eval entry point `{entry_name}` imports from these "
                    f"repo-local files: {deps_list}.\n"
                    f"**Only changes to these files (or files they transitively "
                    f"import) will affect the measured metric.**\n"
                    f"Changes to other repo files will have NO effect on the "
                    f"eval result.\n\n"
                )
            else:
                # Self-contained eval — the framework will splice changes
                # into the eval entry point via AST merge (safe: only
                # replaces changed functions/classes, preserves the rest).
                _eval_is_self_contained = True
                eval_section += (
                    f"## ⚠ SELF-CONTAINED EVAL — MODIFY `{entry_name}` DIRECTLY ⚠\n"
                    f"The eval entry point `{entry_name}` does NOT import from "
                    f"any other repo source files. It is **self-contained** — "
                    f"it has its own training loop, model setup, and evaluation "
                    f"logic.\n\n"
                    f"**Changes to other Python files (e.g. `Train_model.py`) "
                    f"will have ZERO effect on the measured metric.**\n\n"
                    f"**You MUST output your changes as modifications to "
                    f"`{entry_name}` itself.** Output ONLY the functions, "
                    f"classes, or constants you are changing — the framework "
                    f"will safely splice them into the existing file via AST "
                    f"merge (replacing only the definitions you provide while "
                    f"preserving everything else).\n\n"
                    f"Do NOT modify `Train_model.py` or other files that "
                    f"`{entry_name}` does not import — those changes will be "
                    f"ignored.\n\n"
                )
                logger.info(
                    "Eval entry point %s is self-contained — enabling "
                    "AST-splice passthrough for forbidden file filter.",
                    entry_name,
                )

        # When eval is self-contained, update the PROTECTED FILES section
        # to avoid contradicting the splice-allowed guidance above.
        if _eval_is_self_contained and entry_name:
            repo_grounding = repo_grounding.replace(
                f"**BLOCKED (changes will be rejected by the framework):**\n"
                f"- `run_eval.py` or any file matching `**/run_eval.py` (eval entry point)\n",
                f"**BLOCKED (changes will be rejected by the framework):**\n"
                f"- ~~`run_eval.py`~~ **SPLICE-ALLOWED** — self-contained eval; "
                f"your changes to `{entry_name}` will be merged via AST splice "
                f"(only changed functions/classes are replaced)\n",
            )
    problem = (
        f"# AHVS Hypothesis Execution\n\n"
        f"## Hypothesis {hyp_id}\n"
        f"**Type:** {hyp_type}\n"
        f"**Description:** {hyp.get('description', '')}\n"
        f"**Rationale:** {hyp.get('rationale', '')}\n\n"
        f"## Implementation Plan\n"
        f"{plan.get('implementation_approach', 'Implement as described in description.')}\n\n"
        f"## Eval Method\n"
        f"Use: {plan.get('eval_method', 'custom_script')}\n"
        f"Skill: {plan.get('skill', 'sandbox_run')}\n\n"
        f"## Success Criterion\n"
        f"{plan.get('success_criterion', f'{metric_name} > {baseline_value}')}\n\n"
        f"## Target Repository\n"
        f"{config.repo_path}\n\n"
        f"{repo_grounding}"
        f"{eval_section}"
        f"## Primary Metric\n"
        f"Name: {metric_name}\n"
        f"Baseline value: {baseline_value}\n\n"
        f"## Output Requirements\n"
        f"Write the metric result as JSON to: result.json\n"
        f"Format: {{'{metric_name}': <float>, 'eval_method': '<method>'}}\n\n"
        f"{skill_context}"
    )

    # Inject type-specific execution strategy
    type_strategy = _TYPE_EXECUTION_STRATEGIES.get(hyp_type, {})
    if type_strategy:
        problem += (
            f"\n## Type-Specific Execution Strategy ({hyp_type})\n"
            f"{type_strategy['system_addition']}\n\n"
            f"**Constraints:** {type_strategy['constraints']}\n"
            f"**Success Guidance:** {type_strategy['success_guidance']}\n\n"
        )

    # Package hint based on hypothesis type
    pkg_hint_map = {
        "prompt_rewrite": "promptfoo",
        "model_comparison": "openai anthropic",
        "dspy_optimize": "dspy-ai",
        "code_change": "pytest",
        "architecture_change": "pytest",
        "multi_llm_judge": "openai anthropic",
    }
    pkg_hint = pkg_hint_map.get(hyp_type, "")

    # ── Eval-mode intelligence ────────────────────────────────────────
    # Detect when hypothesis type is incompatible with eval-only mode.
    # prompt_rewrite and model_comparison change LLM behaviour, but if the
    # eval command uses --eval-only it reads frozen checkpoint data and the
    # changes will have zero measurable effect.
    _NEEDS_REINFERENCE_TYPES = {"prompt_rewrite", "model_comparison"}
    if hyp_type in _NEEDS_REINFERENCE_TYPES and "--eval-only" in eval_command:
        logger.warning(
            "%s: hypothesis type '%s' modifies LLM prompts/model, but eval "
            "command uses --eval-only (reads frozen checkpoint data). "
            "Changes will have NO measurable effect unless the eval pipeline "
            "re-runs inference. Consider adding --reparse or removing "
            "--eval-only for this hypothesis type.",
            hyp_id, hyp_type,
        )
        # Add a visible warning to the print output
        print(
            f"[AHVS] WARNING: {hyp_id} ({hyp_type}) may be unmeasurable — "
            f"eval command uses --eval-only which reads frozen data. "
            f"Prompt/model changes require re-inference to take effect."
        )

    t0 = time.monotonic()
    metric_value = baseline_value
    measurement_status = "not_executed"
    skill_planned = plan.get("skill") or None
    error: str | None = None
    artifact_paths: list[str] = []
    worktree: HypothesisWorktree | None = None

    # ── Create worktree ──────────────────────────────────────────────
    # AHVS requires a git-backed target repo. Worktree creation failing
    # means the target is not a git repo or has a corrupted .git — fail
    # the hypothesis immediately rather than editing the live repo.
    try:
        wt_path = cycle_dir / "worktrees" / hyp_id
        worktree = HypothesisWorktree(config.repo_path, wt_path)
        worktree.create()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "%s: worktree creation failed (%s) — failing hypothesis.",
            hyp_id, exc,
        )
        return (
            HypothesisResult.make_error(
                hypothesis_id=hyp_id,
                hypothesis_type=hyp_type,
                primary_metric=metric_name,
                baseline_value=baseline_value,
                error=f"Worktree creation failed: {exc}",
            ),
            None,
        )

    try:
        # ── Generate files via Claude Code CLI ─────────────────────
        # Run Claude Code inside the worktree (if available) so the
        # live target repo is never modified.
        generated_files = _generate_files_with_claude_code(
            hyp=hyp,
            hyp_id=hyp_id,
            config=config,
            baseline=baseline,
            work_dir=work_dir,
            worktree_path=worktree.worktree_path,
        )

        # Restore symlinks for gitignored data directories (checkpoints,
        # ground-truth, etc.) — git clean -fd inside Claude Code generation
        # removes untracked entries including these symlinks.
        worktree.restore_data_symlinks()

        # Write generated files to work_dir (with path validation)
        from ahvs.worktree import validate_safe_relpath

        for filename, content in generated_files.items():
            validate_safe_relpath(filename, work_dir)
            fpath = (work_dir / filename).resolve()
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")
            artifact_paths.append(str(fpath.relative_to(cycle_dir)))

        # ── Auto-remap root-level files, then filter forbidden ─────
        # LLM-generated code sometimes writes files to the repo root
        # (e.g. "parsing.py") instead of the correct src/ path (e.g.
        # "src/autoqa/parsing.py"). Before blocking, attempt to remap
        # root-level .py files to their unique src/ counterpart. Only
        # remap when exactly one match exists under src/ to avoid ambiguity.
        _repo_has_src = (config.repo_path / "src").is_dir()
        remapped_files: dict[str, str] = {}
        for filename, content in generated_files.items():
            from pathlib import PurePosixPath as _PP
            _p = _PP(filename)
            if (
                _repo_has_src
                and _p.parent == _PP(".")
                and filename.endswith(".py")
            ):
                remapped = _remap_root_file(_p.name, config.repo_path)
                if remapped is not None:
                    logger.info(
                        "%s: auto-remapped %s → %s (root-level file "
                        "matched unique src/ target)",
                        hyp_id, filename, remapped,
                    )
                    print(
                        f"[AHVS] {hyp_id}: auto-remapped {filename} → "
                        f"{remapped}"
                    )
                    remapped_files[remapped] = content
                    continue
            remapped_files[filename] = content

        filtered_files: dict[str, str] = {}
        blocked_count = 0
        for filename, content in remapped_files.items():
            reason = _is_forbidden_file(filename, repo_has_src=_repo_has_src)
            if reason is not None:
                # ── Self-contained eval splice passthrough ─────────
                # When the eval is self-contained (no repo-local imports),
                # the eval entry point is the ONLY file that matters.
                # Instead of hard-blocking it, allow it through — the
                # existing splice=True in apply_files() will merge only
                # the changed functions/classes via AST splice, preserving
                # the eval infrastructure.
                from pathlib import PurePosixPath as _PP2
                _basename = _PP2(filename).name
                if (
                    _eval_is_self_contained
                    and entry_name is not None
                    and _basename == entry_name
                ):
                    logger.info(
                        "%s: SPLICE-PASSTHROUGH for %s — eval is "
                        "self-contained, allowing AST-spliced changes "
                        "to eval entry point (would normally be blocked: %s)",
                        hyp_id, filename, reason,
                    )
                    print(
                        f"[AHVS] {hyp_id}: allowing {filename} via AST "
                        f"splice (self-contained eval passthrough)"
                    )
                    filtered_files[filename] = content
                    continue

                logger.warning(
                    "%s: BLOCKED %s from worktree apply — %s",
                    hyp_id, filename, reason,
                )
                blocked_count += 1
            else:
                warn_reason = _is_warn_file(filename)
                if warn_reason is not None:
                    logger.warning(
                        "%s: WARNING — applying %s (%s)",
                        hyp_id, filename, warn_reason,
                    )
                filtered_files[filename] = content

        if blocked_count > 0:
            print(
                f"[AHVS] {hyp_id}: blocked {blocked_count} forbidden file(s) "
                f"from worktree apply (eval-harness protection)"
            )

        # Filter out syntactically invalid Python files before applying to
        # worktree — Claude Code may return truncated output that passed
        # hard_validation's max-repair limit.
        valid_files = {}
        for filename, content in filtered_files.items():
            if filename.endswith(".py"):
                try:
                    compile(content, filename, "exec")
                    valid_files[filename] = content
                except SyntaxError as syn_err:
                    logger.warning(
                        "%s: dropping %s from worktree apply — SyntaxError: %s "
                        "(likely truncated LLM output, %d lines generated)",
                        hyp_id, filename, syn_err.msg, content.count("\n") + 1,
                    )
            else:
                valid_files[filename] = content

        # Apply files with splice=True: partial output (only modified
        # functions/classes) is merged into existing files via AST splicing.
        # This avoids the truncation problem from rewriting entire large files.
        if valid_files:
            worktree.apply_files(valid_files, splice=True)

        # ── Pre-eval import sanity check ──────────────────────────────
        # After applying files, verify the eval module can still import.
        # If Claude Code broke imports (circular, missing, syntax in spliced
        # file), fail early with a clear message instead of a cryptic eval
        # crash.
        if eval_command:
            import_err = _run_import_sanity_check(worktree, eval_command)
            if import_err is not None:
                logger.error(
                    "%s: PRE-EVAL IMPORT CHECK FAILED — %s. "
                    "Claude Code changes broke the module structure.",
                    hyp_id, import_err,
                )
                print(
                    f"[AHVS] {hyp_id}: pre-eval import check FAILED — "
                    f"Claude Code changes broke module imports. "
                    f"Skipping eval_command."
                )
                # Mark as extraction_failed — don't run eval on broken code
                measurement_status = "extraction_failed"
                error = f"Pre-eval import check failed: {import_err}"

        # ── Metric extraction ──────────────────────────────────────────

        # Warn if prompt_rewrite uses --eval-only (prompt changes need re-inference)
        if hyp_type == "prompt_rewrite" and eval_command and "--eval-only" in eval_command:
            logger.warning(
                "%s: prompt_rewrite hypothesis using --eval-only eval_command. "
                "Prompt changes require full re-inference to measure; "
                "--eval-only only re-parses from cached analyst_raw. "
                "Consider removing --eval-only for prompt_rewrite hypotheses.",
                hyp_id,
            )

        # When eval_command is configured, it is the ONLY trusted measurement
        # source.  Tiers 1-3 (sandbox self-reports) are unconditionally
        # skipped because Claude Code can fabricate result.json with false
        # metrics (Bug L: false 0.9928 precision when eval actually crashed).
        eval_command_is_authoritative = bool(eval_command)

        # Tier 0: eval_command in worktree (the only trusted source)
        if (
            measurement_status != "measured"
            and measurement_status != "extraction_failed"
            and eval_command
            and worktree is not None
        ):
            eval_timeout = int(baseline.get("eval_timeout", config.eval_timeout_sec))
            logger.info("%s: running eval_command in worktree (timeout=%ds): %s", hyp_id, eval_timeout, eval_command)
            eval_result = worktree.run_eval_command(eval_command, timeout=eval_timeout)
            if eval_result.returncode == 0:
                extracted = _extract_metric_from_output(eval_result.stdout, metric_name)
                if extracted is not None:
                    metric_value = extracted
                    measurement_status = "measured"
                    logger.info(
                        "%s: Tier 0 (eval_command) extracted %s=%.4f",
                        hyp_id, metric_name, metric_value,
                    )
            else:
                logger.warning(
                    "%s: eval_command exited %d — marking extraction_failed. "
                    "stderr: %s",
                    hyp_id, eval_result.returncode,
                    eval_result.stderr[:2000],
                )

        # Fallback: result.json in work_dir (used when no eval_command)
        if not eval_command_is_authoritative:
            result_json_path = work_dir / "result.json"
            if not result_json_path.exists():
                for candidate in sorted(work_dir.glob("agent_runs/*/result.json")):
                    result_json_path = candidate
                    break

            if measurement_status != "measured" and result_json_path.exists():
                raw = result_json_path.read_text(encoding="utf-8")
                extracted = _extract_metric_from_output(raw, metric_name)
                if extracted is not None:
                    metric_value = extracted
                    measurement_status = "measured"

        # ── Eval retry: re-apply files without splice and re-run ────
        # If eval failed and Claude Code produced files, the AST splice
        # may have produced broken code (silent fallback to original).
        # Re-apply the raw Claude Code output directly and retry eval.
        if (
            measurement_status != "measured"
            and generated_files
            and eval_command
            and worktree is not None
        ):
            logger.info(
                "%s: eval failed after splice — retrying with direct "
                "file application (no splice).",
                hyp_id,
            )
            print(
                f"[AHVS] {hyp_id}: eval retry — re-applying files "
                f"without splice and re-running eval"
            )
            try:
                worktree.apply_files(valid_files, splice=False)
                eval_timeout = int(
                    baseline.get("eval_timeout", config.eval_timeout_sec)
                )
                eval_result = worktree.run_eval_command(
                    eval_command, timeout=eval_timeout
                )
                if eval_result.returncode == 0:
                    extracted = _extract_metric_from_output(
                        eval_result.stdout, metric_name
                    )
                    if extracted is not None:
                        metric_value = extracted
                        measurement_status = "measured"
                        logger.info(
                            "%s: eval RETRY succeeded — %s=%.4f",
                            hyp_id, metric_name, metric_value,
                        )
                else:
                    logger.warning(
                        "%s: eval retry also failed (exit %d). stderr: %s",
                        hyp_id, eval_result.returncode,
                        eval_result.stderr[:500],
                    )
            except Exception as retry_exc:  # noqa: BLE001
                logger.warning(
                    "%s: eval retry raised exception: %s",
                    hyp_id, retry_exc,
                )

        # Extraction failed — log warning, keep baseline
        if measurement_status != "measured":
            measurement_status = "extraction_failed"
            logger.warning(
                "%s: metric extraction failed — metric_value remains "
                "at baseline (%.4f). Hypothesis code may not have "
                "produced a valid metric.",
                hyp_id, baseline_value,
            )

    except Exception as exc:  # noqa: BLE001
        error = f"Hypothesis execution failed: {exc}"
        measurement_status = "sandbox_error"
        logger.exception("Hypothesis execution failed for %s", hyp_id)

    duration = time.monotonic() - t0

    # Write canonical result.json so the regression guard always has a file
    # to inspect, regardless of which tier produced the metric.
    canonical_result_path = work_dir / "result.json"
    canonical_result = {
        "hypothesis_id": hyp_id,
        "primary_metric": metric_name,
        metric_name: metric_value,
        "baseline_value": baseline_value,
        "measurement_status": measurement_status,
    }
    canonical_result_path.write_text(
        json.dumps(canonical_result, indent=2), encoding="utf-8"
    )

    # Run regression guard against the canonical result
    guard_passed = _run_regression_guard(config.regression_guard_path, canonical_result_path)

    delta = metric_value - baseline_value
    delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else 0.0

    result = HypothesisResult(
        hypothesis_id=hyp_id,
        hypothesis_type=hyp_type,
        primary_metric=metric_name,
        metric_value=metric_value,
        baseline_value=baseline_value,
        delta=delta,
        delta_pct=delta_pct,
        regression_guard_passed=guard_passed,
        eval_method=plan.get("eval_method", "code_agent"),
        artifact_paths=artifact_paths,
        raw_output_path=str(work_dir),
        duration_seconds=round(duration, 2),
        skill_planned=skill_planned,
        error=error,
        measurement_status=measurement_status,
        execution_mode="repo_grounded",
    )
    return result, worktree


def _execute_report_and_memory(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 7: Write cycle report via LLM; archive lessons to EvolutionStore."""
    for required in ("results.json", "context_bundle.json"):
        if not (cycle_dir / required).exists():
            return AHVSStageResult(
                stage=AHVSStage.AHVS_REPORT_MEMORY,
                status=StageStatus.FAILED,
                artifacts=(),
                error=f"Missing required artifact: {required}",
            )

    from ahvs.result import load_results
    from ahvs.evolution import EvolutionStore, LessonEntry, LessonCategory

    results = load_results(cycle_dir / "results.json")
    bundle = json.loads((cycle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    baseline = bundle["baseline"]
    metric_name = baseline["primary_metric"]

    # Build results summary for prompt
    summary_lines = []
    best_result: HypothesisResult | None = None
    for r in results:
        status_str = "IMPROVED" if r.improved else ("ERROR" if r.error else "no improvement")
        measurement_tag = ""
        if r.measurement_status == "extraction_failed":
            measurement_tag = " [MEASUREMENT FAILED]"
        elif r.measurement_status == "sandbox_error":
            measurement_tag = " [SANDBOX ERROR]"
        summary_lines.append(
            f"- **{r.hypothesis_id}** ({r.hypothesis_type}): "
            f"{metric_name}={r.metric_value:.4f} (Δ{r.delta:+.4f}, {r.delta_pct:+.1f}%) "
            f"| guard={'pass' if r.regression_guard_passed else 'fail'} "
            f"| {status_str}{measurement_tag}"
            + (f"\n  Error: {r.error}" if r.error else "")
        )
        if r.improved and (best_result is None or r.delta > best_result.delta):
            best_result = r
    results_summary = "\n".join(summary_lines) or "No results."

    pm = AHVSPromptManager(config.prompts_override_path)
    prompt = pm.for_stage(
        "ahvs_report",
        question=config.question,
        metric_name=metric_name,
        baseline_value=str(baseline["value"]),
        results_summary=results_summary,
    )

    report_text = ""
    try:
        llm = _make_llm_client(config)
        response = llm.chat(
            [{"role": "user", "content": prompt.user}],
            system=prompt.system,
            max_tokens=prompt.max_tokens,
        )
        report_text = response.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM report generation failed: %s — writing minimal report", exc)
        report_text = (
            f"# AHVS Cycle Report\n\n"
            f"**Question:** {config.question}\n\n"
            f"**Results:**\n{results_summary}\n\n"
            f"*Report generation failed: {exc}*"
        )

    # Ensure the report contains the mandatory Final Summary & Findings section.
    # If the LLM omitted it, append a structured fallback built from raw results.
    if "Final Summary" not in report_text and "## Results table" not in report_text:
        summary_table_lines = [
            "| Hypothesis | Type | Metric | Delta | Verdict |",
            "|---|---|---|---|---|",
            f"| Baseline | — | {baseline['value']:.4f} | — | — |",
        ]
        for r in results:
            verdict = "ADOPT" if r.improved else ("ERROR" if r.error else "No improvement")
            summary_table_lines.append(
                f"| {r.hypothesis_id} | {r.hypothesis_type} | "
                f"{r.metric_value:.4f} | {r.delta:+.4f} ({r.delta_pct:+.1f}%) | {verdict} |"
            )
        recommendations = []
        if best_result:
            recommendations.append(
                f"| 1 | Apply {best_result.hypothesis_id} "
                f"({best_result.hypothesis_type}) | "
                f"{metric_name} +{best_result.delta_pct:.1f}% |"
            )
        rec_table = (
            "| Priority | Action | Expected Impact |\n|---|---|---|\n"
            + "\n".join(recommendations)
        ) if recommendations else "No hypotheses improved the metric this cycle."

        summary_section = (
            "\n\n---\n\n"
            "## Final Summary & Findings\n\n"
            f"**Objective:** {config.question}\n\n"
            "### Results\n\n"
            + "\n".join(summary_table_lines) + "\n\n"
            "### Recommendations\n\n"
            + rec_table + "\n"
        )
        report_text += summary_section

    (cycle_dir / "report.md").write_text(report_text, encoding="utf-8")

    # Write friction log (auto-generated from execution errors)
    # Operators should add their own observations below the auto-generated section.
    friction_items = [
        f"- {r.hypothesis_id}: {r.error}"
        for r in results
        if r.error
    ]
    measurement_items = [
        f"- {r.hypothesis_id}: metric extraction failed (measurement_status={r.measurement_status})"
        for r in results
        if r.measurement_status in ("extraction_failed", "sandbox_error")
    ]
    friction_text = (
        "# Friction Log\n\n"
        "## Execution Errors (auto-generated)\n\n"
        + ("\n".join(friction_items) if friction_items else "No execution errors this cycle.")
        + "\n\n"
        "## Measurement Issues (auto-generated)\n\n"
        + ("\n".join(measurement_items) if measurement_items else "No measurement issues.")
        + "\n\n"
        "## Operator Notes\n\n"
        "<!-- Add your observations below: What felt slow? What felt unclear?\n"
        "     What almost got skipped? What should be automated next? -->\n\n"
    )
    (cycle_dir / "friction_log.md").write_text(friction_text, encoding="utf-8")

    # Archive lessons to EvolutionStore
    config.evolution_dir.mkdir(parents=True, exist_ok=True)
    try:
        store = EvolutionStore(config.evolution_dir)
        now = _utcnow_iso()
        cycle_id = cycle_dir.name
        lessons: list[LessonEntry] = []

        for r in results:
            _structured = dict(
                hypothesis_id=r.hypothesis_id,
                hypothesis_type=r.hypothesis_type,
                metric_name=metric_name,
                metric_baseline=r.baseline_value,
                metric_after=r.metric_value,
                metric_delta=r.delta,
                eval_method=r.eval_method,
            )
            if r.improved:
                lessons.append(LessonEntry(
                    stage_name="ahvs_execution",
                    stage_num=6,
                    category=LessonCategory.EXPERIMENT,
                    severity="info",
                    description=(
                        f"[{cycle_id}] {r.hypothesis_id} ({r.hypothesis_type}) improved "
                        f"{metric_name} by {r.delta:+.4f} ({r.delta_pct:+.1f}%). "
                        f"Eval: {r.eval_method}."
                    ),
                    timestamp=now,
                    run_id=cycle_id,
                    **_structured,
                ))
            elif r.error:
                lessons.append(LessonEntry(
                    stage_name="ahvs_execution",
                    stage_num=6,
                    category=LessonCategory.EXPERIMENT,
                    severity="warning",
                    description=(
                        f"[{cycle_id}] {r.hypothesis_id} ({r.hypothesis_type}) FAILED: {r.error[:150]}"
                    ),
                    timestamp=now,
                    run_id=cycle_id,
                    **_structured,
                ))
            elif r.measurement_status != "measured":
                # Infrastructure failure (extraction_failed, sandbox_error, etc.)
                # — the hypothesis was never actually tested.  Do NOT record as
                # a "Rejected approach" or it will teach AHVS to avoid ideas
                # that were never validated.
                lessons.append(LessonEntry(
                    stage_name="ahvs_execution",
                    stage_num=6,
                    category=LessonCategory.EXPERIMENT,
                    severity="warning",
                    description=(
                        f"[{cycle_id}] {r.hypothesis_id} ({r.hypothesis_type}) "
                        f"measurement failed ({r.measurement_status}). "
                        f"Infrastructure issue — hypothesis was not tested."
                    ),
                    timestamp=now,
                    run_id=cycle_id,
                    **_structured,
                ))
            else:
                lessons.append(LessonEntry(
                    stage_name="ahvs_execution",
                    stage_num=6,
                    category=LessonCategory.EXPERIMENT,
                    severity="info",
                    description=(
                        f"[{cycle_id}] {r.hypothesis_id} ({r.hypothesis_type}) did not improve "
                        f"{metric_name} (Δ{r.delta:+.4f}). Rejected approach."
                    ),
                    timestamp=now,
                    run_id=cycle_id,
                    **_structured,
                ))

        if lessons:
            store.append_many(lessons)
            logger.info("Archived %d lessons to EvolutionStore", len(lessons))
    except Exception as exc:  # noqa: BLE001
        logger.warning("EvolutionStore archival failed (non-fatal): %s", exc)

    return AHVSStageResult(
        stage=AHVSStage.AHVS_REPORT_MEMORY,
        status=StageStatus.DONE,
        artifacts=("report.md", "friction_log.md"),
    )


def _update_lesson_verification(
    config: AHVSConfig,
    cycle_dir: Path,
    summary: dict,
) -> None:
    """Update EvolutionStore lessons with keep/revert verification from Stage 8.

    Sets ``verified="kept"`` on the best hypothesis's lessons and
    ``verified="reverted"`` on all other lessons from this cycle.
    """
    from ahvs.evolution import EvolutionStore

    try:
        store = EvolutionStore(config.evolution_dir)
        cycle_id = cycle_dir.name
        best_id = summary.get("best_hypothesis")
        all_lessons = store.load_all()

        modified = False
        for lesson in all_lessons:
            if lesson.run_id != cycle_id:
                continue
            if not lesson.hypothesis_id:
                continue
            if lesson.hypothesis_id == best_id and best_id is not None:
                lesson.verified = "kept"
            else:
                lesson.verified = "reverted"
            modified = True

        if modified:
            store._lessons_path.write_text(
                "".join(
                    json.dumps(l.to_dict(), ensure_ascii=False) + "\n"
                    for l in all_lessons
                ),
                encoding="utf-8",
            )
            logger.info(
                "Updated lesson verification for cycle %s (best=%s)",
                cycle_id, best_id,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lesson verification update failed (non-fatal): %s", exc)


def _execute_cycle_verify(
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool,
) -> AHVSStageResult:
    """Stage 8: Validate all artifacts exist and write cycle_summary.json."""
    from ahvs.result import load_results

    # Check required artifacts
    required = [
        "cycle_manifest.json", "context_bundle.json", "hypotheses.md",
        "selection.md", "validation_plan.md", "results.json",
        "report.md", "friction_log.md",
    ]
    missing = [f for f in required if not (cycle_dir / f).exists()]
    if missing:
        return AHVSStageResult(
            stage=AHVSStage.AHVS_CYCLE_VERIFY,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Missing artifacts: {missing}",
        )

    # Validate results.json
    try:
        results = load_results(cycle_dir / "results.json")
    except Exception as exc:  # noqa: BLE001
        return AHVSStageResult(
            stage=AHVSStage.AHVS_CYCLE_VERIFY,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"results.json invalid: {exc}",
        )

    selection = json.loads((cycle_dir / "selection.json").read_text(encoding="utf-8"))
    result_ids = {r.hypothesis_id for r in results}
    missing_results = [hid for hid in selection["selected"] if hid not in result_ids]
    if missing_results:
        return AHVSStageResult(
            stage=AHVSStage.AHVS_CYCLE_VERIFY,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Selected hypotheses with no result: {missing_results}",
        )

    # Build cycle summary
    improved = [r for r in results if r.improved]
    best = max(improved, key=lambda r: r.delta) if improved else None
    bundle = json.loads((cycle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    metric_name = bundle["baseline"]["primary_metric"]

    extraction_failures = sum(1 for r in results if r.measurement_status == "extraction_failed")
    measured_count = sum(1 for r in results if r.measurement_status == "measured")
    all_unmeasured = measured_count == 0 and len(results) > 0

    # Build recommendation based on measurement validity
    if all_unmeasured:
        recommendation = (
            "INVALID CYCLE — all hypotheses failed measurement. "
            "No valid comparison to baseline is possible."
        )
    elif best:
        recommendation = (
            f"KEEP {best.hypothesis_id}: {metric_name} improved by "
            f"{best.delta:+.4f} ({best.delta_pct:+.1f}%)"
        )
    else:
        recommendation = "no improvement this cycle — revert all changes"

    summary = {
        "cycle_id": cycle_dir.name,
        "question": config.question,
        "hypotheses_run": len(results),
        "hypotheses_measured": measured_count,
        "hypotheses_improved": len(improved),
        "extraction_failures": extraction_failures,
        "all_unmeasured": all_unmeasured,
        "best_hypothesis": best.hypothesis_id if best else None,
        "best_delta": best.delta if best else 0.0,
        "best_metric_value": best.metric_value if best else None,
        "recommendation": recommendation,
        "kept_worktree": best.worktree_path if best and best.worktree_path else None,
        "kept_patch": best.patch_path if best and best.patch_path else None,
        "all_patches": [r.patch_path for r in results if r.patch_path],
        "per_hypothesis": [
            {
                "id": r.hypothesis_id,
                "execution_mode": r.execution_mode,
                "metric_value": r.metric_value,
                "delta": r.delta,
                "measurement_status": r.measurement_status,
            }
            for r in results
        ],
        "completed_at": _utcnow_iso(),
        "artifacts_verified": required,
    }
    (cycle_dir / "cycle_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Feed keep/revert decision back into evolution lessons
    _update_lesson_verification(config, cycle_dir, summary)

    print(f"\n[AHVS] Cycle complete — {recommendation}")

    # All-unmeasured cycle is a verification failure
    if all_unmeasured:
        return AHVSStageResult(
            stage=AHVSStage.AHVS_CYCLE_VERIFY,
            status=StageStatus.FAILED,
            artifacts=("cycle_summary.json",),
            error="All hypotheses failed measurement — cycle is invalid",
        )

    return AHVSStageResult(
        stage=AHVSStage.AHVS_CYCLE_VERIFY,
        status=StageStatus.DONE,
        artifacts=("cycle_summary.json",),
    )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

_HANDLERS = {
    AHVSStage.AHVS_SETUP:           _execute_setup,
    AHVSStage.AHVS_CONTEXT_LOAD:    _execute_context_load,
    AHVSStage.AHVS_HYPOTHESIS_GEN:  _execute_hypothesis_gen,
    AHVSStage.AHVS_HUMAN_SELECTION: _execute_human_selection,
    AHVSStage.AHVS_VALIDATION_PLAN: _execute_validation_plan,
    AHVSStage.AHVS_EXECUTION:       _execute_hypotheses,
    AHVSStage.AHVS_REPORT_MEMORY:   _execute_report_and_memory,
    AHVSStage.AHVS_CYCLE_VERIFY:    _execute_cycle_verify,
}


def execute_ahvs_stage(
    stage: AHVSStage,
    *,
    cycle_dir: Path,
    config: AHVSConfig,
    skill_library: SkillLibrary,
    auto_approve: bool = False,
) -> AHVSStageResult:
    """Dispatch to the appropriate stage handler."""
    handler = _HANDLERS.get(stage)
    if handler is None:
        return AHVSStageResult(
            stage=stage,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"No handler registered for stage {stage!r}",
        )
    return handler(cycle_dir, config, skill_library, auto_approve)
