"""AHVS-specific pre-flight health checks.

Pre-flight runs twice per cycle:
  1. At AHVS_SETUP — minimal: baseline file + LLM connectivity
  2. After AHVS_HUMAN_SELECTION — full: tools required by selected hypothesis types
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CheckResult and DoctorReport (inlined)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class DoctorReport:
    timestamp: str
    checks: list[CheckResult]
    overall: str

    @property
    def actionable_fixes(self) -> list[str]:
        return [check.fix for check in self.checks if check.fix]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "overall": self.overall,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "detail": check.detail,
                    "fix": check.fix,
                }
                for check in self.checks
            ],
            "actionable_fixes": self.actionable_fixes,
        }


# ---------------------------------------------------------------------------
# Hypothesis type → required external tools
# ---------------------------------------------------------------------------

HYPOTHESIS_TOOL_REQUIREMENTS: dict[str, list[str]] = {
    "prompt_rewrite":      ["promptfoo"],
    "model_comparison":    ["promptfoo"],
    "config_change":       ["promptfoo"],
    "dspy_optimize":       ["dspy", "promptfoo"],
    "phoenix_eval":        ["arize-phoenix"],
    # code_change, architecture_change, multi_llm_judge run via Claude Code.
    # No external tool required.
    "code_change":         [],
    "architecture_change": [],
    "multi_llm_judge":     [],
}


def check_tool(name: str) -> CheckResult:
    """Generic availability check: CLI tool, Python package, or Node package (npx)."""
    # 1. CLI tool — covers docker, promptfoo (global install), etc.
    if shutil.which(name):
        return CheckResult(
            name=f"tool_{name}",
            status="pass",
            detail=f"'{name}' found on PATH: {shutil.which(name)}",
        )

    # 2. Python package — covers dspy, arize-phoenix (import as arize_phoenix), etc.
    pkg_name = name.replace("-", "_")
    if importlib.util.find_spec(pkg_name) is not None:
        return CheckResult(
            name=f"tool_{name}",
            status="pass",
            detail=f"Python package '{name}' is importable",
        )

    # 3. Node package via npx — covers promptfoo (npm install)
    if name == "promptfoo":
        try:
            r = subprocess.run(
                ["npx", name, "--version"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            if r.returncode == 0:
                version = r.stdout.decode(errors="replace").strip().splitlines()[0]
                return CheckResult(
                    name=f"tool_{name}",
                    status="pass",
                    detail=f"'{name}' available via npx: {version}",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    return CheckResult(
        name=f"tool_{name}",
        status="fail",
        detail=f"'{name}' not found (checked: PATH, Python packages, npx)",
        fix=f"Install '{name}'. See project docs for setup instructions.",
    )


def check_baseline_metric(baseline_path: Path) -> CheckResult:
    """Validate .ahvs/baseline_metric.json exists and contains required fields."""
    if not baseline_path.exists():
        return CheckResult(
            name="ahvs_baseline",
            status="fail",
            detail=f"Baseline metric file not found: {baseline_path}",
            fix=(
                "Create .ahvs/baseline_metric.json. Example:\n"
                '  {"primary_metric": "answer_relevance", "answer_relevance": 0.74,\n'
                '   "recorded_at": "2026-03-17T10:00:00Z",\n'
                '   "eval_command": "promptfoo eval --config .ahvs/eval/baseline.yaml"}'
            ),
        )
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return CheckResult(
            name="ahvs_baseline",
            status="fail",
            detail=f"baseline_metric.json is invalid: {exc}",
            fix="Fix JSON syntax in .ahvs/baseline_metric.json",
        )

    required = ("primary_metric", "recorded_at", "eval_command")
    missing = [f for f in required if f not in data]
    if missing:
        return CheckResult(
            name="ahvs_baseline",
            status="fail",
            detail=f"baseline_metric.json missing fields: {missing}",
            fix="Add the missing fields to .ahvs/baseline_metric.json",
        )

    metric_key = data["primary_metric"]
    if metric_key not in data:
        return CheckResult(
            name="ahvs_baseline",
            status="fail",
            detail=f"baseline_metric.json must contain key '{metric_key}' with its numeric value",
            fix=f'Add "{metric_key}": <float> to .ahvs/baseline_metric.json',
        )

    return CheckResult(
        name="ahvs_baseline",
        status="pass",
        detail=(
            f"Baseline: {metric_key}={data[metric_key]} "
            f"(recorded {str(data.get('recorded_at', ''))[:10]})"
        ),
    )


def check_baseline_commit(baseline_path: Path) -> CheckResult | None:
    """Warn if baseline_metric.json lacks a 'commit' field.

    Returns None if the baseline file doesn't exist (that's caught by
    check_baseline_metric). Returns a warn CheckResult if commit is missing.
    """
    if not baseline_path.exists():
        return None
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # Caught by check_baseline_metric
    if "commit" not in data or not data["commit"]:
        return CheckResult(
            name="ahvs_baseline_commit",
            status="warn",
            detail=(
                "baseline_metric.json has no 'commit' field — "
                "cannot verify baseline was measured on the current repo state"
            ),
            fix='Add "commit": "<git-sha>" to .ahvs/baseline_metric.json',
        )
    return CheckResult(
        name="ahvs_baseline_commit",
        status="pass",
        detail=f"Baseline commit: {data['commit']}",
    )


def check_regression_guard(guard_path: Path) -> CheckResult:
    """Validate regression_guard.sh exists and is executable."""
    if not guard_path.exists():
        return CheckResult(
            name="ahvs_regression_guard",
            status="fail",
            detail=f"Regression guard script not found: {guard_path}",
            fix="Create the script or remove --regression-guard from config",
        )
    if not os.access(guard_path, os.X_OK):
        return CheckResult(
            name="ahvs_regression_guard",
            status="fail",
            detail=f"Regression guard not executable: {guard_path}",
            fix=f"chmod +x {guard_path}",
        )
    return CheckResult(
        name="ahvs_regression_guard",
        status="pass",
        detail=f"Regression guard found and executable: {guard_path}",
    )


def check_clean_branch(repo_path: Path) -> CheckResult:
    """Fail if the target repo has uncommitted changes.

    AHVS creates hypothesis worktrees from committed HEAD, so uncommitted
    changes in the working tree are **not** tested.  A dirty repo means
    the operator thinks they are validating the current state, but AHVS
    actually measures an older committed snapshot — a real reproducibility
    trap.  Therefore this is a hard fail, not a warning.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            cwd=repo_path,
            timeout=10,
            check=False,
        )
        if r.returncode != 0:
            return CheckResult(
                name="ahvs_clean_branch",
                status="warn",
                detail="Could not check git status (not a git repo?)",
                fix="Ensure the target repo is a git repository",
            )
        output = r.stdout.decode(errors="replace").strip()
        if output:
            return CheckResult(
                name="ahvs_clean_branch",
                status="fail",
                detail=(
                    "Target repo has uncommitted changes — AHVS worktrees are "
                    "created from committed HEAD, so uncommitted work would not "
                    "be included in the experiment"
                ),
                fix="Commit or stash changes before running an AHVS cycle",
            )
        return CheckResult(
            name="ahvs_clean_branch",
            status="pass",
            detail="Target repo working tree is clean",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return CheckResult(
            name="ahvs_clean_branch",
            status="warn",
            detail="git not found — cannot verify branch cleanliness",
            fix="Install git to enable this check",
        )


def check_llm_connectivity(
    api_key: str,
    model: str,
    base_url: str = "",
    provider: str = "",
    ahvs_config: object | None = None,
) -> CheckResult:
    """Lightweight LLM ping — verify connectivity before the cycle starts.

    Routes through the **same** shared ``create_llm_client()`` factory that
    the runtime path uses, so every provider (anthropic, openai, openrouter,
    acp, ...) is validated identically at preflight and at execution time.

    *ahvs_config* is an ``AHVSConfig`` instance (or None).  When provided
    the factory shim is built from it; otherwise a minimal shim is
    constructed from the individual arguments for backward compatibility.
    """
    effective_provider = provider or (
        getattr(ahvs_config, "llm_provider", "anthropic") if ahvs_config else "anthropic"
    )

    # Fast-fail: non-ACP providers require an API key.
    if effective_provider != "acp" and not api_key:
        return CheckResult(
            name="ahvs_llm_connectivity",
            status="fail",
            detail="No API key configured (checked env var and config)",
            fix="Set the API key env var (e.g. ANTHROPIC_API_KEY) or pass --api-key-env",
        )

    try:
        from ahvs.llm import create_llm_client

        shim = _build_preflight_shim(
            api_key=api_key, model=model, base_url=base_url,
            provider=provider, ahvs_config=ahvs_config,
        )
        client = create_llm_client(shim)
        ok, detail = client.preflight()
        if ok:
            return CheckResult(
                name="ahvs_llm_connectivity",
                status="pass",
                detail=f"LLM reachable ({detail})",
            )
        return CheckResult(
            name="ahvs_llm_connectivity",
            status="fail",
            detail=f"LLM connectivity failed: {detail}",
            fix="Check your API key, model name, and network connectivity",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("LLM connectivity check failed: %s", exc)
        return CheckResult(
            name="ahvs_llm_connectivity",
            status="fail",
            detail=f"LLM connectivity check failed: {exc}",
            fix="Check your API key, model name, and network connectivity",
        )


# -- Preflight shim helpers ------------------------------------------------

class _PreflightAcpShim:
    """Minimal ACP config shim for preflight."""
    __slots__ = ("agent", "cwd", "acpx_command", "session_name", "timeout_sec")

    def __init__(self, ahvs_config: object | None) -> None:
        self.agent = getattr(ahvs_config, "acp_agent", "claude") if ahvs_config else "claude"
        self.cwd = getattr(ahvs_config, "acp_cwd", ".") if ahvs_config else "."
        self.acpx_command = getattr(ahvs_config, "acpx_command", "") if ahvs_config else ""
        self.session_name = getattr(ahvs_config, "acp_session_name", "ahvs") if ahvs_config else "ahvs"
        self.timeout_sec = getattr(ahvs_config, "acp_timeout_sec", 1800) if ahvs_config else 1800


class _PreflightLlmShim:
    """Minimal LLM config shim for preflight."""
    __slots__ = ("provider", "base_url", "api_key", "api_key_env",
                 "primary_model", "fallback_models", "acp")

    def __init__(self, *, api_key: str, model: str, base_url: str,
                 provider: str, ahvs_config: object | None) -> None:
        self.provider = provider or "anthropic"
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_env = ""  # already resolved
        self.primary_model = model
        self.fallback_models: tuple[str, ...] = ()
        self.acp = _PreflightAcpShim(ahvs_config)


class _PreflightConfigShim:
    """Minimal config-shaped shim for ``create_llm_client``."""
    __slots__ = ("llm",)

    def __init__(self, llm_shim: _PreflightLlmShim) -> None:
        self.llm = llm_shim


def _build_preflight_shim(
    *, api_key: str, model: str, base_url: str,
    provider: str, ahvs_config: object | None,
) -> _PreflightConfigShim:
    """Build a shim from either an AHVSConfig or raw arguments.

    If *ahvs_config* is provided its fields take priority (the executor
    shim in ``executor.py`` is the authoritative shape; this mirrors it
    for the preflight-only path).
    """
    if ahvs_config is not None:
        llm = _PreflightLlmShim(
            api_key=getattr(ahvs_config, "llm_api_key", api_key),
            model=getattr(ahvs_config, "llm_model", model),
            base_url=getattr(ahvs_config, "llm_base_url", base_url),
            provider=getattr(ahvs_config, "llm_provider", provider),
            ahvs_config=ahvs_config,
        )
    else:
        llm = _PreflightLlmShim(
            api_key=api_key, model=model, base_url=base_url,
            provider=provider, ahvs_config=None,
        )
    return _PreflightConfigShim(llm)


def run_ahvs_preflight(
    baseline_path: Path,
    repo_path: Path,
    regression_guard_path: Path | None = None,
    hypothesis_types: list[str] | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
    skip_llm_check: bool = False,
    llm_provider: str = "",
    ahvs_config: object | None = None,
) -> DoctorReport:
    """Run AHVS pre-flight checks.

    Args:
        baseline_path: Path to .ahvs/baseline_metric.json
        repo_path: Root of the target repository
        regression_guard_path: Optional path to regression_guard.sh
        hypothesis_types: If provided, also check required tools for these types.
            Pass None for the minimal setup check (before hypothesis selection).
        skip_llm_check: If True, skip LLM connectivity check.  Used by the
            Stage 4 secondary preflight where connectivity was already verified
            at Stage 1 — avoids manufacturing a false failure when LLM
            credentials are not re-passed.
        llm_provider: Provider string (e.g. "anthropic", "acp").  Passed to
            ``check_llm_connectivity`` for provider-aware behaviour.
        ahvs_config: Optional AHVSConfig, forwarded to ``check_llm_connectivity``
            for ACP client construction.
    """
    from datetime import datetime, timezone

    checks: list[CheckResult] = [
        check_baseline_metric(baseline_path),
        check_clean_branch(repo_path),
    ]

    commit_check = check_baseline_commit(baseline_path)
    if commit_check is not None:
        checks.append(commit_check)

    # Always run LLM check at setup (Stage 1).  Skip at Stage 4 secondary
    # preflight where we only care about tool availability.
    if not skip_llm_check:
        checks.append(check_llm_connectivity(
            llm_api_key, llm_model, llm_base_url,
            provider=llm_provider, ahvs_config=ahvs_config,
        ))

    if regression_guard_path is not None:
        checks.append(check_regression_guard(regression_guard_path))

    if hypothesis_types:
        required_tools: set[str] = set()
        for h_type in hypothesis_types:
            required_tools.update(HYPOTHESIS_TOOL_REQUIREMENTS.get(h_type, []))
        for tool in sorted(required_tools):
            checks.append(check_tool(tool))

    overall = "fail" if any(c.status == "fail" for c in checks) else "pass"
    return DoctorReport(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checks=checks,
        overall=overall,
    )
