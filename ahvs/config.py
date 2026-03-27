"""AHVSConfig — configuration for a single AHVS hypothesis-validation cycle."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _default_run_dir(repo_path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return repo_path / ".ahvs" / "cycles" / ts


@dataclass
class AHVSConfig:
    """Full configuration for one AHVS cycle."""

    # ── Target ────────────────────────────────────────────────────────────
    repo_path: Path
    question: str

    # ── Cycle settings ────────────────────────────────────────────────────
    run_dir: Path = field(default=None)  # type: ignore[assignment]
    max_hypotheses: int = 3  # soft default; hard max enforced at 5
    max_lesson_cycles: int = 5  # load lessons from last K recent non-failed cycle IDs (0 = unlimited)

    # ── Guards ────────────────────────────────────────────────────────────
    regression_guard_path: Path | None = None
    apply_best: bool = False          # auto-apply best hypothesis patch after cycle

    # ── Skill system ──────────────────────────────────────────────────────
    skill_registry_path: Path | None = None  # custom skills YAML

    # ── Prompts ───────────────────────────────────────────────────────────
    prompts_override_path: Path | None = None  # override default AHVS prompts

    # ── LLM settings ─────────────────────────────────────────────────────
    llm_provider: str = "anthropic"  # "anthropic" | "openai" | "openai-compatible" | "acp"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "claude-opus-4-6"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"

    # ── Memory lifecycle ──────────────────────────────────────────────
    memory_stale_days: int = 60      # prepend [STALE] marker to old memory files
    memory_archive_days: int = 120   # move very old memory files to archive/

    # ── Cross-project learning ─────────────────────────────────────────
    enable_cross_project: bool = True
    global_evolution_dir: Path = field(default=None)  # type: ignore[assignment]

    # ── Eval settings ─────────────────────────────────────────────────
    eval_timeout_sec: int = 600  # eval_command timeout; baseline_metric.json "eval_timeout" overrides

    # ── ACP settings (only used when llm_provider == "acp") ───────────
    acp_agent: str = "claude"
    acp_cwd: str = "."           # resolved to repo_path in __post_init__
    acpx_command: str = ""       # auto-detect if empty
    acp_session_name: str = "ahvs"
    acp_timeout_sec: int = 1800  # per-prompt timeout

    def __post_init__(self) -> None:
        self.repo_path = Path(self.repo_path).resolve()
        if self.run_dir is None:
            self.run_dir = _default_run_dir(self.repo_path)
        else:
            self.run_dir = Path(self.run_dir).resolve()
        if not self.llm_api_key and self.llm_provider != "acp":
            self.llm_api_key = os.environ.get(self.llm_api_key_env, "")
        if self.acp_cwd == ".":
            self.acp_cwd = str(self.repo_path)
        if self.max_hypotheses > 5:
            raise ValueError("max_hypotheses cannot exceed 5 (AHVS hard cap)")
        if self.max_hypotheses < 1:
            raise ValueError("max_hypotheses must be at least 1")
        if self.global_evolution_dir is None:
            self.global_evolution_dir = Path.home() / ".ahvs" / "global" / "evolution"
        else:
            self.global_evolution_dir = Path(self.global_evolution_dir).resolve()

    # ── Derived paths ─────────────────────────────────────────────────────

    @property
    def baseline_path(self) -> Path:
        return self.repo_path / ".ahvs" / "baseline_metric.json"

    @property
    def evolution_dir(self) -> Path:
        return self.repo_path / ".ahvs" / "evolution"

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "AHVSConfig":
        return cls(
            repo_path=Path(args.repo),
            question=args.question,
            run_dir=Path(args.run_dir) if getattr(args, "run_dir", None) else None,
            max_hypotheses=getattr(args, "max_hypotheses", 3),
            max_lesson_cycles=getattr(args, "max_lesson_cycles", 5),
            regression_guard_path=(
                Path(args.regression_guard)
                if getattr(args, "regression_guard", None)
                else None
            ),
            skill_registry_path=(
                Path(args.skill_registry)
                if getattr(args, "skill_registry", None)
                else None
            ),
            prompts_override_path=(
                Path(args.prompts)
                if getattr(args, "prompts", None)
                else None
            ),
            apply_best=getattr(args, "apply_best", False),
            llm_provider=getattr(args, "provider", "anthropic") or "anthropic",
            llm_base_url=getattr(args, "base_url", "") or "",
            llm_model=getattr(args, "model", "claude-opus-4-6") or "claude-opus-4-6",
            llm_api_key_env=getattr(args, "api_key_env", "ANTHROPIC_API_KEY") or "ANTHROPIC_API_KEY",
            acp_agent=getattr(args, "acp_agent", "claude") or "claude",
            acpx_command=getattr(args, "acpx_command", "") or "",
            acp_session_name=getattr(args, "acp_session_name", "ahvs") or "ahvs",
            acp_timeout_sec=getattr(args, "acp_timeout_sec", 1800) or 1800,
            eval_timeout_sec=getattr(args, "eval_timeout_sec", 600) or 600,
        )
