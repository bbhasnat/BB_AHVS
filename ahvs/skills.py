"""AHVS Skill Library — pre-built skill templates injected into execution context.

Skills describe what tools are available and how to invoke them. Claude Code
reads skill descriptions in its context, picks the right skill for the
hypothesis type, and references it in its implementation plan.

Note: Skills are informational guidance, not executable dispatch. AHVS does
not resolve or enforce skill invocations at runtime — the execution agent
decides what to execute based on the skill descriptions in its prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillSpec:
    """A single skill available during hypothesis execution."""

    name: str
    description: str
    invocation_template: str
    applicable_types: tuple[str, ...]  # hypothesis types, or ("*",) for all
    required_tools: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Built-in skill library
# ---------------------------------------------------------------------------

BUILTIN_SKILLS: list[SkillSpec] = [
    SkillSpec(
        name="promptfoo_eval",
        description=(
            "Run a Promptfoo evaluation against a YAML config and capture the primary metric. "
            "Invokes: promptfoo eval --config <path> --output json. "
            "Parses the JSON output and returns the named metric value. "
            "Use for: prompt_rewrite, model_comparison, config_change."
        ),
        invocation_template=(
            "SKILL: promptfoo_eval\n"
            "  config_path: <path to promptfoo YAML config>\n"
            "  metric_key: <name of metric to extract from output>\n"
            "  output_path: tool_runs/{hypothesis_id}/promptfoo_output.json"
        ),
        applicable_types=("prompt_rewrite", "model_comparison", "config_change"),
        required_tools=("promptfoo",),
    ),
    SkillSpec(
        name="dspy_compile",
        description=(
            "Run DSPy optimization on a module, then run a Promptfoo held-out evaluation. "
            "Two-step: (1) DSPy compile → optimized compiled program, "
            "(2) Promptfoo eval on held-out dataset. "
            "Use for: dspy_optimize."
        ),
        invocation_template=(
            "SKILL: dspy_compile\n"
            "  module_path: <path to DSPy module (.py)>\n"
            "  optimizer: BootstrapFewShot  # or MIPRO\n"
            "  compiled_path: tool_runs/{hypothesis_id}/compiled_program.json\n"
            "  then: promptfoo_eval\n"
            "  promptfoo_config: <path to held-out Promptfoo config YAML>"
        ),
        applicable_types=("dspy_optimize",),
        required_tools=("dspy", "promptfoo"),
    ),
    SkillSpec(
        name="phoenix_eval",
        description=(
            "Run an Arize Phoenix evaluation for LLM response quality. "
            "Supports qa_correctness, hallucination, toxicity, and custom rubrics. "
            "Use when Promptfoo is unavailable or when Phoenix-specific eval types are needed."
        ),
        invocation_template=(
            "SKILL: phoenix_eval\n"
            "  eval_type: qa_correctness  # or hallucination, toxicity, custom\n"
            "  dataset_path: <path to eval dataset (.jsonl or .csv)>\n"
            "  output_path: tool_runs/{hypothesis_id}/phoenix_output.json\n"
            "  metric_key: score"
        ),
        applicable_types=("prompt_rewrite", "model_comparison", "config_change", "code_change"),
        required_tools=("arize-phoenix",),
    ),
    SkillSpec(
        name="sandbox_run",
        description=(
            "Execute a Python script in an isolated sandbox and capture stdout as JSON metrics. "
            "The script must print a JSON object to stdout as its final line, "
            "containing at least the primary metric key as a float. "
            "Use for: code_change, architecture_change, multi_llm_judge."
        ),
        invocation_template=(
            "SKILL: sandbox_run\n"
            "  entry_point: <path to Python entry script>\n"
            "  output_schema: {\"<primary_metric>\": float}  # must match baseline metric name\n"
            "  timeout_seconds: 300\n"
            "  output_path: tool_runs/{hypothesis_id}/sandbox_output.json"
        ),
        applicable_types=("code_change", "architecture_change", "multi_llm_judge"),
        required_tools=(),  # runs locally, no Docker
    ),
    SkillSpec(
        name="regression_guard",
        description=(
            "Run the configured regression guard script against a result file. "
            "Exit 0 = regression check passed. Exit non-zero = regression detected, revert. "
            "Always invoke after hypothesis execution before recording the final result."
        ),
        invocation_template=(
            "SKILL: regression_guard\n"
            "  results_path: tool_runs/{hypothesis_id}/result.json\n"
            "  # guard_script path is resolved from AHVSConfig at runtime"
        ),
        applicable_types=("*",),
        required_tools=(),
    ),
    SkillSpec(
        name="metric_capture",
        description=(
            "Parse a JSON output file and extract the primary metric as a float. "
            "Supports dot-separated key paths (e.g. 'results.0.score'). "
            "Use when the eval tool writes output to JSON and you need to normalise it."
        ),
        invocation_template=(
            "SKILL: metric_capture\n"
            "  source_path: <path to eval output JSON file>\n"
            "  metric_key: <dot-separated key path, e.g. 'results.0.score'>"
        ),
        applicable_types=("*",),
        required_tools=(),
    ),
]


# ---------------------------------------------------------------------------
# SkillLibrary
# ---------------------------------------------------------------------------


class SkillLibrary:
    """Manages the AHVS skill library — built-ins plus optional custom registry."""

    def __init__(self, custom_registry_path: Path | None = None) -> None:
        self._skills: list[SkillSpec] = list(BUILTIN_SKILLS)
        if custom_registry_path and custom_registry_path.exists():
            self._skills.extend(self._load_custom(custom_registry_path))

    def _load_custom(self, path: Path) -> list[SkillSpec]:
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            result = []
            for entry in data.get("skills", []):
                result.append(
                    SkillSpec(
                        name=entry["name"],
                        description=entry["description"],
                        invocation_template=entry["invocation_template"],
                        applicable_types=tuple(entry.get("applicable_types", [])),
                        required_tools=tuple(entry.get("required_tools", [])),
                    )
                )
            logger.info("Loaded %d custom skills from %s", len(result), path)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load custom skills from %s: %s", path, exc)
            return []

    def for_hypothesis_type(
        self,
        hypothesis_type: str,
        available_tools: set[str],
    ) -> list[SkillSpec]:
        """Return skills applicable to this hypothesis type with all required tools available."""
        return [
            s for s in self._skills
            if ("*" in s.applicable_types or hypothesis_type in s.applicable_types)
            and all(t in available_tools for t in s.required_tools)
        ]

    def detect_available_tools(self) -> set[str]:
        """Return the set of tool names currently available in this environment."""
        import importlib.util as ilu
        import subprocess as sp

        available: set[str] = set()
        all_tools: set[str] = set()
        for s in self._skills:
            all_tools.update(s.required_tools)

        for tool in all_tools:
            # CLI check
            if __import__("shutil").which(tool):
                available.add(tool)
                continue
            # Python package check
            pkg = tool.replace("-", "_")
            if ilu.find_spec(pkg) is not None:
                available.add(tool)
                continue
            # npx fallback for Node tools
            if tool == "promptfoo":
                try:
                    r = sp.run(
                        ["npx", tool, "--version"],
                        capture_output=True, timeout=8, check=False,
                    )
                    if r.returncode == 0:
                        available.add(tool)
                except (FileNotFoundError, sp.TimeoutExpired, OSError):
                    pass

        return available

    def to_context_block(self, skills: list[SkillSpec]) -> str:
        """Render skills as a context block for injection into the execution prompt."""
        if not skills:
            return ""
        lines = [
            "=== AVAILABLE AHVS SKILLS ===",
            "These skills describe tools and patterns available in this environment.",
            "Use them as guidance for your implementation — pick the right approach",
            "for the hypothesis type and invoke the underlying tools directly in your code.",
            "",
        ]
        for skill in skills:
            lines.append(f"SKILL: {skill.name}")
            lines.append(f"  Description: {skill.description}")
            lines.append("  Invocation template:")
            for tpl_line in skill.invocation_template.splitlines():
                lines.append(f"    {tpl_line}")
            lines.append("")
        return "\n".join(lines)
