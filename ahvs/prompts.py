"""AHVS prompt manager — standalone prompt registry for the 4 AHVS stage prompts.

Follows the RenderedPrompt / PromptManager pattern but is fully independent.

Supports optional YAML overrides via ahvs_prompts.yaml for customisation
without touching Python source.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _render(template: str, variables: dict[str, str]) -> str:
    """Replace {var_name} placeholders, leaving JSON schemas untouched."""

    def _replacer(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(variables[key]) if key in variables else m.group(0)

    return re.sub(r"\{(\w+)\}", _replacer, template)


# ---------------------------------------------------------------------------
# RenderedPrompt (inlined)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedPrompt:
    """Fully rendered prompt ready for ``llm.chat()``."""

    system: str
    user: str
    json_mode: bool = False
    max_tokens: int | None = None


# ---------------------------------------------------------------------------
# Default AHVS prompt definitions
# ---------------------------------------------------------------------------

_AHVS_STAGES: dict[str, dict[str, Any]] = {
    # ── Stage 3: Hypothesis Generation ────────────────────────────────────
    "ahvs_hypothesis_gen": {
        "system": (
            "You are an expert AI engineer specialising in LLM/RAG system optimisation. "
            "Your task is to generate concrete, testable hypotheses to improve an LLM or "
            "RAG system based on a specific question and the system's current performance context. "
            "Be specific, practical, and grounded in the provided baseline and prior lessons."
        ),
        "user": (
            "# AHVS Hypothesis Generation\n\n"
            "## Cycle Question\n{question}\n\n"
            "## Baseline Performance\n"
            "- Primary metric: {metric_name}\n"
            "- Current value: {baseline_value}\n"
            "- Eval command: {eval_command}\n\n"
            "## Domain Tags\n{domain_tags}\n\n"
            "## Operator Context\n{enriched_context}\n\n"
            "## Prior Lessons (from previous cycles)\n{prior_lessons}\n\n"
            "## Rejected Approaches (do not repeat these)\n{rejected_approaches}\n\n"
            "---\n\n"
            "Generate {max_hypotheses} concrete hypotheses to address the cycle question. "
            "Each hypothesis must:\n"
            "1. Be falsifiable and measurable against the baseline metric\n"
            "2. Have a clear type from: prompt_rewrite | model_comparison | config_change | "
            "dspy_optimize | code_change | architecture_change | multi_llm_judge\n"
            "3. Avoid repeating any rejected approaches listed above\n"
            "4. Be achievable in a single AHVS cycle\n"
            "5. For code_change and architecture_change types: propose NEW algorithms, "
            "data structures, retrieval strategies, or pipeline components — not just "
            "prompt wording changes or config tweaks\n"
            "6. Prefer diverse types across hypotheses — do not generate all prompt_rewrite\n"
            "7. If the eval command uses --eval-only, do NOT generate prompt_rewrite or "
            "model_comparison hypotheses — they are structurally unmeasurable because "
            "--eval-only reads frozen checkpoint data. Also avoid hypotheses that ONLY "
            "modify pre-inference files (post_selector, llm_client) since those changes "
            "are invisible during reparse. Focus on code_change hypotheses that modify "
            "parsing logic, scoring, or classification code.\n\n"
            "Type guidance:\n"
            "- prompt_rewrite: ONLY system prompt, few-shot, instruction changes. No code.\n"
            "- model_comparison: swap model IDs, compare across providers.\n"
            "- config_change: tune hyperparameters (temperature, chunk_size, top_k). Config only.\n"
            "- dspy_optimize: create DSPy modules to optimize prompt pipelines programmatically.\n"
            "- code_change: modify algorithms, retrieval logic, ranking functions. Real code changes.\n"
            "- architecture_change: redesign components — new pipelines, caching, re-rankers, hybrid search.\n"
            "- multi_llm_judge: implement multi-model consensus or judge-based evaluation.\n\n"
            "Format each hypothesis EXACTLY as:\n\n"
            "## H1\n"
            "**Type:** <hypothesis_type>\n"
            "**Description:** <what to change and how>\n"
            "**Rationale:** <why this might improve the metric, referencing prior lessons if relevant>\n"
            "**Estimated Cost:** low | medium | high\n"
            "**Required Tools:** <comma-separated list, e.g. promptfoo, docker>\n\n"
            "## H2\n"
            "..."
        ),
        "json_mode": False,
        "max_tokens": 2000,
    },

    # ── Stage 5: Validation Planning ──────────────────────────────────────
    "ahvs_validation_plan": {
        "system": (
            "You are an expert AI engineer. Your task is to create a detailed validation plan "
            "for selected hypotheses. The plan specifies exactly how each hypothesis will be "
            "implemented and evaluated against the baseline metric."
        ),
        "user": (
            "# AHVS Validation Plan\n\n"
            "## Cycle Question\n{question}\n\n"
            "## Baseline\n"
            "- Metric: {metric_name} = {baseline_value}\n"
            "- Eval command: {eval_command}\n\n"
            "## Selected Hypotheses\n{selected_hypotheses_text}\n\n"
            "## Available Skills\n{available_skills_block}\n\n"
            "---\n\n"
            "For each selected hypothesis, produce a validation plan section.\n"
            "Format EXACTLY as:\n\n"
            "## H1\n"
            "**Implementation Approach:** <step-by-step description of what to build or modify>\n"
            "**Eval Method:** promptfoo | phoenix | custom_script | dspy+promptfoo\n"
            "**Skill:** <skill name from Available Skills>\n"
            "**Success Criterion:** {metric_name} >= <target_value> "
            "(current: {baseline_value}, required delta: <+X.XX>)\n"
            "**Expected Artifacts:** <comma-separated list of files to produce>\n\n"
            "## H2\n"
            "..."
        ),
        "json_mode": False,
        "max_tokens": 2500,
    },

    # ── Stage 7: Report Writing ────────────────────────────────────────────
    "ahvs_report": {
        "system": (
            "You are an expert AI engineer writing a cycle report for an AHVS "
            "hypothesis-validation cycle. Be concise, specific, and evidence-based. "
            "Every claim must reference the results data provided."
        ),
        "user": (
            "# AHVS Cycle Report\n\n"
            "## Cycle Question\n{question}\n\n"
            "## Baseline\n"
            "- Metric: {metric_name} = {baseline_value}\n\n"
            "## Results\n{results_summary}\n\n"
            "---\n\n"
            "Write a structured cycle report answering ALL of the following questions:\n\n"
            "1. **What was the question this cycle?**\n"
            "   Restate it clearly.\n\n"
            "2. **What hypotheses were run?**\n"
            "   List each with its type and description.\n\n"
            "3. **What were the results vs baseline?**\n"
            "   For each hypothesis: metric value, delta, regression guard result.\n\n"
            "4. **What should be kept?**\n"
            "   Which hypothesis (if any) should be applied? Justify with evidence.\n"
            "   If none improved the metric, state that explicitly.\n\n"
            "5. **What was learned?**\n"
            "   Key insight from this cycle, regardless of outcome.\n\n"
            "6. **What approaches are now ruled out?**\n"
            "   List anything that should not be tried again and why.\n\n"
            "7. **What should the next cycle explore?**\n"
            "   One concrete suggestion for the next hypothesis question.\n\n"
            "Be direct and specific. Avoid vague language."
        ),
        "json_mode": False,
        "max_tokens": 1500,
    },
}


# ---------------------------------------------------------------------------
# AHVSPromptManager
# ---------------------------------------------------------------------------


class AHVSPromptManager:
    """Standalone prompt registry for AHVS stage prompts.

    Supports optional YAML overrides. Compatible with RenderedPrompt
    output type.
    """

    def __init__(self, overrides_path: Path | None = None) -> None:
        self._stages: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in _AHVS_STAGES.items()
        }
        if overrides_path and overrides_path.exists():
            self._load_overrides(overrides_path)

    def _load_overrides(self, path: Path) -> None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            logger.warning("Bad AHVS prompts YAML %s: %s — using defaults", path, exc)
            return
        for stage_name, stage_data in (data.get("stages") or {}).items():
            if isinstance(stage_data, dict):
                if stage_name not in self._stages:
                    self._stages[stage_name] = {}
                self._stages[stage_name].update(stage_data)
        logger.info("Loaded AHVS prompt overrides from %s", path)

    def for_stage(self, stage: str, **kwargs: Any) -> RenderedPrompt:
        """Return a fully rendered prompt for *stage* with variables substituted."""
        if stage not in self._stages:
            raise KeyError(f"Unknown AHVS stage prompt: '{stage}'")
        entry = self._stages[stage]
        kw = {k: str(v) for k, v in kwargs.items()}
        return RenderedPrompt(
            system=_render(entry.get("system", ""), kw),
            user=_render(entry.get("user", ""), kw),
            json_mode=entry.get("json_mode", False),
            max_tokens=entry.get("max_tokens"),
        )
