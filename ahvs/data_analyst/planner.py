"""Phase 2: Goal alignment and analysis plan generation.

Uses a single LLM call (via ACP or direct API) to resolve column roles,
determine task type, and select which modules to run. Falls back to a
heuristic planner when no LLM is available.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ahvs.data_analyst.models import AnalysisPlan, DataProfile, ModuleSpec
from ahvs.data_analyst import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM planner prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a data analysis planner. Given a user's goal and a dataset profile,
produce a JSON analysis plan.

Return ONLY valid JSON with this schema:
{
  "task_type": "<e.g. multiclass_classification, binary_classification, regression, ner, absa>",
  "input_columns": ["<col1>", "<col2>"],
  "label_column": "<col or null>",
  "modules": [
    {"name": "<module_name>", "params": {}, "reason": "<why>"}
  ],
  "notes": "<any reasoning>"
}

Available modules: %MODULES%

Rules:
- Only select modules that are relevant to the task and data.
- Always include "eda" as the first module.
- Include "class_balance" if a label column exists.
- Include "text_stats" if text input columns exist.
- Include "duplicates" if text input columns exist.
- Only include "subsample" if the dataset has >5000 rows.
- Only include "split" if the user goal implies training.
- Only include "export" if the user explicitly asks for output files.
- For subsample, set params.target_size and params.strategy.
"""


def plan(
    profile: DataProfile,
    goal: str,
    *,
    llm_client: Any | None = None,
    modules_override: list[str] | None = None,
) -> AnalysisPlan:
    """Generate an analysis plan.

    Args:
        profile: DataProfile from Phase 1.
        goal: User's natural-language goal (e.g. "build ABSA classifier").
        llm_client: Optional LLM client with a ``chat(messages)`` method.
            If None, falls back to heuristic planning.
        modules_override: If provided, use exactly these modules instead of
            asking the LLM.

    Returns:
        AnalysisPlan with ordered module list.
    """
    if modules_override:
        return _plan_from_override(profile, modules_override)

    if llm_client:
        try:
            return _plan_via_llm(profile, goal, llm_client)
        except Exception:
            logger.warning("LLM planner failed — falling back to heuristic", exc_info=True)

    return _plan_heuristic(profile, goal)


# ---------------------------------------------------------------------------
# Heuristic planner (no LLM needed)
# ---------------------------------------------------------------------------


def _plan_heuristic(profile: DataProfile, goal: str) -> AnalysisPlan:
    """Rule-based planner that selects modules based on data characteristics."""
    available = set(registry.available())
    modules: list[ModuleSpec] = []

    # Always start with EDA
    if "eda" in available:
        modules.append(ModuleSpec(name="eda", reason="Always run EDA first."))

    # Class balance if labels exist
    if profile.label_column and "class_balance" in available:
        modules.append(
            ModuleSpec(name="class_balance", reason="Label column detected.")
        )

    # Text stats if text columns
    has_text = any(
        ci.role == "text_input" for ci in profile.columns
    )
    if has_text and "text_stats" in available:
        modules.append(
            ModuleSpec(name="text_stats", reason="Text input columns detected.")
        )

    # Duplicate detection for text
    if has_text and "duplicates" in available:
        modules.append(
            ModuleSpec(name="duplicates", reason="Check for redundant text samples.")
        )

    # Subsampling for large datasets
    if profile.total_rows > 5000 and "subsample" in available:
        target = min(5000, profile.total_rows // 2)
        modules.append(
            ModuleSpec(
                name="subsample",
                params={"target_size": target, "strategy": "stratified"},
                reason=f"Large dataset ({profile.total_rows} rows).",
            )
        )

    # Split if goal mentions training / classifier / model
    train_keywords = {"train", "classifier", "model", "classify", "classification", "fine-tune"}
    if any(kw in goal.lower() for kw in train_keywords) and "split" in available:
        modules.append(
            ModuleSpec(
                name="split",
                params={"train": 0.8, "val": 0.1, "test": 0.1},
                reason="Goal implies model training.",
            )
        )

    # Infer task type from goal
    task_type = _infer_task_type(goal, profile)

    return AnalysisPlan(
        task_type=task_type,
        input_columns=profile.input_columns,
        label_column=profile.label_column,
        modules=modules,
        notes="Generated by heuristic planner (no LLM).",
    )


def _infer_task_type(goal: str, profile: DataProfile) -> str:
    """Best-effort task type inference from goal text."""
    gl = goal.lower()
    if "regression" in gl:
        return "regression"
    if "ner" in gl or "entity" in gl:
        return "ner"
    if "absa" in gl or "aspect" in gl:
        return "absa"
    if "sentiment" in gl:
        return "sentiment_classification"
    if profile.class_distribution:
        n_classes = len(profile.class_distribution)
        if n_classes == 2:
            return "binary_classification"
        return "multiclass_classification"
    return "classification"


# ---------------------------------------------------------------------------
# LLM-assisted planner
# ---------------------------------------------------------------------------


def _plan_via_llm(
    profile: DataProfile, goal: str, llm_client: Any
) -> AnalysisPlan:
    """Call LLM to produce a plan."""
    available_modules = registry.available()
    system = _SYSTEM_PROMPT.replace("%MODULES%", ", ".join(available_modules))
    user_msg = (
        f"Goal: {goal}\n\n"
        f"Dataset profile:\n{profile.summary_for_llm()}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    response = llm_client.chat(messages)
    text = response if isinstance(response, str) else response.get("content", "")

    # Extract JSON from response (handle markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)

    # --- M1: Validate LLM output against profile and registry ---
    valid_columns = {c.name for c in profile.columns}

    # Validate input_columns — reject any that don't exist
    llm_inputs = data.get("input_columns", [])
    validated_inputs = [c for c in llm_inputs if c in valid_columns]
    if len(validated_inputs) != len(llm_inputs):
        rejected = set(llm_inputs) - valid_columns
        logger.warning("LLM planner hallucinated columns, rejected: %s", rejected)
    # Fall back to profile if LLM produced nothing valid
    if not validated_inputs:
        validated_inputs = profile.input_columns

    # Validate label_column
    llm_label = data.get("label_column")
    validated_label = llm_label if llm_label in valid_columns else profile.label_column

    # Validate modules — only accept registered ones, enforce EDA first
    modules = []
    for m in data.get("modules", []):
        name = m.get("name", "")
        if name in available_modules:
            modules.append(
                ModuleSpec(
                    name=name,
                    params=m.get("params", {}),
                    reason=m.get("reason", ""),
                )
            )
        else:
            logger.warning("LLM planner suggested unknown module '%s' — skipped.", name)

    # Enforce: EDA must be first if present
    eda_indices = [i for i, m in enumerate(modules) if m.name == "eda"]
    if eda_indices and eda_indices[0] != 0:
        eda_mod = modules.pop(eda_indices[0])
        modules.insert(0, eda_mod)

    # Fall back to heuristic if LLM produced no valid modules
    if not modules:
        logger.warning("LLM plan had no valid modules — falling back to heuristic.")
        return _plan_heuristic(profile, goal)

    return AnalysisPlan(
        task_type=data.get("task_type", "classification"),
        input_columns=validated_inputs,
        label_column=validated_label,
        modules=modules,
        notes=data.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Override planner (user specifies exact modules)
# ---------------------------------------------------------------------------


def _plan_from_override(
    profile: DataProfile, module_names: list[str]
) -> AnalysisPlan:
    """Build a plan from an explicit module list."""
    available = set(registry.available())
    modules = []
    for name in module_names:
        if name in available:
            modules.append(ModuleSpec(name=name, reason="User-specified."))
        else:
            logger.warning("Module '%s' not found in registry — skipped.", name)

    return AnalysisPlan(
        task_type="",
        input_columns=profile.input_columns,
        label_column=profile.label_column,
        modules=modules,
        notes="User-specified module list.",
    )
