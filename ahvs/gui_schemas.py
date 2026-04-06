"""Predefined GUI form schemas for AHVS skills.

Each schema is a plain dict that can be passed to ``ahvs.gui.run_form()``.
"""

from __future__ import annotations

GENESIS_SCHEMA: dict = {
    "title": "AHVS Genesis",
    "subtitle": "Create a new AHVS project from raw data",
    "submit_label": "Create Project",
    "fields": [
        {
            "name": "problem",
            "type": "textarea",
            "label": "Problem Description",
            "required": True,
            "placeholder": "e.g., Classify customer emails into intent categories",
            "help": "Describe the classification task you want to solve",
        },
        {
            "name": "data_path",
            "type": "text",
            "label": "Data File Path",
            "required": True,
            "placeholder": "/path/to/data.csv",
            "validate_path": True,
            "help": "Path to CSV, TSV, or Parquet file",
        },
        {
            "name": "target_metric",
            "type": "select",
            "label": "Target Metric",
            "required": True,
            "default": "f1_weighted",
            "options": [
                "f1_weighted",
                "accuracy",
                "f1_macro",
                "f1_micro",
                "precision",
                "recall",
            ],
        },
        {
            "name": "output_dir",
            "type": "text",
            "label": "Output Directory",
            "required": True,
            "placeholder": "/home/user/projects/my_classifier",
            "help": "Where the project will be created — must be provided, never auto-generated",
        },
        {
            "name": "mode",
            "type": "radio",
            "label": "Execution Mode",
            "required": True,
            "default": "pipeline",
            "options": [
                {
                    "value": "pipeline",
                    "label": "Pipeline — fast and deterministic, needs classes upfront",
                },
                {
                    "value": "agent",
                    "label": "Agent — smarter but slower, auto-discovers classes",
                },
            ],
        },
        {
            "name": "classes",
            "type": "text",
            "label": "Classes (comma-separated)",
            "placeholder": "urgent, question, feedback, spam",
            "help": "Required for pipeline mode, optional for agent mode",
            "show_when": {"mode": "pipeline"},
        },
        {
            "name": "input_column",
            "type": "text",
            "label": "Input Column Name",
            "default": "text",
            "help": "Column in the CSV containing the text to classify",
        },
    ],
}


MULTIAGENT_SCHEMA: dict = {
    "title": "AHVS Multi-Agent Cycle",
    "subtitle": "Run an AHVS optimization cycle with executor + observer supervision",
    "submit_label": "Start Cycle",
    "fields": [
        {
            "name": "repo_path",
            "type": "text",
            "label": "Target Repository Path",
            "required": True,
            "placeholder": "/path/to/your/repo",
            "validate_path": True,
        },
        {
            "name": "question",
            "type": "textarea",
            "label": "Cycle Question",
            "required": True,
            "placeholder": "How can we improve answer_relevance by 5%?",
            "help": "The optimization question AHVS will try to answer",
        },
        {
            "name": "provider",
            "type": "select",
            "label": "LLM Provider",
            "default": "acp",
            "options": [
                {"value": "acp", "label": "ACP (local Claude agent — recommended)"},
                {"value": "anthropic", "label": "Anthropic API"},
                {"value": "openai", "label": "OpenAI API"},
                {"value": "openrouter", "label": "OpenRouter"},
                {"value": "deepseek", "label": "DeepSeek"},
            ],
        },
        {
            "name": "model",
            "type": "text",
            "label": "LLM Model",
            "default": "anthropic/claude-sonnet-4-20250514",
            "help": "Model identifier (provider-specific)",
        },
        {
            "name": "max_hypotheses",
            "type": "select",
            "label": "Max Hypotheses to Generate",
            "default": "3",
            "options": ["1", "2", "3", "4", "5"],
        },
        {
            "name": "auto_approve",
            "type": "checkbox",
            "label": "Auto-approve all hypotheses (skip selection GUI)",
            "default": False,
            "help": "When checked, skips the hypothesis selection step and runs all generated hypotheses",
        },
        {
            "name": "domain",
            "type": "select",
            "label": "Domain",
            "default": "llm",
            "options": [
                {"value": "llm", "label": "LLM / RAG optimization"},
                {"value": "ml", "label": "Traditional ML (sklearn, etc.)"},
            ],
        },
    ],
}


ONBOARDING_SCHEMA: dict = {
    "title": "AHVS Onboarding",
    "subtitle": "Prepare a repository for AHVS optimization",
    "submit_label": "Start Onboarding",
    "fields": [
        {
            "name": "repo_path",
            "type": "text",
            "label": "Repository Path",
            "required": True,
            "placeholder": "/path/to/your/repo",
            "validate_path": True,
        },
        {
            "name": "metric_name",
            "type": "text",
            "label": "Primary Metric Name",
            "required": True,
            "placeholder": "e.g., f1_weighted, precision, accuracy",
            "help": "The metric AHVS will optimize",
        },
        {
            "name": "eval_command",
            "type": "text",
            "label": "Eval Command",
            "placeholder": "python run_eval.py --eval-only",
            "help": "Leave blank to let the skill discover or create one automatically",
        },
        {
            "name": "notes",
            "type": "textarea",
            "label": "Additional Notes",
            "placeholder": "Any context about the repo, eval setup, or constraints...",
            "help": "Optional — helps the onboarding skill make better decisions",
        },
    ],
}
