"""KD Classifier solver — bridges AHVS genesis to the KD pipeline.

Supports two execution modes:
  - **pipeline** (Option A): Direct call to ``run_auto_ml_pipeline()`` via
    subprocess.  Requires pre-built config + spec YAML.  Deterministic.
  - **agent** (Option B): Drives the KD Agent (claude-agent-sdk) which
    inspects the data, generates spec/config, and runs all stages
    autonomously.  Smarter but requires ``claude-agent-sdk``.

The mode is selected via ``config_overrides["mode"]`` (default: ``"pipeline"``).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ahvs.genesis.contract import GenesisResult

logger = logging.getLogger(__name__)

# Default annotation model — never gpt-4o (cost policy)
_DEFAULT_ANNOTATION_MODEL = "gpt-4.1-mini"

# Default TML model
_DEFAULT_TML_MODEL = "distilbert-base-uncased"

# Valid execution modes
_VALID_MODES = ("pipeline", "agent")


class KDClassifierSolver:
    """Build a text classifier via the KD (Knowledge Distillation) pipeline.

    Supports two modes:
      - ``pipeline``: generates spec/config YAML, calls ``run_auto_ml_pipeline``
        via subprocess.  Reliable, deterministic.
      - ``agent``: uses the KD Agent (claude-agent-sdk) to inspect data,
        auto-generate spec/config, and drive the full pipeline.  Smarter
        (adapts to data), but requires ``claude-agent-sdk``.
    """

    name: str = "kd_classifier"
    problem_types: list[str] = [
        "classification", "multi-label", "sentiment", "intent",
    ]

    def __init__(
        self,
        kd_repo_path: str = "",
        default_model: str = _DEFAULT_TML_MODEL,
        default_annotation_model: str = _DEFAULT_ANNOTATION_MODEL,
        conda_env: str = "",
    ) -> None:
        self.kd_repo_path = Path(kd_repo_path).resolve() if kd_repo_path else None
        self.default_model = default_model
        self.default_annotation_model = default_annotation_model
        self.conda_env = conda_env

    def solve(
        self,
        problem: str,
        data_path: str,
        target_metric: str,
        output_dir: str,
        config_overrides: dict | None = None,
    ) -> GenesisResult:
        """Run the KD pipeline end-to-end and return a GenesisResult.

        Set ``config_overrides["mode"]`` to ``"pipeline"`` (default) or
        ``"agent"`` to choose the execution mode.
        """
        output = Path(output_dir).resolve()
        data = Path(data_path).resolve()
        overrides = config_overrides or {}
        mode = overrides.pop("mode", "pipeline")

        if mode not in _VALID_MODES:
            return GenesisResult(
                project_dir=output,
                baseline_metric={},
                eval_command="",
                errors=[f"Invalid mode {mode!r}. Valid: {_VALID_MODES}"],
            )

        # Validate inputs
        errors = self._validate(data, output)
        if errors:
            return GenesisResult(
                project_dir=output, baseline_metric={}, eval_command="",
                errors=errors,
            )

        # Ensure KD repo is findable
        kd_path = self._resolve_kd_path()
        if not kd_path:
            return GenesisResult(
                project_dir=output, baseline_metric={}, eval_command="",
                errors=["KD repo not found. Set kd_repo_path in solvers.yaml."],
            )

        output.mkdir(parents=True, exist_ok=True)

        # Dispatch to the selected mode
        if mode == "agent":
            result = self._run_agent(kd_path, problem, data, output, target_metric, overrides)
        else:
            result = self._run_pipeline_mode(kd_path, problem, data, output, target_metric, overrides)

        if not result.get("success"):
            return GenesisResult(
                project_dir=output, baseline_metric={}, eval_command="",
                errors=result.get("errors", ["Pipeline failed"]),
            )

        # Build eval command
        eval_cmd = self._build_eval_command(kd_path, output, data, result)

        # Build baseline_metric dict
        metric_value = result.get("metrics", {}).get(target_metric, 0.0)
        baseline = {
            "primary_metric": target_metric,
            target_metric: metric_value,
            "eval_command": eval_cmd,
            "eval_timeout": 600,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "genesis_solver": self.name,
            "genesis_mode": mode,
            "genesis_problem": problem,
            "model_path": result.get("model_path"),
        }

        # Write baseline_metric.json
        ahvs_dir = output / ".ahvs"
        ahvs_dir.mkdir(parents=True, exist_ok=True)
        (ahvs_dir / "baseline_metric.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8",
        )
        logger.info("Wrote baseline: %s", ahvs_dir / "baseline_metric.json")

        # Git init if not already a repo
        self._git_init(output)

        return GenesisResult(
            project_dir=output,
            baseline_metric=baseline,
            eval_command=eval_cmd,
            model_path=result.get("model_path"),
            summary=(
                f"KD classifier ({mode} mode) trained on {data.name}. "
                f"{target_metric}={metric_value:.4f}. "
                f"Model: {result.get('model_path', 'N/A')}"
            ),
        )

    # ==================================================================
    # Mode A: Pipeline (direct API via subprocess)
    # ==================================================================

    def _run_pipeline_mode(
        self,
        kd_path: Path,
        problem: str,
        data: Path,
        output: Path,
        target_metric: str,
        overrides: dict,
    ) -> dict[str, Any]:
        """Generate config/spec, call run_auto_ml_pipeline via subprocess."""
        spec_path = output / "task_spec.yaml"
        config_path = output / "kd_config.yaml"

        classes = overrides.get("classes", [])
        spec = self._build_spec(problem, classes)
        config = self._build_config(data, output, target_metric, overrides)

        spec_path.write_text(yaml.dump(spec, sort_keys=False), encoding="utf-8")
        config_path.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")
        logger.info("Generated spec: %s, config: %s", spec_path, config_path)

        script = f"""\
import sys, json
sys.path.insert(0, {str(kd_path)!r})

from src.auto_ml.pipeline import run_auto_ml_pipeline

result = run_auto_ml_pipeline(
    config_path={str(config_path)!r},
    spec_path={str(spec_path)!r},
    data_source_path={str(data)!r},
    root_output_dir={str(output)!r},
)

out = {{
    "success": result.success,
    "model_path": result.model_path,
    "project_dir": result.project_dir,
    "errors": result.errors,
    "warnings": result.warnings,
    "metrics": {{}},
}}

if result.evaluation_result:
    out["metrics"] = result.evaluation_result

json.dump(out, sys.stdout)
"""
        return self._exec_subprocess(kd_path, script)

    # ==================================================================
    # Mode B: Agent (claude-agent-sdk)
    # ==================================================================

    def _run_agent(
        self,
        kd_path: Path,
        problem: str,
        data: Path,
        output: Path,
        target_metric: str,
        overrides: dict,
    ) -> dict[str, Any]:
        """Run the KD Agent via claude-agent-sdk in SDK (one-shot) mode.

        The agent inspects the data, generates spec/config, and drives
        all pipeline stages autonomously.  After completion, we read
        structured results from .kd_agent_memory.json and output files.
        """
        prompt = (
            f"Build a text classifier for this problem: {problem}\n"
            f"Data file: {data}\n"
            f"Output directory: {output}\n"
            f"Target metric to optimize: {target_metric}\n"
            f"ML backend: tml_classifier (local BERT/DistilBERT)\n"
            f"Annotation model: {self.default_annotation_model} (never use gpt-4o)\n"
        )
        if overrides.get("classes"):
            prompt += f"Classes: {overrides['classes']}\n"

        prompt += (
            "\nRun the FULL pipeline: inspect data → generate spec → "
            "build config → prompt builder → data collector → prompt selector "
            "→ annotator → data analyzer → train TML classifier.\n"
            "Report the final evaluation metrics when done."
        )

        script = f"""\
import sys, json, asyncio, os
sys.path.insert(0, {str(kd_path)!r})
os.chdir({str(kd_path)!r})

from claude_agent_sdk import query, ResultMessage, AssistantMessage
from src.kd_agent.server import create_kd_agent

async def run():
    _, options = create_kd_agent(cwd={str(kd_path)!r}, max_turns=50)
    prompt = {prompt!r}
    prompt += (
        "\\n\\nIMPORTANT: This is SDK mode (non-interactive, one-shot). "
        "Proceed automatically through all pipeline stages without asking for "
        "confirmation or waiting for user input. Do not ask 'Shall I proceed?' — "
        "just execute each step. If a decision is needed, use sensible defaults."
    )

    full_output = []
    cost = 0.0
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            full_output.append(message.content)
        elif isinstance(message, ResultMessage):
            if message.result:
                full_output.append(message.result)
            cost = getattr(message, 'total_cost_usd', 0.0)
            break

    return "\\n".join(full_output), cost

output_text, cost = asyncio.run(run())

# Try to extract metrics from agent output
result = {{
    "success": True,
    "agent_output": output_text[-3000:],
    "cost_usd": cost,
    "model_path": None,
    "metrics": {{}},
    "errors": [],
}}

# Parse metrics from agent text output
import re
for pattern in [
    r"test_f1_weighted[:\\s]+([\\d.]+)",
    r"f1_weighted[:\\s]+([\\d.]+)",
    r"test_accuracy[:\\s]+([\\d.]+)",
    r"accuracy[:\\s]+([\\d.]+)",
    r"test_f1_macro[:\\s]+([\\d.]+)",
    r"f1_macro[:\\s]+([\\d.]+)",
]:
    m = re.search(pattern, output_text, re.IGNORECASE)
    if m:
        metric_name = pattern.split("[")[0].replace("\\\\", "")
        result["metrics"][metric_name] = float(m.group(1))

# Try to find model path in output
model_match = re.search(r"model[_ ]?path[:\\s]+([^\\s,]+)", output_text, re.IGNORECASE)
if model_match:
    result["model_path"] = model_match.group(1)

# Also check .kd_agent_memory.json for structured data
memory_path = {str(kd_path)!r} + "/.kd_agent_memory.json"
try:
    with open(memory_path) as f:
        memory = json.load(f)
    runs = memory.get("run_history", [])
    if runs:
        latest = runs[-1]
        if latest.get("success"):
            m = latest.get("metrics", {{}})
            if m.get("model_path"):
                result["model_path"] = m["model_path"]
            # Merge any metrics from memory
            for k, v in m.items():
                if k != "model_path" and isinstance(v, (int, float)):
                    result["metrics"][k] = v
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    pass

# Check for failure indicators
if any(kw in output_text.lower() for kw in ["pipeline failed", "error:", "traceback"]):
    if not result["metrics"]:
        result["success"] = False
        result["errors"] = ["Agent reported failure. Check agent_output for details."]

json.dump(result, sys.stdout)
"""
        return self._exec_subprocess(kd_path, script, timeout=7200)

    # ==================================================================
    # Shared helpers
    # ==================================================================

    def _exec_subprocess(
        self, cwd: Path, script: str, timeout: int = 3600,
    ) -> dict[str, Any]:
        """Run a Python script in a subprocess and parse JSON output."""
        cmd = [sys.executable, "-c", script]
        if self.conda_env:
            cmd = ["conda", "run", "-n", self.conda_env, "--no-capture-output"] + cmd

        logger.info("Running KD pipeline (this may take several minutes)...")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=str(cwd),
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "errors": [f"KD pipeline timed out ({timeout}s limit)"]}

        if proc.returncode != 0:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
            logger.error("KD pipeline failed:\n%s", stderr_tail)
            return {
                "success": False,
                "errors": [f"KD pipeline exit code {proc.returncode}: {stderr_tail}"],
            }

        # Parse JSON from stdout (may have log lines before it)
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', proc.stdout, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {
                "success": False,
                "errors": [f"Failed to parse pipeline output: {proc.stdout[-500:]}"],
            }

    def _validate(self, data: Path, output: Path) -> list[str]:
        errors: list[str] = []
        if not data.exists():
            errors.append(f"Data file not found: {data}")
        if data.suffix.lower() not in (".csv", ".parquet", ".tsv"):
            errors.append(f"Unsupported data format: {data.suffix}")
        return errors

    def _resolve_kd_path(self) -> Path | None:
        """Find the KD repo — explicit config, then common locations."""
        if self.kd_repo_path and self.kd_repo_path.exists():
            return self.kd_repo_path
        for candidate in [
            Path.home() / "vision" / "hackathon_knowledge_distillation",
            Path.cwd().parent / "hackathon_knowledge_distillation",
        ]:
            if candidate.exists() and (candidate / "src" / "auto_ml").exists():
                self.kd_repo_path = candidate
                return candidate
        return None

    def _build_spec(self, problem: str, classes: list[str]) -> dict:
        """Generate a task spec YAML dict from the problem description."""
        words = problem.lower().split()[:4]
        project_id = "_".join(w for w in words if w.isalnum())[:30] or "genesis"
        task_id = project_id + "_classifier"

        spec: dict[str, Any] = {
            "version": 1,
            "project_name": problem[:80],
            "project_id": project_id,
            "tasks": [
                {
                    "id": task_id,
                    "type": "classification",
                    "name": problem[:80],
                    "input": [{"name": "text", "type": "string"}],
                    "instructions": [
                        f"Classify based on: {problem}",
                        "Provide system prompt only. Do not include user prompt.",
                    ],
                }
            ],
        }
        if classes:
            spec["tasks"][0]["classes"] = classes
        return spec

    def _build_config(
        self, data: Path, output: Path, target_metric: str, overrides: dict,
    ) -> dict:
        """Generate a pipeline config YAML dict."""
        annotation_model = overrides.get(
            "annotation_model", self.default_annotation_model,
        )
        tml_model = overrides.get("tml_model", self.default_model)
        input_column = overrides.get("input_column", "text")

        return {
            "root_output_dir": str(output),
            "data_source": {
                "file_path": str(data),
                "input_column": input_column,
            },
            "prompt_builder": {
                "max_iterations": 1,
                "model": annotation_model,
                "skip_final_critique": False,
                "api_timeout": 120,
            },
            "data_collector": {
                "validation": {
                    "required_columns": [input_column],
                    "min_row_count": 1,
                    "max_null_percentage": 10.0,
                },
                "deduplication": {"method": "lsh", "lsh_threshold": 0.85},
                "sampling": {
                    "target_sample_size": 50,
                    "strategy": "cluster_diverse",
                },
            },
            "prompt_selector": {
                "use_dspy": False,
                "model": annotation_model,
                "max_samples": 20,
            },
            "annotator": {
                "model": annotation_model,
                "rate_limit_delay": 0.0,
                "max_retries": 3,
                "save_every": 10,
                "use_dspy": False,
            },
            "data_analyzer": {
                "enable_llm_insights": True,
                "model": annotation_model,
            },
            "ml": {
                "backend": "tml_classifier",
                "tml_classifier": {
                    "model": tml_model,
                    "input_column": input_column,
                    "train_test_split": 0.8,
                    "validation_split": 0.1,
                    "num_train_epochs": int(overrides.get("num_train_epochs", 3)),
                    "learning_rate": float(overrides.get("learning_rate", 2e-5)),
                    "train_batch_size": int(overrides.get("train_batch_size", 16)),
                    "eval_batch_size": 32,
                    "max_seq_length": 128,
                    "fp16": True,
                    "early_stopping_patience": 2,
                    "metrics": ["accuracy", "f1_macro", "f1_weighted"],
                    "generate_confusion_matrix": True,
                    "per_class_metrics": True,
                    "mlflow_enabled": False,
                },
            },
        }

    def _build_eval_command(
        self, kd_path: Path, output: Path, data: Path, result: dict,
    ) -> str:
        """Build a shell command that re-measures the metric."""
        config_path = output / "kd_config.yaml"
        spec_path = output / "task_spec.yaml"
        return (
            f"cd {kd_path} && "
            f"python -m src.auto_ml.main {config_path} "
            f"--spec {spec_path} "
            f"--data {data}"
        )

    def _git_init(self, project_dir: Path) -> None:
        """Initialize a git repo in the project directory if not already one."""
        if (project_dir / ".git").exists():
            return
        try:
            subprocess.run(
                ["git", "init"], cwd=str(project_dir),
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "add", "-A"], cwd=str(project_dir),
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "genesis: initial project from KD pipeline"],
                cwd=str(project_dir),
                capture_output=True, check=True,
            )
            logger.info("Initialized git repo at %s", project_dir)
        except subprocess.CalledProcessError as exc:
            logger.warning("git init failed: %s", exc.stderr)
