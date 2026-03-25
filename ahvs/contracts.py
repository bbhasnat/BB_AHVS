"""Stage I/O contracts for the 8-stage AHVS cycle.

Each AHVSStageContract declares:
  - input_files: artifacts this stage reads (relative to cycle_dir)
  - output_files: artifacts this stage must produce
  - dod: Definition of Done — human-readable acceptance criterion
  - error_code: unique error identifier for diagnostics
  - max_retries: how many times the stage may be retried on failure
"""

from __future__ import annotations

from dataclasses import dataclass

from ahvs.stages import AHVSStage


@dataclass(frozen=True)
class AHVSStageContract:
    stage: AHVSStage
    input_files: tuple[str, ...]
    output_files: tuple[str, ...]
    dod: str
    error_code: str
    max_retries: int = 1


AHVS_CONTRACTS: dict[AHVSStage, AHVSStageContract] = {
    AHVSStage.AHVS_SETUP: AHVSStageContract(
        stage=AHVSStage.AHVS_SETUP,
        input_files=("baseline_metric.json",),
        output_files=("cycle_manifest.json",),
        dod="Baseline valid, cycle directory created, LLM reachable",
        error_code="EA01_SETUP_FAIL",
        max_retries=0,
    ),
    AHVSStage.AHVS_CONTEXT_LOAD: AHVSStageContract(
        stage=AHVSStage.AHVS_CONTEXT_LOAD,
        input_files=("baseline_metric.json",),
        output_files=("context_bundle.json",),
        dod="context_bundle.json written with baseline, lessons, rejected approaches",
        error_code="EA02_CONTEXT_FAIL",
    ),
    AHVSStage.AHVS_HYPOTHESIS_GEN: AHVSStageContract(
        stage=AHVSStage.AHVS_HYPOTHESIS_GEN,
        input_files=("context_bundle.json",),
        output_files=("hypotheses.md",),
        dod="1–5 typed hypotheses with id, type, description, rationale, required_tools",
        error_code="EA03_HYP_FAIL",
    ),
    AHVSStage.AHVS_HUMAN_SELECTION: AHVSStageContract(
        stage=AHVSStage.AHVS_HUMAN_SELECTION,
        input_files=("hypotheses.md",),
        output_files=("selection.md",),
        dod="Human selected hypotheses recorded in selection.md",
        error_code="EA04_GATE_REJECT",
        max_retries=0,
    ),
    AHVSStage.AHVS_VALIDATION_PLAN: AHVSStageContract(
        stage=AHVSStage.AHVS_VALIDATION_PLAN,
        input_files=("context_bundle.json", "selection.md", "hypotheses.md"),
        output_files=("validation_plan.md",),
        dod="Per-hypothesis implementation spec and eval method defined",
        error_code="EA05_PLAN_FAIL",
    ),
    AHVSStage.AHVS_EXECUTION: AHVSStageContract(
        stage=AHVSStage.AHVS_EXECUTION,
        input_files=("validation_plan.md", "selection.md"),
        output_files=("results.json", "tool_runs/"),
        dod="Every selected hypothesis has a HypothesisResult in results.json",
        error_code="EA06_EXEC_FAIL",
        max_retries=1,
    ),
    AHVSStage.AHVS_REPORT_MEMORY: AHVSStageContract(
        stage=AHVSStage.AHVS_REPORT_MEMORY,
        input_files=("results.json", "context_bundle.json"),
        output_files=("report.md", "friction_log.md"),
        dod="Cycle report written; lessons archived to EvolutionStore",
        error_code="EA07_REPORT_FAIL",
    ),
    AHVSStage.AHVS_CYCLE_VERIFY: AHVSStageContract(
        stage=AHVSStage.AHVS_CYCLE_VERIFY,
        input_files=("results.json", "report.md"),
        output_files=("cycle_summary.json",),
        dod="All artifacts present and valid; cycle_summary.json written",
        error_code="EA08_VERIFY_FAIL",
    ),
}
