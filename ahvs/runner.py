"""AHVS cycle runner — orchestrates the 8-stage hypothesis-validation cycle.

execute_ahvs_cycle() is the public entry point. It:
  1. Iterates AHVS_STAGE_SEQUENCE in order
  2. Calls execute_ahvs_stage() for each stage
  3. Writes a checkpoint after every stage
  4. Stops on failure (with rollback info if at a gate)
  5. Returns the list of AHVSStageResult objects
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ahvs.config import AHVSConfig
from ahvs.executor import AHVSStageResult, execute_ahvs_stage
from ahvs.skills import SkillLibrary
from ahvs.stages import (
    AHVS_STAGE_SEQUENCE,
    AHVS_GATE_ROLLBACK,
    AHVSStage,
    StageStatus,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_FILE = "ahvs_checkpoint.json"


def _write_checkpoint(cycle_dir: Path, stage: AHVSStage, status: str) -> None:
    """Persist the last completed/failed stage so a cycle can be resumed."""
    data = {
        "stage": stage.name,
        "stage_num": int(stage),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (cycle_dir / _CHECKPOINT_FILE).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def read_ahvs_checkpoint(cycle_dir: Path) -> AHVSStage | None:
    """Return the last successfully completed AHVSStage from a checkpoint, or None."""
    cp = cycle_dir / _CHECKPOINT_FILE
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        if data.get("status") == "done":
            return AHVSStage[data["stage"]]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _stage_label(stage: AHVSStage) -> str:
    return f"Stage {int(stage)}/{len(AHVS_STAGE_SEQUENCE)}: {stage.name}"


def execute_ahvs_cycle(
    config: AHVSConfig,
    *,
    auto_approve: bool = False,
    from_stage: AHVSStage | None = None,
    until_stage: AHVSStage | None = None,
    skill_library: SkillLibrary | None = None,
    on_stage_complete: Any = None,
) -> list[AHVSStageResult]:
    """Run a full AHVS hypothesis-validation cycle.

    Args:
        config: Fully populated AHVSConfig.
        auto_approve: If True, skip interactive gate and select all hypotheses.
        from_stage: Resume from this stage (skip earlier stages). If None,
            start from AHVS_SETUP.
        until_stage: Stop after this stage and return. If None, run all
            remaining stages. Useful for running only hypothesis generation
            (``until_stage=AHVS_HYPOTHESIS_GEN``) then resuming after GUI
            selection with ``from_stage=AHVS_HUMAN_SELECTION``.
        skill_library: Optional pre-built SkillLibrary. If None, a default
            one is constructed (using config.skill_registry_path if set).
        on_stage_complete: Optional callback(stage_result: AHVSStageResult).
            Called after each stage regardless of status.

    Returns:
        List of AHVSStageResult objects, one per stage executed.
    """
    cycle_dir = config.run_dir
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # Auto-cleanup: remove stale cycle dirs + compact old lessons
    cycles_root = config.repo_path / ".ahvs" / "cycles"
    if cycles_root.is_dir():
        try:
            from ahvs.evolution import EvolutionStore

            removed = EvolutionStore.cleanup_cycles(cycles_root, keep_complete=3, exclude=cycle_dir)
            if removed:
                print(f"[AHVS] Cleaned up {len(removed)} stale cycle dir(s)")

            if config.evolution_dir.exists():
                store = EvolutionStore(config.evolution_dir)
                compacted = store.compact()
                if compacted:
                    print(f"[AHVS] Compacted evolution store: removed {compacted} stale entries")

            # Summarize friction logs from retained cycles
            processed = EvolutionStore.compact_friction_logs(cycles_root)
            if processed:
                print(f"[AHVS] Summarized {processed} friction logs")

            # Manage memory file lifecycle (stale/archive)
            memory_dir = config.repo_path / ".ahvs" / "memory"
            if memory_dir.is_dir():
                stale, archived = EvolutionStore.compact_memory_files(
                    memory_dir,
                    stale_days=config.memory_stale_days,
                    archive_days=config.memory_archive_days,
                )
                if stale or archived:
                    print(f"[AHVS] Memory lifecycle: {stale} stale-marked, {archived} archived")
        except Exception:  # noqa: BLE001
            pass  # Non-fatal — don't block the cycle for cleanup issues

    if skill_library is None:
        skill_library = SkillLibrary(
            custom_registry_path=config.skill_registry_path
        )

    # Determine which stages to run
    if from_stage is not None:
        start_idx = next(
            (i for i, s in enumerate(AHVS_STAGE_SEQUENCE) if s == from_stage),
            0,
        )
        stages_to_run = AHVS_STAGE_SEQUENCE[start_idx:]
    else:
        stages_to_run = AHVS_STAGE_SEQUENCE

    # Trim to until_stage if specified
    if until_stage is not None:
        until_idx = next(
            (i for i, s in enumerate(stages_to_run) if s == until_stage),
            len(stages_to_run) - 1,
        )
        stages_to_run = stages_to_run[:until_idx + 1]

    print(f"\n[AHVS] Cycle: {cycle_dir.name}")
    print(f"[AHVS] Question: {config.question}")
    print(f"[AHVS] Repo: {config.repo_path}")
    print(f"[AHVS] Max hypotheses: {config.max_hypotheses}")
    print(f"[AHVS] Auto-approve: {auto_approve}")
    print()

    results: list[AHVSStageResult] = []

    for stage in stages_to_run:
        print(f"[AHVS] → {_stage_label(stage)}", flush=True)

        result = execute_ahvs_stage(
            stage,
            cycle_dir=cycle_dir,
            config=config,
            skill_library=skill_library,
            auto_approve=auto_approve,
        )
        results.append(result)

        status_str = result.status.value if hasattr(result.status, "value") else str(result.status)
        _write_checkpoint(cycle_dir, stage, status_str)

        if on_stage_complete is not None:
            try:
                on_stage_complete(result)
            except Exception:  # noqa: BLE001
                pass

        if result.status == StageStatus.DONE:
            artifacts_str = ", ".join(result.artifacts) if result.artifacts else "none"
            print(f"[AHVS]   ✓ done | artifacts: {artifacts_str}")
        else:
            # Failed stage
            error_msg = result.error or "unknown error"
            print(f"[AHVS]   ✗ FAILED — {error_msg}")

            # Gate-specific rollback hint
            if stage in AHVS_GATE_ROLLBACK:
                rollback_to = AHVS_GATE_ROLLBACK[stage]
                print(
                    f"[AHVS]   Gate aborted — to restart from hypothesis generation, "
                    f"resume with --from-stage {rollback_to.name}"
                )

            logger.error(
                "AHVS cycle stopped at %s: %s",
                stage.name,
                error_msg,
            )
            break  # Stop on first failure

    # Final summary
    done_count = sum(1 for r in results if r.status == StageStatus.DONE)
    total = len(results)
    print(f"\n[AHVS] Cycle ended: {done_count}/{total} stages completed")

    if until_stage is not None and done_count == total:
        print(f"[AHVS] Stopped at {until_stage.name} as requested. "
              f"Resume with: --from-stage {AHVS_STAGE_SEQUENCE[AHVS_STAGE_SEQUENCE.index(until_stage) + 1].name}"
              if AHVS_STAGE_SEQUENCE.index(until_stage) + 1 < len(AHVS_STAGE_SEQUENCE)
              else f"[AHVS] Stopped at {until_stage.name} as requested.")
    elif done_count == len(AHVS_STAGE_SEQUENCE):
        print(f"[AHVS] Full cycle complete. See: {cycle_dir / 'cycle_summary.json'}")

    # Promote qualifying lessons to global cross-project store
    if config.enable_cross_project and done_count == len(stages_to_run):
        try:
            from ahvs.evolution import GlobalEvolutionStore, EvolutionStore

            global_store = GlobalEvolutionStore(config.global_evolution_dir)
            local_store = EvolutionStore(config.evolution_dir)
            promoted = global_store.promote_lessons(local_store, config.repo_path.name)
            if promoted:
                print(f"[AHVS] Promoted {promoted} lesson(s) to global store")
        except Exception:  # noqa: BLE001
            pass

    return results
