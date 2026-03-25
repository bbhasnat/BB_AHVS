"""AHVS 8-stage cycle state machine.

Defines the stage sequence, gate stages, and rollback rules for the
AHVS hypothesis-validation cycle.

Includes inlined transition primitives (StageStatus, TransitionEvent,
TransitionOutcome, advance) that are stage-type-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable


# ---------------------------------------------------------------------------
# Transition primitives (inlined, stage-type-agnostic)
# ---------------------------------------------------------------------------


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED_APPROVAL = "blocked_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAUSED = "paused"
    RETRYING = "retrying"
    FAILED = "failed"
    DONE = "done"


class TransitionEvent(str, Enum):
    START = "start"
    SUCCEED = "succeed"
    APPROVE = "approve"
    REJECT = "reject"
    TIMEOUT = "timeout"
    FAIL = "fail"
    RETRY = "retry"
    RESUME = "resume"
    PAUSE = "pause"


TRANSITION_MAP: dict[StageStatus, frozenset[StageStatus]] = {
    StageStatus.PENDING: frozenset({StageStatus.RUNNING}),
    StageStatus.RUNNING: frozenset(
        {StageStatus.DONE, StageStatus.BLOCKED_APPROVAL, StageStatus.FAILED}
    ),
    StageStatus.BLOCKED_APPROVAL: frozenset(
        {StageStatus.APPROVED, StageStatus.REJECTED, StageStatus.PAUSED}
    ),
    StageStatus.APPROVED: frozenset({StageStatus.DONE}),
    StageStatus.REJECTED: frozenset({StageStatus.PENDING}),
    StageStatus.PAUSED: frozenset({StageStatus.RUNNING}),
    StageStatus.RETRYING: frozenset({StageStatus.RUNNING}),
    StageStatus.FAILED: frozenset({StageStatus.RETRYING, StageStatus.PAUSED}),
    StageStatus.DONE: frozenset(),
}


@dataclass(frozen=True)
class TransitionOutcome:
    stage: IntEnum
    status: StageStatus
    next_stage: IntEnum | None
    rollback_stage: IntEnum | None = None
    checkpoint_required: bool = False
    decision: str = "proceed"


# ---------------------------------------------------------------------------
# AHVS 8-stage enum
# ---------------------------------------------------------------------------


class AHVSStage(IntEnum):
    """8-stage AHVS hypothesis-validation cycle."""

    AHVS_SETUP           = 1  # Pre-flight, baseline validation, cycle dir
    AHVS_CONTEXT_LOAD    = 2  # EvolutionStore overlay + baseline → context_bundle.json
    AHVS_HYPOTHESIS_GEN  = 3  # Typed hypothesis generation (1–5)
    AHVS_HUMAN_SELECTION = 4  # GATE: human selects hypotheses to run
    AHVS_VALIDATION_PLAN = 5  # Per-hypothesis implementation spec + eval method
    AHVS_EXECUTION       = 6  # Claude Code executes each selected hypothesis
    AHVS_REPORT_MEMORY   = 7  # LLM report + EvolutionStore lesson archival
    AHVS_CYCLE_VERIFY    = 8  # Contract validation of all artifacts


AHVS_STAGE_SEQUENCE: tuple[AHVSStage, ...] = tuple(AHVSStage)

AHVS_NEXT_STAGE: dict[AHVSStage, AHVSStage | None] = {
    stage: AHVS_STAGE_SEQUENCE[idx + 1] if idx + 1 < len(AHVS_STAGE_SEQUENCE) else None
    for idx, stage in enumerate(AHVS_STAGE_SEQUENCE)
}

AHVS_GATE_STAGES: frozenset[AHVSStage] = frozenset({AHVSStage.AHVS_HUMAN_SELECTION})

AHVS_GATE_ROLLBACK: dict[AHVSStage, AHVSStage] = {
    AHVSStage.AHVS_HUMAN_SELECTION: AHVSStage.AHVS_HYPOTHESIS_GEN,
}


def ahvs_gate_required(stage: AHVSStage) -> bool:
    """Return True if this stage requires human approval."""
    return stage in AHVS_GATE_STAGES


def ahvs_default_rollback(stage: AHVSStage) -> AHVSStage:
    """Return the rollback target for a rejected gate, or the stage itself."""
    return AHVS_GATE_ROLLBACK.get(stage, stage)


# ---------------------------------------------------------------------------
# Generic advance() — works with any IntEnum stage type
# ---------------------------------------------------------------------------


def _compute_next_stage(stage: IntEnum) -> IntEnum | None:
    """Compute the next stage for any IntEnum by finding the successor value."""
    stage_cls = type(stage)
    members = list(stage_cls)
    try:
        idx = members.index(stage)
    except ValueError:
        return None
    if idx + 1 < len(members):
        return members[idx + 1]
    return None


def _compute_previous_stage(stage: IntEnum) -> IntEnum | None:
    """Compute the previous stage for any IntEnum."""
    stage_cls = type(stage)
    members = list(stage_cls)
    try:
        idx = members.index(stage)
    except ValueError:
        return None
    if idx > 0:
        return members[idx - 1]
    return None


def gate_required(
    stage: IntEnum,
    gate_stages: frozenset[IntEnum] | None = None,
    hitl_required_stages: Iterable[int] | None = None,
) -> bool:
    """Check whether a stage requires human-in-the-loop approval.

    *gate_stages* defaults to AHVS_GATE_STAGES when None.
    """
    if gate_stages is None:
        gate_stages = AHVS_GATE_STAGES  # type: ignore[assignment]
    if stage not in gate_stages:
        return False
    if hitl_required_stages is not None:
        return int(stage) in frozenset(hitl_required_stages)
    return True


def default_rollback_stage(
    stage: IntEnum,
    gate_rollback: dict[IntEnum, IntEnum] | None = None,
) -> IntEnum:
    """Return the configured rollback target, or the previous stage.

    *gate_rollback* defaults to AHVS_GATE_ROLLBACK when None.
    """
    if gate_rollback is None:
        gate_rollback = AHVS_GATE_ROLLBACK  # type: ignore[assignment]
    result = gate_rollback.get(stage)  # type: ignore[arg-type]
    if result is not None:
        return result
    prev = _compute_previous_stage(stage)
    return prev if prev is not None else stage


def advance(
    stage: IntEnum,
    status: StageStatus,
    event: TransitionEvent | str,
    *,
    gate_stages: frozenset[IntEnum] | None = None,
    gate_rollback: dict[IntEnum, IntEnum] | None = None,
    hitl_required_stages: Iterable[int] | None = None,
    rollback_stage: IntEnum | None = None,
) -> TransitionOutcome:
    """Compute the next state given current stage, status, and event.

    Works with any IntEnum stage type (not just AHVSStage).
    Raises ValueError on unsupported transitions.
    """
    event = TransitionEvent(event)
    target_rollback = rollback_stage or default_rollback_stage(stage, gate_rollback)
    next_stage = _compute_next_stage(stage)

    # START → RUNNING
    if event is TransitionEvent.START and status in {
        StageStatus.PENDING,
        StageStatus.RETRYING,
        StageStatus.PAUSED,
    }:
        return TransitionOutcome(
            stage=stage, status=StageStatus.RUNNING, next_stage=stage
        )

    # SUCCEED while RUNNING
    if event is TransitionEvent.SUCCEED and status is StageStatus.RUNNING:
        if gate_required(stage, gate_stages, hitl_required_stages):
            return TransitionOutcome(
                stage=stage,
                status=StageStatus.BLOCKED_APPROVAL,
                next_stage=stage,
                checkpoint_required=False,
                decision="block",
            )
        return TransitionOutcome(
            stage=stage,
            status=StageStatus.DONE,
            next_stage=next_stage,
            checkpoint_required=True,
        )

    # APPROVE while BLOCKED
    if event is TransitionEvent.APPROVE and status is StageStatus.BLOCKED_APPROVAL:
        return TransitionOutcome(
            stage=stage,
            status=StageStatus.DONE,
            next_stage=next_stage,
            checkpoint_required=True,
        )

    # REJECT while BLOCKED → rollback
    if event is TransitionEvent.REJECT and status is StageStatus.BLOCKED_APPROVAL:
        return TransitionOutcome(
            stage=target_rollback,
            status=StageStatus.PENDING,
            next_stage=target_rollback,
            rollback_stage=target_rollback,
            checkpoint_required=True,
            decision="pivot",
        )

    # TIMEOUT while BLOCKED → pause
    if event is TransitionEvent.TIMEOUT and status is StageStatus.BLOCKED_APPROVAL:
        return TransitionOutcome(
            stage=stage,
            status=StageStatus.PAUSED,
            next_stage=stage,
            checkpoint_required=True,
            decision="block",
        )

    # FAIL while RUNNING
    if event is TransitionEvent.FAIL and status is StageStatus.RUNNING:
        return TransitionOutcome(
            stage=stage,
            status=StageStatus.FAILED,
            next_stage=stage,
            checkpoint_required=True,
            decision="retry",
        )

    # RETRY while FAILED
    if event is TransitionEvent.RETRY and status is StageStatus.FAILED:
        return TransitionOutcome(
            stage=stage,
            status=StageStatus.RETRYING,
            next_stage=stage,
            decision="retry",
        )

    # RESUME while PAUSED
    if event is TransitionEvent.RESUME and status is StageStatus.PAUSED:
        return TransitionOutcome(
            stage=stage, status=StageStatus.RUNNING, next_stage=stage
        )

    # PAUSE while FAILED
    if event is TransitionEvent.PAUSE and status is StageStatus.FAILED:
        return TransitionOutcome(
            stage=stage,
            status=StageStatus.PAUSED,
            next_stage=stage,
            checkpoint_required=True,
            decision="block",
        )

    raise ValueError(
        f"Unsupported transition: {status.value} + {event.value} for stage {int(stage)}"
    )
