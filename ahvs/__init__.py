"""ahvs — Adaptive Hypothesis Validation System.

AHVS is an 8-stage cyclic hypothesis-validation pipeline that composes
infrastructure (CodeAgent, EvolutionStore, LLMClient, sandbox) to
autonomously generate, select, execute, and evaluate hypotheses that improve
a target LLM/RAG system's primary metric.

Public API
----------
    execute_ahvs_cycle(config, ...)   — run one full AHVS cycle
    read_ahvs_checkpoint(cycle_dir)   — resume from checkpoint
    AHVSConfig                        — cycle configuration
    AHVSStage                         — 8-stage enum
    HypothesisResult                  — tool-agnostic result contract
"""

from ahvs.config import AHVSConfig
from ahvs.result import HypothesisResult
from ahvs.runner import execute_ahvs_cycle, read_ahvs_checkpoint
from ahvs.stages import AHVSStage

__all__ = [
    "AHVSConfig",
    "AHVSStage",
    "HypothesisResult",
    "execute_ahvs_cycle",
    "read_ahvs_checkpoint",
]
