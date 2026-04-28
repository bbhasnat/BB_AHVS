"""Phase 3: Module execution engine.

Runs each module in the plan sequentially, passing the standard ModuleInput
and collecting ModuleResult objects. When a module sets ``transformed_df``
on its result, subsequent modules receive the transformed data.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from ahvs.data_analyst.models import (
    AnalysisPlan,
    DataProfile,
    ModuleInput,
    ModuleResult,
)
from ahvs.data_analyst import registry

logger = logging.getLogger(__name__)


def execute(
    df: pd.DataFrame,
    profile: DataProfile,
    plan: AnalysisPlan,
    output_dir: Path,
) -> list[ModuleResult]:
    """Execute all modules in the plan and return their results.

    Args:
        df: The loaded DataFrame.
        profile: DataProfile from Phase 1.
        plan: AnalysisPlan from Phase 2.
        output_dir: Root output directory for this analysis run.

    Returns:
        List of ModuleResult, one per module in the plan.
    """
    results: list[ModuleResult] = []
    working_df = df  # mutable reference — updated when modules transform data

    for spec in plan.modules:
        runner = registry.get(spec.name)
        if runner is None:
            logger.warning("Module '%s' not found in registry.", spec.name)
            results.append(
                ModuleResult.make_error(spec.name, "Module not found in registry.")
            )
            continue

        inp = ModuleInput(
            df=working_df,
            profile=profile,
            plan=plan,
            task_type=plan.task_type,
            input_cols=plan.input_columns,
            label_col=plan.label_column,
            params=spec.params,
            output_dir=output_dir,
        )

        logger.info("Running module: %s", spec.name)
        t0 = time.time()

        try:
            result = runner(inp)
            elapsed = time.time() - t0
            logger.info(
                "Module %s completed in %.1fs (status=%s)",
                spec.name,
                elapsed,
                result.status,
            )
            # Propagate transformed data to subsequent modules
            if result.transformed_df is not None:
                logger.info(
                    "Module %s transformed data: %d → %d rows",
                    spec.name,
                    len(working_df),
                    len(result.transformed_df),
                )
                working_df = result.transformed_df
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "Module %s failed after %.1fs: %s",
                spec.name,
                elapsed,
                exc,
                exc_info=True,
            )
            result = ModuleResult.make_error(spec.name, str(exc))

        results.append(result)

    return results
