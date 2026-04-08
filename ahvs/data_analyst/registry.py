"""Module registry — discover and load analysis modules.

Built-in modules are listed in ``_BUILTIN_MODULES``. Adding a new module
requires adding its name to this list and creating the corresponding file
in ``ahvs.data_analyst.modules`` with a ``run(ModuleInput) -> ModuleResult``
function.

Custom modules can be registered at runtime via ``register()``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable

from ahvs.data_analyst.models import ModuleInput, ModuleResult

logger = logging.getLogger(__name__)

# Type alias for a module's entry point
ModuleRunner = Callable[[ModuleInput], ModuleResult]

# Global registry populated by discover()
_REGISTRY: dict[str, ModuleRunner] = {}

# Track whether full discovery has been attempted (L2 fix)
_DISCOVERY_DONE = False

# Modules shipped with v1
_BUILTIN_MODULES = [
    "eda",
    "class_balance",
    "text_stats",
    "duplicates",
    "subsample",
    "split",
    "export",
]


def discover() -> dict[str, ModuleRunner]:
    """Import all built-in modules and register their ``run`` functions.

    Safe to call multiple times — only runs full discovery once.
    """
    global _DISCOVERY_DONE

    if _DISCOVERY_DONE:
        return _REGISTRY

    for name in _BUILTIN_MODULES:
        if name in _REGISTRY:
            continue  # already registered (e.g. via register())
        fqn = f"ahvs.data_analyst.modules.{name}"
        try:
            mod = importlib.import_module(fqn)
            runner = getattr(mod, "run", None)
            if runner is None:
                logger.warning("Module %s has no run() function — skipped", fqn)
                continue
            _REGISTRY[name] = runner
            logger.debug("Registered module: %s", name)
        except Exception:
            logger.warning("Failed to import module %s", fqn, exc_info=True)

    _DISCOVERY_DONE = True
    return _REGISTRY


def get(name: str) -> ModuleRunner | None:
    """Get a registered module runner by name."""
    if not _DISCOVERY_DONE:
        discover()
    return _REGISTRY.get(name)


def register(name: str, runner: ModuleRunner) -> None:
    """Register a custom module runner."""
    _REGISTRY[name] = runner
    logger.debug("Registered custom module: %s", name)


def available() -> list[str]:
    """Return names of all registered modules."""
    if not _DISCOVERY_DONE:
        discover()
    return list(_REGISTRY.keys())
