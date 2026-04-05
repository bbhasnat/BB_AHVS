"""Solver registry — YAML-driven discovery and instantiation of solvers."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import yaml

from ahvs.genesis.contract import Solver

logger = logging.getLogger(__name__)

_DEFAULT_SOLVERS_YAML = Path(__file__).parent / "solvers.yaml"


class SolverRegistry:
    """Load solver definitions from YAML and instantiate on demand.

    Each entry in the YAML maps a solver name to its module, class,
    supported problem types, and optional config kwargs::

        solvers:
          kd_classifier:
            module: ahvs.genesis.solvers.kd_classifier
            class: KDClassifierSolver
            problem_types: [classification, sentiment, intent]
            config:
              kd_repo_path: /path/to/kd/repo
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        self._path = Path(yaml_path) if yaml_path else _DEFAULT_SOLVERS_YAML
        self._defs: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, Solver] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("Solver registry not found: %s", self._path)
            return
        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._defs = data.get("solvers", {})
        logger.info(
            "Loaded %d solver definition(s) from %s", len(self._defs), self._path
        )

    def list_solvers(self) -> list[str]:
        """Return registered solver names."""
        return list(self._defs.keys())

    def get(self, name: str) -> Solver:
        """Instantiate and return a solver by name (cached)."""
        if name in self._cache:
            return self._cache[name]

        if name not in self._defs:
            available = ", ".join(self._defs) or "(none)"
            raise KeyError(
                f"Unknown solver {name!r}. Available: {available}"
            )

        defn = self._defs[name]
        module_path = defn["module"]
        class_name = defn["class"]
        config = defn.get("config", {})

        # Security: only allow importing from ahvs.genesis.solvers namespace
        _ALLOWED_PREFIXES = ("ahvs.genesis.solvers",)
        if not any(module_path.startswith(p) for p in _ALLOWED_PREFIXES):
            raise ValueError(
                f"Solver {name!r}: module {module_path!r} is outside allowed "
                f"namespaces {_ALLOWED_PREFIXES}. Custom solvers must live "
                f"under ahvs.genesis.solvers."
            )

        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Solver {name!r}: cannot import module {module_path!r} "
                f"(from {self._path}): {exc}"
            ) from exc

        cls = getattr(mod, class_name)
        instance = cls(**config)

        self._cache[name] = instance
        return instance

    def get_for_problem_type(self, problem_type: str) -> Solver | None:
        """Return the first solver whose problem_types includes *problem_type*."""
        for name, defn in self._defs.items():
            if problem_type in defn.get("problem_types", []):
                return self.get(name)
        return None

    def solver_info(self, name: str) -> dict[str, Any]:
        """Return the raw YAML definition for a solver."""
        return dict(self._defs.get(name, {}))
