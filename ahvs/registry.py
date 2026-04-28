"""AHVS Repo Registry — persistent name-to-path mapping for onboarded repos.

The registry lives at ``~/.ahvs/registry.json`` and is updated automatically
during onboarding and after each cycle.  It allows the CLI to accept short
repo names (e.g. ``--repo autoresearch``) in addition to full filesystem paths.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY_DIR = Path.home() / ".ahvs"
_REGISTRY_PATH = _REGISTRY_DIR / "registry.json"


def _load() -> dict[str, Any]:
    """Load the registry from disk, returning an empty structure on failure."""
    if not _REGISTRY_PATH.exists():
        return {"repos": {}}
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        if "repos" not in data:
            data["repos"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return {"repos": {}}


def _save(data: dict[str, Any]) -> None:
    """Write registry to disk, creating ~/.ahvs/ if needed."""
    _REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


def register(
    repo_path: Path,
    *,
    name: str | None = None,
    primary_metric: str | None = None,
    baseline_value: float | None = None,
) -> str:
    """Register (or update) a repo in the global registry.

    Args:
        repo_path: Absolute path to the onboarded repo.
        name: Short name for the repo.  Defaults to the directory basename.
        primary_metric: The metric being optimised (from baseline_metric.json).
        baseline_value: Current baseline value for the primary metric.

    Returns:
        The short name under which the repo was registered.
    """
    repo_path = Path(repo_path).resolve()
    short_name = name or repo_path.name

    data = _load()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    existing = data["repos"].get(short_name, {})
    entry: dict[str, Any] = {
        "path": str(repo_path),
        "onboarded_at": existing.get("onboarded_at", now),
        "updated_at": now,
    }
    if primary_metric is not None:
        entry["primary_metric"] = primary_metric
    elif "primary_metric" in existing:
        entry["primary_metric"] = existing["primary_metric"]

    if baseline_value is not None:
        entry["baseline_value"] = baseline_value
    elif "baseline_value" in existing:
        entry["baseline_value"] = existing["baseline_value"]

    if "last_cycle" in existing:
        entry["last_cycle"] = existing["last_cycle"]

    data["repos"][short_name] = entry
    _save(data)
    return short_name


def update_last_cycle(repo_path: Path, cycle_id: str) -> None:
    """Update the last_cycle field for a repo after a cycle completes."""
    repo_path = Path(repo_path).resolve()
    data = _load()

    # Find by path (the repo may have been registered under any name)
    for _name, entry in data["repos"].items():
        if Path(entry["path"]).resolve() == repo_path:
            entry["last_cycle"] = cycle_id
            entry["updated_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            _save(data)
            return

    # Not registered yet — read baseline_metric.json for rich registration
    metric_name: str | None = None
    metric_val: float | None = None
    baseline_path = repo_path / ".ahvs" / "baseline_metric.json"
    if baseline_path.exists():
        try:
            bm = json.loads(baseline_path.read_text(encoding="utf-8"))
            metric_name = bm.get("primary_metric")
            if metric_name and metric_name in bm:
                metric_val = float(bm[metric_name])
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass

    register(repo_path, primary_metric=metric_name, baseline_value=metric_val)
    data = _load()
    for _name, entry in data["repos"].items():
        if Path(entry["path"]).resolve() == repo_path:
            entry["last_cycle"] = cycle_id
            _save(data)
            return


def resolve(name_or_path: str) -> Path | None:
    """Resolve a repo name or path to an absolute Path.

    Resolution order:
    1. If *name_or_path* is a valid directory, return it directly.
    2. If it matches a registered short name, return the registered path.
    3. Return None (caller should error).
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()

    data = _load()
    entry = data["repos"].get(name_or_path)
    if entry is not None:
        p = Path(entry["path"])
        if p.is_dir():
            return p.resolve()
        logger.warning(
            "Registry entry '%s' points to %s which no longer exists",
            name_or_path,
            entry["path"],
        )
        return None

    return None


def list_repos() -> dict[str, dict[str, Any]]:
    """Return all registered repos as {name: {path, metric, ...}}."""
    return dict(_load()["repos"])


def unregister(name: str) -> bool:
    """Remove a repo from the registry. Returns True if it was found."""
    data = _load()
    if name in data["repos"]:
        del data["repos"][name]
        _save(data)
        return True
    return False
