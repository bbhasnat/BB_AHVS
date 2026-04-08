"""Pytest configuration — ensure `ahvs` is importable from a fresh checkout."""

import sys
from pathlib import Path

import pytest

# Add the repo root to sys.path so `import ahvs` works without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from writing to the real ~/.ahvs/registry.json."""
    import ahvs.registry as registry

    reg_dir = tmp_path / ".ahvs_test_registry"
    reg_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(registry, "_REGISTRY_DIR", reg_dir)
    monkeypatch.setattr(registry, "_REGISTRY_PATH", reg_dir / "registry.json")
