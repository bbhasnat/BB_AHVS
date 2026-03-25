"""Pytest configuration — ensure `ahvs` is importable from a fresh checkout."""

import sys
from pathlib import Path

# Add the repo root to sys.path so `import ahvs` works without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
