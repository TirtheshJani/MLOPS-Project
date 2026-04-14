"""Test helpers: wire up ``sys.path`` so ``app.main`` can be imported."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_path() -> None:
    """Insert the package root (that contains ``app/``) onto ``sys.path``."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def get_app():
    """Return the FastAPI application instance for tests."""
    ensure_path()
    from app.main import app  # type: ignore[import-not-found]

    return app
