import sys
from pathlib import Path


def get_app():
    repo_root = Path(__file__).resolve().parents[1]
    # Ensure repo root (which contains clinical-note-summarizer/) is on sys.path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.main import app  # type: ignore

    return app


