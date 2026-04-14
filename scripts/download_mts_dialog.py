#!/usr/bin/env python3
"""Download and extract the MTS-Dialog dataset from GitHub.

Tries the primary Microsoft repo first, then falls back to a mirror. The
archive is downloaded via ``requests`` and unpacked with the stdlib
``zipfile`` module so no git clone is needed.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

DEFAULT_REPO = "microsoft/clinical_visit_note_summarization_corpus"
FALLBACK_REPO = "abachaa/MTS-Dialog"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract the MTS-Dialog dataset from GitHub.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub repo to pull from (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path("data/primary/mts-dialog")),
        help="Directory where the dataset will be extracted",
    )
    return parser.parse_args()


def build_zip_urls(repo: str) -> list[str]:
    """Return candidate ZIP URLs for ``main`` and ``master`` branches."""
    owner_repo = repo.strip("/")
    return [
        f"https://github.com/{owner_repo}/archive/refs/heads/main.zip",
        f"https://github.com/{owner_repo}/archive/refs/heads/master.zip",
    ]


def try_download_zip(url: str) -> bytes | None:
    """Attempt to download ``url`` and return bytes, or ``None`` on failure."""
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except requests.RequestException:
        return None
    return None


def download_repo_zip(repo: str) -> tuple[bytes, str]:
    """Download the first successful candidate URL for ``repo``."""
    for url in build_zip_urls(repo):
        data = try_download_zip(url)
        if data:
            return data, url
    raise RuntimeError(f"Failed to download ZIP from {repo} (tried main/master branches)")


def extract_zip_to_dir(zip_bytes: bytes, outdir: Path) -> None:
    """Extract the in-memory zip ``zip_bytes`` under ``outdir``."""
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(outdir)


def main() -> None:
    """Download and extract MTS-Dialog, falling back to the mirror if needed."""
    args = parse_args()
    outdir = Path(args.outdir)

    try:
        zip_bytes, used_url = download_repo_zip(args.repo)
        print(f"Downloaded from: {used_url}")
    except RuntimeError:
        print(
            f"Primary repo failed ({args.repo}). Trying fallback: {FALLBACK_REPO}",
            file=sys.stderr,
        )
        zip_bytes, used_url = download_repo_zip(FALLBACK_REPO)
        print(f"Downloaded from: {used_url}")

    extract_zip_to_dir(zip_bytes, outdir)
    print(f"Extracted to: {outdir}")


if __name__ == "__main__":
    main()
