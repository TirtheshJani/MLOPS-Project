import os
import shutil
from pathlib import Path

import subprocess


DATA_DIR = Path("data/primary/mts-dialog")
REPO_URL = "https://github.com/microsoft/clinical_visit_note_summarization_corpus"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = DATA_DIR / "clinical_visit_note_summarization_corpus-main"

    if target_dir.exists():
        print(f"Dataset already present at: {target_dir}")
        return

    # Prefer zip download to avoid full git history
    zip_url = REPO_URL + "/archive/refs/heads/main.zip"
    zip_path = DATA_DIR / "mts_dialog.zip"

    try:
        import requests  # type: ignore

        print(f"Downloading {zip_url} ...")
        with requests.get(zip_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"Falling back to git clone due to: {e}")
        subprocess.run([
            "git", "clone", "--depth", "1", REPO_URL, str(target_dir)
        ], check=True)
        return

    print("Extracting archive ...")
    shutil.unpack_archive(str(zip_path), str(DATA_DIR))
    zip_path.unlink(missing_ok=True)
    print(f"Done. Data at: {target_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import requests
import zipfile

DEFAULT_REPO = "microsoft/clinical_visit_note_summarization_corpus"
FALLBACK_REPO = "abachaa/MTS-Dialog"


def parse_args() -> argparse.Namespace:
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
    owner_repo = repo.strip("/")
    return [
        f"https://github.com/{owner_repo}/archive/refs/heads/main.zip",
        f"https://github.com/{owner_repo}/archive/refs/heads/master.zip",
    ]


def try_download_zip(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except requests.RequestException:
        return None
    return None


def download_repo_zip(repo: str) -> tuple[bytes, str]:
    for url in build_zip_urls(repo):
        data = try_download_zip(url)
        if data:
            return data, url
    raise RuntimeError(f"Failed to download ZIP from {repo} (tried main/master branches)")


def extract_zip_to_dir(zip_bytes: bytes, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(outdir)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    # Try primary repo, then fallback
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
