# Contributing

Thanks for taking the time to contribute! This guide covers the local
development workflow and the expectations for pull requests.

## Development setup

```bash
# Clone + create a virtual environment
git clone https://github.com/TirtheshJani/MLOPS-Project.git
cd MLOPS-Project
python -m venv .venv && source .venv/bin/activate

# Install runtime + dev dependencies
pip install -r clinical-note-summarizer/requirements.txt
pip install pre-commit

# Install git hooks
pre-commit install
```

For the frontend:

```bash
cd web
npm ci
```

## Running checks locally

Please run all of the following before opening a PR:

```bash
# Python lint + format
ruff check .
ruff format --check .

# Type check (non-blocking but encouraged)
mypy clinical-note-summarizer/app

# Tests (uses a tiny HF model for speed)
MODEL_DIR=hf-internal-testing/tiny-random-t5 \
USE_FAST_TOKENIZER=true \
PYTHONPATH=clinical-note-summarizer \
  pytest -q --cov=app --cov-report=term-missing

# Frontend
cd web && npm run lint && npm run build
```

## Branching and commits

- Branch from `main`. Use a descriptive branch name, e.g.
  `feat/batch-inference` or `fix/cors-env`.
- Use clear, imperative commit messages (`Add …`, `Fix …`, `Refactor …`).
- Keep commits focused; rebase/squash noise before opening the PR.

## Pull requests

- Include a short description of **what** changed and **why**.
- Link any related issue with `Fixes #N` / `Refs #N`.
- Make sure CI (lint, tests, frontend build) is green.
- If you change configuration, update `README.md` and any relevant
  Kubernetes or Docker files.

## Code style

- Python: enforced by `ruff check` and `ruff format` (config in
  `pyproject.toml`). Target Python 3.11.
- JavaScript/React: enforced by ESLint (`web/eslint.config.js`).
- Prefer type hints on new or modified Python code.
- Add or update tests for any behaviour change.

## Security / data handling

- Never commit real PHI, credentials, or `.env` files.
- Run `git diff --cached` before committing if you've touched data files.

Thanks again — happy hacking!
