# Contributing to Open Vernacular AI Kit

Thanks for contributing. This project is India-first today (Sarvam-first integrations) and open to global/community expansion through PRs.

## Ground Rules

- Be respectful and follow `CODE_OF_CONDUCT.md`.
- Do not push directly to protected branches (`main`, `develop`).
- Open an issue before major feature work so design and scope are aligned.
- Keep PRs focused and small when possible.

## Branching Strategy

- `main`: stable and releasable only.
- `develop`: integration branch for upcoming release work.
- Feature branches: create from `develop`.

Recommended branch name format:

- `feat/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `chore/<short-description>`
- `refactor/<short-description>`
- `test/<short-description>`
- `hotfix/<short-description>`
- `release/<short-description>`

Example: `feat/hindi-token-lid-bootstrap`

## Optional: Worktree-Based Parallel Development

If you work on multiple PRs in parallel, prefer `git worktree` instead of stashing repeatedly.

```bash
# From repo root
git worktree add ../ovak-feat-token-lid -b feat/token-lid-improvements develop
cd ../ovak-feat-token-lid
```

Benefits:

- Isolated branch directories
- Cleaner context switching
- Fewer accidental cross-branch changes

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,indic,demo,eval,rag]"
```

## Quality Checks Before PR

```bash
ruff check .
pytest -q
```

If docs changed:

```bash
mkdocs build --strict
```

## Pull Request Requirements

- Base branch should usually be `develop` (or `main` for urgent hotfixes).
- Fill out the PR template completely.
- Include tests for behavior changes.
- Update docs/README when user-facing behavior changes.
- Keep CI green before requesting review.
- At least 1 approval is required; code-owner review is required for owned paths.

## Commit Message Style

Use clear messages that explain why the change exists. Conventional Commit prefixes are recommended:

- `feat:`
- `fix:`
- `docs:`
- `refactor:`
- `test:`
- `chore:`

## Release Flow (Tags + RC)

- Release candidates: `vX.Y.Z-rc.N` (example: `v1.1.0-rc.1`)
- Stable releases: `vX.Y.Z` (example: `v1.1.0`)
- RC tags create GitHub prereleases.
- Stable tags create full GitHub releases and publish package artifacts.

See `RELEASE.md` for details.

## Security

Do not open public issues for vulnerabilities. See `SECURITY.md` for reporting instructions.
