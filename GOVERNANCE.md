# Governance

This repository follows a maintainer-led model with transparent review and protected branches.

## Roles

- Maintainer: project owner (`@SudhirGadhvi`)
- Contributors: anyone submitting issues, docs, code, or review feedback

## Decision Model

- Small changes: maintainers decide during PR review.
- Significant changes (architecture, release process, language/provider roadmap): discuss in an issue before implementation.
- Security-sensitive decisions follow `SECURITY.md` and private disclosure practices.

## Protected Branches

- `main`: no direct pushes; PR-only merges.
- `develop`: integration branch; PR-only merges.

Expected protections:

- Required pull request reviews
- Required passing status checks
- Required conversation resolution
- Force-push and branch deletion blocked

## Merge Policy

- Squash merge is recommended for most contributor PRs.
- Merge only when CI passes and required reviews are approved.
- Hotfixes to `main` should be back-merged to `develop`.

## Versioning and Releases

- Semantic versioning with `vX.Y.Z` tags.
- Pre-releases use `vX.Y.Z-rc.N`, `-beta.N`, or `-alpha.N`.
- See `RELEASE.md` for full flow.

## Community Health

The following files define repository standards:

- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `SUPPORT.md`
- `.github/CODEOWNERS`
