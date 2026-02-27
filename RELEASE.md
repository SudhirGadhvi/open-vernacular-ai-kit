# Release Process

This repository uses semantic versioning and tag-driven GitHub Actions releases.

## Tag Formats

- Stable release: `vX.Y.Z` (example: `v1.2.0`)
- Pre-release:
  - `vX.Y.Z-rc.N` (release candidate)
  - `vX.Y.Z-beta.N`
  - `vX.Y.Z-alpha.N`

## Branch Flow

1. Merge features into `develop` through PRs.
2. Prepare release PR from `develop` to `main`.
3. After merge to `main`, create and push a signed/annotated tag.

Example:

```bash
git checkout main
git pull --ff-only
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0
```

## What Automation Does

- On `v*` tag push:
  - Build source distribution and wheel
  - Create GitHub release with generated notes
  - Publish to PyPI only for stable tags (`vX.Y.Z`)
  - Mark GitHub prerelease for prerelease tags (`-rc`, `-beta`, `-alpha`)

## Release Checklist

- [ ] Changelog/release notes reviewed
- [ ] CI green on `main`
- [ ] Docs updated for user-facing changes
- [ ] Security-impacting changes reviewed
- [ ] Version/tag verified before push
