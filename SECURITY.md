# Security Policy

## Supported Versions

Security fixes are prioritized for the latest stable major/minor release line.

| Version | Supported |
| --- | --- |
| 1.x | yes |
| < 1.0 | no |

## Reporting a Vulnerability

Please do not report security vulnerabilities in public issues.

Use one of these channels:

1. Preferred: GitHub private vulnerability reporting (Security tab -> Report a vulnerability)
2. Fallback: open a private maintainer contact request and ask for an encrypted channel

When reporting, include:

- A clear description of the issue
- Steps to reproduce
- Potential impact
- Any suggested remediation

We aim to:

- Acknowledge reports within 72 hours
- Provide an initial triage timeline
- Coordinate disclosure after a fix is available

## Security Best Practices for Contributors

- Never commit secrets, API keys, or credentials.
- Keep dependencies updated and review Dependabot PRs promptly.
- Validate and sanitize all external inputs.
- Avoid broad, unnecessary permissions in GitHub Actions workflows.
