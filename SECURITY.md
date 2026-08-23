# Security Policy

## Supported versions

Security fixes are applied to the latest `0.2.x` release line. Pre-release
snapshots and older versions may not receive fixes.

## Reporting a vulnerability

Please do not open a public issue for an unpatched vulnerability. Use the
repository's **Security → Advisories → New draft security advisory** flow and
include the affected version, impact, reproduction steps, and any suggested
mitigation. If private advisories are unavailable, contact the repository owner
privately before sharing details publicly.

## Runtime security boundary

- The image API binds to `127.0.0.1` by default and requires a Bearer key of at
  least 16 characters.
- A non-loopback bind requires the explicit `--allow-remote` flag. Do not use it
  without network isolation, transport encryption, and a strong secret.
- The service is a local inference endpoint, not a hardened multi-tenant server.
- API keys must be supplied through the environment and must never be committed,
  logged, or passed as command-line arguments.
- Model weights are external data. Download only the pinned files in the bundled
  manifest and run `sensenova-mlx verify` before inference.

The public repository and release packages must never contain model weights,
private keys, tokens, local environment files, or raw private experiment logs.
