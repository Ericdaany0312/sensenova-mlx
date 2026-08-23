# Contributing

Thanks for improving SenseNova MLX. Keep changes focused, reproducible, and safe
for a public repository.

## Development setup

You need an Apple silicon Mac, Python 3.11–3.13, and `uv`.

```bash
uv sync --python 3.12
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_config_flow tests.test_download tests.test_generate_cli \
  tests.test_packaging tests.test_rope_attention tests.test_server tests.test_weights
uv build
```

The default test suite does not require model weights or PyTorch. Numerical
parity tooling uses the optional `parity` dependency and formal weights, so it
must be run separately on an authorized machine.

## Pull requests

- Add a failing regression test before changing behavior.
- Keep the runtime free of PyTorch and avoid adding dependencies unless needed.
- Preserve the frozen model revisions and hash verification boundary.
- Explain the behavior change and list the exact verification commands.
- Do not lower numerical, memory, or swap thresholds to make a test pass.

## Files that must never be committed

Do not add model weights, `.venv`, caches, generated `artifacts/`, raw profiles,
credentials, private keys, proxy configuration, or machine-specific paths. One
curated sample and the sanitized verification summary under `docs/` are the only
generated evidence intended for source control.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
