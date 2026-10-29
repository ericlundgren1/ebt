# Releasing

Every release is one command:

```bash
scripts/release.sh 0.1.1
```

It bumps the version in `pyproject.toml` and `ebt_rust/Cargo.toml`, commits, tags `v0.1.1` and
pushes. The tag starts the **Release** workflow (`.github/workflows/release.yml`), which builds
the sdist and the wheels (Linux x86_64 and Windows x86_64; one abi3 wheel per platform for
Python 3.9+; macOS installs from the sdist) and publishes them to PyPI as `ebt-backtest`.

## One-time setup

1. Push the repo to GitHub and create an environment named `pypi` under
   Settings → Environments (no secrets needed).
2. On PyPI, Account settings → Publishing → "Add a new pending publisher":
   project `ebt-backtest`, your GitHub owner and repo name, workflow `release.yml`,
   environment `pypi`. The first publish claims the project name.
3. Optional dry runs: do the same on https://test.pypi.org, then Actions → Release →
   "Run workflow" with "Upload to TestPyPI" ticked. Install the result with
   `pip install -i https://test.pypi.org/simple/ ebt-backtest`.

## Local build (optional)

```bash
pip install build twine
python -m build          # dist/ebt_backtest-X.Y.Z.tar.gz + a wheel for this machine
twine check dist/*
```

The local Linux wheel is tagged `linux_x86_64`, which PyPI does not accept; the workflow
builds `manylinux` wheels with cibuildwheel. Installing from the sdist needs a Rust toolchain.
