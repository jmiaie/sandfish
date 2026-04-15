# Installation Guide

SandFish runs on any platform with Python 3.10 or newer. It has no published
PyPI release yet, so install from a local clone.

## From source

```bash
git clone <this-repo>
cd sandfish
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (cmd)
.venv\Scripts\activate.bat
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install .
# For tests, linting, and the security auditor extras:
pip install -e '.[dev]'
```

Verify the CLI is on your path:

```bash
sandfish --version
```

## Platform notes

- **Linux (Debian/Ubuntu):** Install Python with `sudo apt install python3
  python3-venv python3-pip`. No extra system libraries are required for the
  runtime; the build image uses `libsqlite3-dev`/`libsqlite3-0` if you build
  the Docker image locally.
- **macOS:** `brew install python@3.11`, then the `pip install .` flow above.
- **Windows:** Install Python from <https://www.python.org/downloads/>, make
  sure "Add python.exe to PATH" is checked, then use the PowerShell
  activation shown above.
- **WSL2:** Follow the Linux instructions inside your WSL distro.

## Docker

The repo ships a two-stage Dockerfile and a Compose file.

```bash
# build the image
docker compose build

# run the HTTP API on port 8000
docker compose up

# run a long-form simulation alongside the API server
docker compose --profile orchestrator up
```

The container exposes port 8000 and runs as a non-root `sandfish` user. Mount
a volume at `/app/vault` to keep vault data across restarts.

No image is published to a public registry. Build locally or push to your own
registry if you want to deploy it elsewhere.

## Running it

```bash
# a small simulation, no persistence
sandfish orchestrator --rounds 10 --agents 5

# the HTTP API
sandfish api --port 8000 --vault ./sandfish_vault
```

See the main [README](../README.md) for the full CLI and configuration
reference.

## Troubleshooting

**`YAML config requires PyYAML`**

The CLI loads JSON natively; for YAML configs you need `pip install pyyaml`.

**`ompa` failed to install**

```bash
pip install ompa
```

If the default install does not resolve extras your simulation needs, consult
the OMPA project for feature flags.

**Windows path issues with `--vault`**

Use forward slashes or quote the path:

```powershell
sandfish orchestrator --vault "C:/Users/Name/sandfish_vault"
```

## System requirements

SandFish is light; requirements scale with simulation size.

| Resource | Suggested minimum |
| --- | --- |
| Python | 3.10 |
| RAM | 512 MB for small simulations |
| Disk | 100 MB plus vault size |
| CPU | Any modern x86_64 / arm64 core |

For larger simulations, see `benchmarks/run_all.py`, which exercises startup,
agent creation, full-simulation throughput, and (if `psutil` is installed)
process memory.
