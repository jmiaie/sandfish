# Installation Guide

AegisFlow is platform-agnostic and runs on any system with Python 3.10+.

## Quick Install

```bash
pip install aegisflow
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.10+ if needed
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Create virtual environment
python3.11 -m venv aegisflow-env
source aegisflow-env/bin/activate

# Install AegisFlow
pip install aegisflow

# Run
aegisflow --version
```

### macOS

```bash
# Using Homebrew
brew install python@3.11

# Create virtual environment
python3.11 -m venv aegisflow-env
source aegisflow-env/bin/activate

# Install AegisFlow
pip install aegisflow

# Run
aegisflow --version
```

### Windows (Native)

```powershell
# Install Python 3.10+ from python.org
# Create virtual environment
python -m venv aegisflow-env
aegisflow-env\Scripts\activate

# Install AegisFlow
pip install aegisflow

# Run
aegisflow --version
```

### Windows (WSL2) - Recommended

```bash
# In WSL2 terminal
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

python3.11 -m venv aegisflow-env
source aegisflow-env/bin/activate
pip install aegisflow
```

### Docker (All Platforms)

```bash
# Pull from Docker Hub (when published)
docker pull jmiaie/aegisflow:latest

# Or build locally
git clone https://github.com/jmiaie/aegisflow.git
cd aegisflow
docker build -t aegisflow .

# Run
docker run -p 8000:8000 aegisflow

# Or use docker-compose
docker-compose up
```

### Raspberry Pi / ARM64

```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv libsqlite3-dev

# Create virtual environment
python3 -m venv aegisflow-env
source aegisflow-env/bin/activate

# Install AegisFlow
pip install aegisflow

# Note: First install may take longer on ARM due to compilation
```

## Cloud Deployment

### AWS EC2

```bash
# User data script for EC2
#!/bin/bash
apt update
apt install -y docker.io
docker run -d -p 80:8000 jmiaie/aegisflow:latest
```

### Google Cloud Run

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: aegisflow
spec:
  template:
    spec:
      containers:
      - image: jmiaie/aegisflow:latest
        ports:
        - containerPort: 8000
```

### Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Development Install

```bash
git clone https://github.com/jmiaie/aegisflow.git
cd aegisflow
pip install -e ".[dev]"
```

## Verification

```bash
# Check installation
aegisflow --version

# Run tests
pytest tests/

# Run security audit
aegisflow security-audit

# Start API
aegisflow api --port 8000
```

## Troubleshooting

### OMPA Not Found

```bash
pip install 'ompa[semantic]'
```

### Permission Denied (Linux/macOS)

```bash
# Use virtual environment instead of system Python
python3 -m venv myenv
source myenv/bin/activate
pip install aegisflow
```

### Windows Path Issues

```powershell
# Use forward slashes or double backslashes
aegisflow orchestrator --vault "C:/Users/Name/vault"
# or
aegisflow orchestrator --vault "C:\\Users\\Name\\vault"
```

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 512 MB | 2 GB |
| Disk | 100 MB | 1 GB (for vault) |
| CPU | 1 core | 2+ cores |

## Next Steps

- [Quick Start Guide](QUICKSTART.md)
- [API Documentation](API.md)
- [Configuration Reference](CONFIG.md)
