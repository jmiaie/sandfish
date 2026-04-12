# Installation Guide

SandFish is platform-agnostic and runs on any system with Python 3.10+.

## Quick Install

```bash
pip install sandfish
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.10+ if needed
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Create virtual environment
python3.11 -m venv sandfish-env
source sandfish-env/bin/activate

# Install SandFish
pip install sandfish

# Run
sandfish --version
```

### macOS

```bash
# Using Homebrew
brew install python@3.11

# Create virtual environment
python3.11 -m venv sandfish-env
source sandfish-env/bin/activate

# Install SandFish
pip install sandfish

# Run
sandfish --version
```

### Windows (Native)

```powershell
# Install Python 3.10+ from python.org
# Create virtual environment
python -m venv sandfish-env
sandfish-env\Scripts\activate

# Install SandFish
pip install sandfish

# Run
sandfish --version
```

### Windows (WSL2) - Recommended

```bash
# In WSL2 terminal
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

python3.11 -m venv sandfish-env
source sandfish-env/bin/activate
pip install sandfish
```

### Docker (All Platforms)

```bash
# Pull from Docker Hub (when published)
docker pull jmiaie/sandfish:latest

# Or build locally
git clone https://github.com/jmiaie/sandfish.git
cd sandfish
docker build -t sandfish .

# Run
docker run -p 8000:8000 sandfish

# Or use docker-compose
docker-compose up
```

### Raspberry Pi / ARM64

```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv libsqlite3-dev

# Create virtual environment
python3 -m venv sandfish-env
source sandfish-env/bin/activate

# Install SandFish
pip install sandfish

# Note: First install may take longer on ARM due to compilation
```

## Cloud Deployment

### AWS EC2

```bash
# User data script for EC2
#!/bin/bash
apt update
apt install -y docker.io
docker run -d -p 80:8000 jmiaie/sandfish:latest
```

### Google Cloud Run

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: sandfish
spec:
  template:
    spec:
      containers:
      - image: jmiaie/sandfish:latest
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
git clone https://github.com/jmiaie/sandfish.git
cd sandfish
pip install -e ".[dev]"
```

## Verification

```bash
# Check installation
sandfish --version

# Run tests
pytest tests/

# Run security audit
sandfish security-audit

# Start API
sandfish api --port 8000
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
pip install sandfish
```

### Windows Path Issues

```powershell
# Use forward slashes or double backslashes
sandfish orchestrator --vault "C:/Users/Name/vault"
# or
sandfish orchestrator --vault "C:\\Users\\Name\\vault"
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
