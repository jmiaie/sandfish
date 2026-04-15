# SandFish - Multi-agent Swarm Intelligence System
# Multi-stage image: builder produces a virtualenv, runtime ships only it.

# ---------- Builder ----------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build deps for any wheels that need to compile (e.g. SQLite extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv we will copy into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy package metadata + source so `pip install .` succeeds.
COPY pyproject.toml README.md ./
COPY sandfish/ ./sandfish/

RUN pip install --upgrade pip \
    && pip install .


# ---------- Runtime ----------
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime libs only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Run as a non-root user.
RUN groupadd -r sandfish && useradd -r -g sandfish -d /app sandfish

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=sandfish:sandfish sandfish/ ./sandfish/

# Vault directory for OMPA-backed memory; mount a volume here in production.
RUN mkdir -p /app/vault && chown -R sandfish:sandfish /app

USER sandfish

# Health check: hit the API rather than just importing the package.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status == 200 else 1)" \
    || exit 1

EXPOSE 8000

CMD ["sandfish", "api", "--host", "0.0.0.0", "--port", "8000", "--vault", "/app/vault"]
