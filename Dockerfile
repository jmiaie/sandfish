# SandFish - Multi-agent Swarm Intelligence System
# Production-grade container with security hardening

FROM python:3.11-slim-bookworm AS builder

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install build && \
    pip install -e .[dev]

# Production stage
FROM python:3.11-slim-bookworm

# Security: Run as non-root user
RUN groupadd -r sandfish && useradd -r -g sandfish sandfish

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=sandfish:sandfish sandfish/ ./sandfish/
COPY --chown=sandfish:sandfish tests/ ./tests/
COPY --chown=sandfish:sandfish pyproject.toml README.md ./

# Install package
RUN pip install -e .

# Create vault directory
RUN mkdir -p /app/vault && chown -R sandfish:sandfish /app

# Security: Drop to non-root user
USER sandfish

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sandfish; print('OK')" || exit 1

# Expose API port
EXPOSE 8000

# Default command
CMD ["sandfish", "api", "--host", "0.0.0.0", "--port", "8000", "--vault", "/app/vault"]
