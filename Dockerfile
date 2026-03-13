# Stage 1: Fetch the uv binary
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# Stage 2: Final image based on the official Python slim variant
FROM python:3.12-slim

WORKDIR /src

# Install system-level dependencies required by psycopg and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the first stage
COPY --from=uv_bin /uv /uvx /bin/

# Copy dependency manifests first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies in frozen mode (pinned versions from lockfile).
# --no-install-project skips installing the project itself since source is not copied yet.
RUN uv sync --frozen --no-install-project

# Copy application source code
COPY src/ ./src/
COPY ui/ ./ui/

# Ensure Python can resolve the src package
ENV PYTHONPATH=/src

# Default command (overridden by docker-compose per service)
CMD ["python"]