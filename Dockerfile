# Stage 1: Fetch the uv binary
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# Stage 2: Base environment with dependencies
FROM python:3.12-slim AS base

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

# Ensure Python can resolve the src package
ENV PYTHONPATH=/src

# ---------------------------------------------------
# Stage 3: Testing environment
# ---------------------------------------------------
FROM base AS tester

# Copy all source code including tests
COPY src/ ./src/
COPY ui/ ./ui/
COPY tests/ ./tests/

# Default command to execute tests inside the container
CMD ["uv", "run", "pytest", "--cov=src", "-v"]

# ---------------------------------------------------
# Stage 4: Final Production Image
# ---------------------------------------------------
FROM base AS production

# Copy only the necessary application source code (excluding tests)
COPY src/ ./src/
COPY ui/ ./ui/

# Default command to start the FastAPI application
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]