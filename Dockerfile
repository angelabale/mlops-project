# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment (no editable install yet)
RUN uv sync --frozen --no-install-project --no-dev

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install UV in runtime too (needed to run with uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ ./src/
COPY pyproject.toml uv.lock ./

# Copy the trained model artifact
# Make sure model.joblib is present at the project root before building
COPY model.joblib ./model.joblib

# Expose FastAPI port
EXPOSE 8000

# Use the venv's Python directly
ENV PATH="/app/.venv/bin:$PATH"

# Run the API
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]