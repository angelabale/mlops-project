#  builder (dependencies only)
FROM python:3.12-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-install-project 

# trainer (runs training and produces model.joblib)
FROM python:3.12-slim AS trainer

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code and data
COPY src/ ./src/
COPY data/ ./data/
COPY pyproject.toml uv.lock ./

ENV PATH="/app/.venv/bin:$PATH"

# Run training — produces model.joblib at /app/model.joblib
RUN python -m src.models.train

#  runtime
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ ./src/
COPY pyproject.toml uv.lock ./

# Copy the trained model from trainer stage (no local file needed!)
COPY --from=trainer /app/model.joblib ./model.joblib

# Expose FastAPI port
EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"

# Run the API
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]