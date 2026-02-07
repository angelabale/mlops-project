# -----------------------------
# Install development dependencies
# -----------------------------
install:
	uv sync --dev

# -----------------------------
# Run pre-commit hooks on all files
# -----------------------------
precommit:
	uv run pre-commit run --all-files

# -----------------------------
# Run unit tests
# -----------------------------
test:
	uv run pytest

# -----------------------------
# Run tests with coverage
# -----------------------------
coverage:
	uv run pytest --cov=src --cov-report=term-missing

# -----------------------------
# Linting and formatting
# -----------------------------
lint:
	black src tests
	isort src tests
	flake8 src tests
	pylint src

