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
# Linting and formatting
# -----------------------------
lint:
	black src tests
	isort src tests
	flake8 src tests
	pylint src
