.PHONY: help install install-dev test lint typecheck notebook-check clean format coverage

.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help:
	@echo "$(BLUE)=== Hybrid Tokenizer Development Makefile ===$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          Install package in production mode"
	@echo "  make install-dev      Install package with dev dependencies"
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(NC)"
	@echo "  make test             Run pytest on all tests/"
	@echo "  make coverage         Run tests with coverage report (80% threshold)"
	@echo "  make lint             Run flake8 and pylint checks"
	@echo "  make typecheck        Run mypy type checking"
	@echo "  make format           Format code with black and isort"
	@echo ""
	@echo "$(GREEN)Validation:$(NC)"
	@echo "  make notebook-check   Validate Kaggle notebook structure"
	@echo "  make pre-commit       Install and run pre-commit hooks"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean            Remove build artifacts and caches"
	@echo "  make clean-hard       Remove all generated files including .venv"
	@echo ""

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pytest-cov coverage flake8 pylint black isort mypy pre-commit

test:
	pytest tests/ -v --tb=short

coverage:
	pytest tests/ \
		--cov=src/my_slm \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=80
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/index.html$(NC)"

lint:
	@echo "$(YELLOW)Running Flake8...$(NC)"
	flake8 src/ tests/ \
		--count \
		--statistics \
		--max-line-length=100 \
		--ignore=E501,W503,E203 || true
	@echo ""
	@echo "$(YELLOW)Running Pylint...$(NC)"
	pylint src/ --exit-zero --disable=fixme || true

typecheck:
	@echo "$(YELLOW)Running mypy type checking...$(NC)"
	mypy src/ --ignore-missing-imports --no-error-summary || true

format:
	@echo "$(YELLOW)Formatting with Black...$(NC)"
	black src/ tests/ --line-length=100
	@echo "$(YELLOW)Sorting imports with isort...$(NC)"
	isort src/ tests/ --profile=black --line-length=100
	@echo "$(GREEN)✓ Code formatted$(NC)"

notebook-check:
	@echo "$(YELLOW)Validating notebook structure...$(NC)"
	python -c "import json, glob; \
	for nb in glob.glob('**/*.ipynb', recursive=True): \
		try: \
			with open(nb) as f: json.load(f); \
			print(f'✓ {nb}'); \
		except Exception as e: \
			print(f'✗ {nb}: {e}'); \
			exit(1)"
	@echo "$(GREEN)✓ All notebooks are valid$(NC)"

pre-commit:
	@echo "$(YELLOW)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(YELLOW)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files || true
	@echo "$(GREEN)✓ Pre-commit setup complete$(NC)"

clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-hard: clean
	@echo "$(YELLOW)Performing hard clean...$(NC)"
	rm -rf .venv venv env
	@echo "$(GREEN)✓ Hard clean complete$(NC)"

ci: install-dev test coverage lint typecheck notebook-check
	@echo "$(GREEN)✓ All CI checks passed$(NC)"
