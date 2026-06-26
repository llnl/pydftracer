.PHONY: help install test test-parallel test-subprocess test-ci lint type-check format clean build
.DEFAULT_GOAL := help

PYTEST_FLAGS = -v --capture=no
COV_FLAGS_BASE = --cov=dftracer --cov-report=term-missing
COV_FLAGS = $(COV_FLAGS_BASE) --cov-append

help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e .[dev]

test:
	pytest tests/ $(COV_FLAGS_BASE) -v

test-parallel: ## Run tests with parallel execution
	pytest tests/ $(COV_FLAGS_BASE) -v -n 4

test-subprocess: ## Run subprocess tests
	pytest tests/ -m subprocess $(PYTEST_FLAGS) -n 4


test-ci: ## Run complete CI test suite (tests + linting + type checking)
	@echo "=== Running main test suite with coverage ==="
	pytest tests/ $(COV_FLAGS_BASE) --cov-report=xml $(PYTEST_FLAGS) -n 4 --ignore=tests/test_dynamo.py
	@echo ""
	@echo "=== Running linting checks ==="
	$(MAKE) lint
	$(MAKE) format-check
	@echo ""
	@echo "=== Running type checking ==="
	$(MAKE) type-check
	@echo ""
	@echo "✅ All CI checks passed! ✅"

test-dynamo: ## Run Dynamo tests with PyTorch
	@echo "=== Running Dynamo tests ==="
	pytest tests/test_dynamo.py -v -n 4 $(COV_FLAGS) $(PYTEST_FLAGS)
	@echo "✅ Dynamo tests passed! ✅"

lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	ruff format .

format-check:
	ruff format --check .

type-check:
	MYPYPATH=python mypy --explicit-package-bases python/dftracer/python --ignore-missing-imports

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f coverage.xml
	rm -rf htmlcov/
	rm -rf .pytest_cache/

build:
	python -m build

check-all: lint format-check type-check test-parallel
fix-all: lint-fix format type-check

test-ci-quick: ## Run quick CI checks without coverage
	@echo "Running quick tests..."
	pytest tests/ -v -n 4 --ignore=tests/test_dynamo.py
	@echo ""
	@echo "Running linting and type checks..."
	ruff check .
	ruff format --check .
	MYPYPATH=python mypy --explicit-package-bases python/dftracer/python --ignore-missing-imports
	@echo ""
	@echo "Quick checks passed! ✅"
