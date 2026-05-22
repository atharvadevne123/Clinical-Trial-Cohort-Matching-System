.PHONY: install test test-fast test-integration lint lint-fix run docker-build docker-up docker-down clean seed coverage-ci type-check retrain benchmark

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio ruff httpx

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -x --tb=short --ignore=tests/test_recruitment.py

lint:
	ruff check . --select E,F,W,I --ignore E501

lint-fix:
	ruff check . --select E,F,W,I --ignore E501 --fix

run:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t clinical-trial-matching .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f test_clinical.db test_api.db test.db coverage.xml

seed:
	curl -s -X POST http://localhost:8000/admin/seed -H "X-API-Key: $$API_KEY" | python3 -m json.tool

coverage-ci:
	pytest tests/ --cov=src --cov-report=xml --cov-fail-under=60 -q

type-check:
	python -m mypy src/ --ignore-missing-imports --no-error-summary

retrain:
	python pipelines/retrain_pipeline.py

benchmark:
	python scripts/benchmark.py

test-integration:
	pytest tests/test_integration_e2e.py -v --tb=short

validate-env:
	@python3 -c "import os; missing=[v for v in ['API_KEY','DATABASE_URL'] if not os.environ.get(v)]; print('Missing env vars:', missing) if missing else print('All required env vars set.')"

help:
	@echo "Available targets:"
	@echo "  install         Install Python dependencies"
	@echo "  test            Run all tests with coverage"
	@echo "  test-fast       Run tests skipping slow recruitment tests"
	@echo "  test-integration Run only integration tests"
	@echo "  lint            Run ruff linter"
	@echo "  lint-fix        Run ruff linter with auto-fix"
	@echo "  run             Start uvicorn dev server"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-up       Start docker-compose stack"
	@echo "  docker-down     Stop docker-compose stack"
	@echo "  coverage-ci     Run tests with XML coverage report"
	@echo "  type-check      Run mypy type checker"
	@echo "  retrain         Run retraining pipeline"
	@echo "  benchmark       Run performance benchmarks"
	@echo "  validate-env    Check required environment variables"
	@echo "  seed            Seed the database via API"
