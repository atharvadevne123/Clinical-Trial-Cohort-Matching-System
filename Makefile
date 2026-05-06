.PHONY: install test lint run docker-build docker-up clean

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
