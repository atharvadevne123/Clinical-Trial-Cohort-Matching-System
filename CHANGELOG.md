# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Full test suite covering eligibility, ML prediction, NLP, FHIR, schemas, and API endpoints.
- GitHub Actions CI workflow with ruff linting and pytest coverage.
- Makefile with install, test, lint, run, docker targets.
- CONTRIBUTING.md with development workflow documentation.
- CHANGELOG.md tracking all notable changes.
- `.pre-commit-config.yaml` with ruff and trailing-whitespace hooks.
- Type annotations across all source modules.
- Google-style docstrings for all public classes and functions.
- Structured logging replacing bare print() calls.
- `/version` endpoint returning API version metadata.
- `/metrics` endpoint returning runtime statistics.
- Input validation with Pydantic field validators on PatientCreate and TrialCreate.
- Rate limiting middleware (SlowAPI) on mutation endpoints.
- Correlation ID middleware for distributed tracing.
- Environment variable support for SMTP configuration in RecruitmentEngine.
- Connection pool settings on the SQLAlchemy engine.
- `lru_cache` on NLP keyword lookup for repeated calls.

## [1.0.0] - 2024-01-01

### Added
- Rule-based eligibility matching engine with ICD-10 criteria support.
- XGBoost ML enrollment probability prediction with 14 features.
- Keyword NLP with negation detection for clinical entity extraction.
- FHIR R4 client with mock fallback for offline development.
- Async recruitment engine with SMTP outreach support.
- FastAPI REST API with patients, trials, matching, NLP, ML, and FHIR endpoints.
- PostgreSQL persistence via SQLAlchemy ORM.
- Docker and docker-compose configuration for local deployment.
- Metabase analytics dashboard integration.
- Seed data generator for synthetic patient and trial records.
