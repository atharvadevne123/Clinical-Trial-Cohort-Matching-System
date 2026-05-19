# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-05-19

### Added
- `FHIR_TIMEOUT` env var controls FHIR HTTP client timeout (default 5.0s).
- `DRIFT_THRESHOLD` env var controls KS-test p-value threshold (default 0.05).
- `SMTP_TIMEOUT` env var controls SMTP connection timeout (default 5s).
- `lru_cache(maxsize=512)` on `is_valid_icd10` and `is_valid_atc` validators.
- Database indexes on `patients.gender`, `patients.date_of_birth`, `trials.phase`, and `trials.status`.
- `__all__` exports to `monitoring`, `validators`, `features`, and `eligibility` modules.
- Startup warning logged when `API_KEY` env var is not set.
- API key authentication on all `/monitoring/*` endpoints.
- `_NEGATION_WINDOW_CHARS` and `_SEVERITY_WINDOW_CHARS` constants extracted in `nlp.py`.
- `_strip_timezone` helper and `DEFAULT_AGE` constant extracted in `features.py`.
- `_INCLUSION_WEIGHT` and `_EXCLUSION_WEIGHT` constants extracted in `eligibility.py`.
- `Field(...)` descriptions added to all `MatchResponse` schema fields.

### Fixed
- `MODEL_PATH` in `retrain_pipeline.py` resolved relative to `src/` correctly.
- Bare `except Exception` replaced with specific `httpx` exception types in `fhir.py`.
- ICD-10 regex extended to accept alphanumeric suffixes (e.g. `Z00.00A`).
- Input type guard added to `_dict_to_features()` in `ml_prediction.py`.

### Changed
- `logger.warning` used instead of `logger.debug` for FHIR unreachable server messages.
- `print()` calls replaced with `logger.info()` in `benchmark.py`, `seed_data.py`, and `retrain_pipeline.py`.
- Email validator tightened to require `local@domain.tld` format.
- PatientDict type alias added to `features.py` and `recruitment.py`.

### Added (tests)
- Extended test coverage: eligibility operators, validator cache, feature timezone, ML fallback,
  FHIR fallback, monitoring threshold env var, NLP negation window, schema email/gender,
  monitoring router API key, pipeline custom min_samples and MODEL_PATH.

## [1.0.0] - 2024-01-01

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
