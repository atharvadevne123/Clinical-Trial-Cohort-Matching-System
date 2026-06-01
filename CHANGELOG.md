# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.2] - 2026-06-01

### Added
- `GET /patients/count` endpoint returning patient count with optional gender filter.
- `GET /trials/count` endpoint returning trial count with optional phase filter.
- `GET /api/v1/operators` versioned endpoint listing all supported eligibility operators.
- `GET /api/v1/version` versioned endpoint returning API version metadata.
- `operator_names` property on `EligibilityMatcher` returning sorted operator list.
- `__repr__` method on `EligibilityMatcher` for readable string representation.
- `count_eligible()` helper on `EligibilityMatcher` counting eligible patients for a trial.
- `batch_record()`, `__len__()`, `__repr__()`, `percentile()`, and `clear_reference()` on `PredictionMonitor`.
- `is_fitted`, `feature_names`, and `reset()` on `ClinicalFeaturePipeline`.
- `get_condition_count()` and `get_medication_count()` utility functions in `features.py`.
- `_FEATURE_COUNT` constant in `features.py` documenting the feature vector size.
- `_MAX_TEXT_LENGTH` constant in `nlp.py` limiting input text size.
- `validate_probability()` static method on `RecruitmentEngine`.
- `_DEFAULT_SCORE_THRESHOLD` and `_DEFAULT_BATCH_SIZE` constants in `recruitment.py`.
- `tests/test_retrain_pipeline.py` with 9 pipeline tests (synthetic data, skip, success).
- `tests/test_api_v1_extended.py` with 11 v1 router endpoint tests.
- Type annotations added to `tests/conftest.py` fixtures.
- `format`, `check`, and `coverage-html` Makefile targets.
- Branch coverage and HTML report configured in `pyproject.toml`.

### Fixed
- Version mismatch: `GET /` root endpoint now returns `1.2.0` (was `1.1.0`).

## [1.2.1] - 2026-05-22

### Added
- `NOT_IN` operator: excludes patients whose field value appears in a comma-separated list.
- `CONTAINS` operator: case-insensitive substring check on string fields.
- `BETWEEN` operator: inclusive numeric range check using `"lo,hi"` value syntax.
- `SUPPORTED_OPERATORS` list exported from `src/eligibility.py` for introspection.
- `GET /trials/{id}/eligible-patients` endpoint returning eligible patient IDs for a trial.
- `GET /operators` meta endpoint listing all supported eligibility criterion operators.
- `X-Request-ID` response header echoed (or generated) on every response.
- CI workflow: `ruff format --check .` added to lint step.
- CI workflow: smoke test step verifying key endpoints on Python 3.11.

### Changed
- HTTP middleware docstring updated to reflect `X-Request-ID` injection.

## [1.2.0] - 2026-05-22

### Added
- `DELETE /patients/{id}` and `DELETE /trials/{id}` endpoints for record removal.
- `PATCH /patients/{id}` and `PATCH /trials/{id}` endpoints for partial updates.
- `POST /patients/bulk` (up to 100) and `POST /trials/bulk` (up to 50) batch creation endpoints.
- `GET /matches` endpoint to list all match records with optional status filter and pagination.
- `GET /patients/{id}/eligible-trials` endpoint returning matched trial IDs.
- `GET /summary` endpoint with aggregated patient-by-gender, trials-by-phase, matches-by-status.
- `GET /ping` ultra-lightweight liveness probe returning `{"ping": "pong"}`.
- `GET /readyz` database readiness probe querying the DB.
- `POST /monitoring/set-reference` endpoint to set reference distribution for drift detection.
- `GZipMiddleware` for automatic response compression (>1 KB).
- `CORS_ORIGINS` env var for configurable allowed origins (default `*`).
- `X-Process-Time-Ms` header added to all responses via middleware.
- OpenAPI tags metadata with descriptions for all endpoint groups.
- Startup banner log message with version, DB type, and CORS config.
- `PatientUpdate` and `TrialUpdate` Pydantic schemas for partial update validation.
- `MatchStatus` enum (`PENDING`, `ELIGIBLE`, `INELIGIBLE`, `ENROLLED`, `WITHDRAWN`) in models.
- `DriftResult` and `SummaryResult` TypedDicts in `monitoring.py`.
- `validate_patient_medications` and `validate_criteria_list` functions in `validators.py`.
- Exponential-backoff retry logic (`_get_with_retry`) in FHIR client.
- `_patient_to_dict` and `_trial_to_dict` helpers extracted in `main.py`.
- Monitoring router now properly included in the main FastAPI app.
- `val_samples` field added to successful retraining pipeline result.

### Fixed
- Test isolation bug: `app.dependency_overrides.clear()` replaced with `.pop(get_db, None)`.
- `Trial.recruitment_status` attribute reference corrected to `Trial.status`.
- FHIR client now uses specific `httpx.ConnectError`/`httpx.TimeoutException` instead of bare `except`.
- OSError handling added for model persistence failures in retraining pipeline.
- `send_recruitment_email` now catches `smtplib.SMTPException` and `OSError` specifically.

### Changed
- Version bumped to `1.2.0` in `pyproject.toml`, `src/__init__.py`, and `src/main.py`.

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
