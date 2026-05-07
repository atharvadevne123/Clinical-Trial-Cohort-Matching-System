"""Tests for database migration runner."""
import pytest
from sqlalchemy import create_engine, text

from scripts.migrate import MIGRATIONS, apply_migrations


@pytest.fixture
def test_engine():
    eng = create_engine("sqlite:///:memory:")
    # Create required tables
    with eng.connect() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS patient_trial_matches ("
            "  id INTEGER PRIMARY KEY,"
            "  patient_id TEXT,"
            "  trial_id TEXT,"
            "  letter_sent_date TEXT,"
            "  enrollment_date TEXT,"
            "  combined_score REAL"
            ")"
        ))
        conn.commit()
    return eng


def test_apply_migrations_creates_log_table(test_engine):
    apply_migrations(test_engine)
    with test_engine.connect() as conn:
        rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        table_names = [r[0] for r in rows]
    assert "migration_log" in table_names


def test_apply_migrations_records_applied(test_engine):
    apply_migrations(test_engine)
    with test_engine.connect() as conn:
        rows = conn.execute(text("SELECT version FROM migration_log")).fetchall()
    applied_versions = {r[0] for r in rows}
    for m in MIGRATIONS:
        assert m["version"] in applied_versions


def test_apply_migrations_idempotent(test_engine):
    apply_migrations(test_engine)
    apply_migrations(test_engine)
    with test_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM migration_log")).scalar()
    assert count == len(MIGRATIONS)


def test_migrations_have_required_keys():
    for m in MIGRATIONS:
        assert "version" in m
        assert "description" in m
        assert "up" in m
        assert "down" in m
