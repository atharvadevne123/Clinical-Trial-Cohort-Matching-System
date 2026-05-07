"""Database migration helpers using Alembic-style raw SQL migrations.

Provides a minimal migration runner for applying incremental schema changes
to the PostgreSQL/SQLite database without a full Alembic setup.
"""

import logging
from datetime import datetime, timezone
from typing import Any, List

from sqlalchemy import text

logger = logging.getLogger(__name__)

MIGRATIONS: List[dict] = [
    {
        "version": "001",
        "description": "Add letter_sent_date index to patient_trial_matches",
        "up": "CREATE INDEX IF NOT EXISTS idx_ptm_letter_sent_date ON patient_trial_matches (letter_sent_date);",
        "down": "DROP INDEX IF EXISTS idx_ptm_letter_sent_date;",
    },
    {
        "version": "002",
        "description": "Add enrollment_date index to patient_trial_matches",
        "up": "CREATE INDEX IF NOT EXISTS idx_ptm_enrollment_date ON patient_trial_matches (enrollment_date);",
        "down": "DROP INDEX IF EXISTS idx_ptm_enrollment_date;",
    },
    {
        "version": "003",
        "description": "Add combined_score index for ranking queries",
        "up": "CREATE INDEX IF NOT EXISTS idx_ptm_combined_score ON patient_trial_matches (combined_score DESC);",
        "down": "DROP INDEX IF EXISTS idx_ptm_combined_score;",
    },
]


def apply_migrations(engine: Any) -> None:
    """Apply all pending migrations in order.

    Creates a migration_log table if it does not exist, then executes
    any migrations not yet recorded.

    Args:
        engine: SQLAlchemy engine instance.
    """
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS migration_log ("
            "  version VARCHAR(10) PRIMARY KEY,"
            "  description TEXT,"
            "  applied_at TIMESTAMP"
            ")"
        ))
        conn.commit()

        applied = {
            row[0] for row in conn.execute(text("SELECT version FROM migration_log"))
        }

        for migration in MIGRATIONS:
            if migration["version"] not in applied:
                logger.info("Applying migration %s: %s", migration["version"], migration["description"])
                try:
                    conn.execute(text(migration["up"]))
                    conn.execute(text(
                        "INSERT INTO migration_log (version, description, applied_at) "
                        "VALUES (:v, :d, :t)"
                    ), {"v": migration["version"], "d": migration["description"],
                        "t": datetime.now(timezone.utc).isoformat()})
                    conn.commit()
                    logger.info("Migration %s applied.", migration["version"])
                except Exception as exc:
                    logger.error("Migration %s failed: %s", migration["version"], exc)
                    conn.rollback()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.models import engine
    apply_migrations(engine)
    logger.info("All migrations applied.")
