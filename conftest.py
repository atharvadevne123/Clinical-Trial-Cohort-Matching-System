"""Root-level pytest configuration and shared fixtures."""
import os

import pytest

# Set test database URL before any imports
os.environ.setdefault("DATABASE_URL", "sqlite:///./pytest_test.db")


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Ensure test environment variables are set for the entire test session."""
    os.environ["DATABASE_URL"] = "sqlite:///./pytest_test.db"
    yield
    # Cleanup test database files after session
    import glob
    for db_file in glob.glob("*.db"):
        try:
            os.remove(db_file)
        except OSError:
            pass
