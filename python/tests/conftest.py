"""Shared pytest configuration for Flask application tests."""

import pytest

from app import app


@pytest.fixture(autouse=True)
def configure_csrf_for_tests(request):
    """Disable CSRF only for legacy tests, not security tests."""
    original_csrf = app.config.get("WTF_CSRF_ENABLED", True)

    if request.module.__name__ != "test_csrf":
        app.config["WTF_CSRF_ENABLED"] = False

    try:
        yield
    finally:
        app.config["WTF_CSRF_ENABLED"] = original_csrf
