"""HTTPS deployment entry point; local development still uses app.py."""

import os

from app import app

if not os.environ.get("FLASK_SECRET_KEY"):
    raise RuntimeError("Set FLASK_SECRET_KEY before starting the hosted app.")

app.config.update(
    DEBUG=False,
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
