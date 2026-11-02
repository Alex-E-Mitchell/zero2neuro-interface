"""Temporary server-side storage for generated configurations."""

from uuid import uuid4
from flask import session

# Local development only. Contents are lost when the server restarts.
_configurations = {}


def _get_session_id():
    """Return a unique identifier for the current browser session."""
    if "config_session_id" not in session:
        session["config_session_id"] = uuid4().hex

    return session["config_session_id"]


def save_config(name, content):
    """Save a generated configuration for the current session."""
    if name not in ("network", "data", "experiment"):
        raise ValueError("Unknown configuration type.")

    session_id = _get_session_id()

    if session_id not in _configurations:
        _configurations[session_id] = {}

    _configurations[session_id][name] = content


def get_config(name):
    """Retrieve one configuration from the current session."""
    session_id = _get_session_id()
    return _configurations.get(session_id, {}).get(name)


def get_all_configs():
    """Retrieve all three configurations for the Review page."""
    session_id = _get_session_id()
    saved = _configurations.get(session_id, {})

    return {
        "network": saved.get("network"),
        "data": saved.get("data"),
        "experiment": saved.get("experiment"),
    }


def clear_config(name):
    """Remove a saved configuration for the current session."""
    if name not in ("network", "data", "experiment"):
        raise ValueError("Unknown configuration type.")

    session_id = _get_session_id()
    saved = _configurations.get(session_id, {})
    saved.pop(name, None)
