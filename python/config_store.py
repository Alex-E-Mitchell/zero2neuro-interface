"""Temporary server-side storage for generated configurations."""

from uuid import uuid4
from flask import session

# Local development only; server restarts lose saved configurations.
_configurations = {}
# Duplicate submissions awaiting a decision on the Review page.
_duplicate_attempts = {}

VALID_CONFIG_TYPES = ("network", "data", "experiment")


def _validate_name(name):
    if name not in VALID_CONFIG_TYPES:
        raise ValueError("Unknown configuration type.")


def _get_session_id():
    if "config_session_id" not in session:
        session["config_session_id"] = uuid4().hex
    return session["config_session_id"]


def _get_session_configs():
    session_id = _get_session_id()
    return _configurations.setdefault(
        session_id,
        {name: [] for name in VALID_CONFIG_TYPES},
    )


def save_config(name, content, allow_duplicate=False):
    """Save a configuration, unless it duplicates an existing one."""
    _validate_name(name)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Cannot save an empty configuration.")

    configs = _get_session_configs()[name]

    if not allow_duplicate:
        for existing in configs:
            if existing["content"] == content:
                session_id = _get_session_id()

                _duplicate_attempts.setdefault(session_id, []).append({
                    "id": uuid4().hex,
                    "name": name,
                    "existing_config_id": existing["id"],
                    "content": content,
                })

                return {
                    "status": "duplicate",
                    "config": existing.copy(),
                }

    saved = {
        "id": uuid4().hex,
        "content": content,
    }
    configs.append(saved)

    return {
        "status": "saved",
        "config": saved.copy(),
    }


def get_configs(name):
    """Return all saved configurations of a given type."""
    _validate_name(name)
    return [
        config.copy()
        for config in _get_session_configs()[name]
    ]


def get_config(name, config_id=None):
    """Return a specific configuration, or the latest if no ID is given."""
    configs = get_configs(name)

    if config_id is None:
        return configs[-1]["content"] if configs else None

    for config in configs:
        if config["id"] == config_id:
            return config["content"]

    return None


def get_all_configs():
    """Return all saved configurations grouped by type."""
    saved = _get_session_configs()
    return {
        name: [config.copy() for config in saved[name]]
        for name in VALID_CONFIG_TYPES
    }


def delete_config(name, config_id):
    """Remove only the configuration with the matching ID."""
    _validate_name(name)

    configs = _get_session_configs()[name]

    for index, config in enumerate(configs):
        if config["id"] == config_id:
            del configs[index]
            return True

    return False


def clear_config(name):
    """Clear all saved configurations of a given type."""
    _validate_name(name)
    _get_session_configs()[name].clear()


def get_duplicate_attempts():
    """Return duplicate submissions awaiting user decisions."""
    session_id = _get_session_id()
    return [
        attempt.copy()
        for attempt in _duplicate_attempts.get(session_id, [])
    ]


def keep_duplicate(attempt_id):
    """Save a pending duplicate as another configuration."""
    session_id = _get_session_id()
    attempts = _duplicate_attempts.get(session_id, [])

    for index, attempt in enumerate(attempts):
        if attempt["id"] == attempt_id:
            result = save_config(
                attempt["name"],
                attempt["content"],
                allow_duplicate=True,
            )

            del attempts[index]
            return result

    return None


def discard_duplicate(attempt_id):
    """Discard a pending duplicate without deleting saved configurations."""
    session_id = _get_session_id()
    attempts = _duplicate_attempts.get(session_id, [])

    for index, attempt in enumerate(attempts):
        if attempt["id"] == attempt_id:
            del attempts[index]
            return True

    return False
