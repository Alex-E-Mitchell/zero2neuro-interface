"""Tests for CSRF protection on configuration forms."""

import pytest

from app import app


@pytest.fixture
def client():
    original_testing = app.config["TESTING"]
    original_csrf = app.config["WTF_CSRF_ENABLED"]

    app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=True,
    )

    try:
        with app.test_client() as client:
            yield client
    finally:
        app.config.update(
            TESTING=original_testing,
            WTF_CSRF_ENABLED=original_csrf,
        )


@pytest.mark.parametrize(
    "path",
    ["/", "/data", "/experiment"],
)
def test_missing_csrf_token_is_rejected(client, path):
    """Configuration forms must reject POSTs without CSRF tokens."""
    response = client.post(path, data={})

    assert response.status_code == 400


@pytest.mark.parametrize(
    "path",
    ["/", "/data", "/experiment"],
)
def test_invalid_csrf_token_is_rejected(client, path):
    """Configuration forms must reject invalid CSRF tokens."""
    response = client.post(
        path,
        data={"csrf_token": "invalid-token"},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "path",
    [
        "/review/delete/network/test-id",
        "/review/duplicates/test-id/keep",
        "/review/duplicates/test-id/discard",
    ],
)
def test_review_actions_reject_missing_csrf_token(client, path):
    """Review POST actions must reject requests without CSRF tokens."""
    response = client.post(path)

    assert response.status_code == 400


@pytest.mark.parametrize(
    "path",
    [
        "/review/delete/network/test-id",
        "/review/duplicates/test-id/keep",
        "/review/duplicates/test-id/discard",
    ],
)
def test_review_actions_reject_invalid_csrf_token(client, path):
    """Review POST actions must reject invalid CSRF tokens."""
    response = client.post(
        path,
        data={"csrf_token": "invalid-token"},
    )

    assert response.status_code == 400


def test_valid_csrf_token_allows_network_submission(client):
    """A valid CSRF token should allow the Network form to submit."""
    from html.parser import HTMLParser

    class CSRFTokenParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.token = None

        def handle_starttag(self, tag, attrs):
            if tag != "input":
                return

            attributes = dict(attrs)
            if attributes.get("name") == "csrf_token":
                self.token = attributes.get("value")

    response = client.get("/")
    assert response.status_code == 200

    parser = CSRFTokenParser()
    parser.feed(response.get_data(as_text=True))

    assert parser.token is not None

    response = client.post(
        "/",
        data={
            "csrf_token": parser.token,
        },
    )

    # Missing configuration fields may cause validation errors, but a valid CSRF token should allow the request through.
    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/data", "/experiment"])
def test_valid_csrf_token_allows_form_submission(client, path):
    """Data and Experiment forms should accept valid CSRF tokens."""
    from html.parser import HTMLParser

    class CSRFTokenParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.token = None

        def handle_starttag(self, tag, attrs):
            if tag == "input":
                attributes = dict(attrs)
                if attributes.get("name") == "csrf_token":
                    self.token = attributes.get("value")

    response = client.get(path)
    assert response.status_code == 200

    parser = CSRFTokenParser()
    parser.feed(response.get_data(as_text=True))

    assert parser.token is not None

    response = client.post(
        path,
        data={"csrf_token": parser.token},
    )

    # The form may report missing required fields, but CSRF validation should succeed.
    assert response.status_code == 200


def test_valid_csrf_token_allows_remove_config(client):
    """A valid CSRF token should allow removing a saved configuration."""
    from html.parser import HTMLParser

    from config_store import get_configs, save_config

    class CSRFTokenParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.token = None

        def handle_starttag(self, tag, attrs):
            if tag == "input":
                attributes = dict(attrs)
                if attributes.get("name") == "csrf_token":
                    self.token = attributes.get("value")

    # Save a configuration in the test client's session.
    with client.session_transaction() as session:
        session["config_session_id"] = "csrf-remove-test"

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = "csrf-remove-test"
        saved = save_config(
            "network",
            "--network_type=fully_connected\n",
        )
        config_id = saved["config"]["id"]

    # Get a valid token from the Review page.
    response = client.get("/review")
    assert response.status_code == 200

    parser = CSRFTokenParser()
    parser.feed(response.get_data(as_text=True))
    assert parser.token is not None

    # Submit the Remove action using the valid token.
    response = client.post(
        f"/review/delete/network/{config_id}",
        data={"csrf_token": parser.token},
    )

    assert response.status_code == 302

    # Verify the saved configuration was actually removed.
    response = client.get("/review")
    assert b"No network configuration has been generated yet." in response.data


def test_valid_csrf_token_allows_keep_duplicate(client):
    """A valid CSRF token should allow keeping a duplicate."""
    from html.parser import HTMLParser
    from uuid import uuid4

    from config_store import get_configs, get_duplicate_attempts, save_config

    class CSRFTokenParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.token = None

        def handle_starttag(self, tag, attrs):
            if tag == "input":
                attributes = dict(attrs)
                if attributes.get("name") == "csrf_token":
                    self.token = attributes.get("value")

    session_id = uuid4().hex

    with client.session_transaction() as session:
        session["config_session_id"] = session_id

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = session_id

        content = "--batch=32\n"
        save_config("data", content)
        save_config("data", content)

        attempts = get_duplicate_attempts()
        assert len(attempts) == 1
        attempt_id = attempts[0]["id"]

    response = client.get("/review")
    assert response.status_code == 200

    parser = CSRFTokenParser()
    parser.feed(response.get_data(as_text=True))
    assert parser.token is not None

    response = client.post(
        f"/review/duplicates/{attempt_id}/keep",
        data={"csrf_token": parser.token},
    )

    assert response.status_code == 302

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = session_id

        configs = get_configs("data")
        assert len(configs) == 2
        assert configs[0]["id"] != configs[1]["id"]
        assert configs[0]["content"] == configs[1]["content"]
        assert get_duplicate_attempts() == []


def test_valid_csrf_token_allows_discard_duplicate(client):
    """A valid CSRF token should allow discarding a duplicate."""
    from html.parser import HTMLParser
    from uuid import uuid4

    from config_store import get_configs, get_duplicate_attempts, save_config

    class CSRFTokenParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.token = None

        def handle_starttag(self, tag, attrs):
            if tag == "input":
                attributes = dict(attrs)
                if attributes.get("name") == "csrf_token":
                    self.token = attributes.get("value")

    session_id = uuid4().hex

    with client.session_transaction() as session:
        session["config_session_id"] = session_id

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = session_id

        content = "--batch=32\n"
        save_config("data", content)
        save_config("data", content)

        attempts = get_duplicate_attempts()
        assert len(attempts) == 1
        attempt_id = attempts[0]["id"]

    response = client.get("/review")
    assert response.status_code == 200

    parser = CSRFTokenParser()
    parser.feed(response.get_data(as_text=True))
    assert parser.token is not None

    response = client.post(
        f"/review/duplicates/{attempt_id}/discard",
        data={"csrf_token": parser.token},
    )

    assert response.status_code == 302

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = session_id

        configs = get_configs("data")
        assert len(configs) == 1
        assert configs[0]["content"] == content
        assert get_duplicate_attempts() == []
