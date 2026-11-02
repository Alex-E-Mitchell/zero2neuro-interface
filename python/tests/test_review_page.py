
"""Tests for Review page and configuration exports."""

from io import BytesIO
from zipfile import ZipFile

import pytest

from app import app
from config_store import save_config
from config_store import get_config


@pytest.fixture
def client():
    app.config.update(TESTING=True)

    with app.test_client() as client:
        yield client


def test_review_page_loads(client):
    """The Review page should load successfully."""
    response = client.get("/review")

    assert response.status_code == 200
    assert b"Review and Export" in response.data


def test_incomplete_zip_export(client):
    """Incomplete configurations cannot be exported together."""
    response = client.get("/review/download-all")

    assert response.status_code == 400


def test_complete_zip_export(client):
    """A complete export contains all three configuration files."""
    with client.session_transaction() as session:
        session["config_session_id"] = "review-test-complete"

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = "review-test-complete"
        save_config("network", "--network_type=fully_connected\n")
        save_config("data", "--data_format=csv\n")
        save_config("experiment", "--experiment_name=test\n")

    response = client.get("/review/download-all")

    assert response.status_code == 200
    assert response.mimetype == "application/zip"

    with ZipFile(BytesIO(response.data)) as archive:
        assert set(archive.namelist()) == {
            "network.txt",
            "data.txt",
            "experiment.txt",
        }

        assert archive.read("network.txt").decode() == (
            "--network_type=fully_connected\n"
        )
        assert archive.read("data.txt").decode() == (
            "--data_format=csv\n"
        )
        assert archive.read("experiment.txt").decode() == (
            "--experiment_name=test\n"
        )


def test_invalid_data_submission_clears_saved_config(client):
    """An invalid Data submission removes the previous configuration."""

    valid_fields = {
        "data_format": "tabular",
        "data_directory": "",
        "data_files": "xor_data.csv",
        "data_inputs": "In 0\nIn 1",
        "data_outputs": "Out 0",
        "data_fold_split": "identity",
        "data_set_type": "fixed",
    }

    # First, generate a valid Data configuration.
    response = client.post("/data", data=valid_fields)

    assert response.status_code == 200

    review_response = client.get("/review")
    assert b"--data_format=tabular" in review_response.data

    # Submit invalid Data settings directly to Flask.
    invalid_fields = dict(valid_fields, data_outputs="-bad")

    response = client.post("/data", data=invalid_fields)

    assert response.status_code == 200
    assert b'class="error-message"' in response.data

    # The previously saved configuration should be gone.
    review_response = client.get("/review")

    assert b"No data configuration has been generated yet." in (
        review_response.data
    )
    assert b"--data_format=tabular" not in review_response.data


def test_invalid_data_preserves_network_config(client):
    """Invalid Data submissions must not remove Network settings."""
    from flask import session
    from config_store import save_config

    # Save a Network configuration for this browser session.
    with client.session_transaction() as test_session:
        test_session["config_session_id"] = "review-test-independent"

    with app.test_request_context("/"):
        session["config_session_id"] = "review-test-independent"
        save_config("network", "--network_type=fully_connected\n")

    # Submit invalid Data settings.
    response = client.post("/data", data={
        "data_format": "tabular",
        "data_files": "xor_data.csv",
        "data_inputs": "In 0\nIn 1",
        "data_outputs": "-bad",
        "data_fold_split": "identity",
        "data_set_type": "fixed",
    })

    assert response.status_code == 200
    assert b'class="error-message"' in response.data

    # Network remains saved, while Data is unavailable.
    review_response = client.get("/review")

    assert b"--network_type=fully_connected" in review_response.data
    assert b"No data configuration has been generated yet." in (
        review_response.data
    )
