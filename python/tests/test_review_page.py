"""Tests for Review page and configuration exports."""

from io import BytesIO
from zipfile import ZipFile

import pytest

from app import app
from config_store import save_config
from config_store import get_config


@pytest.fixture
def client():
    original_testing = app.config["TESTING"]
    original_csrf = app.config["WTF_CSRF_ENABLED"]

    app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
    )

    try:
        with app.test_client() as client:
            yield client
    finally:
        app.config.update(
            TESTING=original_testing,
            WTF_CSRF_ENABLED=original_csrf,
        )


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
            "network_1.txt",
            "data_1.txt",
            "experiment_1.txt",
        }

        assert archive.read("network_1.txt").decode() == (
            "--network_type=fully_connected\n"
        )
        assert archive.read("data_1.txt").decode() == (
            "--data_format=csv\n"
        )
        assert archive.read("experiment_1.txt").decode() == (
            "--experiment_name=test\n"
        )


def test_invalid_data_submission_preserves_saved_config(client):
    """An invalid Data submission preserves the previous configuration."""

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

    # The previously saved configuration should remain available.
    review_response = client.get("/review")

    assert review_response.status_code == 200
    assert b"--data_format=tabular" in review_response.data
    assert b"No data configuration has been generated yet." not in (
        review_response.data
    )


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


def test_experiment_config_saved_to_review(client):
    """Generated Experiment configurations appear in Review and can be downloaded."""

    valid_fields = {
        "experiment_name": "xor-demo",
        "loss": "mse",
        "learning_rate": "0.001",
        "epochs": "5000",
        "early_stopping_monitor": "loss",
        "early_stopping_patience": "2000",
        "early_stopping_min_delta": "0.01",
    }

    # Generate a valid Experiment configuration.
    response = client.post("/experiment", data=valid_fields)

    assert response.status_code == 200
    assert b"--experiment_name=xor-demo" in response.data

    # Verify that Review displays the saved configuration.
    review_response = client.get("/review")

    assert review_response.status_code == 200
    assert b"--experiment_name=xor-demo" in review_response.data

    # Retrieve the saved Experiment configuration and its unique ID.
    with client.session_transaction() as test_session:
        session_id = test_session["config_session_id"]

    from flask import session

    with app.test_request_context("/"):
        session["config_session_id"] = session_id
        from config_store import get_configs
        saved_configs = get_configs("experiment")

    assert len(saved_configs) == 1

    config_id = saved_configs[0]["id"]

    # Verify that the configuration can be downloaded by its ID.
    download_response = client.get(
        f"/review/download/experiment/{config_id}"
    )

    assert download_response.status_code == 200
    assert b"--experiment_name=xor-demo" in download_response.data
    assert (
        f"experiment_{config_id}.txt"
        in download_response.headers["Content-Disposition"]
    )


def test_invalid_experiment_submission_preserves_saved_config(client):
    """Invalid Experiment submissions preserve saved configurations."""

    valid_fields = {
        "experiment_name": "xor-demo",
        "loss": "mse",
        "learning_rate": "0.001",
        "epochs": "5000",
    }

    # Generate and save a valid Experiment configuration.
    response = client.post("/experiment", data=valid_fields)
    assert response.status_code == 200

    review_response = client.get("/review")
    assert b"--experiment_name=xor-demo" in review_response.data

    # Submit an invalid Experiment configuration.
    invalid_fields = dict(valid_fields, epochs="-1")

    response = client.post("/experiment", data=invalid_fields)

    assert response.status_code == 200
    assert b'id="form-errors"' in response.data

    # Verify the successful Experiment configuration is still saved.
    review_response = client.get("/review")

    assert review_response.status_code == 200
    assert b"--experiment_name=xor-demo" in review_response.data


def test_full_configuration_workflow(client):
    """Generate all three configurations and export them together."""
    from io import BytesIO
    from zipfile import ZipFile

    # Generate Network configuration.
    network_response = client.post("/", data={
        "network_type": "fully_connected",
        "input_shape": "34",
        "hidden_units": "64,32",
        "hidden_activation": "elu",
        "output_shape": "1",
        "output_activation": "linear",
    })

    assert network_response.status_code == 200
    assert b"--network_type=fully_connected" in network_response.data

    # Generate Data configuration.
    data_response = client.post("/data", data={
        "data_format": "tabular",
        "data_directory": "",
        "data_files": "xor_data.csv",
        "data_inputs": "In 0\nIn 1",
        "data_outputs": "Out 0",
        "data_fold_split": "identity",
        "data_set_type": "fixed",
    })

    assert data_response.status_code == 200
    assert b"--data_format=tabular" in data_response.data

    # Generate Experiment configuration.
    experiment_response = client.post("/experiment", data={
        "experiment_name": "xor-demo",
        "loss": "mse",
        "learning_rate": "0.001",
        "epochs": "5000",
    })

    assert experiment_response.status_code == 200
    assert b"--experiment_name=xor-demo" in experiment_response.data

    # Verify all three configurations appear on Review.
    review_response = client.get("/review")

    assert review_response.status_code == 200
    assert b"--network_type=fully_connected" in review_response.data
    assert b"--data_format=tabular" in review_response.data
    assert b"--experiment_name=xor-demo" in review_response.data

    # Download the combined ZIP file.
    zip_response = client.get("/review/download-all")

    assert zip_response.status_code == 200
    assert zip_response.mimetype == "application/zip"

    # Verify ZIP contents.
    with ZipFile(BytesIO(zip_response.data)) as archive:
        assert set(archive.namelist()) == {
            "network_1.txt",
            "data_1.txt",
            "experiment_1.txt",
        }

        assert b"--network_type=fully_connected" in archive.read("network_1.txt")
        assert b"--data_format=tabular" in archive.read("data_1.txt")
        assert b"--experiment_name=xor-demo" in archive.read("experiment_1.txt")


def test_multiple_data_configs_saved_and_exported(client):
    """Multiple successful Data configurations remain saved."""

    from config_store import get_configs

    with client.session_transaction() as test_session:
        test_session["config_session_id"] = "review-test-multiple-data"

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = "review-test-multiple-data"

        save_config("data", "--batch=32\n")
        save_config("data", "--batch=64\n")

        saved_configs = get_configs("data")

    # Both configurations should exist independently.
    assert len(saved_configs) == 2
    assert saved_configs[0]["content"] == "--batch=32\n"
    assert saved_configs[1]["content"] == "--batch=64\n"
    assert saved_configs[0]["id"] != saved_configs[1]["id"]

    # Both should appear on the Review page.
    review_response = client.get("/review")

    assert review_response.status_code == 200
    assert b"--batch=32" in review_response.data
    assert b"--batch=64" in review_response.data

    # Both should be included in the ZIP export.
    zip_response = client.get("/review/download-all")

    assert zip_response.status_code == 200

    with ZipFile(BytesIO(zip_response.data)) as archive:
        assert set(archive.namelist()) == {
            "data_1.txt",
            "data_2.txt",
        }

        assert archive.read("data_1.txt") == b"--batch=32\n"
        assert archive.read("data_2.txt") == b"--batch=64\n"


def test_duplicate_config_requires_confirmation(client):
    """Duplicates are detected without automatically saving another copy."""
    from config_store import get_configs

    with client.session_transaction() as test_session:
        test_session["config_session_id"] = "review-test-duplicates"

    with app.test_request_context("/"):
        from flask import session

        session["config_session_id"] = "review-test-duplicates"

        # Save the first configuration.
        first_result = save_config("data", "--batch=32\n")

        assert first_result["status"] == "saved"

        # Attempt to save the exact same configuration.
        duplicate_result = save_config("data", "--batch=32\n")

        assert duplicate_result["status"] == "duplicate"
        assert duplicate_result["config"]["id"] == (
            first_result["config"]["id"]
        )

        # The duplicate should not be saved automatically.
        assert len(get_configs("data")) == 1

        # Simulate the user choosing "Keep Duplicate".
        kept_result = save_config(
            "data",
            "--batch=32\n",
            allow_duplicate=True,
        )

        assert kept_result["status"] == "saved"

        # Both copies now exist with different IDs.
        saved_configs = get_configs("data")

        assert len(saved_configs) == 2
        assert saved_configs[0]["id"] != saved_configs[1]["id"]
        assert saved_configs[0]["content"] == saved_configs[1]["content"]



def test_keep_duplicate_from_review(client):
    """Keeping a duplicate creates a second saved configuration."""
    from config_store import (
        get_configs,
        get_duplicate_attempts,
        save_config,
    )

    with client.session_transaction() as session:
        session["config_session_id"] = "test-keep-duplicate-review"

    with client.application.test_request_context():
        from flask import session

        session["config_session_id"] = "test-keep-duplicate-review"

        first = save_config("data", "--batch=32\n")
        duplicate = save_config("data", "--batch=32\n")

        assert first["status"] == "saved"
        assert duplicate["status"] == "duplicate"
        assert len(get_configs("data")) == 1

        attempts = get_duplicate_attempts()
        assert len(attempts) == 1
        attempt_id = attempts[0]["id"]

    response = client.post(
        f"/review/duplicates/{attempt_id}/keep",
        follow_redirects=True,
    )

    assert response.status_code == 200

    with client.application.test_request_context():
        from flask import session

        session["config_session_id"] = "test-keep-duplicate-review"

        configs = get_configs("data")
        assert len(configs) == 2
        assert configs[0]["id"] != configs[1]["id"]
        assert configs[0]["content"] == configs[1]["content"]
        assert get_duplicate_attempts() == []



def test_discard_duplicate_from_review(client):
    """Discarding a duplicate preserves the original configuration."""
    from flask import session
    from config_store import (
        get_configs,
        get_duplicate_attempts,
        save_config,
    )
    from uuid import uuid4

    session_id = uuid4().hex

    with client.session_transaction() as test_session:
        test_session["config_session_id"] = session_id

    with client.application.test_request_context():
        session["config_session_id"] = session_id

        first = save_config("data", "--batch=32\n")
        duplicate = save_config("data", "--batch=32\n")

        assert first["status"] == "saved"
        assert duplicate["status"] == "duplicate"

        original_id = first["config"]["id"]

        attempts = get_duplicate_attempts()
        assert len(attempts) == 1
        attempt_id = attempts[0]["id"]

    response = client.post(
        f"/review/duplicates/{attempt_id}/discard",
        follow_redirects=True,
    )

    assert response.status_code == 200

    with client.application.test_request_context():
        session["config_session_id"] = session_id

        configs = get_configs("data")

        # The original configuration must remain saved.
        assert len(configs) == 1
        assert configs[0]["id"] == original_id
        assert configs[0]["content"] == "--batch=32\n"

        # The pending duplicate notification must disappear.
        assert get_duplicate_attempts() == []
