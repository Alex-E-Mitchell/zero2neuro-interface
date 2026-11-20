"""Review saved configurations and export them."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from flask import abort, redirect, render_template, Response, send_file, url_for
from config_store import (
    delete_config,
    discard_duplicate,
    get_all_configs,
    get_config,
    get_duplicate_attempts,
    keep_duplicate,
)


def review():
    """Display saved configurations and pending duplicate notifications."""
    configurations = get_all_configs()
    duplicate_attempts = get_duplicate_attempts()

    total_configs = sum(
        len(configs) for configs in configurations.values()
    )

    return render_template(
        "review.html",
        configurations=configurations,
        duplicate_attempts=duplicate_attempts,
        ready_for_export=total_configs > 0,
        total_configs=total_configs,
    )


def download_config(name, config_id):
    """Download one saved configuration."""
    if name not in ("network", "data", "experiment"):
        abort(404)

    content = get_config(name, config_id)

    if content is None:
        abort(404, description="Configuration not found.")

    return Response(
        content,
        mimetype="text/plain",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{name}_{config_id}.txt"'
            ),
            "Cache-Control": "no-store",
        },
    )


def download_all_configs():
    """Export every saved configuration in one ZIP."""
    configurations = get_all_configs()

    if not any(configurations.values()):
        abort(400, description="No configurations have been generated.")

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as archive:
        for name, configs in configurations.items():
            for index, config in enumerate(configs, start=1):
                filename = f"{name}_{index}.txt"
                archive.writestr(filename, config["content"])

    zip_buffer.seek(0)

    response = send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="zero2neuro_config.zip",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def remove_config(name, config_id):
    """Delete one saved configuration."""
    if name not in ("network", "data", "experiment"):
        abort(404)

    if not delete_config(name, config_id):
        abort(404, description="Configuration not found.")

    return redirect(url_for("review"))


def keep_duplicate_config(attempt_id):
    """Keep a duplicate as an additional saved configuration."""
    result = keep_duplicate(attempt_id)

    if result is None:
        abort(404, description="Duplicate request not found.")

    return redirect(url_for("review"))


def discard_duplicate_config(attempt_id):
    """Discard a pending duplicate without removing the original."""
    if not discard_duplicate(attempt_id):
        abort(404, description="Duplicate request not found.")

    return redirect(url_for("review"))


def register_review_page(app):
    app.add_url_rule(
        "/review",
        endpoint="review",
        view_func=review,
        methods=["GET"],
    )

    app.add_url_rule(
        "/review/download-all",
        endpoint="download_all_configs",
        view_func=download_all_configs,
        methods=["GET"],
    )

    app.add_url_rule(
        "/review/download/<name>/<config_id>",
        endpoint="download_config",
        view_func=download_config,
        methods=["GET"],
    )

    app.add_url_rule(
        "/review/delete/<name>/<config_id>",
        endpoint="remove_config",
        view_func=remove_config,
        methods=["POST"],
    )

    app.add_url_rule(
        "/review/duplicates/<attempt_id>/keep",
        endpoint="keep_duplicate_config",
        view_func=keep_duplicate_config,
        methods=["POST"],
    )

    app.add_url_rule(
        "/review/duplicates/<attempt_id>/discard",
        endpoint="discard_duplicate_config",
        view_func=discard_duplicate_config,
        methods=["POST"],
    )
