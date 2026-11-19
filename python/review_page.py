
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from config_store import get_all_configs, get_config
from flask import abort, render_template, Response, send_file


def get_available_configs(configurations):
    """Return only non-empty generated configurations."""
    return {
        name: content
        for name, content in configurations.items()
        if content and content.strip()
    }


def review():
    """Display saved configurations and allow partial exports."""
    configurations = get_all_configs()
    available = get_available_configs(configurations)

    missing = [
        name
        for name in configurations
        if name not in available
    ]

    return render_template(
        "review.html",
        configurations=configurations,
        missing_configs=missing,
        ready_for_export=bool(available),
    )


def download_config(name):
    """Download an individual generated configuration."""
    if name not in ("network", "data", "experiment"):
        abort(404)

    config = get_config(name)

    if not config or not config.strip():
        abort(
            404,
            description="Configuration has not been generated.",
        )

    return Response(
        config,
        mimetype="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{name}.txt"',
            "Cache-Control": "no-store",
        },
    )


def download_all_configs():
    """Download all available configurations as a ZIP archive."""
    configurations = get_all_configs()
    available = get_available_configs(configurations)

    if not available:
        abort(
            400,
            description="No configurations available for export.",
        )

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in available.items():
            archive.writestr(f"{name}.txt", content)

    zip_buffer.seek(0)

    response = send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="zero2neuro_config.zip",
    )

    response.headers["Cache-Control"] = "no-store"
    return response


def register_review_page(app):
    """Register the Review page routes."""
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
        "/review/download/<name>",
        endpoint="download_config",
        view_func=download_config,
        methods=["GET"],
    )