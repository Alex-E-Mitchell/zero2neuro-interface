import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from config_generator import generate_network_config
from network_form import SELECT_FIELDS, prepare_network_form

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def network():
    config = None
    error = None
    values, arguments, errors = prepare_network_form(
        request.form if request.method == "POST" else None
    )

    if request.method == "POST" and not errors:
        try:
            config = generate_network_config(**arguments)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "network.html",
        config=config,
        error=error,
        errors=errors,
        values=values,
        select_fields=SELECT_FIELDS,
    )


@app.route("/data")
def data():
    return render_template(
        "in_progress.html",
        page_name="Data",
        planned_features="Dataset format, file paths, columns, and splitting settings will be available here.",
    )


@app.route("/experiment")
def experiment():
    return render_template(
        "in_progress.html",
        page_name="Experiment",
        planned_features="Experiment name, loss, metrics, learning rate, epochs, and early stopping settings will be available here.",
    )


if __name__ == "__main__":
    app.run(debug=True)
