import math
import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from config_generator import generate_network_config
from config_rules import NETWORK_ARCHITECTURES
from network_form import SELECT_FIELDS, prepare_network_form

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def network():
    config = None
    error = None

    values, arguments, errors = prepare_network_form(
        request.form if request.method == "POST" else None
    )
    selected_network_type = values["network_type"]

    if request.method == "POST":
        if (
            len(request.form.getlist("network_type")) > 1
            or selected_network_type not in NETWORK_ARCHITECTURES
        ):
            errors["network_type"] = "Unknown network type. Please choose Fully Connected or CNN."
            
        optional_values = {}

        for name in (
            "dropout",
            "dropout_input",
            "L1_regularization",
            "L2_regularization",
        ):
            raw = request.form.get(name, "").strip()

            if not raw:
                optional_values[name] = None
                continue

            try:
                value = float(raw)
            except ValueError:
                errors[name] = "Enter a valid number."
                continue

            if (
                not math.isfinite(value)
                or value < 0
                or (name in ("dropout", "dropout_input") and value >= 1)
            ):
                if name in ("dropout", "dropout_input"):
                    errors[name] = (
                        "Enter a finite number greater than or equal "
                        "to 0 and less than 1."
                    )
                else:
                    errors[name] = "Enter a finite, non-negative number."
            else:
                optional_values[name] = value

        if not errors:
            if selected_network_type == "cnn":
                error = (
                    "CNN configuration is under development. "
                    "No configuration was generated. "
                    "Choose Fully Connected to generate a configuration now."
                )
            else:
                try:
                    config = generate_network_config(
                        **arguments,
                        **optional_values,
                    )
                except (ValueError, KeyError) as exc:
                    error = str(exc)

    return render_template(
        "network.html",
        config=config,
        error=error,
        errors=errors,
        values=values,
        select_fields=SELECT_FIELDS,
        selected_network_type=selected_network_type,
    )


@app.route("/data")
def data():
    return render_template(
        "in_progress.html",
        page_name="Data",
        planned_features=(
            "Dataset format, file paths, columns, and splitting "
            "settings will be available here."
        ),
    )


@app.route("/experiment")
def experiment():
    return render_template(
        "in_progress.html",
        page_name="Experiment",
        planned_features=(
            "Experiment name, loss, metrics, learning rate, epochs, "
            "and early stopping settings will be available here."
        ),
    )


if __name__ == "__main__":
    app.run(debug=True)
