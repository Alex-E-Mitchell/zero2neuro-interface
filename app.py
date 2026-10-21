import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from config_generator import generate_network_config
from config_rules import (
    NETWORK_ARCHITECTURES,
    DROPOUT_RULES,
    REGULARIZATION_RULES,
    validate_config_values,
)
from network_form import SELECT_FIELDS, prepare_network_form
from data_page import register_data_page

app = Flask(__name__)
register_data_page(app)


@app.route("/", methods=["GET", "POST"])
def network():
    config = None
    error = None

    values, arguments, errors = prepare_network_form(
        request.form if request.method == "POST" else None
    )
    selected_network_type = values["network_type"]

    if (
        request.method == "POST"
        and selected_network_type == "cnn"
        and len(request.form.getlist("network_type")) == 1
    ):
        errors.clear()
        error = (
            "CNN configuration is under development. "
            "No configuration was generated. "
            "Choose Fully Connected to generate a configuration now."
        )
    elif request.method == "POST":
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
                optional_values[name] = float(raw)
            except ValueError:
                errors[name] = "Enter a valid number."

        for rules in (DROPOUT_RULES, REGULARIZATION_RULES):
            for message in validate_config_values(optional_values, rules):
                field_name = message.split(" ", 1)[0]
                errors[field_name] = message

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