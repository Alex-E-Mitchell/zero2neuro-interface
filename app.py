import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from config_generator import generate_network_config, parse_integer_list
from config_rules import NETWORK_ARCHITECTURES
from data_page import register_data_page

app = Flask(__name__)
register_data_page(app)


@app.route("/", methods=["GET", "POST"])
def network():
    config = None
    error = None
    selected_network_type = "fully_connected"

    if request.method == "POST":
        try:
            network_type = request.form.get("network_type", "fully_connected")
            if len(request.form.getlist("network_type")) > 1 or network_type not in NETWORK_ARCHITECTURES:
                raise ValueError("Unknown network type. Please choose Fully Connected or CNN.")
            selected_network_type = network_type
            if network_type == "cnn":
                raise ValueError(
                    "CNN configuration is under development. No configuration was generated. "
                    "Choose Fully Connected to generate a configuration now."
                )

            input_shape = parse_integer_list(request.form["input_shape"], "Input shape")
            hidden_units = parse_integer_list(
                request.form.get("hidden_units", ""), "Hidden layer sizes", allow_empty=True
            )
            hidden_activation = request.form["hidden_activation"]
            output_shape = parse_integer_list(request.form["output_shape"], "Output shape")
            output_activation = request.form["output_activation"]
            batch_normalization = "batch_normalization" in request.form
            batch_normalization_input = "batch_normalization_input" in request.form
            optional_values = {}
            for name, label, expected in (
                ("dropout", "Dropout", "a number greater than or equal to 0 and less than 1"),
                ("dropout_input", "Input dropout", "a number greater than or equal to 0 and less than 1"),
                ("L1_regularization", "L1 regularization", "a non-negative number"),
                ("L2_regularization", "L2 regularization", "a non-negative number"),
            ):
                text = request.form.get(name, "").strip()
                try:
                    optional_values[name] = float(text) if text else None
                except ValueError:
                    raise ValueError(f"{label} must be {expected}.") from None

            config = generate_network_config(
                network_type=network_type,
                input_shape=input_shape,
                hidden_units=hidden_units,
                hidden_activation=hidden_activation,
                batch_normalization=batch_normalization,
                batch_normalization_input=batch_normalization_input,
                **optional_values,
                output_shape=output_shape,
                output_activation=output_activation,
            )

        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "network.html",
        config=config,
        error=error,
        selected_network_type=selected_network_type,
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