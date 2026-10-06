import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from config_generator import generate_network_config

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def network():
    config = None
    error = None

    if request.method == "POST":
        try:
            input_shape = int(request.form["input_shape"])
            hidden_units = [
                int(value.strip())
                for value in request.form["hidden_units"].split(",")
            ]
            hidden_activation = request.form["hidden_activation"]
            output_shape = int(request.form["output_shape"])
            output_activation = request.form["output_activation"]
            batch_normalization = "batch_normalization" in request.form
            batch_normalization_input = "batch_normalization_input" in request.form

            config = generate_network_config(
                network_type="fully_connected",
                input_shape=input_shape,
                hidden_units=hidden_units,
                hidden_activation=hidden_activation,
                batch_normalization=batch_normalization,
                batch_normalization_input=batch_normalization_input,
                output_shape=output_shape,
                output_activation=output_activation,
            )

        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "network.html",
        config=config,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)