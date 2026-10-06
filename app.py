import sys
from pathlib import Path

from flask import Flask, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from config_generator import generate_network_config, parse_integer_list

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def network():
    config = None
    error = None

    if request.method == "POST":
        try:
            input_shape = parse_integer_list(request.form["input_shape"], "Input shape")
            hidden_units = parse_integer_list(
                request.form.get("hidden_units", ""), "Hidden layer sizes", allow_empty=True
            )
            hidden_activation = request.form["hidden_activation"]
            output_shape = parse_integer_list(request.form["output_shape"], "Output shape")
            output_activation = request.form["output_activation"]

            config = generate_network_config(
                network_type="fully_connected",
                input_shape=input_shape,
                hidden_units=hidden_units,
                hidden_activation=hidden_activation,
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
