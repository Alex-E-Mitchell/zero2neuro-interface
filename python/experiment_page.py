"""Experiment form handling using the shared layout and validation feedback."""

from flask import render_template, request
from config_store import save_config, clear_config
from parser_validator import validate_config

from experiment_generator import (
    DEFAULT_VALUES, LOSS_CHOICES, METRIC_CHOICES, MONITOR_CHOICES,
    generate_experiment_config, validate_experiment_arguments,
)


def prepare_experiment_form(submitted=None):
    values = dict(DEFAULT_VALUES, metrics=[])
    arguments, errors = {}, {}
    if submitted is None:
        return values, arguments, errors

    for name in DEFAULT_VALUES:
        if name != "early_stopping":
            values[name] = submitted.get(name, "")
            if len(submitted.getlist(name)) > 1:
                errors[name] = "Submit only one value for this setting."
    values["early_stopping"] = "early_stopping" in submitted
    values["metrics"] = submitted.getlist("metrics")
    if len(submitted.getlist("early_stopping")) > 1:
        errors["early_stopping"] = "Submit only one early stopping option."
    arguments.update(values)
    arguments["experiment_name"] = values["experiment_name"].strip()

    numeric_fields = {
        "learning_rate": (float, "Learning rate"),
        "epochs": (int, "Epochs"),
    }
    if values["early_stopping"]:
        numeric_fields.update({
            "early_stopping_patience": (int, "Patience"),
            "early_stopping_min_delta": (float, "Minimum improvement"),
        })
    for field, (convert, label) in numeric_fields.items():
        try:
            arguments[field] = convert(values[field].strip())
            if convert is int:
                # The shared finite-number check converts integers to floats.
                # Reject values too large for that conversion as field errors.
                float(arguments[field])
        except (ValueError, OverflowError):
            arguments[field] = None
            expected = "whole number" if convert is int else "number"
            errors[field] = f"{label}: enter a valid {expected}."

    for field, message in validate_experiment_arguments(arguments).items():
        errors.setdefault(field, message)
    return values, arguments, errors


def experiment():
    if request.method == "POST":
        clear_config("experiment")

    values, arguments, errors = prepare_experiment_form(
        request.form if request.method == "POST" else None
    )

    config = None
    error = None
    if request.method == "POST" and not errors:
        config = generate_experiment_config(arguments)
        validation = validate_config(config)
        if validation["status"] == "invalid":
            error = validation["message"]
            config = None
        else:
            # Save if valid, or if parser validation is unavailable.
            save_config("experiment", config)

    return render_template(
        "experiment.html", values=values, errors=errors, error=error, config=config,
        loss_choices=LOSS_CHOICES, metric_choices=METRIC_CHOICES,
        monitor_choices=MONITOR_CHOICES,
    )


def register_experiment_page(app):
    app.add_url_rule(
        "/experiment", endpoint="experiment", view_func=experiment,
        methods=["GET", "POST"],
    )
