"""Phase 1 experiment settings for Zero2Neuro v0.11.4.

Argument names/defaults were checked against upstream src/parser.py at
55638d7eab68e8dc1a173af15f567ba607a383d2. Choices below are a supported UI
subset from the XOR, breast-cancer and iris examples, not all Keras options.
"""

import re

from config_rules import validate_config_values


LOSS_CHOICES = {
    "mse": "Mean squared error (numeric predictions)",
    "binary_crossentropy": "Binary cross-entropy (two classes)",
    "sparse_categorical_crossentropy": "Sparse categorical cross-entropy (integer class labels)",
}
METRIC_CHOICES = {
    "mse": "Mean squared error (MSE)",
    "mae": "Mean absolute error (MAE)",
    "binary_accuracy": "Binary accuracy (two classes)",
    "sparse_categorical_accuracy": "Sparse categorical accuracy (integer class labels)",
}
MONITOR_CHOICES = {
    "loss": "Training loss",
    "val_loss": "Validation loss (requires validation data)",
}

DEFAULT_VALUES = {
    "experiment_name": "experiment",
    "loss": "mse",
    "learning_rate": "0.0001",
    "epochs": "100",
    "early_stopping": False,
    "early_stopping_monitor": "loss",
    "early_stopping_patience": "20",
    "early_stopping_min_delta": "0.01",
}
EARLY_STOPPING_FIELDS = (
    "early_stopping_monitor", "early_stopping_patience", "early_stopping_min_delta",
)
FIELD_LABELS = {
    "loss": "Loss", "learning_rate": "Learning rate", "epochs": "Epochs",
    "early_stopping_monitor": "Monitor", "early_stopping_patience": "Patience",
    "early_stopping_min_delta": "Minimum improvement",
}


def validate_experiment_arguments(arguments):
    """Return field-keyed errors for parsed arguments before serialization."""
    errors = {}
    name = arguments.get("experiment_name", "")
    # Interface policy: a simple file-friendly name also prevents config injection.
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        errors["experiment_name"] = "Start the name with a letter or number; use only letters, numbers, underscores and hyphens."

    rules = {
        "required": {"loss", "learning_rate", "epochs"},
        "choices": {"loss": LOSS_CHOICES},
        "minimum": {"learning_rate": 0, "epochs": 1},
    }
    if arguments.get("early_stopping"):
        rules["required"].update(EARLY_STOPPING_FIELDS)
        rules["choices"]["early_stopping_monitor"] = MONITOR_CHOICES
        rules["minimum"].update(early_stopping_patience=0, early_stopping_min_delta=0)

    for message in validate_config_values(arguments, rules):
        field, detail = message.split(" ", 1)
        errors.setdefault(field, f"{FIELD_LABELS[field]} {detail}")

    rate = arguments.get("learning_rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate == 0:
        errors["learning_rate"] = "Learning rate must be greater than zero."

    integer_fields = {"epochs": "Epochs"}
    if arguments.get("early_stopping"):
        integer_fields["early_stopping_patience"] = "Patience"
    for field, label in integer_fields.items():
        if type(arguments.get(field)) is not int:
            errors[field] = f"{label} must be a whole number."

    if type(arguments.get("early_stopping")) is not bool:
        errors["early_stopping"] = "Choose whether to enable early stopping."

    metrics = arguments.get("metrics", [])
    if not isinstance(metrics, list) or any(
        not isinstance(metric, str) or metric not in METRIC_CHOICES for metric in metrics
    ):
        errors["metrics"] = "Choose metrics from the available options."
    return errors


def generate_experiment_config(arguments):
    """Validate and serialize settings; optional lists/flags are omitted when off."""
    errors = validate_experiment_arguments(arguments)
    if errors:
        raise ValueError(" ".join(errors.values()))

    lines = [
        f"--experiment_name={arguments['experiment_name']}",
        f"--loss={arguments['loss']}",
    ]
    metrics = list(dict.fromkeys(arguments.get("metrics", [])))
    if metrics:
        lines.extend(["--metrics", *metrics])
    lines.extend([
        f"--learning_rate={arguments['learning_rate']}",
        f"--epochs={arguments['epochs']}",
    ])
    if arguments["early_stopping"]:
        lines.append("--early_stopping")
        for field in EARLY_STOPPING_FIELDS:
            lines.append(f"--{field}={arguments[field]}")
    return "\n".join(lines) + "\n"
