"""Values and feedback for the current Network web form.

These choices describe this form's supported options, not every Zero2Neuro
option. Keep generation and underlying network rules in their existing modules.
"""

from config_generator import parse_integer_list, validate_hidden_units, validate_shape
from config_rules import ACTIVATION_CHOICES, CNN_LIST_MINIMUMS, CNN_CHOICES, validate_cnn_arguments

SELECT_FIELDS = {
    "network_type": {
        "label": "Network Type",
        "default": "fully_connected",
        "choices": (
    ("fully_connected", "Fully Connected"),
    ("cnn", "CNN"),
        ),
    },
    "hidden_activation": {
        "label": "Hidden Activation",
        "default": "elu",
        "choices": tuple(
            (value, value.upper() if value != "tanh" else "Tanh")
            for value in ACTIVATION_CHOICES["hidden_activation"]
        ),
    },
    "output_activation": {
        "label": "Output Activation",
        "default": "elup1",
        "choices": tuple(
            (value, "ELU+1" if value == "elup1" else value.capitalize())
            for value in ACTIVATION_CHOICES["output_activation"]
        ),
    },
}

INTEGER_FIELDS = {
    "input_shape": "Input Shape",
    "hidden_units": "Hidden Layer Sizes",
    "output_shape": "Output Shape",
}
CHECKBOX_FIELDS = ("batch_normalization", "batch_normalization_input")

CNN_LABELS = {
    "conv_number_filters": "Filters per Layer",
    "conv_kernel_size": "Kernel Sizes per Layer",
    "conv_pool_size": "Max-Pooling Sizes per Layer",
}
for name, choices in CNN_CHOICES.items():
    SELECT_FIELDS[name] = {
        "label": "Convolution Padding" if name == "conv_padding" else "Convolution Activation",
        "default": choices[0],
        "choices": tuple((value, {
            "valid": "Valid (no added border)", "same": "Same (add border zeros)",
            "elu": "ELU", "relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh",
        }[value]) for value in choices),
    }


def prepare_network_form(submitted=None):
    """Return display values, parsed generator arguments, and field errors.

    Keep raw text separate from parsed integers so correcting one field never
    discards the user's other entries. A missing hidden-layer field means zero
    layers, and a missing network type retains the existing Fully Connected
    default for older clients.
    """
    values = {name: "" for name in INTEGER_FIELDS}
    values.update({name: field["default"] for name, field in SELECT_FIELDS.items()})
    values.update({name: False for name in CHECKBOX_FIELDS})
    values.update({name: "" for name in CNN_LIST_MINIMUMS})
    parsed, errors = {}, {}
    if submitted is None:
        return values, parsed, errors

    for name, label in INTEGER_FIELDS.items():
        values[name] = submitted.get(name, "")
        optional = name == "hidden_units"
        if not optional and not values[name].strip():
            errors[name] = f"Enter {label.lower()} using whole numbers greater than zero (for example, 34 or 128, 128, 3)."
            continue
        try:
            numbers = parse_integer_list(values[name], label, allow_empty=optional)
            valid = validate_hidden_units(numbers) if optional else validate_shape(numbers)
            if not valid:
                raise ValueError
            parsed[name] = numbers
        except ValueError:
            errors[name] = f"{label}: use whole numbers greater than zero, separated by commas (for example, 64, 32)."
            if optional:
                errors[name] += " Leave blank for zero hidden layers."

    for name, field in SELECT_FIELDS.items():
        if name in CNN_CHOICES and submitted.get("network_type") != "cnn":
            values[name] = submitted.get(name, field["default"])
            continue
        default = field["default"] if name == "network_type" else ""
        if name in CNN_CHOICES:
            default = field["default"]
        values[name] = submitted.get(name, default)
        if values[name] not in dict(field["choices"]):
            errors[name] = f"Choose a supported {field['label'].lower()} from the list."
        else:
            parsed[name] = values[name]

    for name in CHECKBOX_FIELDS:
        values[name] = name in submitted
        parsed[name] = values[name]

    for name in CNN_LIST_MINIMUMS:
        values[name] = submitted.get(name, "")
        if values["network_type"] == "cnn":
            try:
                parsed[name] = parse_integer_list(values[name], CNN_LABELS[name])
            except ValueError as exc:
                errors[name] = str(exc)
    if values["network_type"] == "cnn":
        for name, message in validate_cnn_arguments(parsed.get("input_shape"), parsed).items():
            errors.setdefault(name, message)

    return values, parsed, errors
