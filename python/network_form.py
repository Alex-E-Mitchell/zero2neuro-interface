"""Values and feedback for the current Network web form.

These choices describe this form's supported options, not every Zero2Neuro
option. Keep generation and underlying network rules in their existing modules.
"""

from config_generator import parse_integer_list, validate_hidden_units, validate_shape


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
        "choices": (("elu", "ELU"), ("relu", "ReLU"),
                    ("sigmoid", "Sigmoid"), ("tanh", "Tanh")),
    },
    "output_activation": {
        "label": "Output Activation",
        "default": "elup1",
        "choices": (("elup1", "ELU+1"), ("linear", "Linear"),
                    ("sigmoid", "Sigmoid"), ("softmax", "Softmax")),
    },
}

INTEGER_FIELDS = {
    "input_shape": "Input Shape",
    "hidden_units": "Hidden Layer Sizes",
    "output_shape": "Output Shape",
}
CHECKBOX_FIELDS = ("batch_normalization", "batch_normalization_input")


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
        default = field["default"] if name == "network_type" else ""
        values[name] = submitted.get(name, default)
        if values[name] not in dict(field["choices"]):
            errors[name] = f"Choose a supported {field['label'].lower()} from the list."
        else:
            parsed[name] = values[name]

    for name in CHECKBOX_FIELDS:
        values[name] = name in submitted
        parsed[name] = values[name]

    return values, parsed, errors
