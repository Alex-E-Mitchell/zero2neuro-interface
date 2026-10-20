from config_rules import (
    ACTIVATION_RULES, DROPOUT_RULES, REGULARIZATION_RULES,
    validate_config_values, validate_required_arguments,
)


def validate_hidden_units(units):
    """Return True for a list of positive integers, including zero layers ([])."""
    if not isinstance(units, list):
        return False

    return all(isinstance(unit, int) and not isinstance(unit, bool) and unit > 0 for unit in units)


def validate_shape(shape):
    """Shapes need at least one dimension; every dimension must be a positive int."""
    return validate_hidden_units(shape) and bool(shape)


def parse_integer_list(text, field_name, allow_empty=False):
    """Convert a comma-separated form field without silently dropping empty items."""
    if allow_empty and not text.strip():
        return []
    try:
        return [int(value.strip()) for value in text.split(",")]
    except ValueError:
        raise ValueError(f"{field_name} must contain comma-separated integers.") from None


def generate_network_config(
    network_type,
    input_shape,
    hidden_units,
    hidden_activation,
    output_shape,
    output_activation,
    batch_normalization=False,
    batch_normalization_input=False,
    dropout=None,
    dropout_input=None,
    L1_regularization=None,
    L2_regularization=None,
):
    """Generate a small Zero2Neuro-style network configuration string."""

    validate_required_arguments(network_type, {
        "network_type": network_type,
        "input_shape": input_shape,
        "number_hidden_units": hidden_units,
        "hidden_activation": hidden_activation,
        "output_shape": output_shape,
        "output_activation": output_activation,
    })

    if not validate_shape(input_shape):
        raise ValueError("Input shape must be a non-empty list of positive integers.")

    if network_type == "fully_connected" and len(input_shape) != 1:
        raise ValueError(
            "Fully Connected input shape must contain exactly one positive integer, "
            "such as 34 (the number of input features)."
        )

    if not validate_hidden_units(hidden_units):
        raise ValueError("Hidden layer sizes must be positive integers.")

    if not validate_shape(output_shape):
        raise ValueError("Output shape must be a non-empty list of positive integers.")

    dropout_rates = {"dropout": dropout, "dropout_input": dropout_input}
    regularization = {"L1_regularization": L1_regularization, "L2_regularization": L2_regularization}
    errors = validate_config_values(dropout_rates, DROPOUT_RULES)
    errors.extend(validate_config_values(regularization, REGULARIZATION_RULES))
    errors.extend(validate_config_values({
        "hidden_activation": hidden_activation, "output_activation": output_activation,
    }, ACTIVATION_RULES))
    if network_type == "fully_connected" and L1_regularization is not None and L2_regularization is not None:
        errors.append(
            "L1 and L2 regularization cannot be supplied together with the installed "
            "Zero2Neuro/Keras version. Leave one field blank; zero still counts as supplied."
        )
    if errors:
        raise ValueError(" ".join(errors))

    # A bare hidden-units option parses as [] with nargs='*'; omitting it gives None.
    lines = [f"--network_type={network_type}", "--input_shape"]
    lines.extend(str(dimension) for dimension in input_shape)
    lines.append("--number_hidden_units")
    lines.extend(str(unit) for unit in hidden_units)
    lines.append(f"--hidden_activation={hidden_activation}")

    if batch_normalization:
        lines.append("--batch_normalization")

    if batch_normalization_input:
        lines.append("--batch_normalization_input")

    for name, value in {**dropout_rates, **regularization}.items():
        if value is not None:
            lines.append(f"--{name}={float(value)}")

    lines.append("--output_shape")
    lines.extend(str(dimension) for dimension in output_shape)
    lines.append(f"--output_activation={output_activation}")

    return "\n".join(lines) + "\n"
