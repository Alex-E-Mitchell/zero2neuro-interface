from config_rules import validate_required_arguments


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

    if not validate_hidden_units(hidden_units):
        raise ValueError("Hidden layer sizes must be positive integers.")

    if not validate_shape(output_shape):
        raise ValueError("Output shape must be a non-empty list of positive integers.")

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

    lines.append("--output_shape")
    lines.extend(str(dimension) for dimension in output_shape)
    lines.append(f"--output_activation={output_activation}")

    return "\n".join(lines) + "\n"
