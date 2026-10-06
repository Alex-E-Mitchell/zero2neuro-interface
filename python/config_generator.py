def validate_hidden_units(units):
    """Return True when units is a non-empty list of positive integers."""
    if not isinstance(units, list) or len(units) == 0:
        return False

    return all(isinstance(unit, int) and not isinstance(unit, bool) and unit > 0 for unit in units)


def generate_network_config(
    network_type,
    input_shape,
    hidden_units,
    hidden_activation,
    batch_normalization,
    batch_normalization_input,
    output_shape,
    output_activation,
):
    """Generate a small Zero2Neuro-style network configuration string."""

    if not isinstance(input_shape, int) or isinstance(input_shape, bool) or input_shape <= 0:
        raise ValueError("Input shape must be a positive integer.")

    if not validate_hidden_units(hidden_units):
        raise ValueError("Hidden layer sizes must be positive integers.")

    if not isinstance(output_shape, int) or isinstance(output_shape, bool) or output_shape <= 0:
        raise ValueError("Output shape must be a positive integer.")

    hidden_units_text = "\n".join(str(unit) for unit in hidden_units)

    batch_normalization_text = ""
    batch_normalization_input_text = ""

    if batch_normalization:
        batch_normalization_text = "--batch_normalization\n"

    if batch_normalization_input:
        batch_normalization_input_text = "--batch_normalization_input\n"

    return (
        f"--network_type={network_type}\n"
        f"--input_shape\n"
        f"{input_shape}\n"
        f"--number_hidden_units\n"
        f"{hidden_units_text}\n"
        f"--hidden_activation={hidden_activation}\n"
        f"{batch_normalization_text}"
        f"{batch_normalization_input_text}"
        f"--output_shape\n"
        f"{output_shape}\n"
        f"--output_activation={output_activation}\n"
    )
