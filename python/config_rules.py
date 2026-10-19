"""Manually maintained architecture relationships, separate from parser metadata.

Inspected E:/zero2neuro/src/{parser,network_builder}.py at commit
55638d7eab68e8dc1a173af15f567ba607a383d2, and the local
keras3_tools/src/cnn_tools.py (create_cnn_network/create_cnn_stack).

Names here are config option names without '--', not always argparse destinations:
input_shape/output_shape alias input_shape0/output_shape0 in the inspected parser.
This is a deliberately small registry for newly built networks, not a complete
catalog of training, tokenizer, plugin, or model-loading options.

Required shapes are interface policy: require explicit sizes rather than silently
use parser defaults [10]. Activations have defaults and hidden layers may be empty.
CNN requirements are provisional: the builder passes None defaults through to a
stack that calls len/zip on filters, kernels, and pooling. Pooling entries can be 0
to disable pooling. Length relationships and conditional dimensions need mentor
review before adding a CNN UI. No CNN semantic validation is implemented here.
"""
import math

_COMMON_REQUIRED = {"network_type", "input_shape", "output_shape"}
_COMMON_OPTIONAL = {
    "number_hidden_units", "hidden_activation", "output_activation",
    "batch_normalization", "batch_normalization_input", "dropout", "dropout_input",
    "L1_regularization", "L2_regularization",
}

NETWORK_ARCHITECTURES = {
    "fully_connected": {
        "required": _COMMON_REQUIRED.copy(),
        "optional": _COMMON_OPTIONAL.copy(),
    },
    "cnn": {
        "required": _COMMON_REQUIRED | {
            "conv_kernel_size", "conv_number_filters", "conv_pool_size",
        },
        "optional": _COMMON_OPTIONAL | {
            "conv_pool_average_size", "conv_padding", "conv_activation",
            "conv_batch_normalization", "conv_strides", "spatial_dropout",
        },
    },
}

def validate_required_arguments(network_type, arguments):
    """Check presence through registry data; value semantics are checked elsewhere.

    Empty lists count as supplied so this helper does not impose a layer count.
    """
    if network_type not in NETWORK_ARCHITECTURES:
        raise ValueError(f"Unknown network architecture: {network_type}.")
    missing = [
        name for name in sorted(NETWORK_ARCHITECTURES[network_type]["required"])
        if name not in arguments or arguments[name] is None
    ]
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(missing) + ".")

# Documented string options from the Zero2Neuro parser.
# These are suggested UI choices, not necessarily exhaustive.
STRING_CHOICES = {
    "data_format": (
        "tabular",
        "tabular-indirect",
        "netcdf",
        "pickle",
        "tf-dataset",
    ),
    "data_representation": (
        "numpy",
        "tf-dataset",
    ),
    "data_fold_split": (
        "identity",
        "group-by-file",
        "group-by-example",
        "random",
        "random-stratify",
    ),
}

def get_string_choices(argument_name):
    """Return documented choices for an argument, if available."""
    return STRING_CHOICES.get(argument_name, ())

def validate_config_values(arguments, rules):
    """Validate values using reusable configuration rules."""
    errors = []

    for name in rules.get("required", ()):
        value = arguments.get(name)

        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            errors.append(f"{name} is required.")

    for name, minimum in rules.get("minimum", {}).items():
        value = arguments.get(name)

        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number.")
        elif not math.isfinite(value):
            errors.append(f"{name} must be a finite number.")
        elif value < minimum:
            errors.append(f"{name} must be at least {minimum}.")

    for name, allowed_values in rules.get("choices", {}).items():
        value = arguments.get(name)

        if value is None:
            continue

        if value not in allowed_values:
            errors.append(
                f"{name} must be one of: {', '.join(allowed_values)}."
            )

    return errors
