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
CNN rules follow upstream release v0.11.4 and keras3_tools v0.2.1.
See docs/CNN_CONFIGURATION.md for source revisions, interface policies, and gaps.
"""
import math

# Zero2Neuro --dropout and --dropout_input are optional floats (default None), passed to Keras
# Dropout. The TensorFlow backend requires 0 <= rate < 1 during training.
DROPOUT_RULES = {
    "minimum": {"dropout": 0.0, "dropout_input": 0.0},
    "exclusive_maximum": {"dropout": 1.0, "dropout_input": 1.0},
}

ACTIVATION_CHOICES = {
    "hidden_activation": ("elu", "relu", "sigmoid", "tanh"),
    "output_activation": ("elup1", "linear", "sigmoid", "softmax"),
}
ACTIVATION_RULES = {"required": set(ACTIVATION_CHOICES), "choices": ACTIVATION_CHOICES}

CNN_LIST_MINIMUMS = {
    "conv_number_filters": 1, "conv_kernel_size": 1, "conv_pool_size": 0,
}
CNN_CHOICES = {
    "conv_padding": ("valid", "same"),
    "conv_activation": ("elu", "relu", "sigmoid", "tanh"),
}


def validate_cnn_arguments(input_shape, arguments):
    """Return field errors for the supported channels-last CNN subset.

    Equal nonempty lists are interface policy: upstream zip silently truncates.
    Stride is 1, average pooling is omitted, and max pooling uses its size as stride.
    """
    errors = {}
    if (not isinstance(input_shape, list) or len(input_shape) not in (2, 3, 4)
            or any(type(d) is not int or d <= 0 for d in input_shape)):
        errors["input_shape"] = (
            "CNN input shape must contain 2, 3, or 4 positive integers: "
            "spatial dimensions followed by channels, without batch size."
        )
    for name, minimum in CNN_LIST_MINIMUMS.items():
        values = arguments.get(name)
        if (not isinstance(values, list) or not values
                or any(type(v) is not int or v < minimum for v in values)):
            errors[name] = f"{name} must be a non-empty list of integers at least {minimum}."
    for name, choices in CNN_CHOICES.items():
        if arguments.get(name) not in choices:
            errors[name] = f"{name} must be one of: {', '.join(choices)}."
    if any(name in errors for name in CNN_LIST_MINIMUMS):
        return errors
    if len({len(arguments[name]) for name in CNN_LIST_MINIMUMS}) != 1:
        for name in CNN_LIST_MINIMUMS:
            errors[name] = "Filters, kernel sizes, and max-pooling sizes must have the same number of layers."
    if errors:
        return errors

    spatial = input_shape[:-1]
    for layer, (kernel, pool) in enumerate(zip(
            arguments["conv_kernel_size"], arguments["conv_pool_size"]), start=1):
        if arguments["conv_padding"] == "valid":
            spatial = [d - kernel + 1 for d in spatial]
            if any(d <= 0 for d in spatial):
                errors["conv_kernel_size"] = f"Convolution layer {layer} leaves no spatial cells; reduce its kernel size or use same padding."
                break
        if pool > 1:
            spatial = ([(d + pool - 1) // pool for d in spatial]
                       if arguments["conv_padding"] == "same"
                       else [d // pool for d in spatial])
            if any(d <= 0 for d in spatial):
                errors["conv_pool_size"] = f"Max pooling at layer {layer} leaves no spatial cells; reduce its size or use same padding."
                break
    return errors

# Zero2Neuro's optional float coefficients default to None; Keras requires
# finite, non-negative values and does not impose an upper bound.
REGULARIZATION_RULES = {
    "minimum": {"L1_regularization": 0.0, "L2_regularization": 0.0},
}

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

    for bound in ("minimum", "maximum", "exclusive_maximum"):
        for name, limit in rules.get(bound, {}).items():
            value = arguments.get(name)

            if value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                message = f"{name} must be a number."
            elif not math.isfinite(value):
                message = f"{name} must be a finite number."
            elif bound == "minimum" and value < limit:
                message = f"{name} must be at least {limit}."
            elif bound == "maximum" and value > limit:
                message = f"{name} must be at most {limit}."
            elif bound == "exclusive_maximum" and value >= limit:
                message = f"{name} must be less than {limit}."
            else:
                continue

            if message not in errors:
                errors.append(message)

    for name, allowed_values in rules.get("choices", {}).items():
        value = arguments.get(name)

        if value is None:
            continue

        if value not in allowed_values:
            errors.append(
                f"{name} must be one of: {', '.join(allowed_values)}."
            )

    return errors
