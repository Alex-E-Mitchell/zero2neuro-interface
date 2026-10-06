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
