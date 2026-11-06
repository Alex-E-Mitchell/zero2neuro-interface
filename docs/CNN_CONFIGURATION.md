# CNN configuration generator

The Network page generates CNN configuration text for the five existing convolution
fields: filters, kernel sizes, max-pooling sizes, padding, and activation. It also
supports the existing dense/output settings after upstream's global max pooling.
Fully Connected configuration behavior remains supported.

## Inspected sources

Inspected before implementation on November 5, 2026. GitHub's current default
branch for both repositories was `release`:

- Zero2Neuro v0.11.4, commit `55638d7eab68e8dc1a173af15f567ba607a383d2`:
  [parser.py](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/src/parser.py),
  [network_builder.py](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/src/network_builder.py),
  [CNN documentation](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/docs/modules/networkbuilder/cnn.md),
  [CNN details](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/docs/modules/networkbuilder/cnn_details.md),
  and [argument reference](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/docs/api/args_list.md).
- The builder imports the actual CNN from sibling `keras3_tools`.
  Its current release v0.2.1, commit `c7d63be937a6889fd5b2b4cfe9caa34e0c715f95`:
  [cnn_tools.py](https://github.com/Symbiotic-Computing-Laboratory/keras3_tools/blob/c7d63be937a6889fd5b2b4cfe9caa34e0c715f95/src/cnn_tools.py)
  (`create_cnn_network`, `create_cnn_stack`) and
  [fully_connected_tools.py](https://github.com/Symbiotic-Computing-Laboratory/keras3_tools/blob/c7d63be937a6889fd5b2b4cfe9caa34e0c715f95/src/fully_connected_tools.py)
  (activation translation and regularization).
- Keras official [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)
  and [MaxPooling2D](https://keras.io/api/layers/pooling_layers/max_pooling2d/)
  documentation for padding, channel order, and spatial-size arithmetic.

No upstream files were modified. Implementation is based on these code paths,
with explicit interface policies below, rather than relying on documentation alone.

## Validation and serialization

| Field | Rule |
| --- | --- |
| Input shape | A list of 2, 3, or 4 positive integers, including channels last, excluding batch size. Examples: `100, 2`; `128, 128, 3`; `16, 16, 16, 1`. |
| Filters | Non-empty list of positive integers, one count per convolution layer. |
| Kernels | Non-empty list of positive integers, one per layer. A scalar expands equally over all spatial axes. |
| Max pooling | Non-empty list of non-negative integers, one per layer. Both 0 and 1 skip pooling; values above 1 pool with stride equal to size. |
| Layer-list lengths | Filters, kernels, and max pooling must have equal lengths. This is an additional beginner guardrail; upstream `zip` silently truncates to the shortest list. No broadcasting or truncation is performed by the interface. |
| Padding | Exactly `valid` or `same`, shared by convolution and pooling. |
| Convolution activation | Exactly `elu`, `relu`, `sigmoid`, or `tanh`, matching the existing form choices. This is a supported interface subset, not all upstream/Keras activations. |
| Dense/output settings | Existing positive hidden units (blank means zero layers), positive output dimensions, activation choices, dropout, and non-negative finite regularization rules apply. CNN also rejects simultaneous supplied L1 and L2, including zero, because it uses the same upstream regularizer helper as FC. |

Booleans, fractional numbers, and non-list values are rejected for integer lists by
the generator. Form lists must contain comma-separated whole numbers; missing,
blank, malformed, or trailing/empty items are rejected. Invalid submissions retain
entered text and show field-linked feedback. CNN validation runs only for CNN;
retained CNN values cannot invalidate an FC submission. Invalid submissions clear
the previously saved network section so Review cannot export stale CNN text.

Spatial dimensions are checked through every layer in order. With `valid` padding,
convolution produces `d - k + 1`; pooling produces `floor(d / p)` when `p > 1`.
With `same` padding, convolution preserves spatial dimensions (stride 1) and pooling
produces `ceil(d / p)`. Every intermediate dimension must remain positive. This
also catches later-layer collapse, rather than checking only the original input.

Lists serialize as an option followed by one integer on each line. Scalar options
use `--name=value`; output ends with a newline. An empty dense stack emits a bare
`--number_hidden_units`, which parses to `[]` rather than the parser's `None`.

## Assumptions and discrepancies

- The parser declares convolution lists as optional (`None`), but the builder passes
  these values explicitly into a stack that calls `len`/`zip` on them. Python helper
  defaults therefore do not rescue missing parser values. The interface requires
  explicit non-empty lists and shapes as a beginner policy.
- Upstream permits an input rank of one with tokenizer/embedding modes. This form
  has neither feature, so only numeric CNN inputs with ranks 2–4 are supported.
- Channel order assumes Keras `channels_last`, matching official Zero2Neuro examples.
  Upstream does not explicitly set `data_format`; a user's Keras configuration can
  change it. Confirm this setting in the actual execution environment.
- Average pooling and strides are not fields in the current CNN form. They are
  omitted, yielding no average pooling and convolution stride 1. Spatial dropout
  and convolution batch normalization are also omitted. The existing shared batch
  normalization control applies to the dense stack, not convolution layers.
- The argument reference calls max pooling and strides single integers, but the
  current parser uses `nargs='+'` lists. The details page spells average pooling
  `conv_average_pool_size`; the parser/builder use `conv_pool_average_size`.
  The details page also calls convolution batch normalization `batch_normalization`;
  the actual convolution flag is `conv_batch_normalization`.
- Documentation highlights pooling 0 as disabled; the implementation skips both 0
  and 1 (`p > 1`). Negative values also skip upstream, but the interface rejects them.
- Upstream accepts more activation names and translates custom activations; this
  first generator intentionally supports the existing four convolution choices.
- Upstream CNN currently ignores input dropout (the implementation is commented
  out) and input batch normalization (passed via unused kwargs). Existing options
  still serialize if supplied, but have no effect there; the CNN form explains this.
- The regularizer helper calls `keras.regularizers.l1l2` when both L1 and L2 are
  supplied. The interface's existing installed-version compatibility guard is also
  applied to CNN, which uses the same helper. Verify against the target Keras
  version before relaxing this rule.

## Verification and remaining manual work

The tests cover direct generator validation, exact text, supported CNN ranks and
choices, pooling boundaries, spatial collapse, mismatched lists, malformed forms,
preserved values, Review persistence, and FC isolation. An argparse fixture matches
the inspected argument declarations and checks text round trips. It does not import
or run Zero2Neuro or Keras.

Baseline: `pytest -q` reported **132 passed, 494 subtests passed**; the two current
JavaScript test files reported **10 passed**. After implementation: **148 passed,
605 subtests passed**, plus **10 JavaScript tests passed**.

Manual verification still needed:

1. In a browser, switch between FC and CNN, generate/copy a CNN, correct linked
   errors, and confirm Review/download contain the generated network section.
2. In the target Zero2Neuro environment, verify installed Zero2Neuro/keras3_tools
   revisions, Keras version and channel order. Parse a generated file using the real
   parser, build a model through `NetworkBuilder`, and inspect layer/output shapes.
3. Supply compatible data and experiment settings and run an actual training smoke
   test (including the 1D/2D/3D variants intended for use). Data/loss/target compatibility,
   memory limits, and successful training are not established by these tests.

No generated CNN has been built or trained against real Zero2Neuro in this change.
