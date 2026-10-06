# Implementation Plan

The current capstone prototype generates Zero2Neuro network configuration files
through a local Flask web application. Preserve this small working implementation
while adding reusable rules and parser metadata.

## Current Implementation

- **Flask / Python**: `app.py` handles GET/POST requests, parses form values, calls
  configuration generation, and renders validation errors through Jinja templates.
- **Python validation and generation**: `python/config_generator.py` validates
  shapes and hidden layers and emits Zero2Neuro argument-file text.
- **HTML / CSS / JavaScript**: `templates/network.html` and `static/style.css`
  provide the existing form, tooltips, generated config display, and Copy Config.
- **unittest**: tests cover backend logic, Flask requests, and parser prototypes.

The earlier Java/JavaScript validation exercises are archived coursework, not
the active backend. There is no React, TypeScript, FastAPI, or Pydantic dependency.
Pydantic could support future models if there is a concrete benefit; it is not
needed for this update.

## Current Network Workflow

1. Enter input/output dimensions as comma-separated positive integers.
2. Enter comma-separated hidden-layer sizes, or leave blank for zero layers.
3. Select hidden/output activation functions using existing dropdowns.
4. Submit the form for Python validation and configuration generation.
5. Review errors or copy the generated network configuration.

Shapes are non-empty lists such as `[34]`, `[128, 128, 3]`, or `[10, 20]`.
Zero, negative, boolean, and non-integer dimensions are rejected. Hidden-layer
lists use the same positive-integer rule but may be empty. Each list element is
written on its own value line. A bare `--number_hidden_units` option explicitly
represents zero layers with no blank values.

## Intended Extensible Architecture

```text
Zero2Neuro src/parser.py
    -> argument metadata (types, nargs, defaults, help, choices, flags)

plus

data-driven constraint / architecture registries
    -> required / optional relationships and semantic constraints

then

generic validation / config generation
    -> Flask web interface
```

`python/parser_metadata.py` inspects a supplied `argparse.ArgumentParser` and
returns one record per action, preserving aliases and actions with shared
destinations. It does not import Zero2Neuro, parse help text into choices, or
infer architecture constraints. Returned type callables/defaults remain Python
objects. The prototype isolates argparse's private action enumeration so future
compatibility changes have one place to be handled.

`python/config_rules.py` holds initial `fully_connected` and `cnn` required and
optional sets. A generic presence checker is already used by the current
generator. The registry stores option names; metadata preserves actual argparse
destinations. A future integration must reconcile aliases, notably `input_shape`
to `input_shape0` and `output_shape` to `output_shape0`.

Parser metadata extraction remains a separate reusable prototype, rather than a
runtime dependency of the form. Future work can combine it with field validators
and constraint registries to reduce duplicated metadata. Positivity, supported
string values absent from argparse choices, and architecture relationships still
need explicit semantic rules; `nargs` alone cannot establish them.

## Source Evidence and Assumptions

Inspected local Zero2Neuro `src/parser.py` and `src/network_builder.py` at commit
`55638d7eab68e8dc1a173af15f567ba607a383d2`, its network documentation and CNN
example, and local `keras3_tools/src/cnn_tools.py`.

- Shapes use `nargs='+'`; hidden units use `nargs='*'` and default to `None`.
- Requiring explicit input/output sizes is interface policy, since the upstream
  parser supplies `[10]` defaults. Activation options retain upstream defaults.
- CNN filter, kernel, and pooling lists are provisionally required for new model
  construction: the parser defaults them to `None`, the network builder passes
  them explicitly, and the CNN stack calls `len`/`zip` on them. A pooling entry of
  zero can disable pooling. The initial registry does not claim to cover every
  upstream option or pretrained-model path.
- CNN dimensionality and list-length relationships need mentor confirmation.
  They are documented here rather than implemented as speculative validation.

The fully connected UI remains the only supported form workflow. A CNN call to
the existing generator cannot supply the extra required CNN fields and is
rejected by the generic presence check.

## Testing and Near-Term Decisions

Run the full active Python suite with:

```powershell
python -m unittest discover -s python/tests -v
```

Verify one-/multidimensional shapes, invalid dimensions, empty/invalid hidden
layers, exact generated text, form validation/errors, registry structure, and
parser metadata including flags and aliases. A lightweight argparse fixture
checks generated-text round trips without requiring TensorFlow or Keras.

Before integrating metadata into production validation, agree on the supported
Zero2Neuro version, canonical argument names, semantic string choices, and CNN
requirements. Model building/training compatibility is a later integration check.

## Later Work (Outside This Update)

CNN controls, Data and Experiment configuration, complete file exports, and
direct experiment execution can be planned after this foundation is reviewed.
U-Net, RNN, and scikit-learn workflows remain later extensions. No framework
migration or frontend redesign is planned as part of this work.
