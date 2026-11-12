# Zero2Neuro Interface

A capstone project focused on creating a beginner-friendly graphical interface for the existing Zero2Neuro deep neural network toolbox.

## Project Goal

Zero2Neuro already allows users to configure neural-network experiments without writing model code, but users still need to understand configuration-file syntax, valid argument choices, and experiment options.

This project aims to make that process easier by providing a guided web interface that:

- presents Zero2Neuro settings in a beginner-friendly form
- validates user input before configuration files are generated
- provides guardrails against invalid or inappropriate values
- translates form inputs into Zero2Neuro-compatible configuration text
- eventually supports network, data, and experiment configuration workflows

The current planned workflow is:

```text
Beginner User
     |
     v
Web Interface
     |
     v
Validation / Configuration Generation
     |
     v
Zero2Neuro-Compatible Configuration Files
     |
     v
Existing Zero2Neuro System
```

The interface is intended to remain separate from the core Zero2Neuro codebase. Zero2Neuro continues to provide the underlying neural-network functionality while this project focuses on making configuration easier and more accessible.

## Current Prototype

The current prototype implements the first end-to-end vertical slice of the interface: **Fully Connected and CNN Network Configuration**.

A user can open the local web application, enter basic network settings, and generate Zero2Neuro-compatible network configuration text.

The current form includes:

- Input Shape (comma-separated positive dimensions, e.g. `34` or `128, 128, 3`)
- Hidden Layer Sizes (comma-separated positive integers; blank means zero layers)
- Hidden Activation
- Output Shape (comma-separated positive dimensions, e.g. `1` or `10, 20`)
- Output Activation

For example, the following form values:

```text
Input Shape: 34
Hidden Layer Sizes: 64, 32
Hidden Activation: ELU
Output Shape: 1
Output Activation: ELU+1
```

generate:

```text
--network_type=fully_connected
--input_shape
34
--number_hidden_units
64
32
--hidden_activation=elu
--output_shape
1
--output_activation=elup1
```

The interface also includes:

- validation for invalid input values
- visible error messages for rejected settings
- required shape fields and backend validation of comma-separated dimensions
- a Copy button that copies only the generated configuration text
- basic styling for a cleaner beginner-facing workflow

## Experiment Configuration

The Experiment page at `/experiment` supports experiment name, loss, optional
metrics, learning rate, epochs and early stopping settings. Submissions are
validated on the server and produce configuration text with a Copy Config button.
Invalid submissions preserve entered values and show field-specific feedback.

See [Experiment page notes](docs/EXPERIMENT_PAGE.md) for supported choices,
upstream source references, manual checks and remaining Phase 2 work.

## Current Guardrails

The prototype currently validates several beginner-facing settings before generating configuration output.

Examples include:

- input and output shapes must be non-empty lists of positive integers
- hidden-layer sizes may be empty (zero hidden layers)
- hidden-layer sizes must contain only positive integers

Examples:

```text
[64, 32] -> valid
[128]    -> valid
[0, 32]  -> invalid
[-4, 32] -> invalid
[]       -> valid (zero hidden layers)
```

These checks are an early example of the larger project goal: helping users avoid invalid Zero2Neuro configurations before they reach the underlying system.

For multidimensional shapes, each dimension is written on its own line:

```text
--input_shape
128
128
3
--output_shape
10
20
```

For zero hidden layers, the generated `--number_hidden_units` line is immediately
followed by the next option, with no blank value lines. This explicitly supplies
`[]` to Zero2Neuro's `nargs='*'` argument rather than relying on its `None` default.

## Extensible Rules and Parser Metadata

`python/config_rules.py` contains initial `fully_connected` and `cnn` architecture
entries with required/optional argument sets. The generator uses a generic
registry-based presence check; value validation remains separate. CNN generation
supports the existing convolution fields, matching layer lists, and spatial-size checks.

CNN rules were verified against current upstream Zero2Neuro v0.11.4 and
keras3_tools v0.2.1. See [CNN configuration notes](docs/CNN_CONFIGURATION.md) for
source links, validation rules, documentation discrepancies, assumptions, test
results, and remaining manual verification. Generated CNNs have not been built
or trained against real Zero2Neuro.

`python/parser_metadata.py` provides `extract_argument_metadata(parser)`. Pass an
existing `argparse.ArgumentParser`, for example the result of Zero2Neuro's
`create_parser()`, to derive destinations, option aliases, type callables, `nargs`,
defaults, help, choices, argparse-required status, and boolean flag information.
Records remain Python objects, not JSON. This utility does not import Zero2Neuro
or guess semantic string choices from help text. Parser declarations and manually
maintained architecture constraints stay separate.

## Repository Structure

```text
zero2neuro-interface/
|
|-- README.md
|-- .gitignore
|-- app.py
|-- requirements.txt
|
|-- python/
|   |-- config_generator.py
|   |-- config_rules.py
|   |-- parser_metadata.py
|   `-- tests/
|       |-- test_config_generator.py
|       |-- test_config_rules.py
|       |-- test_parser_metadata.py
|       `-- test_app.py
|
|-- templates/
|   `-- network.html
|
|-- static/
|   `-- style.css
|
|-- docs/
|   `-- IMPLEMENTATION_PLAN.md
|
|-- examples/
|   `-- sample_network_config.txt
|
`-- archive/
    `-- ticket2-three-language-tests/
        |-- java/
        `-- javascript/
```

The `archive/` directory contains earlier course-assignment work demonstrating equivalent validation logic in multiple programming languages. Current capstone development is focused on the Python backend and web interface.

Local development files such as `.venv/`, `__pycache__/`, and compiled Python files should not be committed.

## Running the Prototype

### 1. Create a virtual environment

From the repository root:

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

and then activate the environment again.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the web application

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

in a web browser.

## Running the Python Tests

From the repository root:

```powershell
pytest -q
```

Install the test runner with `pip install pytest pytest-subtests` if needed.
Run the current JavaScript tests with:

```powershell
node --test python/tests/network_selector.test.js python/tests/dropout_validation.test.js
```

The tests cover shape/hidden-layer validation, exact configuration output, Flask
form submissions and errors, registry structure and presence checks, parser metadata,
and generated argument round trips using a lightweight argparse fixture. They do
not build or train neural networks.

## Current Technology Stack

The current working prototype uses:

- **Python** - backend logic and Zero2Neuro configuration generation
- **Flask** - lightweight local web application
- **HTML/CSS** - beginner-facing interface and styling
- **JavaScript** - small browser-side features such as copying generated configuration text
- **Python unittest** - automated tests for validation and configuration generation

The project may adopt additional technologies later if they provide a clear benefit, but the current priority is building a small working configuration workflow before expanding the architecture.

## Near-Term Development Plan

1. Refine Fully Connected network configuration and validation.
2. Add more beginner-friendly explanations and tooltips.
3. Add a visible multi-step workflow for Network, Data, Experiment, and Review.
4. Add support for CNN network configuration.
5. Add Data configuration generation.
6. Add Experiment configuration generation.
7. Export complete Zero2Neuro configuration files.
8. Test generated files directly with Zero2Neuro.
9. Expand guardrails based on Zero2Neuro argument requirements and mentor feedback.
10. Treat recurrent networks, U-Net, and scikit-learn pipelines as later extensions.

## Relationship to Zero2Neuro

This repository contains the capstone group's interface work rather than a replacement for the Zero2Neuro project.

The intended relationship is:

```text
Zero2Neuro Interface
        |
        | generates validated configuration
        v
Existing Zero2Neuro
        |
        v
Training / Evaluation / Results
```

The current required goal is to make Zero2Neuro configuration easier for beginners. Directly launching Zero2Neuro experiments from the GUI may be explored later, but it is not required for the initial configuration-generation workflow.

## Verify exported files with XOR training

The [XOR walkthrough](examples/xor/README.md) shows how to generate Network, Data,
and Experiment files in the interface, run them through Zero2Neuro in a separate
training environment, and check the saved model's predictions. It includes a
four-row dataset, reference configurations, and a repeatable verification script.
Training runs outside the web app; the example does not add a training button.

## Browser workflow tests

See [Browser testing](docs/BROWSER_TESTING.md) for setup and commands to watch
Chromium fill forms, check errors and navigation, and verify downloaded files.
These tests use the real Zero2Neuro parser and run separately from XOR training.
