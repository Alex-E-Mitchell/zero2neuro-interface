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

The current prototype implements the first end-to-end vertical slice of the interface: **Fully Connected Network Configuration**.

A user can open the local web application, enter basic network settings, and generate Zero2Neuro-compatible network configuration text.

The current form includes:

- Input Shape
- Hidden Layer Sizes
- Hidden Activation
- Output Shape
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
- browser-level restrictions for simple numeric fields
- a Copy button that copies only the generated configuration text
- basic styling for a cleaner beginner-facing workflow

## Current Guardrails

The prototype currently validates several beginner-facing settings before generating configuration output.

Examples include:

- input shape must be a positive integer
- output shape must be a positive integer
- hidden-layer sizes must contain at least one value
- hidden-layer sizes must contain only positive integers

Examples:

```text
[64, 32] -> valid
[128]    -> valid
[0, 32]  -> invalid
[-4, 32] -> invalid
[]       -> invalid
```

These checks are an early example of the larger project goal: helping users avoid invalid Zero2Neuro configurations before they reach the underlying system.

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
|   `-- tests/
|       `-- test_config_generator.py
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
python -m unittest discover -s python/tests
```

A successful test run should end with:

```text
OK
```

The tests currently cover hidden-layer validation, network-configuration generation, and rejection of invalid input values.

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
