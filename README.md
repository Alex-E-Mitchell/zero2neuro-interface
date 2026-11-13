# Zero2Neuro Interface

A beginner-friendly web interface for creating configuration files for the [Zero2Neuro](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro) deep neural network toolbox.

This project is developed as part of the CS4273 Capstone course. It provides guided forms for configuring neural networks, datasets, and training experiments without requiring users to manually write Zero2Neuro configuration arguments.

## Features

The Flask application currently provides four pages:

| Page | Route | Description |
|---|---|---|
| Network | `/` | Configure Fully Connected or Convolutional Neural Networks |
| Data | `/data` | Configure dataset formats, columns, and splitting methods |
| Experiment | `/experiment` | Configure training parameters and evaluation settings |
| Review | `/review` | Review and download generated configuration files |

### Network Configuration

The Network page supports Fully Connected and Convolutional Neural Network (CNN) configurations.

Available settings include:

- Input and output shapes
- Hidden layer sizes and activation functions
- Batch normalization
- Dropout and input dropout
- L1 and L2 regularization
- CNN filters, kernel sizes, pooling, padding, and activation functions

The interface validates required fields, supported options, numeric values, and architecture-specific constraints.

For example, a Fully Connected network with input shape `34`, hidden layers `64, 32`, and output shape `1` can generate:

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

For multidimensional shapes, each dimension is written on a separate line:

```text
--input_shape
128
128
3
--output_shape
10
20
```

Hidden layers may also be empty. In that case, the generator emits `--number_hidden_units` without additional values, allowing the parser to receive an empty list.

CNN validation includes matching layer-list lengths and spatial-size checks. The supported CNN options are a subset of the options available in Zero2Neuro.

See [CNN Configuration](docs/CNN_CONFIGURATION.md) for additional details.

### Data Configuration

The Data page supports:

- Tabular (CSV/XLSX) and pickle datasets
- Dataset directories and file names
- Input and output column selection
- One fold per file
- Random folds
- Class-balanced random folds
- Grouping files into folds
- Assigning folds using a data column
- Dataset set-type options

The interface validates required settings based on the selected data format and splitting method.

### Experiment Configuration

The Experiment page supports:

- Experiment name
- Loss function
- Evaluation metrics
- Learning rate
- Number of epochs
- Optional early stopping
- Early stopping monitor, patience, and minimum improvement

The page validates numeric ranges and supported options before generating configuration text.

See [Experiment Page](docs/EXPERIMENT_PAGE.md) for supported settings and implementation details.

### Review and Export

The Review page displays the generated Network, Data, and Experiment configurations for the current session.

Users can download individual files:

```text
network.txt
data.txt
experiment.txt
```

Once all three configurations have been generated, the interface can export them together as:

```text
zero2neuro_config.zip
```

The ZIP contains all three configuration files for use with the existing Zero2Neuro framework.

## Validation

The interface uses built-in validation rules and optional validation through Zero2Neuro's existing argument parser.

### Built-in Validation

Validation rules check:

- Required fields
- Positive integer dimensions and layer sizes
- Supported activation functions and other choices
- Dropout values in the range `0 <= rate < 1`
- Nonnegative regularization coefficients
- CNN layer-list lengths and spatial dimensions
- Data configuration requirements
- Experiment settings and numeric ranges

For example, hidden-layer values are validated as follows:

```text
[64, 32] -> valid
[128]    -> valid
[0, 32]  -> invalid
[-4, 32] -> invalid
[]       -> valid (zero hidden layers)
```

Architecture-specific rules are maintained in `python/config_rules.py`.

The rules were developed against Zero2Neuro v0.11.4 and keras3_tools v0.2.1. See the CNN documentation for source revisions, assumptions, and limitations.

### Parser Metadata

`python/parser_metadata.py` provides utilities for extracting information from an existing `argparse.ArgumentParser`.

The metadata includes:

- Argument names and aliases
- Types and defaults
- Required status
- Supported choices
- Boolean flags
- Help information

Parser metadata is kept separate from the interface's manually maintained architecture rules.

### Zero2Neuro Parser Integration

`python/parser_validator.py` validates generated configuration arguments using Zero2Neuro's existing `create_parser()` function.

When `NEURO_REPOSITORY_PATH` is configured, the interface:

1. Validates submitted form values.
2. Generates the corresponding configuration text.
3. Loads the existing Zero2Neuro argument parser.
4. Checks whether the generated arguments are accepted.
5. Rejects invalid configurations and displays an error.
6. Saves accepted configurations for review and export.

If the external parser is unavailable, the interface continues using its built-in validation rules.

This integration does not require modifications to the main Zero2Neuro repository.

**Note:** Parser validation confirms that arguments can be parsed. It does not guarantee successful data loading, network construction, or training.

## User Interface

The application uses shared page layouts and navigation.

Current interface improvements include:

- Updated visual styling
- Field-specific validation messages
- Preserved form values after submissions
- Browser-side form persistence
- Clear Inputs controls
- Configuration previews
- Individual file downloads and ZIP export

Unfinished form inputs are stored in the browser using `localStorage`, allowing users to return to their entries after navigating between pages or reloading.

The Clear Inputs control resets the current page's draft without clearing drafts on other pages.

Generated configurations are stored separately in temporary server-side memory for the current browser session. They are not permanently saved and are lost when the Flask server restarts.

## Repository Structure

```text
zero2neuro-interface/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- python/
|   |-- config_generator.py
|   |-- config_rules.py
|   |-- config_store.py
|   |-- data_generator.py
|   |-- data_page.py
|   |-- data_rules.py
|   |-- experiment_generator.py
|   |-- experiment_page.py
|   |-- network_form.py
|   |-- parser_metadata.py
|   |-- parser_validator.py
|   |-- review_page.py
|   `-- tests/
|
|-- templates/
|   |-- base.html
|   |-- network.html
|   |-- data.html
|   |-- experiment.html
|   `-- review.html
|
|-- static/
|   |-- style.css
|   |-- data.css
|   |-- network_selector.js
|   |-- dropout_validation.js
|   |-- form_persistence.js
|   `-- experiment.js
|
|-- docs/
|   |-- IMPLEMENTATION_PLAN.md
|   |-- CNN_CONFIGURATION.md
|   `-- EXPERIMENT_PAGE.md
|
|-- examples/
|   |-- sample_network_config.txt
|   `-- xor/
|
`-- archive/
```

The `python/` directory contains backend logic, configuration generation, validation, and automated tests.

The `templates/` and `static/` directories contain the Flask user interface.

The `docs/` directory contains additional implementation documentation.

The `examples/` directory contains configuration examples and XOR training verification resources.

The `archive/` directory contains earlier course-assignment work.

## Installation

### Requirements

To run the web interface locally:

- Python
- Git
- A web browser

The Python packages required by the interface are listed in `requirements.txt`.

The main Zero2Neuro repository is optional for running the web application but is needed to enable parser integration.

Actual neural network training requires additional dependencies and a separate Zero2Neuro environment.

### 1. Clone the Repository

```powershell
git clone https://github.com/Alex-E-Mitchell/zero2neuro-interface.git
cd zero2neuro-interface
```

If the repository is already cloned, open a terminal in its root directory.

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

The execution-policy change applies only to the current PowerShell session.

### 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The application requirements include Flask, NumPy, and their supporting packages.

### 5. Configure Parser Integration (Optional)

To enable validation through Zero2Neuro's existing parser, clone the main repository separately:

```powershell
git clone https://github.com/Symbiotic-Computing-Laboratory/zero2neuro.git
```

Set the `NEURO_REPOSITORY_PATH` environment variable to the root of the main Zero2Neuro repository.

Example:

```powershell
$env:NEURO_REPOSITORY_PATH = "D:\CS4273\zero2neuro"
```

Replace the example path with the actual location of your checkout.

The expected structure is:

```text
zero2neuro/
`-- src/
    `-- parser.py
```

The parser's required Python dependencies must also be available in the active environment.

This environment variable is set for the current PowerShell session. It must be configured again in a new session if needed.

If the parser is unavailable, the web application can still generate configurations using its built-in validation rules.

**Important:** The interface's parser integration expects `NEURO_REPOSITORY_PATH` to point directly to the Zero2Neuro checkout. The upstream training framework uses this variable differently. The XOR verification runner handles the training environment separately.

### 6. Run the Application

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Use the navigation menu to access the Network, Data, Experiment, and Review pages.

The Flask development server is intended for local development and testing.

## Testing

### Python Tests

The repository contains Python tests for configuration generation, validation, Flask routes, parser integration, and review/export behavior.

Install the test runner if needed:

```powershell
python -m pip install pytest pytest-subtests
```

Run the test suite from the repository root:

```powershell
python -m pytest -q
```

Alternatively, run the unittest suite:

```powershell
python -m unittest discover -s python/tests -v
```

Some parser tests use mocks and do not require the full Zero2Neuro training environment.

### JavaScript Tests

The repository includes Node.js tests for network selection, dropout validation, and form persistence.

With Node.js installed, run:

```powershell
node --test python/tests/*.test.js
```

On Windows PowerShell, if wildcard expansion is not supported by the Node.js command, specify the test files:

```powershell
node --test python/tests/network_selector.test.js python/tests/dropout_validation.test.js python/tests/form_persistence.test.js
```

These tests verify browser-side behavior independently of the Python test suite.

Passing the automated tests does not guarantee that every generated configuration will successfully train in Zero2Neuro.

## XOR Training Verification

The repository includes an XOR example demonstrating how configurations generated by the interface can be used for actual training with Zero2Neuro.

See [XOR Verification Walkthrough](examples/xor/README.md).

The example includes:

- A four-row XOR dataset
- Reference Network, Data, and Experiment configurations
- A Python training and verification script
- Recorded results from a successful run

The runner:

1. Loads the configuration files.
2. Validates them using the upstream parser.
3. Runs training through Zero2Neuro.
4. Reloads the trained model.
5. Checks predictions for all four XOR inputs.

The example was verified using pinned Zero2Neuro and keras3_tools revisions, with Python 3.12 on Apple Silicon macOS.

The recorded verification completed successfully and classified all four training rows correctly.

The example is a small training demonstration, not a held-out evaluation or proof that every configuration supported by the interface will train successfully.

### Running the XOR Example

The XOR walkthrough uses a separate Python 3.12 environment because training requires additional dependencies, including TensorFlow.

Follow the setup instructions in:

```text
examples/xor/README.md
```

The documented runner is:

```text
examples/xor/run_demo.py
```

It can use either the included reference configurations or files exported from the web interface.

Training runs outside the Flask application. The web interface does not directly start or manage training jobs.

## Current Limitations

The current implementation supports configuration generation, validation, review, and export.

Known limitations include:

- The interface does not directly launch Zero2Neuro training.
- Training results are not displayed in the web interface.
- Parser validation does not guarantee successful model training.
- Only a subset of Zero2Neuro's available configuration options is exposed.
- Generated configurations are stored temporarily on the server.
- The XOR verification covers a specific Fully Connected example, not CNN training or every possible configuration.
- Additional end-to-end testing and usability improvements are still needed.

## Additional Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [CNN Configuration](docs/CNN_CONFIGURATION.md)
- [Experiment Page](docs/EXPERIMENT_PAGE.md)
- [XOR Training Verification](examples/xor/README.md)

## Project Scope

The Zero2Neuro Interface is a separate application from the main Zero2Neuro framework.

Its purpose is to simplify configuration creation, provide validation feedback, and export compatible configuration files.

The [XOR walkthrough](examples/xor/README.md) shows how to generate Network, Data,
and Experiment files in the interface, run them through Zero2Neuro in a separate
training environment, and check the saved model's predictions. It includes a
four-row dataset, reference configurations, and a repeatable verification script.
Training runs outside the web app; the example does not add a training button.

## Browser workflow tests

See [Browser testing](docs/BROWSER_TESTING.md) for setup and commands to watch
Chromium fill forms, check errors and navigation, and verify downloaded files.
These tests use the real Zero2Neuro parser and run separately from XOR training.
