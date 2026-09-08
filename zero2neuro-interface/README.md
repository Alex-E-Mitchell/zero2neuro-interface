# Zero2Neuro Interface

A capstone project focused on creating a beginner-friendly interface for the existing Zero2Neuro deep neural network toolbox.

## Project Goal

Zero2Neuro already allows users to configure neural-network experiments without writing model code, but users still need to understand configuration syntax and experiment options. This project aims to add a beginner-friendly interface that guides users through those choices and translates them into Zero2Neuro-compatible configuration data.

A likely long-term architecture is:

```text
Beginner User
     |
     v
Web Interface
     |
     v
Interface Backend
     |
     v
Configuration Generation / Validation
     |
     v
Existing Zero2Neuro System
     |
     v
Training / Evaluation / Results
```

## Current Feature

### Network Configuration Validation

The first feature validates hidden-layer sizes before a Zero2Neuro network configuration is generated.

A valid hidden-layer list must:

- contain at least one value
- contain only integer values
- contain only values greater than zero

Examples:

```text
[64, 32] -> valid
[128]    -> valid
[0, 32]  -> invalid
[-4, 32] -> invalid
[]       -> invalid
```

This is a small but realistic part of a future graphical configuration workflow. A beginner-facing form should prevent invalid network settings before passing them to Zero2Neuro.

## Why Three Languages?

The capstone assignment asks the group to demonstrate one unit test in three different programming languages. The same validation rule is implemented and tested here in:

- Python
- JavaScript
- Java

Python is the most relevant language for eventual integration with Zero2Neuro. JavaScript is relevant to a browser-based interface. Java is included to satisfy the assignment's three-language testing requirement.

## Repository Structure

```text
zero2neuro-interface/
|
|-- README.md
|-- .gitignore
|-- docs/
|   `-- IMPLEMENTATION_PLAN.md
|-- examples/
|   `-- sample_network_config.txt
|-- python/
|   |-- config_generator.py
|   `-- tests/
|       `-- test_config_generator.py
|-- javascript/
|   |-- configGenerator.js
|   |-- configGenerator.test.js
|   `-- package.json
`-- java/
    |-- pom.xml
    `-- src/
        |-- main/java/
        |   `-- ConfigGenerator.java
        `-- test/java/
            `-- ConfigGeneratorTest.java
```

## Running the Tests

### Python

From the repository root:

```bash
python -m unittest discover -s python/tests
```

### JavaScript

Requires a recent version of Node.js.

```bash
cd javascript
npm test
```

The JavaScript tests use Node's built-in test runner, so there are no third-party packages to install.

### Java

Requires Java and Maven.

```bash
cd java
mvn test
```

The Java tests use JUnit 5 through Maven.

## Planned Technologies

The final technology stack should be confirmed with the group and mentor, but a reasonable starting direction is:

- **Python** - backend logic and Zero2Neuro integration
- **Flask or FastAPI** - possible web backend
- **HTML/CSS/JavaScript** - browser interface
- **Bootstrap** - possible responsive UI toolkit
- **Zero2Neuro** - existing neural-network experiment engine

## Near-Term Progress Plan

1. Validate common beginner-facing network inputs.
2. Generate Zero2Neuro-compatible network configuration text.
3. Prototype a basic web form for model configuration.
4. Connect the web form to Python backend logic.
5. Test configuration generation and error handling.
6. Integrate the interface with the existing Zero2Neuro workflow.
7. Refine the interface using mentor and user feedback.

## Relationship to Zero2Neuro

This repository is intended to contain the capstone group's interface work rather than duplicate the full Zero2Neuro project. The existing Zero2Neuro repository can remain the underlying neural-network system that this project integrates with.
