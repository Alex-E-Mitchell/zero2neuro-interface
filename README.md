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
    `-- src/
        |-- main/java/
        |   `-- ConfigGenerator.java
        `-- test/java/
            `-- ConfigGeneratorTest.java
```

The Java `out/` directory is generated when the Java files are compiled and should not be committed to the repository.

## Running the Tests

### Python

Requires Python 3.

From the repository root:

```bash
python -m unittest discover -s python/tests
```

A successful run should end with:

```text
OK
```

### JavaScript

Requires a recent version of Node.js.

From the repository root:

```bash
cd javascript
npm test
```

The JavaScript tests use Node's built-in test runner, so there are no third-party testing packages to install.

On Windows PowerShell, if `npm test` is blocked by the PowerShell execution policy, use:

```powershell
npm.cmd test
```

A successful run should report that all JavaScript tests passed.

### Java

Requires a Java Development Kit (JDK). Maven and JUnit are not required for the current Java test.

From the repository root:

```powershell
cd java
javac src\main\java\ConfigGenerator.java src\test\java\ConfigGeneratorTest.java -d out
java -cp out ConfigGeneratorTest
```

The first command compiles both Java files into the generated `out` directory. The second command runs `ConfigGeneratorTest`.

A successful run should display:

```text
PASS: Valid hidden units [64, 32]
PASS: Valid hidden units [128]
PASS: Zero should be invalid
PASS: Negative values should be invalid
PASS: Empty array should be invalid
```

## Planned Technologies

The current proposed technology stack for the Zero2Neuro interface is:

- **FastAPI** - Python-based backend and API layer that will receive requests from the frontend, handle application logic, and connect the interface to the existing Zero2Neuro system.
- **React + TypeScript** - frontend framework and language combination for building the beginner-friendly web interface, including forms, configuration controls, validation feedback, and experiment-related views.
- **Pydantic** - configuration validation and data modeling for checking user-provided Zero2Neuro settings before they are processed or passed to the backend.
- **Python** - underlying language used for backend development and integration with Zero2Neuro.
- **Zero2Neuro** - the existing deep neural network toolbox that will provide the core model configuration, training, evaluation, and experiment functionality.

At a high level, the planned architecture is:

```text
React + TypeScript Frontend
            |
            v
        FastAPI Backend
            |
            v
    Pydantic Validation
            |
            v
       Zero2Neuro
```

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

The exact integration method has not yet been finalized. Possible approaches include generating configuration files for Zero2Neuro or calling Zero2Neuro functionality from the interface's Python backend, depending on the existing project's API and the mentor's expectations.
