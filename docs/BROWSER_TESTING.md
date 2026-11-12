# Testing the interface

These checks cover different stages of the user's workflow. Run application tests and real-browser tests separately. Training is a separate, heavier
manual demonstration in [the XOR guide](../examples/xor/README.md).

## 1. Set up the test tools

From the repository root (Python 3.13 and Node.js 24 are recommended):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
```

On Linux, use `python -m playwright install --with-deps chromium` to install the
browser's system dependencies too. If you moved the repository, recreate the
virtual environment rather than relying on activation scripts with old paths.
Do not install the TensorFlow/XOR requirements into this environment: the training
example uses a separate Python 3.12 environment and a different NumPy version.

## 2. Run the existing application tests

```sh
python -m pytest python/tests -q
node --test python/tests/*.test.js
```

Use pytest, not only unittest discovery: the Review/export tests include pytest
functions. The tests cover form handling, generated text, validation rules,
parser integration, navigation, and JavaScript behavior in a simulated DOM.
They do not all exercise a real browser or an installed upstream parser.

## 3. Run the real-browser tests

Download the exact upstream parser version used by this suite:

```sh
mkdir -p .test-deps
git clone https://github.com/Symbiotic-Computing-Laboratory/zero2neuro.git .test-deps/zero2neuro
git -C .test-deps/zero2neuro checkout 55638d7eab68e8dc1a173af15f567ba607a383d2
export NEURO_REPOSITORY_PATH="$PWD/.test-deps/zero2neuro"
python -m pytest tests/browser -q --tracing=retain-on-failure --screenshot=only-on-failure
```

The suite starts and stops its own Flask server on an available local port. You
do not need an existing app running. Each test gets a separate browser session.
Setup fails if the real parser cannot load; parser validation is not mocked here.

To watch the tests operate the browser:

```sh
python -m pytest tests/browser --headed --slowmo 300
```

The seven test cases cover:

1. Fully connected Network → Data → Experiment → Review → individual and ZIP
   downloads. Downloaded bytes must match generated text and the XOR reference files.
2. CNN generation and export, including convolution fields. This checks export,
   not training a CNN on the XOR dataset.
3. Invalid Network input, retained entries, error-link focus, correction, and
   removal of the stale downloadable configuration.
4. Dynamic Data split fields and correction of an invalid output-column entry.
5. Experiment early-stopping visibility, multiple metrics, and error correction.
6. Isolation: a new browser session cannot access another session's saved config.
7. Unsubmitted drafts return after navigation; cancelling Clear Inputs preserves
   the draft, and confirming it clears Network without clearing Data or Experiment.

Failures retain traces/screenshots in ignored `test-results/`. Open a trace with:

```sh
python -m playwright show-trace /path/to/trace.zip
```

## Scope

Chromium is the initial browser target. The tests cover representative
workflows, not every configuration or successful CNN training. The Experiment-page
download button is not yet merged; downloads are tested through Review.

Two existing CNN test selectors were updated to accept the section element used
by the new UI while keeping their field, default, and visibility assertions.

Clear Inputs currently clears a browser draft, not an already generated file in
Review. The test checks the draft-clearing behavior and does not claim otherwise.

## Automatic GitHub checks

The workflow in `.github/workflows/tests.yml` runs on pull requests, pushes to
`main`, and manual dispatch. It runs two separate jobs on Ubuntu:

- Python and JavaScript application tests with Python 3.13 and Node.js 24.
- Chromium browser tests against the pinned Zero2Neuro parser version above.

Open a pull request's **Checks** tab to see each job and its output. Failed browser
tests upload available traces and screenshots as `browser-failure-evidence`,
retained for seven days. Training the XOR model is not part of these checks.

The workflow requires the browser-test files from PR #23. Merge that PR before
merging this workflow into `main`. A successful local run does not verify the
GitHub runner setup; check the first hosted run before treating CI as complete.
