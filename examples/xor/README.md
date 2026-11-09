# Run GUI-generated configurations with Zero2Neuro: XOR

This example connects the interface's exported files to actual model training.
XOR has two inputs: the target is 1 when they differ and 0 when they match.
The four-row CSV in this folder contains the complete truth table.

The included text files match the configurations exported during the November 3,
2026 manual GUI walkthrough. You can reproduce that walkthrough and test your own
exported files using `--config-dir`. Downloading a file alone does not prove it can
train a model; the runner also loads the saved model and checks its predictions.

## 1. Prepare a separate training environment

Run these commands from the root of `zero2neuro-interface`. Install Python 3.12
first if `python3.12` is not available. Keep the app's existing `.venv` separate:
TensorFlow training needs additional dependencies and was tested with Python 3.12.

```sh
python3.12 -m venv .xor-venv
source .xor-venv/bin/activate
python -m pip install -r examples/xor/requirements.txt
mkdir -p .xor-runtime
git clone https://github.com/Symbiotic-Computing-Laboratory/zero2neuro.git .xor-runtime/zero2neuro
git -C .xor-runtime/zero2neuro checkout 55638d7eab68e8dc1a173af15f567ba607a383d2
git clone https://github.com/Symbiotic-Computing-Laboratory/keras3_tools.git .xor-runtime/keras3_tools
git -C .xor-runtime/keras3_tools checkout c7d63be937a6889fd5b2b4cfe9caa34e0c715f95
```

The commits pin Zero2Neuro v0.11.4 and its companion library to the versions used
in the successful local run. The Zero2Neuro startup banner at this tag still says
0.11.3. TensorFlow is a substantial download. CPU training is sufficient; GPU,
Graphviz, and a Weights & Biases account are not required for this example.
The tested platform is Apple Silicon macOS; other platforms have not been verified.

## 2. Generate the files in the interface

Start the app in a separate Terminal using its normal `.venv`, then enter:

| Page | Setting | Value |
|---|---|---|
| Network | Network type | Fully Connected |
| Network | Input shape | `2` |
| Network | Hidden layer sizes | `4` |
| Network | Hidden activation | Tanh |
| Network | Output shape | `1` |
| Network | Output activation | Sigmoid |
| Data | File format | Tabular |
| Data | File names | `xor_data.csv` |
| Data | Input columns | `In 0` and `In 1`, on separate lines |
| Data | Output columns | `Out 0` |
| Data | Fold split | One fold per file |
| Data | Set type | Fixed (train / validation / test) |
| Experiment | Name | `xor-demo` |
| Experiment | Loss | Mean squared error |
| Experiment | Learning rate | `0.001` |
| Experiment | Epochs | `5000` |

Leave normalization and early stopping unchecked, metrics unselected, and optional
numeric settings and data directory blank. Generate each configuration, visit
Review, download the ZIP, and extract it into a folder. It must contain
`network.txt`, `data.txt`, and `experiment.txt`.

## 3. Run and verify

From the repository root, with `.xor-venv` activated:

```sh
python examples/xor/run_demo.py \
  --zero2neuro .xor-runtime/zero2neuro \
  --keras3-tools .xor-runtime/keras3_tools \
  --output xor-runs/first-run
```

That command uses the included reference configurations. To check files downloaded
from the GUI, supply their folder too (replace the example path):

```sh
python examples/xor/run_demo.py \
  --zero2neuro .xor-runtime/zero2neuro \
  --keras3-tools .xor-runtime/keras3_tools \
  --config-dir /path/to/extracted-configurations \
  --output xor-runs/gui-export-run
```

Use a new output directory for each run. Existing directories are rejected so a
previous result cannot be mistaken for a new successful training run.

The script copies the three files unchanged and supplies the CSV separately. It
checks them with the actual upstream parser, sets a random seed of 42, invokes
Zero2Neuro's training entry point, reloads the saved model, and checks all four
predictions. It adds output/report options on the command line because the GUI
currently does not expose those settings. These additions are listed in the run
manifest. It does not add the training engine to the web application.

## 4. Inspect the evidence

A successful run ends with `PASS: all four training rows classified correctly`.
The output folder contains:

- Copies of the configurations and dataset used.
- `run-manifest.json`: upstream commit IDs, Python/Keras versions, seed, configuration
  hashes, and additional reporting arguments.
- `verified-predictions.json`: expected outputs, actual predictions, classes using
  a 0.5 cutoff, and mean squared error.
- `results/xor-demo_model.keras`: the trained model.
- `results/xor-demo_report.xlsx`: Zero2Neuro's training report.
- `results/xor-demo_results.pkl`: Zero2Neuro's saved run results.

For the original November 3 run, predictions were approximately 0.019, 0.956,
0.983, and 0.044 for inputs (0,0), (0,1), (1,0), and (1,1), respectively. All four
classes were correct and mean squared error was about 0.00112. Floating-point
results may differ by machine or dependency versions even with the same seed.
A nonzero exit or missing PASS message is not a successful verification.

## Scope and limitations

This is a training-set demonstration with four rows, not a held-out evaluation.
It verifies this fully connected XOR configuration, not CNN training or every
possible combination of Network, Data, and Experiment settings. The GUI's normal
Python/JavaScript tests are separate and do not require these training libraries.
The runner is for trusted local configuration files, not uploads from strangers.

If imports fail, check that `.xor-venv` is active and requirements installed. If
training fails, inspect the exception and retained run files before changing the
configuration. Parser acceptance alone does not establish model compatibility.

References: [upstream XOR example](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/tree/55638d7eab68e8dc1a173af15f567ba607a383d2/examples/xor),
[Zero2Neuro setup](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/docs/getting_started/zero2neuro_config.md).

The portable runner was also verified on November 6, 2026 using the original
GUI exports. It completed successfully and classified all four rows correctly.
See [recorded example results](verified-example.json) for predictions, exact
configuration hashes, and upstream revisions. This file is evidence of that run;
it is not a substitute for executing the script on your own machine.

## Parser path when starting the web app

The web interface's parser-validation integration expects `NEURO_REPOSITORY_PATH`
to point directly to the Zero2Neuro checkout (the folder containing `src`). For
example, start the app with `NEURO_REPOSITORY_PATH="$PWD/.xor-runtime/zero2neuro"`.
The upstream training engine uses the same variable name with a different meaning:
the parent of `keras3_tools`. The demo runner sets that value inside its own process
automatically, so you do not need to change your shell's setting to run training.
