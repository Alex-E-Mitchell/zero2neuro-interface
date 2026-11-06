# Experiment page (Phase 1)

`/experiment` now accepts GET/POST and provides experiment name, loss, optional
metrics, learning rate, epochs and early stopping controls. It extends the shared
layout and uses the shared error macros and `validate_config_values` helper.
Entries are preserved in the submission response, not across navigation/reloads.

## Source and scope

Names and defaults were verified against Zero2Neuro v0.11.4, commit
`55638d7eab68e8dc1a173af15f567ba607a383d2`:

- [Parser](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/src/parser.py)
- [XOR example](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro/blob/55638d7eab68e8dc1a173af15f567ba607a383d2/examples/xor/experiment.txt)
- The breast-cancer and iris experiment examples supply the classification choices.

The dropdowns are a small supported subset, not an exhaustive Keras catalog.
Defaults match upstream: MSE loss, no extra metrics, learning rate 0.0001,
100 epochs, early stopping off, monitor `loss`, patience 20, minimum delta 0.01.
The interface requires a file-friendly experiment name, positive finite learning
rate, positive whole-number epochs, nonnegative whole-number patience and finite
nonnegative minimum improvement. These stricter bounds are interface guardrails,
not argparse declarations. Disabled early stopping values are ignored and omitted.
An empty metrics selection omits `--metrics`, since upstream uses `nargs='+'`.

Generated text can be copied or downloaded as `experiment.txt` using the controls
beside the preview. Downloads reuse the Review page's `download_config` route and
the configuration saved for the current browser session. Output/result settings
remain Phase 2 work. This page does not train a model or prove compatibility between
loss/metrics, Network outputs and Data targets. Validation-loss monitoring needs a
validation set; the page explains that requirement but cannot inspect other pages.

## Files

- `python/experiment_generator.py`: supported choices, defaults, value rules, serialization.
- `python/experiment_page.py`: raw form values, numeric parsing and Flask route.
- `templates/experiment.html`: form and generated output using shared layout/errors.
- `static/experiment.js`: early stopping visibility, feedback focus and clipboard feedback.

## Check the page

Run `python -m unittest discover -s python/tests`.
Start on macOS with `python -m flask --app app run --debug --port 5001`.

1. Open `/experiment`; confirm Network/Data/Experiment navigation works.
2. Set name `xor-demo`, loss MSE, metrics MAE and MSE, rate `0.001`, epochs `5000`.
3. Enable early stopping with training loss, patience `2000`, minimum improvement `0.01`.
4. Generate and copy the text; metrics must appear on separate lines. Click
   Download .txt and confirm `experiment.txt` matches the generated text, including
   line breaks. Both controls should remain usable at narrow screen widths.
5. Submit rate `0` and epochs `1.5`; both fields should have errors and retain entries.
6. Turn early stopping off; generation should omit all early stopping options.

Browser JavaScript improves interaction; form submission and server validation
still work without it. Clipboard failure displays instructions for manual copying.

Implementation validation: 124 Python tests passed. Four generated configurations
(no optional settings, multiple metrics with early stopping, binary classification,
and sparse categorical classification with validation-loss monitoring) were also
parsed by the actual upstream v0.11.4 parser. This checks syntax and parsed values,
not training or cross-page model/data compatibility.
