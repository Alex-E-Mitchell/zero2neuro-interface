"""Check GUI exports, train through Zero2Neuro, and verify all four XOR rows."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--zero2neuro', type=Path, required=True, help='Zero2Neuro checkout directory')
    cli.add_argument('--keras3-tools', type=Path, required=True, help='keras3_tools checkout directory')
    cli.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent,
                     help='Directory containing network.txt, data.txt, experiment.txt from the GUI')
    cli.add_argument('--output', type=Path, required=True, help='New directory for this run (must not exist)')
    options = cli.parse_args()
    upstream = options.zero2neuro.resolve()
    companion = options.keras3_tools.resolve()
    inputs = options.config_dir.resolve()
    output = options.output.resolve()
    for path in (upstream / 'src/zero2neuro.py', companion / 'src/fully_connected_tools.py',
                 *(inputs / name for name in ('network.txt', 'data.txt', 'experiment.txt'))):
        if not path.is_file():
            cli.error(f'Missing required file: {path}')
    # Upstream locates keras3_tools relative to NEURO_REPOSITORY_PATH.
    if companion.name != 'keras3_tools':
        cli.error('The companion checkout directory must be named keras3_tools.')
    if output.exists():
        cli.error('Output already exists. Choose a new directory to avoid overwriting results.')
    output.mkdir(parents=True)
    for name in ('network.txt', 'data.txt', 'experiment.txt'):
        shutil.copy2(inputs / name, output / name)
    shutil.copy2(Path(__file__).with_name('xor_data.csv'), output / 'xor_data.csv')
    os.environ.update(NEURO_REPOSITORY_PATH=str(companion.parent),
                      MPLCONFIGDIR=str(output / '.matplotlib'), MPLBACKEND='Agg',
                      WANDB_MODE='disabled', TF_CPP_MIN_LOG_LEVEL='2',
                      TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
    sys.path.insert(0, str(upstream / 'src'))
    from parser import create_parser
    files = ['@' + str(output / name) for name in ('network.txt', 'data.txt', 'experiment.txt')]
    parsed = create_parser().parse_args(files)
    if (parsed.input_shape0 != [2] or parsed.output_shape0 != [1]
            or parsed.data_inputs != ['In 0', 'In 1'] or parsed.data_outputs != ['Out 0']
            or parsed.data_file != 'xor_data.csv' or parsed.data_fold_split != 'identity'
            or parsed.data_set_type != 'fixed' or parsed.data_format != 'tabular'):
        cli.error('This demonstration requires the two-input, one-output XOR data settings in README.md.')
    import keras
    import numpy as np
    keras.utils.set_random_seed(42)
    def revision(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {
        'zero2neuro_commit': revision(upstream), 'keras3_tools_commit': revision(companion),
        'python': sys.version, 'keras': keras.__version__, 'seed': 42,
        'config_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ('network.txt', 'data.txt', 'experiment.txt')},
    }
    extra = ['--results_path=' + str(output / 'results'), '--output_file_base=xor-demo',
             '--save_model', '--log_training_set', '--report', '--report_training', '--report_training_ins']
    manifest['additional_output_arguments'] = extra
    (output / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    os.chdir(output)
    sys.argv = [str(upstream / 'src/zero2neuro.py'), *files, *extra]
    print('Parser accepted the files. Starting Zero2Neuro training.', flush=True)
    runpy.run_path(str(upstream / 'src/zero2neuro.py'), run_name='__main__')
    model = keras.models.load_model(output / 'results/xor-demo_model.keras')
    with (output / 'xor_data.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(row['In 0']), float(row['In 1'])] for row in rows], dtype='float32')
    expected = np.array([int(row['Out 0']) for row in rows])
    predictions = model([x], training=False).numpy().reshape(-1)
    classes = (predictions >= 0.5).astype(int)
    result = {'correct': int((classes == expected).sum()), 'total': len(rows),
              'mse': float(np.mean((predictions - expected) ** 2)),
              'predictions': [{'inputs': ins.tolist(), 'expected': int(target),
                               'prediction': float(pred), 'class_at_0_5': int(label)}
                              for ins, target, pred, label in zip(x, expected, predictions, classes)]}
    (output / 'verified-predictions.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['correct'] != len(rows) or not np.isfinite(predictions).all():
        raise SystemExit('Training ran, but the saved model did not classify all four XOR rows correctly.')
    print('PASS: all four training rows classified correctly. Results:', output)


if __name__ == '__main__':
    main()
