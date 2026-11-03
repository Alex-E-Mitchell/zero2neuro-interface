"""Exercise the Experiment form, validation, escaping and generated output."""

import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

from werkzeug.datastructures import MultiDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app import app
from experiment_generator import generate_experiment_config
from experiment_page import prepare_experiment_form


VALID_FORM = {
    "experiment_name": "xor-demo",
    "loss": "mse",
    "learning_rate": "0.001",
    "epochs": "5000",
    "early_stopping_monitor": "loss",
    "early_stopping_patience": "2000",
    "early_stopping_min_delta": "0.01",
}


class ConfigParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.in_config = False
        self.config = ""
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "pre" and dict(attrs).get("id") == "config-text":
            self.in_config = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_config = False

    def handle_data(self, data):
        if self.in_config:
            self.config += data


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def submit(self, **changes):
        return self.client.post('/experiment', data={**VALID_FORM, **changes}).get_data(as_text=True)

    def test_initial_page_shows_defaults_without_output(self):
        page = self.client.get('/experiment').get_data(as_text=True)
        self.assertIn('value="0.0001"', page)
        self.assertIn('value="100"', page)
        self.assertNotIn('id="generated-config"', page)
        self.assertNotIn('id="form-errors"', page)

    def test_minimal_config_omits_metrics_and_disabled_early_stopping(self):
        self.assertEqual(ConfigParser(self.submit()).config,
                         '--experiment_name=xor-demo\n--loss=mse\n--learning_rate=0.001\n--epochs=5000\n')

    def test_multiple_metrics_and_enabled_early_stopping(self):
        page = self.submit(metrics=['mae', 'mse'], early_stopping='on')
        self.assertEqual(ConfigParser(page).config,
                         '--experiment_name=xor-demo\n--loss=mse\n--metrics\nmae\nmse\n'
                         '--learning_rate=0.001\n--epochs=5000\n--early_stopping\n'
                         '--early_stopping_monitor=loss\n--early_stopping_patience=2000\n'
                         '--early_stopping_min_delta=0.01\n')
        self.assertIn('value="mae" checked', page)

    def test_invalid_numeric_values_are_rejected(self):
        for field, invalid in {
            'learning_rate': ['', 'abc', '0', '-1', 'nan', 'inf', '1e999'],
            'epochs': ['', 'abc', '0', '-1', '1.5', '9' * 400],
            'early_stopping_patience': ['', '-1', '1.5'],
            'early_stopping_min_delta': ['', '-1', 'nan', 'inf'],
        }.items():
            for value in invalid:
                with self.subTest(field=field, value=value):
                    page = self.submit(**{field: value, 'early_stopping': 'on'})
                    self.assertIn(f'id="{field}-error"', page)
                    self.assertNotIn('id="generated-config"', page)

    def test_zero_patience_and_delta_and_scientific_rate_are_valid(self):
        page = self.submit(early_stopping='on', early_stopping_patience='0',
                           early_stopping_min_delta='0', learning_rate='1e-4')
        self.assertIn('--learning_rate=0.0001', page)
        self.assertIn('--early_stopping_patience=0', page)
        self.assertIn('--early_stopping_min_delta=0.0', page)

    def test_disabled_early_stopping_ignores_its_invalid_values(self):
        page = self.submit(early_stopping_monitor='unsupported',
                           early_stopping_patience='bad', early_stopping_min_delta='nan')
        self.assertIn('id="generated-config"', page)
        self.assertNotIn('--early_stopping', ConfigParser(page).config)
        self.assertIn('value="bad"', page)

    def test_unsupported_choices_and_config_injection_are_rejected(self):
        for change in [
            {'loss': 'unsupported'}, {'metrics': ['mse', '--epochs=1']},
            {'early_stopping': 'on', 'early_stopping_monitor': 'unsupported'},
            {'experiment_name': ''}, {'experiment_name': '../outside'},
            {'experiment_name': 'demo\n--epochs=1'},
        ]:
            with self.subTest(change=change):
                page = self.submit(**change)
                self.assertIn('id="form-errors"', page)
                self.assertNotIn('id="generated-config"', page)

    def test_errors_keep_other_entries_and_escape_html(self):
        page = self.submit(experiment_name='<script>alert(1)</script>', epochs='0', metrics=['mae'])
        self.assertIn('id="experiment_name-error"', page)
        self.assertIn('id="epochs-error"', page)
        self.assertIn('value="0.001"', page)
        self.assertIn('value="mae" checked', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertNotIn('<script>alert(1)</script>', page)

    def test_empty_post_is_invalid_and_fresh_get_resets(self):
        page = self.client.post('/experiment', data={}).get_data(as_text=True)
        self.assertIn('id="form-errors"', page)
        self.assertNotIn('id="generated-config"', page)
        self.submit(experiment_name='custom')
        fresh = self.client.get('/experiment').get_data(as_text=True)
        self.assertIn('value="experiment"', fresh)
        self.assertNotIn('value="custom"', fresh)

    def test_duplicate_scalar_fields_are_rejected(self):
        form = MultiDict(VALID_FORM)
        form.add('epochs', '1')
        page = self.client.post('/experiment', data=form).get_data(as_text=True)
        self.assertIn('id="epochs-error"', page)
        self.assertNotIn('id="generated-config"', page)

    def test_generator_checks_values_even_when_called_directly(self):
        _, args, errors = prepare_experiment_form(MultiDict(VALID_FORM))
        self.assertFalse(errors)
        for change in [{'learning_rate': float('nan')}, {'epochs': 1.5},
                       {'epochs': True}, {'loss': 'bad'}, {'metrics': ['bad']}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                generate_experiment_config({**args, **change})


if __name__ == '__main__':
    unittest.main()
