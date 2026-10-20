import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import app


class TestNetworkForm(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.fields = dict(network_type="fully_connected", input_shape="34", hidden_units="64, 32",
                           hidden_activation="elu", output_shape="10, 20",
                           output_activation="linear")

    def test_form_and_copy_controls(self):
        page = self.client.get("/").get_data(as_text=True)
        for name in ("input_shape", "output_shape", "hidden_units"):
            tag = re.search(r'<input[^>]*name="' + name + r'"[^>]*>', page).group()
            self.assertIn('type="text"', tag)
            self.assertEqual("required" in tag, name != "hidden_units")
        self.assertIn("tooltip", page)
        self.assertIn("navigator.clipboard.writeText(configText)", page)

    def test_valid_submissions(self):
        for input_shape, output_shape in (("34", "1"), ("128", "10, 20")):
            for hidden in ("64, 32", "", "  ", None):
                with self.subTest(input_shape=input_shape, hidden=hidden):
                    fields = dict(self.fields, input_shape=input_shape, output_shape=output_shape)
                    if hidden is None:
                        del fields["hidden_units"]
                    else:
                        fields["hidden_units"] = hidden
                    response = self.client.post("/", data=fields)
                    self.assertEqual(response.status_code, 200)
                    page = response.get_data(as_text=True)
                    self.assertIn('--input_shape\n' + '\n'.join(value.strip() for value in input_shape.split(',')), page)
                    self.assertIn('--output_shape\n' + '\n'.join(value.strip() for value in output_shape.split(',')), page)
                    self.assertIn('id="copy-button"', page)
                    self.assertNotIn('class="error-message"', page)
                    if not hidden or not hidden.strip():
                        self.assertIn("--number_hidden_units\n--hidden_activation=elu", page)

    def test_batch_normalization_submissions(self):
        cases = (
            (),
            ("batch_normalization",),
            ("batch_normalization_input",),
            ("batch_normalization", "batch_normalization_input"),
        )
        for selected_fields in cases:
            with self.subTest(selected=selected_fields):
                fields = dict(self.fields)
                # Browsers omit unchecked checkboxes from form submissions.
                fields.update({name: "on" for name in selected_fields})
                response = self.client.post("/", data=fields)
                self.assertEqual(response.status_code, 200)
                page = response.get_data(as_text=True)
                self.assertNotIn('class="error-message"', page)
                config = re.search(r'<pre id="config-text">(.*?)</pre>', page, re.DOTALL)
                self.assertIsNotNone(config)
                flags = [line for line in config.group(1).splitlines()
                         if line.startswith("--batch_normalization")]
                self.assertEqual(flags, ["--" + name for name in selected_fields])

    def test_dropout_field(self):
        page = self.client.get("/").get_data(as_text=True)
        tag = re.search(r'<input[^>]*name="dropout"[^>]*>', page).group()
        for attribute in ('type="number"', 'min="0"', 'max="0.9999999999999999"', 'step="any"'):
            self.assertIn(attribute, tag)
        self.assertNotIn("required", tag)
        self.assertIn("0.2 means 20%", page)

    def test_valid_dropout_submissions(self):
        for value, expected in (("0", "0.0"), ("0.2", "0.2"), ("0.9999999999999999", "0.9999999999999999"),
                                (" 0.25 ", "0.25")):
            with self.subTest(dropout=value):
                response = self.client.post("/", data=dict(self.fields, dropout=value))
                self.assertEqual(response.status_code, 200)
                page = response.get_data(as_text=True)
                self.assertNotIn('class="error-message"', page)
                config = re.search(r'<pre id="config-text">(.*?)</pre>', page, re.DOTALL)
                self.assertIsNotNone(config)
                self.assertEqual(
                    [line for line in config.group(1).splitlines() if line.startswith("--dropout")],
                    ["--dropout=" + expected],
                )

    def test_blank_or_missing_dropout(self):
        for value in (None, "", "   "):
            with self.subTest(dropout=value):
                fields = dict(self.fields)
                if value is not None:
                    fields["dropout"] = value
                page = self.client.post("/", data=fields).get_data(as_text=True)
                self.assertNotIn('class="error-message"', page)
                self.assertIn('id="config-text"', page)
                self.assertNotIn("--dropout", page)

    def test_invalid_dropout_submissions(self):
        for value in ("-0.01", "1", "1.0", "1.01", "abc", "0,2", "nan", "inf", "-inf", "1e309"):
            with self.subTest(dropout=value):
                page = self.client.post("/", data=dict(self.fields, dropout=value)).get_data(as_text=True)
                self.assertIn('class="error-message"', page)
                self.assertIn("dropout", page.lower())
                self.assertNotIn('id="config-text"', page)

    def test_input_dropout_field(self):
        page = self.client.get("/").get_data(as_text=True)
        tag = re.search(r'<input[^>]*name="dropout_input"[^>]*>', page).group()
        for attribute in ('type="number"', 'min="0"', 'max="0.9999999999999999"', 'step="any"'):
            self.assertIn(attribute, tag)
        self.assertNotIn("required", tag)
        self.assertIn("Fraction of input features", page)

    def test_input_dropout_submissions_and_combinations(self):
        cases = (
            ({"dropout_input": "0"}, ["--dropout_input=0.0"]),
            ({"dropout_input": " 0.25 "}, ["--dropout_input=0.25"]),
            ({"dropout_input": "0.9999999999999999"}, ["--dropout_input=0.9999999999999999"]),
            ({"dropout_input": "0.2", "dropout": "0.3"},
             ["--dropout=0.3", "--dropout_input=0.2"]),
            ({"dropout_input": "0", "dropout": "0"},
             ["--dropout=0.0", "--dropout_input=0.0"]),
        )
        for rates, expected in cases:
            with self.subTest(rates=rates):
                response = self.client.post("/", data=dict(self.fields, **rates))
                self.assertEqual(response.status_code, 200)
                page = response.get_data(as_text=True)
                self.assertNotIn('class="error-message"', page)
                config = re.search(r'<pre id="config-text">(.*?)</pre>', page, re.DOTALL)
                self.assertIsNotNone(config)
                self.assertEqual(
                    [line for line in config.group(1).splitlines() if line.startswith("--dropout")],
                    expected,
                )

    def test_blank_or_missing_input_dropout(self):
        for value in (None, "", "   "):
            for dropout in ("", "0.3"):
                with self.subTest(dropout_input=value, dropout=dropout):
                    fields = dict(self.fields, dropout=dropout)
                    if value is not None:
                        fields["dropout_input"] = value
                    page = self.client.post("/", data=fields).get_data(as_text=True)
                    self.assertNotIn('class="error-message"', page)
                    self.assertIn('id="config-text"', page)
                    self.assertNotIn("--dropout_input", page)
                    if dropout:
                        self.assertIn("--dropout=0.3", page)

    def test_invalid_input_dropout_submissions(self):
        for value in ("-0.01", "1", "1.0", "1.01", "abc", "0,2", "nan", "inf", "-inf", "1e309"):
            with self.subTest(dropout_input=value):
                page = self.client.post("/", data=dict(self.fields, dropout_input=value)).get_data(as_text=True)
                self.assertIn('class="error-message"', page)
                self.assertRegex(page.lower(), r"input dropout|dropout_input")
                self.assertNotIn('id="config-text"', page)
                tag = re.search(r'<input[^>]*name="dropout_input"[^>]*>', page).group()
                self.assertIn(f'value="{value}"', tag)

    def test_input_dropout_preserved_after_other_validation_errors(self):
        for invalid_fields in ({"input_shape": "0"}, {"dropout": "abc"}, {"dropout": "1.1"}):
            with self.subTest(invalid_fields=invalid_fields):
                fields = dict(self.fields, dropout_input="0.250", **invalid_fields)
                page = self.client.post("/", data=fields).get_data(as_text=True)
                self.assertIn('class="error-message"', page)
                self.assertNotIn('id="config-text"', page)
                tag = re.search(r'<input[^>]*name="dropout_input"[^>]*>', page).group()
                self.assertIn('value="0.250"', tag)

    def test_regularization_fields(self):
        page = self.client.get("/").get_data(as_text=True)
        for name in ("L1_regularization", "L2_regularization"):
            with self.subTest(argument=name):
                tag = re.search(r'<input[^>]*name="' + name + r'"[^>]*>', page).group()
                for attribute in ('type="number"', 'min="0"', 'step="any"'):
                    self.assertIn(attribute, tag)
                self.assertNotIn("required", tag)
                self.assertNotIn("max=", tag)

    def test_regularization_submissions(self):
        for name in ("L1_regularization", "L2_regularization"):
            for value, expected in (("0", "0.0"), (" 0.001 ", "0.001"),
                                    ("1e-8", "1e-08"), ("2.5", "2.5")):
                with self.subTest(argument=name, value=value):
                    response = self.client.post("/", data=dict(self.fields, **{name: value}))
                    self.assertEqual(response.status_code, 200)
                    page = response.get_data(as_text=True)
                    self.assertNotIn('class="error-message"', page)
                    config = re.search(r'<pre id="config-text">(.*?)</pre>', page, re.DOTALL)
                    self.assertIsNotNone(config)
                    self.assertEqual(
                        [line for line in config.group(1).splitlines()
                         if line.startswith(("--L1", "--L2"))],
                        [f"--{name}={expected}"],
                    )

    def test_regularization_omission_and_combinations(self):
        cases = (
            ({}, []),
            ({"L1_regularization": "", "L2_regularization": "   "}, []),
            ({"L1_regularization": "0.001", "L2_regularization": ""}, ["--L1_regularization=0.001"]),
            ({"L1_regularization": "   ", "L2_regularization": "0.002"}, ["--L2_regularization=0.002"]),
            ({"L1_regularization": "0", "L2_regularization": ""}, ["--L1_regularization=0.0"]),
            ({"L1_regularization": "", "L2_regularization": "0"}, ["--L2_regularization=0.0"]),
        )
        for values, expected in cases:
            with self.subTest(values=values):
                page = self.client.post("/", data=dict(self.fields, **values)).get_data(as_text=True)
                self.assertNotIn('class="error-message"', page)
                config = re.search(r'<pre id="config-text">(.*?)</pre>', page, re.DOTALL)
                self.assertIsNotNone(config)
                self.assertEqual(
                    [line for line in config.group(1).splitlines() if line.startswith(("--L1", "--L2"))],
                    expected,
                )

    def test_invalid_regularization_submissions(self):
        for name in ("L1_regularization", "L2_regularization"):
            for value in ("-0.001", "abc", "0,1", "nan", "inf", "-inf", "1e309"):
                with self.subTest(argument=name, value=value):
                    page = self.client.post("/", data=dict(self.fields, **{name: value})).get_data(as_text=True)
                    self.assertIn('class="error-message"', page)
                    self.assertIn(name[:2], page.split('class="error-message"')[1])
                    self.assertNotIn('id="config-text"', page)
                    tag = re.search(r'<input[^>]*name="' + name + r'"[^>]*>', page).group()
                    self.assertIn(f'value="{value}"', tag)

    def test_regularization_preserved_after_validation_errors(self):
        for invalid in ({"input_shape": "0"}, {"dropout": "abc"},
                        {"L1_regularization": "-0.01"}, {"L2_regularization": "abc"}):
            with self.subTest(invalid=invalid):
                fields = dict(self.fields, L1_regularization="0.0010", L2_regularization="0.0020")
                fields.update(invalid)
                page = self.client.post("/", data=fields).get_data(as_text=True)
                self.assertIn('class="error-message"', page)
                self.assertNotIn('id="config-text"', page)
                for name in ("L1_regularization", "L2_regularization"):
                    tag = re.search(r'<input[^>]*name="' + name + r'"[^>]*>', page).group()
                    self.assertIn(f'value="{fields[name]}"', tag)

    def test_combined_regularization_rejected(self):
        for l1, l2 in (("0.001", "0.002"), ("0", "0.002"), ("0.001", "0"), ("0", "0")):
            with self.subTest(l1=l1, l2=l2):
                page = self.client.post("/", data=dict(
                    self.fields, L1_regularization=l1, L2_regularization=l2,
                )).get_data(as_text=True)
                self.assertIn('class="error-message"', page)
                self.assertIn("L1 and L2 regularization cannot be supplied together", page)
                self.assertIn("Leave one field blank; zero still counts as supplied", page)
                self.assertNotIn('id="config-text"', page)
                for name, value in (("L1_regularization", l1), ("L2_regularization", l2)):
                    tag = re.search(r'<input[^>]*name="' + name + r'"[^>]*>', page).group()
                    self.assertIn(f'value="{value}"', tag)

    def test_multidimensional_input_rejected(self):
        for shape in ("10, 20", "128, 128, 3"):
            with self.subTest(shape=shape):
                page = self.client.post("/", data=dict(self.fields, input_shape=shape)).get_data(as_text=True)
                self.assertIn('class="error-message"', page)
                self.assertIn("exactly one positive integer", page)
                self.assertNotIn('id="config-text"', page)

    def test_activation_choices_match_gui_and_accept_submissions(self):
        from config_rules import ACTIVATION_CHOICES

        page = self.client.get("/").get_data(as_text=True)
        for name, choices in ACTIVATION_CHOICES.items():
            select = re.search(r'<select[^>]*name="' + name + r'"[^>]*>(.*?)</select>', page, re.DOTALL).group(1)
            self.assertEqual(tuple(re.findall(r'<option value="([^"]+)"', select)), choices)
            for choice in choices:
                with self.subTest(argument=name, choice=choice):
                    result = self.client.post("/", data=dict(self.fields, **{name: choice})).get_data(as_text=True)
                    self.assertNotIn('class="error-message"', result)
                    self.assertIn(f"--{name}={choice}", result)

    def test_crafted_activation_submissions_rejected(self):
        for name in ("hidden_activation", "output_activation"):
            for value in ("unsupported", "", " ", "ReLU", "relu\n--dropout=0.2"):
                with self.subTest(argument=name, value=value):
                    page = self.client.post("/", data=dict(self.fields, **{name: value})).get_data(as_text=True)
                    self.assertIn('class="error-message"', page)
                    self.assertIn(name, page.split('class="error-message"')[1])
                    self.assertNotIn('id="config-text"', page)

    def test_invalid_submissions(self):
        for field in ("input_shape", "output_shape", "hidden_units"):
            values = ["0", "-4", "1,0", "1.5", "abc", "1,", "1,,2"]
            if field != "hidden_units":
                values += ["", " "]
            for value in values:
                with self.subTest(field=field, value=value):
                    page = self.client.post("/", data=dict(self.fields, **{field: value})).get_data(as_text=True)
                    self.assertIn('class="error-message"', page)
                    self.assertNotIn('id="config-text"', page)

    def test_missing_required_field(self):
        del self.fields["input_shape"]
        response = self.client.post("/", data=self.fields)
        self.assertEqual(response.status_code, 200)
        self.assertIn('class="error-message"', response.get_data(as_text=True))

    def test_network_type_selector_present(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('name="network_type"', page)
        self.assertIn('value="fully_connected"', page)
