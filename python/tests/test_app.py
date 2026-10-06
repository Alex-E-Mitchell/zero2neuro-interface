import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import app


class TestNetworkForm(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.fields = dict(network_type="fully_connected", input_shape="128, 128, 3", hidden_units="64, 32",
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
        for input_shape, output_shape in (("34", "1"), ("128, 128, 3", "10, 20")):
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
