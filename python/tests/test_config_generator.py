import sys
import unittest
from pathlib import Path

# Allow imports from the parent python/ directory when tests are run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_generator import generate_network_config, validate_hidden_units, validate_shape, parse_integer_list


class TestConfigGenerator(unittest.TestCase):
    def test_validate_hidden_units(self):
        self.assertTrue(validate_hidden_units([64, 32]))
        self.assertTrue(validate_hidden_units([128]))

        self.assertFalse(validate_hidden_units([0, 32]))
        self.assertFalse(validate_hidden_units([-4, 32]))
        self.assertTrue(validate_hidden_units([]))
        self.assertFalse(validate_hidden_units([64, 3.5]))
        self.assertFalse(validate_hidden_units([True, 32]))

    def test_generate_network_config(self):
        config = generate_network_config(
            network_type="fully_connected",
            input_shape=[34],
            hidden_units=[64, 32],
            hidden_activation="elu",
            output_shape=[1],
            output_activation="elup1",
        )

        self.assertIn("--network_type=fully_connected", config)
        self.assertIn("--input_shape\n34", config)
        self.assertIn("--number_hidden_units\n64\n32", config)
        self.assertIn("--hidden_activation=elu", config)
        self.assertIn("--output_shape\n1", config)
        self.assertIn("--output_activation=elup1", config)

    def test_generate_network_config_rejects_invalid_hidden_units(self):
        with self.assertRaises(ValueError):
            generate_network_config(
                network_type="fully_connected",
                input_shape=[34],
                hidden_units=[64, 0],
                hidden_activation="elu",
                output_shape=[1],
                output_activation="elup1",
            )

    def test_generate_network_config_rejects_invalid_input_shape(self):
        with self.assertRaises(ValueError):
            generate_network_config(
                network_type="fully_connected",
                input_shape=[0],
                hidden_units=[64, 32],
                hidden_activation="elu",
                output_shape=[1],
                output_activation="elup1",
            )

    def test_shape_validation(self):
        for shape in ([34], [128, 128, 3], [10, 20]):
            with self.subTest(shape=shape):
                self.assertTrue(validate_shape(shape))
        for shape in ([], [0], [-1], [10, 0], [10, -20], [True],
                      [1.5], ["3"], 34, (34,), None):
            with self.subTest(shape=shape):
                self.assertFalse(validate_shape(shape))
                for field in ("input_shape", "output_shape"):
                    arguments = dict(network_type="fully_connected", input_shape=[34],
                                     hidden_units=[], hidden_activation="elu",
                                     output_shape=[1], output_activation="linear")
                    arguments[field] = shape
                    with self.assertRaises(ValueError):
                        generate_network_config(**arguments)

    def test_invalid_hidden_units(self):
        for units in ([0, 32], [-4], [3.5], [True], ["64"], None, 64, (64,)):
            with self.subTest(units=units):
                self.assertFalse(validate_hidden_units(units))
                with self.assertRaises(ValueError):
                    generate_network_config("fully_connected", [34], units, "elu", [1], "linear")

    def test_multidimensional_config(self):
        self.assertEqual(
            generate_network_config("fully_connected", [128, 128, 3], [64, 32], "elu", [10, 20], "linear"),
            "--network_type=fully_connected\n--input_shape\n128\n128\n3\n"
            "--number_hidden_units\n64\n32\n--hidden_activation=elu\n"
            "--output_shape\n10\n20\n--output_activation=linear\n",
        )

    def test_zero_layers_config(self):
        config = generate_network_config("fully_connected", [34], [], "elu", [1], "linear")
        self.assertIn("--number_hidden_units\n--hidden_activation=elu\n", config)
        self.assertNotIn("\n\n", config)

    def test_form_list_parsing(self):
        self.assertEqual(parse_integer_list("128, 128, 3", "Shape"), [128, 128, 3])
        self.assertEqual(parse_integer_list("34", "Shape"), [34])
        self.assertEqual(parse_integer_list("  ", "Layers", allow_empty=True), [])
        for value in ("", " ", "1,", ",1", "1,,3", "1.5", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_integer_list(value, "Shape")


if __name__ == "__main__":
    unittest.main()
