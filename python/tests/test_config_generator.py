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

    def test_batch_normalization_flags(self):
        cases = (
            (False, False, []),
            (True, False, ["--batch_normalization"]),
            (False, True, ["--batch_normalization_input"]),
            (True, True, ["--batch_normalization", "--batch_normalization_input"]),
        )
        for hidden_enabled, input_enabled, expected_flags in cases:
            with self.subTest(hidden=hidden_enabled, input=input_enabled):
                config = generate_network_config(
                    network_type="fully_connected",
                    input_shape=[34],
                    hidden_units=[64, 32],
                    hidden_activation="elu",
                    output_shape=[1],
                    output_activation="linear",
                    batch_normalization=hidden_enabled,
                    batch_normalization_input=input_enabled,
                )
                flags = [line for line in config.splitlines()
                         if line.startswith("--batch_normalization")]
                self.assertEqual(flags, expected_flags)

    def test_dropout_output(self):
        for value in (0, 0.0, 0.2, 0.999, 0.9999999999999999):
            with self.subTest(dropout=value):
                config = generate_network_config(
                    "fully_connected", [34], [64, 32], "elu", [1], "linear",
                    dropout=value,
                )
                self.assertEqual(
                    [line for line in config.splitlines() if line.startswith("--dropout")],
                    [f"--dropout={float(value)}"],
                )

    def test_dropout_omitted_by_default_or_none(self):
        arguments = ("fully_connected", [34], [64], "elu", [1], "linear")
        self.assertNotIn("--dropout", generate_network_config(*arguments))
        self.assertNotIn("--dropout", generate_network_config(*arguments, dropout=None))

    def test_invalid_dropout(self):
        for value in (-0.01, 1.0, 1.01, float("nan"), float("inf"), float("-inf"),
                      True, False, "0.2", "", [], {}):
            with self.subTest(dropout=value):
                with self.assertRaisesRegex(ValueError, "dropout"):
                    generate_network_config(
                        "fully_connected", [34], [64], "elu", [1], "linear",
                        dropout=value,
                    )

    def test_input_dropout_output_and_combinations(self):
        cases = (
            (None, None, []),
            (0, None, ["--dropout_input=0.0"]),
            (0.2, None, ["--dropout_input=0.2"]),
            (0.999, None, ["--dropout_input=0.999"]),
            (0.9999999999999999, None, ["--dropout_input=0.9999999999999999"]),
            (None, 0.3, ["--dropout=0.3"]),
            (0.2, 0.3, ["--dropout=0.3", "--dropout_input=0.2"]),
            (0, 0, ["--dropout=0.0", "--dropout_input=0.0"]),
        )
        for input_rate, hidden_rate, expected in cases:
            with self.subTest(dropout_input=input_rate, dropout=hidden_rate):
                config = generate_network_config(
                    "fully_connected", [34], [64], "elu", [1], "linear",
                    dropout_input=input_rate, dropout=hidden_rate,
                )
                self.assertEqual(
                    [line for line in config.splitlines() if line.startswith("--dropout")],
                    expected,
                )

    def test_input_dropout_omitted_by_default(self):
        config = generate_network_config("fully_connected", [34], [64], "elu", [1], "linear")
        self.assertNotIn("--dropout_input", config)

    def test_invalid_input_dropout(self):
        for value in (-0.01, 1.0, 1.01, float("nan"), float("inf"), float("-inf"),
                      True, False, "0.2", "", [], {}):
            with self.subTest(dropout_input=value):
                with self.assertRaisesRegex(ValueError, "dropout_input"):
                    generate_network_config(
                        "fully_connected", [34], [64], "elu", [1], "linear",
                        dropout_input=value, dropout=0.3,
                    )

    def test_regularization_output(self):
        for name in ("L1_regularization", "L2_regularization"):
            for value in (0, 0.001, 1e-8, 2.5, 1e100):
                with self.subTest(argument=name, value=value):
                    config = generate_network_config(
                        "fully_connected", [34], [64], "elu", [1], "linear", **{name: value},
                    )
                    self.assertEqual(
                        [line for line in config.splitlines() if line.startswith(("--L1", "--L2"))],
                        [f"--{name}={float(value)}"],
                    )

    def test_regularization_omission(self):
        arguments = ("fully_connected", [34], [64], "elu", [1], "linear")
        for optional in ({}, {"L1_regularization": None, "L2_regularization": None}):
            with self.subTest(optional=optional):
                config = generate_network_config(*arguments, **optional)
                self.assertNotIn("--L1_regularization", config)
                self.assertNotIn("--L2_regularization", config)

    def test_regularization_together_with_dropout(self):
        for l1, l2 in ((0.001, None), (None, 0.002), (0, None), (None, 0)):
            with self.subTest(l1=l1, l2=l2):
                config = generate_network_config(
                    "fully_connected", [34], [64], "elu", [1], "linear",
                    L1_regularization=l1, L2_regularization=l2, dropout=0.3, dropout_input=0.2,
                )
                expected = ["--dropout=0.3", "--dropout_input=0.2"]
                if l1 is not None:
                    expected.append(f"--L1_regularization={float(l1)}")
                if l2 is not None:
                    expected.append(f"--L2_regularization={float(l2)}")
                self.assertEqual(
                    [line for line in config.splitlines()
                     if line.startswith(("--L1", "--L2", "--dropout"))],
                    expected,
                )

    def test_combined_regularization_rejected(self):
        for l1, l2 in ((0.001, 0.002), (0, 0.002), (0.001, 0), (0, 0)):
            with self.subTest(l1=l1, l2=l2):
                with self.assertRaisesRegex(ValueError, "Leave one field blank; zero still counts"):
                    generate_network_config(
                        "fully_connected", [34], [64], "elu", [1], "linear",
                        L1_regularization=l1, L2_regularization=l2,
                    )

    def test_fully_connected_input_requires_one_dimension(self):
        for shape in ([10, 20], [128, 128, 3]):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "exactly one positive integer"):
                    generate_network_config("fully_connected", shape, [64], "elu", [1], "linear")

    def test_supported_activations(self):
        for hidden in ("elu", "relu", "sigmoid", "tanh"):
            for output in ("elup1", "linear", "sigmoid", "softmax"):
                with self.subTest(hidden=hidden, output=output):
                    config = generate_network_config("fully_connected", [34], [64], hidden, [10, 20], output)
                    self.assertIn(f"--hidden_activation={hidden}\n", config)
                    self.assertIn(f"--output_activation={output}\n", config)

    def test_invalid_activations(self):
        for name in ("hidden_activation", "output_activation"):
            for value in ("unsupported", "", " ", None, "ReLU", "relu\n--dropout=0.2"):
                with self.subTest(argument=name, value=value):
                    arguments = dict(network_type="fully_connected", input_shape=[34], hidden_units=[],
                                     hidden_activation="elu", output_shape=[1], output_activation="linear")
                    arguments[name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        generate_network_config(**arguments)

    def test_invalid_regularization(self):
        for name in ("L1_regularization", "L2_regularization"):
            for value in (-0.001, float("nan"), float("inf"), float("-inf"),
                          True, False, "0.001", "", [], {}):
                with self.subTest(argument=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        generate_network_config(
                            "fully_connected", [34], [64], "elu", [1], "linear", **{name: value},
                        )

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

    def test_multidimensional_output_config(self):
        self.assertEqual(
            generate_network_config("fully_connected", [34], [64, 32], "elu", [10, 20], "linear"),
            "--network_type=fully_connected\n--input_shape\n34\n"
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
