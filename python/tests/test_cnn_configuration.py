"""CNN generation/form contracts; these tests do not build or train a model."""
import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import app
from config_generator import generate_network_config
from config_rules import validate_cnn_arguments


def cnn_arguments(**overrides):
    arguments = dict(
        network_type="cnn", input_shape=[32, 32, 3], hidden_units=[],
        hidden_activation="elu", output_shape=[2], output_activation="softmax",
        conv_number_filters=[8, 16], conv_kernel_size=[3, 3], conv_pool_size=[0, 2],
        conv_padding="valid", conv_activation="elu",
    )
    return dict(arguments, **overrides)


class TestCnnGenerator(unittest.TestCase):
    def test_exact_config_text(self):
        self.assertEqual(generate_network_config(**cnn_arguments()), """--network_type=cnn
--input_shape
32
32
3
--conv_number_filters
8
16
--conv_kernel_size
3
3
--conv_pool_size
0
2
--conv_padding=valid
--conv_activation=elu
--number_hidden_units
--hidden_activation=elu
--output_shape
2
--output_activation=softmax
""")

    def test_valid_spatial_ranks_and_choices(self):
        for shape in ([100, 2], [32, 24, 3], [16, 16, 16, 1]):
            for padding in ("valid", "same"):
                for activation in ("elu", "relu", "sigmoid", "tanh"):
                    with self.subTest(shape=shape, padding=padding, activation=activation):
                        text = generate_network_config(**cnn_arguments(
                            input_shape=shape, conv_padding=padding, conv_activation=activation))
                        self.assertIn(f"--conv_activation={activation}\n", text)
                        self.assertIn(f"--conv_padding={padding}\n", text)

    def test_invalid_input_shapes(self):
        for shape in (None, [], [34], [2, 2, 2, 2, 1], [0, 3], [-2, 3],
                      [2.5, 3], [True, 3], "32, 3", (32, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                generate_network_config(**cnn_arguments(input_shape=shape))

    def test_invalid_layer_lists(self):
        for name in ("conv_number_filters", "conv_kernel_size", "conv_pool_size"):
            for value in (None, [], "3, 3", (3, 3), [True, 3], [3.5, 3],
                          [-1, 3], [None, 3], [float("inf"), 3], [float("nan"), 3]):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    generate_network_config(**cnn_arguments(**{name: value}))
        for name in ("conv_number_filters", "conv_kernel_size"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                generate_network_config(**cnn_arguments(**{name: [0, 3]}))

    def test_equal_layer_lengths_required(self):
        for name in ("conv_number_filters", "conv_kernel_size", "conv_pool_size"):
            for value in ([3], [3, 3, 3]):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "same number of layers"):
                        generate_network_config(**cnn_arguments(**{name: value}))

    def test_invalid_choices_and_injection(self):
        for name in ("conv_padding", "conv_activation"):
            for value in (None, "", "unknown", "same\n--force", [], True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        generate_network_config(**cnn_arguments(**{name: value}))

    def test_pool_zero_and_one_skip_pooling(self):
        for value in (0, 1):
            with self.subTest(pool=value):
                text = generate_network_config(**cnn_arguments(
                    input_shape=[1, 1], conv_number_filters=[1],
                    conv_kernel_size=[1], conv_pool_size=[value]))
                self.assertIn(f"--conv_pool_size\n{value}\n", text)

    def test_spatial_collapse_and_boundaries(self):
        cases = (
            ([2, 3], [3], [0], "conv_kernel_size"),
            ([3, 3], [3], [2], "conv_pool_size"),
            ([5, 3], [3, 3, 3], [0, 0, 0], "conv_kernel_size"),
            ([7, 3], [1, 1, 1], [2, 2, 2], "conv_pool_size"),
            ([32, 2, 3], [3], [0], "conv_kernel_size"),
        )
        for shape, kernels, pools, field in cases:
            arguments = cnn_arguments(input_shape=shape, conv_number_filters=[1] * len(kernels),
                                      conv_kernel_size=kernels, conv_pool_size=pools)
            with self.subTest(arguments=arguments):
                self.assertIn(field, validate_cnn_arguments(shape, arguments))
                with self.assertRaisesRegex(ValueError, "layer"):
                    generate_network_config(**arguments)
                # SAME padding permits kernels/pools larger than the remaining domain.
                generate_network_config(**dict(arguments, conv_padding="same"))
        generate_network_config(**cnn_arguments(input_shape=[4, 1], conv_number_filters=[1],
                                               conv_kernel_size=[3], conv_pool_size=[2]))

    def test_same_pooling_rounds_up_before_next_convolution(self):
        # With SAME the 3 cells become 2, then 1; they must never round down to zero.
        generate_network_config(**cnn_arguments(input_shape=[3, 1], conv_pool_size=[2, 2],
                                               conv_padding="same"))

    def test_dense_settings_and_common_validation(self):
        text = generate_network_config(**cnn_arguments(
            hidden_units=[64, 32], output_shape=[2, 3], hidden_activation="tanh",
            batch_normalization=True, dropout=0.2, L2_regularization=0.001))
        for fragment in ("--number_hidden_units\n64\n32\n", "--output_shape\n2\n3\n",
                         "--batch_normalization\n", "--dropout=0.2\n", "--L2_regularization=0.001\n"):
            self.assertIn(fragment, text)
        for overrides in (dict(hidden_units=[0]), dict(output_shape=[]), dict(hidden_activation="bad"),
                          dict(output_activation="bad"), dict(dropout=1), dict(L1_regularization=-1),
                          dict(L1_regularization=0, L2_regularization=0)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                generate_network_config(**cnn_arguments(**overrides))

    def test_generated_text_parser_contract(self):
        # Lightweight fixture follows inspected upstream types/nargs/aliases.
        # It verifies serialization only, not compatibility with a running model.
        parser = argparse.ArgumentParser()
        for name in ("network_type", "hidden_activation", "output_activation", "conv_padding", "conv_activation"):
            parser.add_argument("--" + name)
        parser.add_argument("--input_shape", dest="input_shape0", nargs="+", type=int)
        parser.add_argument("--output_shape", dest="output_shape0", nargs="+", type=int)
        parser.add_argument("--number_hidden_units", nargs="*", type=int)
        for name in ("conv_number_filters", "conv_kernel_size", "conv_pool_size"):
            parser.add_argument("--" + name, nargs="+", type=int)
        for units in ([], [64, 32]):
            with self.subTest(units=units):
                result = parser.parse_args(generate_network_config(**cnn_arguments(hidden_units=units)).splitlines())
                self.assertEqual(result.input_shape0, [32, 32, 3])
                self.assertEqual(result.output_shape0, [2])
                self.assertEqual(result.number_hidden_units, units)
                self.assertEqual(result.conv_number_filters, [8, 16])
                self.assertEqual(result.conv_kernel_size, [3, 3])
                self.assertEqual(result.conv_pool_size, [0, 2])
                self.assertEqual(result.conv_padding, "valid")
                self.assertEqual(result.conv_activation, "elu")


class TestCnnForm(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        self.fields = dict(network_type="cnn", input_shape="32, 32, 3", hidden_units="",
                           hidden_activation="elu", output_shape="2", output_activation="softmax",
                           conv_number_filters="8, 16", conv_kernel_size="3, 3", conv_pool_size="0, 2")

    def test_valid_form_generates_and_saves_review_section(self):
        page = self.client.post("/", data=self.fields).get_data(as_text=True)
        self.assertIn('id="config-text"', page)
        self.assertIn(generate_network_config(**cnn_arguments()), page)
        self.assertNotIn('id="form-errors"', page)
        self.assertNotIn("under development", page)
        self.assertNotIn('id="fully-connected-fields" hidden disabled', page)
        self.assertIn("--network_type=cnn", self.client.get("/review").get_data(as_text=True))

    def test_invalid_fields_have_links_and_preserve_values(self):
        for name, value in (("input_shape", "34"), ("input_shape", "0, 3"),
                            ("conv_number_filters", ""), ("conv_number_filters", "0, 16"),
                            ("conv_number_filters", "8,,16"), ("conv_kernel_size", "3.5, 3"),
                            ("conv_kernel_size", "3,"), ("conv_pool_size", "-1, 2"),
                            ("conv_pool_size", "2"), ("conv_padding", "causal"),
                            ("conv_activation", "elup1")):
            with self.subTest(name=name, value=value):
                page = self.client.post("/", data=dict(self.fields, **{name: value})).get_data(as_text=True)
                self.assertNotIn('id="config-text"', page)
                self.assertIn(f'href="#{name}"', page)
                self.assertIn(f'id="{name}-error"', page)
                self.assertIn('value="32, 32, 3"' if name != "input_shape" else f'value="{value}"', page)
                if name in ("conv_padding", "conv_activation"):
                    self.assertIn('selected disabled>Choose an option', page)

    def test_invalid_cnn_clears_previous_saved_network(self):
        self.client.post("/", data=self.fields)
        self.client.post("/", data=dict(self.fields, conv_kernel_size="oops"))
        self.assertNotIn("--network_type=cnn", self.client.get("/review").get_data(as_text=True))

    def test_cnn_common_dense_fields_are_used(self):
        page = self.client.post("/", data=dict(self.fields, hidden_units="64, 32", dropout="0.2",
                                              L1_regularization="0.001", batch_normalization="on")).get_data(as_text=True)
        self.assertIn("--number_hidden_units\n64\n32\n", page)
        self.assertIn("--dropout=0.2", page)
        self.assertIn("--L1_regularization=0.001", page)
        self.assertIn("--batch_normalization\n", page)

    def test_fc_ignores_invalid_cnn_values(self):
        page = self.client.post("/", data=dict(self.fields, network_type="fully_connected", input_shape="34",
                                              conv_kernel_size="bad", conv_padding="bad",
                                              conv_activation="bad")).get_data(as_text=True)
        self.assertIn("--network_type=fully_connected", page)
        self.assertNotIn("--conv_", page)


if __name__ == "__main__":
    unittest.main()
