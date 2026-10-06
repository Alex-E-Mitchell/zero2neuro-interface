import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser_metadata import extract_argument_metadata
from config_generator import generate_network_config


def network_parser():
    """Small fixture matching inspected Zero2Neuro declarations, no heavy imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--network_type", type=str, default=None)
    parser.add_argument("--input_shape0", "--input_shape", type=int, nargs="+", default=[10], help="Input dimensions")
    parser.add_argument("--output_shape0", "--output_shape", type=int, nargs="+", default=[10])
    parser.add_argument("--number_hidden_units", type=int, nargs="*", default=None)
    parser.add_argument("--hidden_activation", type=str, default="elu")
    parser.add_argument("--output_activation", type=str, default=None)
    return parser


class TestParserMetadata(unittest.TestCase):
    def test_metadata(self):
        records = extract_argument_metadata(network_parser())
        shape = next(record for record in records if record["dest"] == "input_shape0")
        self.assertEqual(shape["option_strings"], ["--input_shape0", "--input_shape"])
        self.assertIs(shape["type"], int)
        self.assertEqual(shape["nargs"], "+")
        self.assertEqual(shape["default"], [10])
        self.assertEqual(shape["help"], "Input dimensions")
        self.assertIsNone(shape["choices"])
        self.assertFalse(shape["required"])
        self.assertFalse(shape["is_boolean_flag"])
        hidden = next(record for record in records if record["dest"] == "number_hidden_units")
        self.assertEqual(hidden["nargs"], "*")
        self.assertIsNone(hidden["default"])

    def test_flags_choices_and_shared_destination(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--gpu", action="store_true")
        parser.add_argument("--no-gpu", action="store_false", dest="gpu")
        parser.add_argument("--feature", action=argparse.BooleanOptionalAction)
        parser.add_argument("--count", action="count")
        parser.add_argument("--mode", choices=["a", "b"], required=True)
        parser.add_argument("file", type=str)
        records = extract_argument_metadata(parser)
        self.assertEqual([record["dest"] for record in records[:2]], ["gpu", "gpu"])
        self.assertTrue(records[0]["is_store_true"])
        self.assertFalse(records[1]["is_store_true"])
        for record in records[:3]:
            self.assertTrue(record["is_boolean_flag"])
            self.assertEqual(record["nargs"], 0)
        self.assertFalse(records[3]["is_boolean_flag"])
        self.assertEqual(records[4]["choices"], ["a", "b"])
        self.assertTrue(records[4]["required"])
        self.assertEqual(records[5]["option_strings"], [])

    def test_does_not_mutate_parser(self):
        parser = network_parser()
        records = extract_argument_metadata(parser)
        shape = next(record for record in records if record["dest"] == "input_shape0")
        shape["default"].append(99)
        shape["option_strings"].clear()
        self.assertEqual(parser.parse_args([]).input_shape0, [10])
        self.assertEqual(parser.parse_args(["--input_shape", "34"]).input_shape0, [34])

    def test_generated_text_round_trip(self):
        for input_shape, output_shape in (([34], [1]), ([128, 128, 3], [10, 20])):
            for hidden in ([], [128], [64, 32]):
                with self.subTest(input_shape=input_shape, output_shape=output_shape, hidden=hidden):
                    config = generate_network_config("fully_connected", input_shape, hidden, "elu", output_shape, "linear")
                    parsed = network_parser().parse_args(config.splitlines())
                    self.assertEqual(parsed.input_shape0, input_shape)
                    self.assertEqual(parsed.output_shape0, output_shape)
                    self.assertEqual(parsed.number_hidden_units, hidden)
