import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_rules import NETWORK_ARCHITECTURES, validate_required_arguments
from config_generator import generate_network_config


class TestConfigRules(unittest.TestCase):
    def test_registry_structure(self):
        self.assertEqual(set(NETWORK_ARCHITECTURES), {"fully_connected", "cnn"})
        for rules in NETWORK_ARCHITECTURES.values():
            self.assertEqual(set(rules), {"required", "optional"})
            self.assertIsInstance(rules["required"], set)
            self.assertIsInstance(rules["optional"], set)
            self.assertFalse(rules["required"] & rules["optional"])
            self.assertTrue({"network_type", "input_shape", "output_shape"} <= rules["required"])
            self.assertIn("number_hidden_units", rules["optional"])
        self.assertTrue({"conv_kernel_size", "conv_number_filters", "conv_pool_size"}
                        <= NETWORK_ARCHITECTURES["cnn"]["required"])

    def test_required_arguments_are_data_driven(self):
        for architecture, rules in NETWORK_ARCHITECTURES.items():
            supplied = dict.fromkeys(rules["required"], [])
            validate_required_arguments(architecture, supplied)
            for name in rules["required"]:
                with self.subTest(architecture=architecture, name=name):
                    missing = supplied.copy()
                    del missing[name]
                    with self.assertRaisesRegex(ValueError, name):
                        validate_required_arguments(architecture, missing)
                    missing[name] = None
                    with self.assertRaisesRegex(ValueError, name):
                        validate_required_arguments(architecture, missing)

    def test_generator_uses_registry(self):
        with self.assertRaisesRegex(ValueError, "Unknown network architecture"):
            generate_network_config("unknown", [34], [], "elu", [1], "linear")
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            generate_network_config("cnn", [128, 128, 3], [], "elu", [1], "linear")

    def test_string_choices(self):
        from config_rules import get_string_choices

        self.assertIn("tabular", get_string_choices("data_format"))
        self.assertIn("numpy", get_string_choices("data_representation"))
        self.assertIn("random", get_string_choices("data_fold_split"))
        self.assertEqual(get_string_choices("unknown"), ())

    def test_generalized_validation(self):
        from config_rules import validate_config_values

        rules = {
            "required": {"epochs", "learning_rate"},
            "minimum": {
                "epochs": 1,
                "learning_rate": 0.000001,
            },
        }

        valid = {
            "epochs": 100,
            "learning_rate": 0.001,
        }

        self.assertEqual(validate_config_values(valid, rules), [])

        invalid = {
            "epochs": 0,
            "learning_rate": -0.1,
        }

        errors = validate_config_values(invalid, rules)

        self.assertEqual(len(errors), 2)
        self.assertTrue(any("epochs" in error for error in errors))
        self.assertTrue(any("learning_rate" in error for error in errors))

        missing = validate_config_values({}, rules)
        self.assertEqual(len(missing), 2)

    def test_generalized_string_choices(self):
        from config_rules import (
            get_string_choices,
            validate_config_values,
        )

        rules = {
            "choices": {
                "data_format": get_string_choices("data_format"),
                "data_fold_split": get_string_choices("data_fold_split"),
            }
        }

        valid = {
            "data_format": "tabular",
            "data_fold_split": "random",
        }

        self.assertEqual(validate_config_values(valid, rules), [])

        invalid = {
            "data_format": "invalid-format",
            "data_fold_split": "unknown",
        }

        errors = validate_config_values(invalid, rules)

        self.assertEqual(len(errors), 2)
        self.assertTrue(any("data_format" in error for error in errors))
        self.assertTrue(any("data_fold_split" in error for error in errors))

    def test_required_empty_strings(self):
        from config_rules import validate_config_values

        rules = {"required": {"name"}}

        self.assertEqual(
            validate_config_values({"name": "Valid"}, rules),
            [],
        )

        self.assertEqual(
            len(validate_config_values({"name": ""}, rules)),
            1,
        )

        self.assertEqual(
            len(validate_config_values({"name": "   "}, rules)),
            1,
        )

    def test_maximum_validation(self):
        from config_rules import validate_config_values

        rules = {"maximum": {"rate": 1.0}}
        for value in (None, 0, 0.5, 1):
            with self.subTest(rate=value):
                self.assertEqual(validate_config_values({"rate": value}, rules), [])
        for value in (1.01, True, "0.5", float("nan"), float("inf"), float("-inf")):
            with self.subTest(rate=value):
                self.assertEqual(len(validate_config_values({"rate": value}, rules)), 1)

    def test_dropout_rules(self):
        from config_rules import DROPOUT_RULES, validate_config_values

        for name in ("dropout", "dropout_input"):
            for value in (None, 0, 0.25, 0.9999999999999999):
                with self.subTest(argument=name, value=value):
                    self.assertEqual(validate_config_values({name: value}, DROPOUT_RULES), [])
            for value in (-0.01, 1.0, 1.01, True, "0.2", float("nan"), float("inf")):
                with self.subTest(argument=name, value=value):
                    self.assertEqual(len(validate_config_values({name: value}, DROPOUT_RULES)), 1)

    def test_regularization_rules(self):
        from config_rules import REGULARIZATION_RULES, validate_config_values

        for name in ("L1_regularization", "L2_regularization"):
            self.assertIn(name, NETWORK_ARCHITECTURES["fully_connected"]["optional"])
            for value in (None, 0, 0.001, 2.5, 1e100):
                with self.subTest(argument=name, value=value):
                    self.assertEqual(validate_config_values({name: value}, REGULARIZATION_RULES), [])
            for value in (-0.001, True, False, "0.001", float("nan"), float("inf"), float("-inf")):
                with self.subTest(argument=name, value=value):
                    errors = validate_config_values({name: value}, REGULARIZATION_RULES)
                    self.assertEqual(len(errors), 1)
                    self.assertIn(name, errors[0])

    def test_exclusive_maximum(self):
        from config_rules import validate_config_values

        rules = {"exclusive_maximum": {"rate": 1.0}}
        for value in (None, 0, 0.9999999999999999):
            with self.subTest(value=value):
                self.assertEqual(validate_config_values({"rate": value}, rules), [])
        for value in (1, 1.01, True, "0.5", float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertEqual(len(validate_config_values({"rate": value}, rules)), 1)

    def test_activation_rules(self):
        from config_rules import ACTIVATION_CHOICES, ACTIVATION_RULES, validate_config_values

        for name, choices in ACTIVATION_CHOICES.items():
            for choice in choices:
                arguments = {"hidden_activation": "elu", "output_activation": "linear", name: choice}
                self.assertEqual(validate_config_values(arguments, ACTIVATION_RULES), [])
            for invalid in (None, "", "unsupported"):
                arguments = {"hidden_activation": "elu", "output_activation": "linear", name: invalid}
                self.assertTrue(validate_config_values(arguments, ACTIVATION_RULES))

    def test_non_finite_numbers(self):
        from config_rules import validate_config_values

        rules = {"minimum": {"learning_rate": 0.0}}

        for value in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value):
                errors = validate_config_values(
                    {"learning_rate": value},
                    rules,
                )
                self.assertEqual(len(errors), 1)
                self.assertIn("finite", errors[0])
