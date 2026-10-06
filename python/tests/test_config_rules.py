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
