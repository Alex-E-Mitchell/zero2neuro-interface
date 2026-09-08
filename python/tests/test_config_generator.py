import sys
import unittest
from pathlib import Path

# Allow imports from the parent python/ directory when tests are run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_generator import generate_network_config, validate_hidden_units


class TestConfigGenerator(unittest.TestCase):
    def test_validate_hidden_units(self):
        self.assertTrue(validate_hidden_units([64, 32]))
        self.assertTrue(validate_hidden_units([128]))

        self.assertFalse(validate_hidden_units([0, 32]))
        self.assertFalse(validate_hidden_units([-4, 32]))
        self.assertFalse(validate_hidden_units([]))
        self.assertFalse(validate_hidden_units([64, 3.5]))
        self.assertFalse(validate_hidden_units([True, 32]))

    def test_generate_network_config(self):
        config = generate_network_config(
            network_type="fully_connected",
            hidden_units=[64, 32],
            hidden_activation="elu",
            output_shape=1,
            output_activation="elup1",
        )

        self.assertIn("--network_type=fully_connected", config)
        self.assertIn("--number_hidden_units\n64\n32", config)
        self.assertIn("--hidden_activation=elu", config)
        self.assertIn("--output_shape\n1", config)
        self.assertIn("--output_activation=elup1", config)

    def test_generate_network_config_rejects_invalid_hidden_units(self):
        with self.assertRaises(ValueError):
            generate_network_config(
                network_type="fully_connected",
                hidden_units=[64, 0],
                hidden_activation="elu",
                output_shape=1,
                output_activation="elup1",
            )


if __name__ == "__main__":
    unittest.main()
