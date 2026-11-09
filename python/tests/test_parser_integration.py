
"""Test Flask integration with Zero2Neuro parser validation."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import app


class TestParserIntegration(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.fields = {
            "network_type": "fully_connected",
            "input_shape": "34",
            "hidden_units": "64, 32",
            "hidden_activation": "elu",
            "output_shape": "10, 20",
            "output_activation": "linear",
        }


    def test_valid_parser_result(self):
        with patch("app.validate_config", return_value={
            "status": "valid",
            "message": "Configuration accepted.",
        }) as validator, patch("app.save_config") as mock_save:

            response = self.client.post("/", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_called_once()

        self.assertEqual(mock_save.call_args.args[0], "network")
        self.assertIn('id="config-text"', page)


    def test_invalid_parser_result(self):
        with patch("app.validate_config", return_value={
            "status": "invalid",
            "message": "Invalid configuration.",
        }) as validator, patch("app.save_config") as mock_save:

            response = self.client.post("/", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_not_called()

        self.assertNotIn('id="config-text"', page)
        self.assertIn("Invalid configuration.", page)


    def test_unavailable_parser_result(self):
        with patch("app.validate_config", return_value={
            "status": "unavailable",
            "message": "Parser unavailable.",
        }) as validator, patch("app.save_config") as mock_save:

            response = self.client.post("/", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_called_once()

        self.assertEqual(mock_save.call_args.args[0], "network")
        self.assertIn('id="config-text"', page)


class TestDataParserIntegration(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.fields = {
            "data_format": "tabular",
            "data_directory": "",
            "data_files": "xor_data.csv",
            "data_inputs": "In 0\nIn 1",
            "data_outputs": "Out 0",
            "data_fold_split": "identity",
            "data_set_type": "fixed",
        }

    def test_valid_data_parser_result(self):
        with patch("data_page.validate_config", return_value={
            "status": "valid",
            "message": "Configuration accepted.",
        }) as validator, patch("data_page.save_config") as mock_save:

            response = self.client.post("/data", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.args[0], "data")
        self.assertIn("--data_format=tabular", page)

    def test_invalid_data_parser_result(self):
        with patch("data_page.validate_config", return_value={
            "status": "invalid",
            "message": "Invalid Data configuration.",
        }) as validator, patch("data_page.save_config") as mock_save:

            response = self.client.post("/data", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_not_called()
        self.assertIn("Invalid Data configuration.", page)
        self.assertNotIn('id="config-text"', page)

    def test_unavailable_data_parser_result(self):
        with patch("data_page.validate_config", return_value={
            "status": "unavailable",
            "message": "Parser unavailable.",
        }) as validator, patch("data_page.save_config") as mock_save:

            response = self.client.post("/data", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.args[0], "data")
        self.assertIn("--data_format=tabular", page)


class TestExperimentParserIntegration(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.fields = {
            "experiment_name": "test-experiment",
            "loss": "mse",
            "learning_rate": "0.001",
            "epochs": "10",
        }

    def test_valid_experiment_parser_result(self):
        with patch("experiment_page.validate_config", return_value={
            "status": "valid",
            "message": "Configuration accepted.",
        }) as validator, patch("experiment_page.save_config") as mock_save:

            response = self.client.post("/experiment", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.args[0], "experiment")
        self.assertIn("--experiment_name", page)

    def test_invalid_experiment_parser_result(self):
        with patch("experiment_page.validate_config", return_value={
            "status": "invalid",
            "message": "Invalid Experiment configuration.",
        }) as validator, patch("experiment_page.save_config") as mock_save:

            response = self.client.post("/experiment", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_not_called()
        self.assertIn("Invalid Experiment configuration.", page)
        self.assertNotIn('href="#parser"', page)
        self.assertNotIn('id="config-text"', page)

    def test_unavailable_experiment_parser_result(self):
        with patch("experiment_page.validate_config", return_value={
            "status": "unavailable",
            "message": "Parser unavailable.",
        }) as validator, patch("experiment_page.save_config") as mock_save:

            response = self.client.post("/experiment", data=self.fields)

        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        validator.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.args[0], "experiment")
        self.assertIn("--experiment_name", page)


if __name__ == "__main__":
    unittest.main()
