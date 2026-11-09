import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser_validator import validate_config


class TestParserValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.repository = Path(self.temp_dir.name)
        src = self.repository / "src"
        src.mkdir()

        (src / "zero2neuro_debug.py").write_text("")

        (src / "parser.py").write_text("""
import argparse

class CommentedArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, line):
        if "=" in line and line.startswith("--"):
            return line.split("=", 1)
        return [line]

def create_parser():
    parser = CommentedArgumentParser()
    parser.add_argument("--input_shape", type=int, nargs="+")
    return parser
""")

    def test_valid_configuration(self):
        with patch.dict(
            os.environ,
            {"NEURO_REPOSITORY_PATH": str(self.repository)}
        ):
            result = validate_config("--input_shape\n34\n")

        self.assertEqual(result["status"], "valid")

    def test_invalid_configuration(self):
        with patch.dict(
            os.environ,
            {"NEURO_REPOSITORY_PATH": str(self.repository)}
        ):
            result = validate_config("--input_shape\nabc\n")

        self.assertEqual(result["status"], "invalid")
        self.assertIn("invalid int value", result["message"])

    def test_missing_repository_path(self):
        with patch.dict(os.environ, {}, clear=True):
            result = validate_config("--input_shape\n34\n")

        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
