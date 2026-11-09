"""Validate configuration text using Zero2Neuro's parser."""

import argparse
import importlib.util
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path


# Protect parser loading and validation from overlapping requests.
_parser_lock = threading.RLock()


@lru_cache(maxsize=4)
def _load_parser(repository_path):
    """Load and cache the parser for a configured repository."""

    src_path = Path(repository_path) / "src"
    parser_path = src_path / "parser.py"

    if not parser_path.is_file():
        raise FileNotFoundError("Zero2Neuro parser.py was not found.")

    spec = importlib.util.spec_from_file_location(
        "zero2neuro_parser", parser_path
    )

    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Zero2Neuro parser.")

    module = importlib.util.module_from_spec(spec)

    # The framework parser imports zero2neuro_debug.
    # Only change sys.path while holding the shared lock.
    source_directory = str(src_path)
    sys.path.insert(0, source_directory)

    try:
        # Avoid reusing a debug module from another repository.
        cached_debug = sys.modules.pop("zero2neuro_debug", None)

        try:
            spec.loader.exec_module(module)
        finally:
            if cached_debug is not None:
                sys.modules["zero2neuro_debug"] = cached_debug
            else:
                sys.modules.pop("zero2neuro_debug", None)
    finally:
        sys.path.remove(source_directory)

    parser = module.create_parser()
    parser.exit_on_error = False

    return parser


def validate_config(config_text):
    """Return valid, invalid, or unavailable with an explanation."""

    repository = os.environ.get("NEURO_REPOSITORY_PATH")

    if not repository:
        return {
            "status": "unavailable",
            "message": "NEURO_REPOSITORY_PATH is not configured.",
        }

    repository_path = str(Path(repository).expanduser().resolve())

    try:
        with _parser_lock:
            parser = _load_parser(repository_path)

            arguments = []
            for line in config_text.splitlines():
                arguments.extend(
                    parser.convert_arg_line_to_args(line)
                )

            parser.parse_args(arguments)

        return {
            "status": "valid",
            "message": "Configuration accepted by Zero2Neuro's parser.",
        }

    except (argparse.ArgumentError, SystemExit) as error:
        message = str(error)

        if isinstance(error, SystemExit):
            message = (
                "Zero2Neuro rejected this configuration. "
                "Please check the generated settings."
            )

        return {
            "status": "invalid",
            "message": message,
        }

    except (ImportError, OSError, AttributeError) as error:
        return {
            "status": "unavailable",
            "message": str(error),
        }
