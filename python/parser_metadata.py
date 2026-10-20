"""Inspect an existing parser without importing Zero2Neuro or semantic rules."""

import argparse
import ast
from copy import deepcopy
from pathlib import Path

def extract_argument_metadata(parser):
    """Return one metadata record per argparse action, including aliases/positionals.

    Usage: extract_argument_metadata(zero2neuro_parser.create_parser()). The caller
    supplies the parser so this prototype adds no Zero2Neuro runtime dependency.
    Records retain Python type callables and defaults; this is not a JSON exporter.
    Multiple actions may share a destination (e.g. --gpu and --no-gpu), so a list
    preserves both. argparse exposes no public action iterator: _actions and flag
    action classes are isolated here and covered by tests. Required means argparse
    presence only; architecture relationships and string semantics live elsewhere.
    """
    records = []
    for action in parser._actions:
        records.append({
            "dest": action.dest,
            "option_strings": list(action.option_strings),
            "type": action.type,
            "nargs": action.nargs,
            "default": deepcopy(action.default),
            "help": action.help,
            "choices": deepcopy(action.choices),
            "required": action.required,
            "action": type(action).__name__,
            "is_boolean_flag": isinstance(action, (
                argparse._StoreTrueAction, argparse._StoreFalseAction,
                argparse.BooleanOptionalAction,
            )),
            "is_store_true": isinstance(action, argparse._StoreTrueAction),
        })
    return records

def read_static_value(node):
    """Read literal values without executing source code."""
    if isinstance(node, ast.Name) and node.id in {"int", "float", "str", "bool"}:
        return node.id

    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return "<dynamic>"

def inspect_parser_source(parser_path):
    """Inspect parser declarations without importing Zero2Neuro."""
    parser_path = Path(parser_path)

    if not parser_path.is_file():
        raise FileNotFoundError(f"Parser file not found: {parser_path}")

    source = parser_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(parser_path))

    create_parser = next(
        (
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_parser"
        ),
        None
    )

    if create_parser is None:
        raise ValueError("Could not find create_parser() in the source file.")

    arguments = []

    for node in ast.walk(create_parser):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Attribute):
            continue

        if node.func.attr != "add_argument":
            continue

        if not isinstance(node.func.value, ast.Name):
            continue

        if node.func.value.id != "parser":
            continue

        names = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]

        if not names:
            continue

        keywords = {keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None}

        arguments.append({
            "option_strings": names,
            "line": node.lineno,
            "type": read_static_value(keywords["type"]) if "type" in keywords else None,
            "default": read_static_value(keywords["default"]) if "default" in keywords else None,
            "choices": read_static_value(keywords["choices"]) if "choices" in keywords else None,
            "nargs": read_static_value(keywords["nargs"]) if "nargs" in keywords else None,
            "action": read_static_value(keywords["action"]) if "action" in keywords else None,
        })

    return arguments
