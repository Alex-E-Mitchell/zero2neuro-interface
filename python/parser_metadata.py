"""Inspect an existing parser without importing Zero2Neuro or semantic rules."""

import argparse
from copy import deepcopy


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
