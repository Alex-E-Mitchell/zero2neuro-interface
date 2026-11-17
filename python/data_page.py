from flask import render_template, request
from config_store import save_config, clear_config

from data_generator import generate_data_config, parse_lines, parse_optional_int
from data_rules import (
    DATA_FORMATS, FOLD_SPLITS, REPRESENTATIONS, SET_TYPES,
    TF_CACHE_MODES,
)
from parser_validator import validate_config


def parse_int_list(text, field_name):
    """Whole numbers separated by spaces, commas or new lines."""
    try:
        return [int(piece) for piece in (text or "").replace(",", " ").split()]
    except ValueError:
        raise ValueError(f"{field_name} must be whole numbers.") from None


def data():
    config = None
    error = None
    form = request.form

    if request.method == "POST":
        clear_config("data")
        try:
            config = generate_data_config(
                data_format=form.get("data_format", ""),
                data_directory=form.get("data_directory", ""),
                data_files=parse_lines(form.get("data_files", ""), "Data files"),
                data_inputs=parse_lines(form.get("data_inputs", ""), "Data Inputs", allow_empty=True),
                data_outputs=parse_lines(form.get("data_outputs", ""), "Data Outputs", allow_empty=True),
                data_weights=form.get("data_weights", ""),
                tag_examples=parse_lines(form.get("tag_examples", ""), "Data Tag Examples", allow_empty=True),
                representation=form.get("representation", "numpy"),
                tf_cache_mode=form.get("tf_cache_mode", "none"),
                tf_cache_path=form.get("tf_cache_path", ""),
                batch=parse_optional_int(form.get("batch"), "Batch size"),
                rotation=parse_optional_int(form.get("data_rotation"), "Rotation"),
                n_validation_folds=parse_optional_int(form.get("n_validation_folds"), "Validation folds"),
                n_training_folds=parse_optional_int(form.get("n_training_folds"), "Training folds"),
                tabular_header_row=parse_optional_int(form.get("tabular_header_row"), "Header row"),
                tabular_header_names=parse_lines(form.get("tabular_header_names", ""), "Header names", allow_empty=True),
                tabular_column_range=parse_int_list(form.get("tabular_column_range"), "Column range"),
                tabular_column_list=parse_int_list(form.get("tabular_column_list"), "Column list"),
                fold_split=form.get("data_fold_split", ""),
                set_type=form.get("data_set_type", ""),
                n_folds=parse_optional_int(form.get("data_n_folds"), "Number of folds"),
                data_groups=form.get("data_groups", ""),
                data_stratify=form.get("data_stratify", ""),
            )
            validation = validate_config(config)

            if validation["status"] == "invalid":
                error = validation["message"]
                config = None
            else:
                # Save if valid, or if the parser is unavailable.
                save_config("data", config)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "data.html",
        config=config,
        error=error,
        form=form,
        data_formats=DATA_FORMATS,
        fold_splits=FOLD_SPLITS,
        set_types=SET_TYPES,
        representations=REPRESENTATIONS,
        tf_cache_modes=TF_CACHE_MODES,
        # The page reads this dictionary to decide which extra fields to show.
        fold_split_rules={
            key: {
                "requires": rule["requires"],
                "file_fold_numbers": rule.get("file_fold_numbers", False),
                "default": rule.get("default", False),
                "help": rule["help"],
            }
            for key, rule in FOLD_SPLITS.items()
        },
        set_type_rules={
            key: {
                "help": rule["help"],
                "cross_validation": rule.get("cross_validation", False),
                "notice": rule.get("notice", ""),
            }
            for key, rule in SET_TYPES.items()
        },
        format_rules={
            key: {
                "help": rule["help"],
                "tabular_options": rule.get("tabular_options", False),
                "columns_optional": rule.get("columns_optional", False),
                "unsupported_splits": rule.get("unsupported_splits", []),
                "forces_representation": rule.get("forces_representation", ""),
            }
            for key, rule in DATA_FORMATS.items()
        },
        representation_rules={
            key: {"help": rule["help"], "tf_options": rule.get("tf_options", False)}
            for key, rule in REPRESENTATIONS.items()
        },
    )


def register_data_page(app):
    """Add the /data route to the app.

    The endpoint is named "data" so the shared navigation in base.html, which
    calls url_for('data'), finds it.
    """
    app.add_url_rule("/data", endpoint="data", view_func=data, methods=["GET", "POST"])