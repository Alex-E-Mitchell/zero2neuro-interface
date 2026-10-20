from flask import render_template, request

from data_generator import generate_data_config, parse_lines, parse_optional_int
from data_rules import DATA_FORMATS, FOLD_SPLITS, SET_TYPES

def data():
    config = None
    error = None
    form = request.form

    if request.method == "POST":
        try:
            config = generate_data_config(
                data_format=form.get("data_format", ""),
                data_directory=form.get("data_directory", ""),
                data_files=parse_lines(form.get("data_files", ""), "Data files"),
                data_inputs=parse_lines(form.get("data_inputs", ""), "Input columns"),
                data_outputs=parse_lines(form.get("data_outputs", ""), "Output columns"),
                fold_split=form.get("data_fold_split", ""),
                set_type=form.get("data_set_type", ""),
                n_folds=parse_optional_int(form.get("data_n_folds"), "Number of folds"),
                data_groups=form.get("data_groups", ""),
                data_stratify=form.get("data_stratify", ""),
            )
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
        # The page reads this dictionary to decide which extra fields to show.
        fold_split_rules={
            key: {
                "requires": rule["requires"],
                "file_fold_numbers": rule.get("file_fold_numbers", False),
                "help": rule["help"],
            }
            for key, rule in FOLD_SPLITS.items()
        },
        set_type_help={key: rule["help"] for key, rule in SET_TYPES.items()},
        format_help={key: rule["help"] for key, rule in DATA_FORMATS.items()},
    )


def register_data_page(app):
    """Add the /data route to the app.

    The endpoint is named "data" so the shared navigation in base.html, which
    calls url_for('data'), finds it.
    """
    app.add_url_rule("/data", endpoint="data", view_func=data, methods=["GET", "POST"])