from data_rules import (
    DATA_FORMATS, EXCLUSIVE_OPTIONS, FOLD_SPLITS, OPTION_LABELS, OPTIONS,
    REPRESENTATIONS, SET_TYPES, TF_CACHE_MODES, missing_required,
)


def check_safe_value(value, field_name):
    """Reject text that Zero2Neuro's config reader would misread.

    Its reader treats '#' as the start of a comment, treats a line starting with
    '-' as an option name, and treats an argument starting with '@' as a file to
    include.
    """
    if "#" in value:
        raise ValueError(
            f"{field_name}: '{value}' contains '#', which Zero2Neuro reads as the start of a comment."
        )
    if value.startswith(("-", "@")):
        raise ValueError(f"{field_name}: '{value}' cannot start with '-' or '@'.")


def parse_lines(text, field_name, allow_empty=False):
    """Split a text box into one entry per non-blank line, with spaces trimmed."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines and not allow_empty:
        raise ValueError(f"{field_name} needs at least one entry (one per line).")
    for line in lines:
        check_safe_value(line, field_name)
    return lines


def parse_optional_int(text, field_name):
    """Blank means 'not supplied'; anything else must be a whole number."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{field_name} must be a whole number.") from None


def _split_file_and_fold(line):
    name, separator, fold = line.rpartition(",")
    try:
        fold_number = int(fold.strip())
    except ValueError:
        fold_number = -1
    if not separator or not name.strip() or fold_number < 0:
        raise ValueError(
            f"Data files: '{line}' must look like file,fold with a whole fold number such as data_A.csv,0."
        )
    return name.strip(), fold_number


def generate_data_config(
    data_format,
    data_files,
    data_inputs,
    data_outputs,
    fold_split,
    set_type,
    data_directory="",
    n_folds=None,
    data_groups="",
    data_stratify="",
    data_weights="",
    tag_examples=(),
    representation="numpy",
    tf_cache_mode="none",
    tf_cache_path="",
    batch=None,
    rotation=None,
    n_validation_folds=None,
    n_training_folds=None,
    tabular_header_row=None,
    tabular_header_names=(),
    tabular_column_range=(),
    tabular_column_list=(),
):
    """Generate Zero2Neuro-style data configuration text.

    data_files, data_inputs, and data_outputs are lists of strings. For the
    group-by-file split, each entry in data_files must be written as file,fold.
    Options that the chosen split, format, set type or representation does not
    use are left out of the file (see OPTIONS in data_rules.py).

    """
    if data_format not in DATA_FORMATS:
        raise ValueError(f"Unknown data format: {data_format!r}.")
    if fold_split not in FOLD_SPLITS:
        raise ValueError(f"Unknown fold split: {fold_split!r}.")
    if set_type not in SET_TYPES:
        raise ValueError(f"Unknown dataset split type: {set_type!r}.")
    if not data_files:
        raise ValueError("Data files needs at least one entry (one per line).")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown data representation: {representation!r}.")
    if tf_cache_mode not in TF_CACHE_MODES:
        raise ValueError(f"Unknown cache choice: {tf_cache_mode!r}.")

    format_rule = DATA_FORMATS[data_format]
    split_rule = FOLD_SPLITS[fold_split]
    set_type_rule = SET_TYPES[set_type]
    representation_rule = REPRESENTATIONS[representation]

    if fold_split in format_rule.get("unsupported_splits", []):
        raise ValueError(
            f"{split_rule['label']} is not available for {format_rule['label']}."
        )
    forced = format_rule.get("forces_representation")
    if forced and representation != forced:
        raise ValueError(
            f"{format_rule['label']} must use the {REPRESENTATIONS[forced]['label']} representation."
        )

    for option in ("data_inputs", "data_outputs"):
        if not (data_inputs if option == "data_inputs" else data_outputs) \
                and not format_rule.get("columns_optional"):
            raise ValueError(f"{OPTIONS[option]['label']} needs at least one entry (one per line).")

    data_directory = (data_directory or "").strip()
    if data_directory:
        check_safe_value(data_directory, "Data directory")

    # The cache choice becomes the value of --data_tf_cache:
    # none -> no line, memory -> an empty value, disk -> the folder.
    tf_cache_path = (tf_cache_path or "").strip()
    uses_tf_options = representation_rule.get("tf_options", False)
    if uses_tf_options and tf_cache_mode == "disk" and not tf_cache_path:
        raise ValueError("Cache folder is needed when caching to disk.")
    cache_value = {"none": None, "memory": "", "disk": tf_cache_path}[tf_cache_mode]

    # Every option this function can write, by flag name.
    values = {
        "data_n_folds": n_folds,
        "data_groups": (data_groups or "").strip(),
        "data_stratify": (data_stratify or "").strip(),
        "data_rotation": rotation,
        "data_n_validation_folds": n_validation_folds,
        "data_n_training_folds": n_training_folds,
        "data_inputs": list(data_inputs),
        "data_outputs": list(data_outputs),
        "data_weights": (data_weights or "").strip(),
        "data_tag_examples": list(tag_examples),
        "tabular_header_row": tabular_header_row,
        "tabular_header_names": list(tabular_header_names),
        "tabular_column_range": list(tabular_column_range),
        "tabular_column_list": list(tabular_column_list),
        "data_representation": None if representation_rule.get("default") else representation,
        "data_tf_cache": cache_value if uses_tf_options else None,
        "data_batch": batch,
    }

    cv = set_type_rule.get("cross_validation", False)
    flags = {
        "tabular_options": format_rule.get("tabular_options", False),
        "cross_validation": cv,
        "tf_options": uses_tf_options,
    }

    def applies(name):
        condition = OPTIONS[name].get("applies_when", "always")
        if condition == "always":
            return True
        if condition == "required_by_split":
            return name in split_rule["requires"]
        return flags[condition]

    def is_filled(name):
        value = values[name]
        if value is None:
            return False
        if OPTIONS[name].get("allow_empty"):
            return True
        return value != "" and value != []

    # Options the split or set type needs must be given.
    missing = missing_required(
        FOLD_SPLITS, fold_split,
        {name: values[name] for name in split_rule["requires"]},
    )
    if missing:
        names = ", ".join(OPTION_LABELS[name] for name in missing)
        raise ValueError(f"{split_rule['label']} needs: {names}.")
    missing = missing_required(SET_TYPES, set_type, {"data_rotation": rotation})
    if missing:
        names = ", ".join(OPTION_LABELS[name] for name in missing)
        raise ValueError(f"{set_type_rule['label']} needs: {names}.")

    # Check each option by its row in OPTIONS.
    for name, rule in OPTIONS.items():
        if not is_filled(name):
            continue
        if not applies(name):
            if rule.get("error_if_unused"):
                raise ValueError(rule["error_if_unused"])
            continue
        label, value, kind = rule["label"], values[name], rule["kind"]
        if kind == "text":
            if value:
                check_safe_value(value, label)
        elif kind == "list":
            for entry in value:
                check_safe_value(entry, label)
        else:
            numbers = [value] if kind == "int" else value
            if "count" in rule and len(numbers) != rule["count"]:
                raise ValueError(f"{label} needs exactly {rule['count']} numbers.")
            for number in numbers:
                if not isinstance(number, int) or isinstance(number, bool) or number < rule["min"]:
                    raise ValueError(
                        f"{label} must be a whole number of at least {rule['min']}."
                    )
            if rule.get("ascending") and numbers != sorted(numbers):
                raise ValueError(f"{label} must go from the smaller number to the larger one.")

    for first, second in EXCLUSIVE_OPTIONS:
        if is_filled(first) and is_filled(second):
            raise ValueError(
                f"Use either {OPTIONS[first]['label'].lower()} or "
                f"{OPTIONS[second]['label'].lower()}, not both."
            )

    # How many folds will exist, when we can tell (used for the set-type limits).
    known_folds = None
    file_lines = list(data_files)
    if split_rule.get("file_fold_numbers"):
        pairs = [_split_file_and_fold(line) for line in data_files]
        file_lines = [f"{name},{fold}" for name, fold in pairs]
        fold_numbers = {fold for _, fold in pairs}
        # Zero2Neuro stops with "data_files missing groups" if a number is skipped.
        if fold_numbers != set(range(len(fold_numbers))):
            raise ValueError(
                "Data files: fold numbers must start at 0 and have no gaps "
                f"(you used {sorted(fold_numbers)})."
            )
        known_folds = len(fold_numbers)
    elif split_rule.get("folds_from") == "data_n_folds":
        known_folds = n_folds
    elif split_rule.get("folds_from") == "files":
        known_folds = len(data_files)

    # With a list of files, Zero2Neuro cuts each line at spaces and commas, so a
    # file name containing a space would be read as two separate values.
    uses_list = split_rule.get("file_fold_numbers") or len(file_lines) > 1
    if uses_list:
        for line in file_lines:
            if len(line.split(",")[0].split()) != 1:
                raise ValueError(
                    f"Data files: '{line}' has a space in the file name, which Zero2Neuro "
                    "cannot read when several files are listed. Rename the file."
                )

    min_folds = set_type_rule.get("min_folds")
    if min_folds is not None and known_folds is not None and known_folds < min_folds:
        raise ValueError(
            f"{set_type_rule['label']} needs at least {min_folds} folds, "
            f"but this setup creates {known_folds}."
        )

    max_folds = set_type_rule.get("max_folds")
    if max_folds is not None and known_folds is not None and known_folds > max_folds:
        raise ValueError(
            f"{set_type_rule['label']} supports at most {max_folds} folds, "
            f"but this setup creates {known_folds}."
        )

    if cv and known_folds is not None:
        fewer = set_type_rule.get("rotations_fewer_than_folds_by", 0)
        rotations = known_folds - fewer
        if rotation >= rotations:
            raise ValueError(
                f"{set_type_rule['label']} with {known_folds} folds has {rotations} "
                f"rotations, so Rotation can be 0 to {rotations - 1}."
            )
        if n_training_folds is not None:
            if n_training_folds + (n_validation_folds or 1) > known_folds - 1:
                raise ValueError(
                    "Training folds plus validation folds cannot be more than the number of "
                    f"folds minus 1 ({known_folds - 1})."
                )
        elif n_validation_folds is not None and n_validation_folds > known_folds - 2:
            raise ValueError(
                f"Validation folds can be at most {known_folds - 2} with {known_folds} folds "
                "(at least one fold is left for training and one for testing)."
            )

    lines = [f"--data_format={data_format}"]
    if data_directory:
        lines.append(f"--dataset_directory={data_directory}")

    if uses_list:
        lines.append("--data_files")
        lines.extend(file_lines)
    else:
        lines.append(f"--data_file={file_lines[0]}")

    # The default split is left out of the file. Every option the split
    # requires (see FOLD_SPLITS) is written right after it.
    if not split_rule.get("default"):
        lines.append(f"--data_fold_split={fold_split}")
    for name in split_rule["requires"]:
        lines.append(f"--{name}={values[name]}")

    lines.append(f"--data_set_type={set_type}")

    # Everything else is written from its row in OPTIONS, in table order.
    for name, rule in OPTIONS.items():
        if rule.get("written_by") or not is_filled(name) or not applies(name):
            continue
        if rule["kind"] in ("list", "int_list"):
            lines.append(f"--{name}")
            lines.extend(str(entry) for entry in values[name])
        else:
            lines.append(f"--{name}={values[name]}")

    return "\n".join(lines) + "\n"