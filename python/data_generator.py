from data_rules import (
    DATA_FORMATS, FOLD_SPLITS, OPTION_LABELS, SET_TYPES, missing_required,
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
):
    """Generate Zero2Neuro-style data configuration text.

    data_files, data_inputs, and data_outputs are lists of strings. For the
    group-by-file split, each entry in data_files must be written as file,fold.
    Options the chosen split does not use (n_folds, data_groups, data_stratify)
    are left out.
    """
    if data_format not in DATA_FORMATS:
        raise ValueError(f"Unknown data format: {data_format!r}.")
    if fold_split not in FOLD_SPLITS:
        raise ValueError(f"Unknown fold split: {fold_split!r}.")
    if set_type not in SET_TYPES:
        raise ValueError(f"Unknown dataset split type: {set_type!r}.")
    if not data_files:
        raise ValueError("Data files needs at least one entry (one per line).")
    if not data_inputs:
        raise ValueError("Input columns needs at least one entry (one per line).")
    if not data_outputs:
        raise ValueError("Output columns needs at least one entry (one per line).")

    data_directory = (data_directory or "").strip()
    data_groups = (data_groups or "").strip()
    data_stratify = (data_stratify or "").strip()
    if data_directory:
        check_safe_value(data_directory, "Data directory")
    if data_groups:
        check_safe_value(data_groups, "Group column")
    if data_stratify:
        check_safe_value(data_stratify, "Class column")

    missing = missing_required(
        FOLD_SPLITS, fold_split,
        {"data_n_folds": n_folds, "data_groups": data_groups, "data_stratify": data_stratify},
    )
    if missing:
        names = ", ".join(OPTION_LABELS[name] for name in missing)
        raise ValueError(f"{FOLD_SPLITS[fold_split]['label']} needs: {names}.")

    # How many folds will exist, when we can tell (used for the set-type limit).
    known_folds = None
    file_lines = list(data_files)
    if FOLD_SPLITS[fold_split].get("file_fold_numbers"):
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
    elif fold_split in ("random", "random-stratify"):
        if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
            raise ValueError("Number of folds must be a whole number of at least 2.")
        known_folds = n_folds
    elif fold_split == "identity":
        known_folds = len(data_files)

    # With a list of files, Zero2Neuro cuts each line at spaces and commas, so a
    # file name containing a space would be read as two separate values.
    uses_list = FOLD_SPLITS[fold_split].get("file_fold_numbers") or len(file_lines) > 1
    if uses_list:
        for line in file_lines:
            if len(line.split(",")[0].split()) != 1:
                raise ValueError(
                    f"Data files: '{line}' has a space in the file name, which Zero2Neuro "
                    "cannot read when several files are listed. Rename the file."
                )

    min_folds = SET_TYPES[set_type].get("min_folds")
    if min_folds is not None and known_folds is not None and known_folds < min_folds:
        raise ValueError(
            f"{SET_TYPES[set_type]['label']} needs at least {min_folds} folds, "
            f"but this setup creates {known_folds}."
        )

    max_folds = SET_TYPES[set_type].get("max_folds")
    if max_folds is not None and known_folds is not None and known_folds > max_folds:
        raise ValueError(
            f"{SET_TYPES[set_type]['label']} supports at most {max_folds} folds, "
            f"but this setup creates {known_folds}."
        )

    lines = [f"--data_format={data_format}"]
    if data_directory:
        lines.append(f"--dataset_directory={data_directory}")

    if uses_list:
        lines.append("--data_files")
        lines.extend(file_lines)
    else:
        lines.append(f"--data_file={file_lines[0]}")

    # identity is Zero2Neuro's default, so it is left out of the file.
    if fold_split != "identity":
        lines.append(f"--data_fold_split={fold_split}")
    if fold_split in ("random", "random-stratify"):
        lines.append(f"--data_n_folds={n_folds}")
    if fold_split == "group-by-example":
        lines.append(f"--data_groups={data_groups}")
    if fold_split == "random-stratify":
        lines.append(f"--data_stratify={data_stratify}")

    lines.append(f"--data_set_type={set_type}")

    lines.append("--data_inputs")
    lines.extend(data_inputs)
    lines.append("--data_outputs")
    lines.extend(data_outputs)

    return "\n".join(lines) + "\n"