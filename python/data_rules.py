"""Data-configuration rules stored as dictionaries rather than if-statements.

Option names are written without the leading '--'. Each table maps a choice the
user can make (a value of --data_format, --data_fold_split, or --data_set_type)
to what that choice needs. To support a new choice, add an entry here; the
generator and the Data page both read these tables.

"""

OPTION_LABELS = {
    "data_n_folds": "Number of folds",
    "data_groups": "Group column",
    "data_stratify": "Class column",
}

DATA_FORMATS = {
    "tabular": {
        "label": "Tabular (CSV or XLSX)",
        "level": "Beginner",
        "help": "Spreadsheet-style files. Each row is one example; the first row holds the column names.",
    },
    "pickle": {
        "label": "Pickle (Python dictionary)",
        "level": "Intermediate",
        "help": "A file holding one Python dictionary. Its keys play the role of column names.",
    },
}

FOLD_SPLITS = {
    "identity": {
        "label": "One fold per file",
        "requires": [],
        "help": "Each data file becomes its own fold. Use this with a single file, or with files already separated the way you want.",
    },
    "random": {
        "label": "Random folds",
        "requires": ["data_n_folds"],
        "help": "All examples are shuffled and divided at random into the number of folds you choose. Avoid this when examples in one file are strongly related (for example, all from one subject).",
    },
    "random-stratify": {
        "label": "Random folds, balanced by class",
        "requires": ["data_n_folds", "data_stratify"],
        "help": "Like random folds, but every fold gets about the same share of each class. Use it for classification when some classes are much rarer than others.",
    },
    "group-by-file": {
        "label": "Group files into folds",
        "requires": [],
        "file_fold_numbers": True,
        "help": "You choose which fold each file belongs to by writing file,fold on each line (folds count from 0).",
    },
    "group-by-example": {
        "label": "Folds from a column",
        "requires": ["data_groups"],
        "help": "One column holds the fold number (0, 1, 2, ...) for every example.",
    },
}

SET_TYPES = {
    "fixed": {
        "label": "Fixed (train / validation / test)",
        "level": "Beginner",
        "max_folds": 3,
        "help": "Fold 0 trains the model, fold 1 (if present) validates it, and fold 2 (if present) tests it. Good for quick, informal experiments. At most three folds.",
    },
    "holistic-cross-validation": {
        "label": "Holistic cross-validation",
        "level": "Intermediate",
        "min_folds": 3,
        "help": "Trains one model per fold rotation. Every fold is used exactly once for validation and once for testing. Needs at least three folds.",
    },
    "hold-out-cross-validation": {
        "label": "Hold-out cross-validation",
        "level": "Intermediate",
        "min_folds": 3,
        "help": "Like holistic cross-validation, but the last fold is always the test set. Needs at least three folds.",
    },
}


def missing_required(table, choice, supplied):
    """Return the option names the chosen entry requires that were left blank.

    Works for any of the tables above: it only reads table[choice]["requires"].
    """
    if choice not in table:
        raise ValueError(f"Unknown choice: {choice!r}.")
    return [
        name for name in table[choice].get("requires", [])
        if supplied.get(name) in (None, "", [])
    ]