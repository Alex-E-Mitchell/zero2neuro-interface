"""Data-configuration rules stored as dictionaries rather than if-statements.

Option names are written without the leading '--'. Each table maps a choice the
user can make (a value of --data_format, --data_fold_split, or --data_set_type)
to what that choice needs. To support a new choice, add an entry here; the
generator and the Data page both read these tables.

Keys used by the tables:
  default        the choice Zero2Neuro already assumes, so no line is written
  requires       options the choice needs; each is also written to the file
  folds_from     how many folds a split creates ("files", "data_n_folds",
                 or "fold_numbers")
  tf_options     the choice uses the cache and batch size
  OPTIONS        one row per option: how it is checked and written (see below)

"""

TABULAR_ONLY = "The table header and column options only apply to tabular files."

# One row per option the generator can check and write (flag name = key).
#   label        the name used in error messages
#   kind         text, int, list (of text) or int_list (of whole numbers)
#   min          smallest allowed whole number (int and int_list)
#   count        exact number of entries (int_list)
#   ascending    entries must go from smaller to larger (int_list)
#   allow_empty  an empty string is a real value (written as --flag=)
#   applies_when "always" (default), "required_by_split", or a flag on the
#                chosen format / set type / representation:
#                tabular_options, cross_validation, tf_options.
#                An option that does not apply is left out of the file.
#   error_if_unused  raise this message instead if the option is given but
#                does not apply
#   written_by   a table elsewhere writes this option, so the generic
#                writer skips it (the fold split writes its own options)
# Rows are written to the file in this order.
OPTIONS = {
    "data_n_folds": {"label": "Number of folds", "kind": "int", "min": 2,
                     "applies_when": "required_by_split", "written_by": "fold_split"},
    "data_groups": {"label": "Data Groups", "kind": "text",
                    "applies_when": "required_by_split", "written_by": "fold_split"},
    "data_stratify": {"label": "Class column", "kind": "text",
                      "applies_when": "required_by_split", "written_by": "fold_split"},
    "data_rotation": {"label": "Rotation", "kind": "int", "min": 0,
                      "applies_when": "cross_validation"},
    "data_n_validation_folds": {"label": "Validation folds", "kind": "int", "min": 1,
                                "applies_when": "cross_validation"},
    "data_n_training_folds": {"label": "Training folds", "kind": "int", "min": 1,
                              "applies_when": "cross_validation"},
    "data_inputs": {"label": "Data Inputs", "kind": "list"},
    "data_outputs": {"label": "Data Outputs", "kind": "list"},
    "data_weights": {"label": "Data Weights", "kind": "text"},
    "data_tag_examples": {"label": "Data Tag Examples", "kind": "list"},
    "tabular_header_row": {"label": "Header row", "kind": "int", "min": 0,
                           "applies_when": "tabular_options", "error_if_unused": TABULAR_ONLY},
    "tabular_header_names": {"label": "Header names", "kind": "list",
                             "applies_when": "tabular_options", "error_if_unused": TABULAR_ONLY},
    "tabular_column_range": {"label": "Column range", "kind": "int_list", "min": 0,
                             "count": 2, "ascending": True,
                             "applies_when": "tabular_options", "error_if_unused": TABULAR_ONLY},
    "tabular_column_list": {"label": "Column list", "kind": "int_list", "min": 0,
                            "applies_when": "tabular_options", "error_if_unused": TABULAR_ONLY},
    "data_representation": {"label": "Data representation", "kind": "text"},
    "data_tf_cache": {"label": "Cache folder", "kind": "text", "allow_empty": True,
                      "applies_when": "tf_options"},
    "data_batch": {"label": "Batch size", "kind": "int", "min": 1,
                   "applies_when": "tf_options"},
}

# Pairs of options that cannot be used together.
EXCLUSIVE_OPTIONS = [
    ("tabular_header_row", "tabular_header_names"),
    ("tabular_column_range", "tabular_column_list"),
]

# The names used in "needs: ..." messages.
OPTION_LABELS = {name: rule["label"] for name, rule in OPTIONS.items()}

DATA_FORMATS = {
    "tabular": {
        "label": "Tabular",
        "level": "Beginner",
        "tabular_options": True,
        "help": "Spreadsheet-style files. Each row is one example; the first row holds the column names.",
    },
    "pickle": {
        "label": "Pickle",
        "level": "Intermediate",
        "help": "A file holding one Python dictionary. Its keys play the role of column names.",
    },
    "tf-dataset": {
        "label": "TF-Dataset",
        "level": "Advanced",
        "columns_optional": True,
        "forces_representation": "tf-dataset",
        "unsupported_splits": ["random", "random-stratify", "group-by-example"],
        "help": "Data already saved as TensorFlow datasets. Inputs and outputs are built in, so column names are optional. Random and column-based fold splits are not available.",
    },
}

FOLD_SPLITS = {
    "identity": {
        "label": "Identity",
        "default": True,        # Zero2Neuro's default, so it is not written to the file
        "folds_from": "files",  # one fold per data file
        "requires": [],
        "help": "Each data file becomes its own fold. Use this with a single file, or with files already separated the way you want.",
    },
    "random": {
        "label": "Random Split",
        "folds_from": "data_n_folds",
        "requires": ["data_n_folds"],
        "help": "All examples are shuffled and divided at random into the number of folds you choose. Avoid this when examples in one file are strongly related (for example, all from one subject).",
    },
    "random-stratify": {
        "label": "Random-Stratify Split",
        "folds_from": "data_n_folds",
        "requires": ["data_n_folds", "data_stratify"],
        "help": "Like random folds, but every fold gets about the same share of each class. Use it for classification when some classes are much rarer than others.",
    },
    "group-by-file": {
        "label": "Group-by-File",
        "requires": [],
        "file_fold_numbers": True,
        "folds_from": "fold_numbers",  # counted from the numbers written after each file
        "help": "You choose which fold each file belongs to by writing file,fold on each line (folds count from 0).",
    },
    "group-by-example": {
        "label": "Group-by-Example",
        "requires": ["data_groups"],
        "help": "One column holds the fold number (0, 1, 2, ...) for every example.",
    },
}

# How the data is held in memory while training (--data_representation).
REPRESENTATIONS = {
    "numpy": {
        "label": "Numpy Arrays",
        "default": True,  # Zero2Neuro's default, so it is not written to the file
        "help": "The default. Best for small data sets that fit entirely in RAM, and the most flexible way to handle the data.",
    },
    "tf-dataset": {
        "label": "Tensorflow Datasets",
        "tf_options": True,  # the cache and batch size apply to this choice
        "help": "Data processing pipelines, best for large data sets that are expensive to load from disk. Training can begin before all the data are loaded, and data can be cached to high-speed storage.",
    },
}

# What to do with a TF Dataset's cache (--data_tf_cache).
TF_CACHE_MODES = {
    "none": "No cache",
    "memory": "Cache in memory (RAM)",
    "disk": "Cache on disk (give a folder)",
}

SET_TYPES = {
    "fixed": {
        "label": "Fixed Assignment",
        "level": "Beginner",
        "max_folds": 3,
        "cross_validation": False,
        "help": "Fold 0 trains the model, fold 1 (if present) validates it, and fold 2 (if present) tests it. Good for quick, informal experiments. At most three folds.",
    },
    "holistic-cross-validation": {
        "label": "Holistic Cross-Validation",
        "level": "Intermediate",
        "min_folds": 3,
        "cross_validation": True,
        "requires": ["data_rotation"],
        "notice": "Cross-validation rotation is commonly set on the command line when you run Zero2Neuro (one run per rotation). Here you set the starting rotation.",
        "help": "Trains one model per fold rotation. Every fold is used exactly once for validation and once for testing. Needs at least three folds.",
    },
    "hold-out-cross-validation": {
        "label": "Hold-Out N-1 Fold Cross-Validation",
        "level": "Intermediate",
        "min_folds": 3,
        "cross_validation": True,
        "rotations_fewer_than_folds_by": 1,
        "requires": ["data_rotation"],
        "notice": "Cross-validation rotation is commonly set on the command line when you run Zero2Neuro (one run per rotation). Here you set the starting rotation.",
        "help": "Like holistic cross-validation, but the last fold is always the test set, so only N-1 rotations are possible (rotation 0 up to N-2). Needs at least three folds.",
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