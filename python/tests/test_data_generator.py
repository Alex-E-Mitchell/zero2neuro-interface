import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generator import generate_data_config, parse_lines, parse_optional_int
from data_rules import DATA_FORMATS, FOLD_SPLITS, SET_TYPES, missing_required


def xor_arguments(**changes):
    arguments = dict(
        data_format="tabular",
        data_files=["xor_data.csv"],
        data_inputs=["In 0", "In 1"],
        data_outputs=["Out 0"],
        fold_split="identity",
        set_type="fixed",
        rotation=0,
    )
    arguments.update(changes)
    return arguments


class TestGenerateDataConfig(unittest.TestCase):
    def test_xor_example_matches_zero2neuro_example(self):
        self.assertEqual(
            generate_data_config(**xor_arguments()),
            "--data_format=tabular\n--data_file=xor_data.csv\n--data_set_type=fixed\n"
            "--data_inputs\nIn 0\nIn 1\n--data_outputs\nOut 0\n",
        )

    def test_random_cross_validation_with_directory(self):
        config = generate_data_config(**xor_arguments(
            data_files=["iris_data.csv"], fold_split="random", n_folds=10,
            set_type="holistic-cross-validation", data_directory=" ./data ",
        ))
        self.assertEqual(
            config,
            "--data_format=tabular\n--dataset_directory=./data\n--data_file=iris_data.csv\n"
            "--data_fold_split=random\n--data_n_folds=10\n"
            "--data_set_type=holistic-cross-validation\n--data_rotation=0\n"
            "--data_inputs\nIn 0\nIn 1\n--data_outputs\nOut 0\n",
        )

    def test_multiple_files_use_data_files_list(self):
        config = generate_data_config(**xor_arguments(data_files=["a.csv", "b.csv"]))
        self.assertIn("--data_files\na.csv\nb.csv\n", config)
        self.assertNotIn("--data_file=", config)

    def test_group_by_file_requires_fold_numbers_and_always_uses_list(self):
        config = generate_data_config(**xor_arguments(
            data_format="pickle", data_files=["a.pkl,0", "b.pkl, 0", "c.pkl,1"],
            fold_split="group-by-file",
        ))
        self.assertIn("--data_files\na.pkl,0\nb.pkl,0\nc.pkl,1\n", config)
        self.assertIn("--data_fold_split=group-by-file\n", config)
        for bad in (["a.pkl"], ["a.pkl,x"], ["a.pkl,-1"], [",0"]):
            with self.subTest(files=bad):
                with self.assertRaisesRegex(ValueError, "file,fold"):
                    generate_data_config(**xor_arguments(data_files=bad, fold_split="group-by-file"))

    def test_group_by_example_needs_group_column(self):
        config = generate_data_config(**xor_arguments(
            fold_split="group-by-example", data_groups="fold",
            set_type="hold-out-cross-validation",
        ))
        self.assertIn("--data_groups=fold\n", config)
        with self.assertRaisesRegex(ValueError, "Data Groups"):
            generate_data_config(**xor_arguments(fold_split="group-by-example"))

    def test_unused_options_are_left_out(self):
        config = generate_data_config(**xor_arguments(n_folds=5, data_groups="fold"))
        self.assertNotIn("data_n_folds", config)
        self.assertNotIn("data_groups", config)
        self.assertNotIn("data_fold_split", config)

    def test_random_split_needs_a_valid_number_of_folds(self):
        with self.assertRaisesRegex(ValueError, "Number of folds"):
            generate_data_config(**xor_arguments(fold_split="random"))
        for bad in (0, 1, -3):
            with self.subTest(n_folds=bad):
                with self.assertRaisesRegex(ValueError, "at least 2"):
                    generate_data_config(**xor_arguments(fold_split="random", n_folds=bad))

    def test_fixed_assignment_allows_at_most_three_folds(self):
        generate_data_config(**xor_arguments(fold_split="random", n_folds=3))
        with self.assertRaisesRegex(ValueError, "at most 3"):
            generate_data_config(**xor_arguments(fold_split="random", n_folds=4))
        with self.assertRaisesRegex(ValueError, "at most 3"):
            generate_data_config(**xor_arguments(data_files=["a", "b", "c", "d"]))
        with self.assertRaisesRegex(ValueError, "at most 3"):
            generate_data_config(**xor_arguments(
                data_files=["a,0", "b,1", "c,2", "d,3"], fold_split="group-by-file"))
        generate_data_config(**xor_arguments(
            fold_split="random", n_folds=4, set_type="holistic-cross-validation"))

    def test_random_stratify_needs_folds_and_class_column(self):
        config = generate_data_config(**xor_arguments(
            fold_split="random-stratify", n_folds=5, data_stratify="Class",
            set_type="holistic-cross-validation",
        ))
        self.assertIn(
            "--data_fold_split=random-stratify\n--data_n_folds=5\n--data_stratify=Class\n"
            "--data_set_type=holistic-cross-validation\n--data_rotation=0\n", config)
        with self.assertRaisesRegex(ValueError, "Class column"):
            generate_data_config(**xor_arguments(
                fold_split="random-stratify", n_folds=5, set_type="holistic-cross-validation"))
        with self.assertRaisesRegex(ValueError, "Number of folds"):
            generate_data_config(**xor_arguments(
                fold_split="random-stratify", data_stratify="Class",
                set_type="holistic-cross-validation"))

    def test_cross_validation_needs_at_least_three_folds(self):
        for set_type in ("holistic-cross-validation", "hold-out-cross-validation"):
            with self.subTest(set_type=set_type):
                with self.assertRaisesRegex(ValueError, "at least 3"):
                    generate_data_config(**xor_arguments(
                        fold_split="random", n_folds=2, set_type=set_type))
                with self.assertRaisesRegex(ValueError, "at least 3"):
                    generate_data_config(**xor_arguments(set_type=set_type))  # one file, one fold
                generate_data_config(**xor_arguments(
                    fold_split="random", n_folds=3, set_type=set_type))

    def test_group_by_file_fold_numbers_cannot_skip(self):
        with self.assertRaisesRegex(ValueError, "no gaps"):
            generate_data_config(**xor_arguments(
                data_files=["a.csv,0", "b.csv,2"], fold_split="group-by-file"))
        with self.assertRaisesRegex(ValueError, "no gaps"):
            generate_data_config(**xor_arguments(
                data_files=["a.csv,1", "b.csv,2"], fold_split="group-by-file"))

    def test_spaces_in_file_names_are_rejected_for_file_lists(self):
        with self.assertRaisesRegex(ValueError, "space"):
            generate_data_config(**xor_arguments(data_files=["my data.csv", "b.csv"]))
        with self.assertRaisesRegex(ValueError, "space"):
            generate_data_config(**xor_arguments(
                data_files=["my data.csv,0"], fold_split="group-by-file"))
        # A single file is written with --data_file=, which keeps the name whole.
        config = generate_data_config(**xor_arguments(data_files=["my data.csv"]))
        self.assertIn("--data_file=my data.csv\n", config)

    def test_missing_pieces_are_rejected(self):
        for field in ("data_files", "data_inputs", "data_outputs"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    generate_data_config(**xor_arguments(**{field: []}))
        for field, value in (("data_format", "csv"), ("fold_split", "shuffle"), ("set_type", "magic")):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    generate_data_config(**xor_arguments(**{field: value}))

    def test_text_zero2neuro_would_misread_is_rejected(self):
        # '#' starts a comment, a leading '-' looks like an option, '@' includes a file.
        for text in ("data#1.csv", "-weight", "@file"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_lines(text, "Columns")
        for field, value in (("data_directory", "./my#data"), ("data_groups", "-fold")):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    generate_data_config(**xor_arguments(
                        fold_split="group-by-example", **{field: value}))

    def test_column_names_with_spaces_are_kept(self):
        config = generate_data_config(**xor_arguments(data_inputs=["sepal length", "petal width"]))
        self.assertIn("--data_inputs\nsepal length\npetal width\n", config)


class TestHelpers(unittest.TestCase):
    def test_parse_lines(self):
        self.assertEqual(parse_lines("  a \n\n b b\r\n", "Columns"), ["a", "b b"])
        for blank in ("", "  \n \n"):
            with self.assertRaises(ValueError):
                parse_lines(blank, "Columns")
        self.assertEqual(parse_lines("", "Columns", allow_empty=True), [])

    def test_parse_optional_int(self):
        self.assertIsNone(parse_optional_int("", "Folds"))
        self.assertIsNone(parse_optional_int(None, "Folds"))
        self.assertEqual(parse_optional_int(" 10 ", "Folds"), 10)
        for bad in ("1.5", "ten"):
            with self.assertRaises(ValueError):
                parse_optional_int(bad, "Folds")


class TestRuleTables(unittest.TestCase):
    def test_tables_have_labels_and_help(self):
        for table in (DATA_FORMATS, FOLD_SPLITS, SET_TYPES):
            for key, rule in table.items():
                with self.subTest(choice=key):
                    self.assertTrue(rule["label"])
                    self.assertTrue(rule["help"])

    def test_missing_required_reads_the_table(self):
        self.assertEqual(missing_required(FOLD_SPLITS, "identity", {}), [])
        self.assertEqual(missing_required(FOLD_SPLITS, "random", {}), ["data_n_folds"])
        self.assertEqual(missing_required(FOLD_SPLITS, "random", {"data_n_folds": 5}), [])
        with self.assertRaises(ValueError):
            missing_required(FOLD_SPLITS, "nope", {})



class TestNewDataOptions(unittest.TestCase):
    def test_cross_validation_needs_rotation_and_checks_range(self):
        with self.assertRaisesRegex(ValueError, "Rotation"):
            generate_data_config(**xor_arguments(
                fold_split="random", n_folds=5, set_type="holistic-cross-validation", rotation=None))
        with self.assertRaisesRegex(ValueError, "0 to 4"):
            generate_data_config(**xor_arguments(
                fold_split="random", n_folds=5, set_type="holistic-cross-validation", rotation=5))
        config = generate_data_config(**xor_arguments(
            fold_split="random", n_folds=5, set_type="hold-out-cross-validation",
            rotation=2, n_validation_folds=2))
        self.assertIn("--data_rotation=2\n--data_n_validation_folds=2\n", config)

    def test_validation_folds_leave_room_for_training_and_testing(self):
        with self.assertRaisesRegex(ValueError, "at most 3"):
            generate_data_config(**xor_arguments(
                fold_split="random", n_folds=5, set_type="holistic-cross-validation",
                n_validation_folds=4))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            generate_data_config(**xor_arguments(
                fold_split="random", n_folds=5, set_type="holistic-cross-validation",
                n_validation_folds=0))

    def test_rotation_options_left_out_for_fixed(self):
        config = generate_data_config(**xor_arguments(rotation=3, n_validation_folds=2))
        self.assertNotIn("rotation", config)
        self.assertNotIn("validation_folds", config)

    def test_weights_and_tags(self):
        config = generate_data_config(**xor_arguments(
            data_weights="weight", tag_examples=["subject", "trial"]))
        self.assertIn("--data_weights=weight\n--data_tag_examples\nsubject\ntrial\n", config)
        with self.assertRaisesRegex(ValueError, "Data Weights"):
            generate_data_config(**xor_arguments(data_weights="#bad"))
        with self.assertRaisesRegex(ValueError, "Data Tag Examples"):
            generate_data_config(**xor_arguments(tag_examples=["-bad"]))

    def test_tabular_options(self):
        config = generate_data_config(**xor_arguments(
            tabular_header_row=2, tabular_column_range=[1, 4]))
        self.assertIn("--tabular_header_row=2\n--tabular_column_range\n1\n4\n", config)
        config = generate_data_config(**xor_arguments(
            tabular_header_names=["In 0", "In 1", "Out 0"], tabular_column_list=[0, 1, 4]))
        self.assertIn("--tabular_header_names\nIn 0\nIn 1\nOut 0\n--tabular_column_list\n0\n1\n4\n", config)
        for bad, message in (
            (dict(tabular_header_row=1, tabular_header_names=["a"]), "not both"),
            (dict(tabular_column_range=[1, 2], tabular_column_list=[1]), "not both"),
            (dict(tabular_column_range=[1]), "exactly 2"),
            (dict(tabular_column_range=[4, 1]), "Column range"),
            (dict(tabular_header_row=-1), "Header row"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, message):
                    generate_data_config(**xor_arguments(**bad))
        with self.assertRaisesRegex(ValueError, "only apply to tabular"):
            generate_data_config(**xor_arguments(data_format="pickle", tabular_header_row=1))

    def test_tf_dataset_representation_options(self):
        base = xor_arguments(representation="tf-dataset")
        self.assertIn("--data_representation=tf-dataset\n",
                      generate_data_config(**base))
        self.assertIn("--data_tf_cache=\n--data_batch=64\n",
                      generate_data_config(**dict(base, tf_cache_mode="memory", batch=64)))
        self.assertIn("--data_tf_cache=./cache\n",
                      generate_data_config(**dict(base, tf_cache_mode="disk", tf_cache_path="./cache")))
        with self.assertRaisesRegex(ValueError, "Cache folder"):
            generate_data_config(**dict(base, tf_cache_mode="disk"))
        with self.assertRaisesRegex(ValueError, "Batch size"):
            generate_data_config(**dict(base, batch=0))
        # numpy is the default, so cache and batch are ignored and left out.
        config = generate_data_config(**xor_arguments(tf_cache_mode="memory", batch=64))
        self.assertNotIn("data_representation", config)
        self.assertNotIn("data_tf_cache", config)
        self.assertNotIn("data_batch", config)

    def test_tf_dataset_incoming_format(self):
        config = generate_data_config(**xor_arguments(
            data_format="tf-dataset", data_files=["a", "b"], data_inputs=[], data_outputs=[],
            representation="tf-dataset"))
        self.assertIn("--data_format=tf-dataset\n", config)
        self.assertNotIn("--data_inputs", config)
        with self.assertRaisesRegex(ValueError, "must use"):
            generate_data_config(**xor_arguments(
                data_format="tf-dataset", data_inputs=[], data_outputs=[]))
        with self.assertRaisesRegex(ValueError, "not available"):
            generate_data_config(**xor_arguments(
                data_format="tf-dataset", representation="tf-dataset",
                fold_split="random", n_folds=3, data_inputs=[], data_outputs=[]))

    def test_columns_still_required_for_other_formats(self):
        with self.assertRaisesRegex(ValueError, "Data Inputs"):
            generate_data_config(**xor_arguments(data_inputs=[]))
        with self.assertRaisesRegex(ValueError, "Data Outputs"):
            generate_data_config(**xor_arguments(data_outputs=[]))


if __name__ == "__main__":
    unittest.main()


class TestDocsRules(unittest.TestCase):
    def test_hold_out_has_one_fewer_rotation(self):
        generate_data_config(**xor_arguments(
            fold_split="random", n_folds=5, set_type="holistic-cross-validation", rotation=4))
        generate_data_config(**xor_arguments(
            fold_split="random", n_folds=5, set_type="hold-out-cross-validation", rotation=3))
        with self.assertRaisesRegex(ValueError, "0 to 3"):
            generate_data_config(**xor_arguments(
                fold_split="random", n_folds=5, set_type="hold-out-cross-validation", rotation=4))

    def test_training_folds(self):
        cv = dict(fold_split="random", n_folds=6, set_type="holistic-cross-validation")
        config = generate_data_config(**xor_arguments(**cv, n_training_folds=3, n_validation_folds=2))
        self.assertIn("--data_n_validation_folds=2\n--data_n_training_folds=3\n", config)
        with self.assertRaisesRegex(ValueError, "cannot be more than"):
            generate_data_config(**xor_arguments(**cv, n_training_folds=4, n_validation_folds=2))
        with self.assertRaisesRegex(ValueError, "Training folds"):
            generate_data_config(**xor_arguments(**cv, n_training_folds=0))
        self.assertNotIn("training_folds", generate_data_config(**xor_arguments(n_training_folds=3)))


class TestDictionaryDriven(unittest.TestCase):
    def test_new_split_needs_only_a_dictionary_entry(self):
        from unittest import mock
        import data_rules
        new_split = {
            "label": "Random Split (test copy)",
            "folds_from": "data_n_folds",
            "requires": ["data_n_folds"],
            "help": "A split added only in the dictionary.",
        }
        with mock.patch.dict(data_rules.FOLD_SPLITS, {"random-copy": new_split}):
            config = generate_data_config(**xor_arguments(
                fold_split="random-copy", n_folds=4, set_type="holistic-cross-validation"))
            self.assertIn("--data_fold_split=random-copy\n--data_n_folds=4\n", config)
            with self.assertRaisesRegex(ValueError, "Number of folds"):
                generate_data_config(**xor_arguments(
                    fold_split="random-copy", set_type="holistic-cross-validation"))

class TestOptionsTable(unittest.TestCase):
    def test_options_the_choice_does_not_use_are_ignored(self):
        # The identity split has no use for a group or class column, so they
        # are neither checked nor written.
        config = generate_data_config(**xor_arguments(
            data_groups="#x", data_stratify="@c", n_folds=1))
        self.assertNotIn("data_groups", config)
        self.assertNotIn("data_stratify", config)
        self.assertNotIn("data_n_folds", config)

    def test_every_option_row_is_complete(self):
        import data_rules
        for name, rule in data_rules.OPTIONS.items():
            with self.subTest(option=name):
                self.assertIn(rule["kind"], ("text", "int", "list", "int_list"))
                self.assertTrue(rule["label"])
                if rule["kind"] in ("int", "int_list"):
                    self.assertIn("min", rule)
                self.assertIn(rule.get("applies_when", "always"),
                              ("always", "required_by_split", "tabular_options",
                               "cross_validation", "tf_options"))
        for first, second in data_rules.EXCLUSIVE_OPTIONS:
            self.assertIn(first, data_rules.OPTIONS)
            self.assertIn(second, data_rules.OPTIONS)