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
            "--data_set_type=holistic-cross-validation\n"
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
        with self.assertRaisesRegex(ValueError, "Group column"):
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
            "--data_set_type=holistic-cross-validation\n", config)
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


if __name__ == "__main__":
    unittest.main()