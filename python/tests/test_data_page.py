import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import app


class TestDataPage(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.fields = dict(
            data_format="tabular", data_directory="", data_files="xor_data.csv",
            data_inputs="In 0\nIn 1", data_outputs="Out 0",
            data_fold_split="identity", data_set_type="fixed",
        )

    def test_page_has_expected_controls(self):
        response = self.client.get("/data")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        for name in ("data_format", "data_directory", "data_files", "data_inputs",
                     "data_outputs", "data_fold_split", "data_n_folds",
                     "data_groups", "data_stratify", "data_set_type"):
            with self.subTest(name=name):
                self.assertRegex(page, r'name="' + name + '"')
        self.assertIn("tooltip", page)
        self.assertIn("navigator.clipboard.writeText(configText)", page)

    def test_options_come_from_the_rule_tables(self):
        page = self.client.get("/data").get_data(as_text=True)
        for value in ("tabular", "pickle", "identity", "random", "random-stratify",
                      "group-by-file", "group-by-example", "fixed",
                      "holistic-cross-validation", "hold-out-cross-validation"):
            with self.subTest(value=value):
                self.assertIn(f'value="{value}"', page)
        # The extra fields start hidden for the default (identity) split.
        for option in ("data_n_folds", "data_groups", "data_stratify"):
            tag = re.search(r'<div[^>]*data-requires-option="' + option + r'"[^>]*>', page).group()
            self.assertIn("hidden", tag)

    def test_valid_submission_shows_config_and_copy_button(self):
        response = self.client.post("/data", data=self.fields)
        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("--data_format=tabular\n--data_file=xor_data.csv\n--data_set_type=fixed\n", page)
        self.assertIn("--data_inputs\nIn 0\nIn 1\n--data_outputs\nOut 0\n", page)
        self.assertIn('id="copy-button"', page)
        self.assertNotIn('class="error-message"', page)

    def test_random_split_submission(self):
        fields = dict(self.fields, data_fold_split="random", data_n_folds="10",
                      data_set_type="holistic-cross-validation", data_rotation="0")
        page = self.client.post("/data", data=fields).get_data(as_text=True)
        self.assertIn("--data_fold_split=random\n--data_n_folds=10\n", page)
        # The chosen split stays selected and its extra field stays visible.
        self.assertRegex(page, r'<option value="random"\s+selected')
        tag = re.search(r'<div[^>]*data-requires-option="data_n_folds"[^>]*>', page).group()
        self.assertNotIn("hidden", tag)

    def test_random_stratify_submission(self):
        fields = dict(self.fields, data_fold_split="random-stratify", data_n_folds="5",
                      data_stratify="Class", data_set_type="holistic-cross-validation", data_rotation="0")
        page = self.client.post("/data", data=fields).get_data(as_text=True)
        self.assertIn("--data_n_folds=5\n--data_stratify=Class\n", page)
        tag = re.search(r'<div[^>]*data-requires-option="data_stratify"[^>]*>', page).group()
        self.assertNotIn("hidden", tag)

    def test_invalid_submissions_show_error_and_keep_input(self):
        cases = {
            "data_files": "",
            "data_inputs": "   ",
            "data_outputs": "-bad",
            "data_n_folds": "abc",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                fields = dict(self.fields, **{field: value})
                if field == "data_n_folds":
                    fields.update(data_fold_split="random")
                page = self.client.post("/data", data=fields).get_data(as_text=True)
                self.assertIn('class="error-message"', page)
                self.assertNotIn('id="config-text"', page)
        page = self.client.post("/data", data=dict(self.fields, data_outputs="-bad")).get_data(as_text=True)
        self.assertIn("In 0", page)  # earlier input is not lost

    def test_random_split_without_folds_is_rejected(self):
        page = self.client.post("/data", data=dict(self.fields, data_fold_split="random")).get_data(as_text=True)
        self.assertIn('class="error-message"', page)
        self.assertIn("Number of folds", page)

    def test_network_page_still_works(self):
        self.assertEqual(self.client.get("/").status_code, 200)


    def test_new_fields_are_on_the_page(self):
        page = self.client.get("/data").get_data(as_text=True)
        for name in ("data_weights",
                     "tag_examples", "tabular_header_row", "tabular_header_names",
                     "tabular_column_range", "tabular_column_list", "data_rotation",
                     "n_validation_folds", "representation", "tf_cache_mode",
                     "tf_cache_path", "batch"):
            with self.subTest(name=name):
                self.assertRegex(page, r'name="' + name + '"')
        self.assertIn("commonly set on the command line", page)

    def test_cross_validation_submission_with_new_options(self):
        fields = dict(self.fields, data_fold_split="random", data_n_folds="5",
                      data_set_type="holistic-cross-validation", data_rotation="1",
                      n_validation_folds="2", data_weights="w", tag_examples="id",
                      representation="tf-dataset", tf_cache_mode="disk",
                      tf_cache_path="./cache", batch="16")
        page = self.client.post("/data", data=fields).get_data(as_text=True)
        for line in ("--data_rotation=1", "--data_n_validation_folds=2", "--data_weights=w",
                     "--data_tag_examples\nid", "--data_representation=tf-dataset",
                     "--data_tf_cache=./cache", "--data_batch=16"):
            with self.subTest(line=line):
                self.assertIn(line, page)

    def test_missing_rotation_shows_error(self):
        fields = dict(self.fields, data_fold_split="random", data_n_folds="5",
                      data_set_type="holistic-cross-validation")
        page = self.client.post("/data", data=fields).get_data(as_text=True)
        self.assertIn("needs: Rotation", page)

    def test_non_numbers_show_error(self):
        page = self.client.post("/data", data=dict(self.fields, tabular_column_list="a b")).get_data(as_text=True)
        self.assertIn("Column list must be whole numbers", page)


if __name__ == "__main__":
    unittest.main()