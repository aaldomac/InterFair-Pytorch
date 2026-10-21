import importlib.util
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from modules.data.compas import FEATURES, preprocess_compas

def sample():
    return pd.DataFrame({
        "age": [24, 35, 42, 61, 50],
        "juv_fel_count": [0]*5, "juv_misd_count": [0]*5,
        "juv_other_count": [0]*5, "priors_count": [0, 1, 2, 3, 4],
        "c_charge_degree": ["F", "M", "F", "M", "F"],
        "race": ["Caucasian", "Caucasian", "African-American", "African-American", "Hispanic"],
        "sex": ["Male", "Female", "Male", "Female", "Male"],
        "two_year_recid": [0, 1, 1, 0, 1],
        "days_b_screening_arrest": [-30, 30, 0, 0, 0],
        "is_recid": [0, 1, 1, 0, 1], "score_text": ["Low"]*5,
        "name": ["excluded"]*5, "decile_score": [9]*5,
        "r_days_from_arrest": [100]*5,
    })

class CompasTests(unittest.TestCase):
    def test_all_races_and_allowlist(self):
        frame, _, meta = preprocess_compas(sample())
        self.assertEqual(frame.group_id.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(set(frame), set(FEATURES) | {"race", "sex", "two_year_recid", "group", "group_id"})
        self.assertEqual(meta["row_counts"]["after_filters"], 5)

    def test_binary_subset(self):
        frame, mapping, _ = preprocess_compas(sample(), race_mode="black_white")
        self.assertEqual(len(frame), 4)
        self.assertEqual(mapping, {"0_0": 0, "0_1": 1, "1_0": 2, "1_1": 3})

    def test_filter_rules(self):
        for col, value in [("days_b_screening_arrest", 31),
                           ("days_b_screening_arrest", -31),
                           ("is_recid", -1), ("c_charge_degree", "O"),
                           ("score_text", "N/A")]:
            with self.subTest(col=col, value=value):
                df = sample()
                df.loc[0, col] = value
                self.assertEqual(len(preprocess_compas(df)[0]), 4)

    def test_missing_policy(self):
        df = sample()
        df.loc[0, "age"] = float("nan")
        self.assertEqual(len(preprocess_compas(df)[0]), 4)
        with self.assertRaises(ValueError):
            preprocess_compas(df, drop_na=False)

    def test_bad_data(self):
        for col, value in [("race", "unknown"), ("sex", "unknown"),
                           ("two_year_recid", 2), ("age", float("inf")),
                           ("priors_count", -1), ("priors_count", 1.5)]:
            with self.subTest(col=col):
                df = sample()
                df[col] = df[col].astype(object)
                df.loc[0, col] = value
                with self.assertRaises(ValueError):
                    preprocess_compas(df)

    def test_stable_groups_and_no_filter_columns(self):
        df = sample().iloc[[3, 2]]
        frame, _, _ = preprocess_compas(df.drop(columns=["score_text", "is_recid", "days_b_screening_arrest"]),
                                         apply_filters=False)
        self.assertEqual(frame.group_id.tolist(), [3, 2])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_loader_and_predictor_features(self):
        from modules.utils.dataset_utils import load_dataset_by_name, _feature_columns_from_spec
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"compas.csv"
            sample().to_csv(path, index=False)
            loaded = load_dataset_by_name("compas", csv_path=path)
            self.assertEqual(_feature_columns_from_spec(loaded.df, loaded.metadata["spec"]), list(FEATURES))
            self.assertEqual(loaded.metadata["spec"].protected_cols, ("race", "sex"))
            self.assertEqual(len(loaded.metadata["source_sha256"]), 64)

if __name__ == "__main__":
    unittest.main()
