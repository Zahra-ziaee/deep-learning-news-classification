import unittest
from unittest.mock import patch

import pandas as pd

from src.data_loader import load_news_dataset
from src.preprocessing import prepare_text_data, split_dataset


class TestDataSplitting(unittest.TestCase):
    def test_disjoint_stratified_splits(self):
        labels = ["graphics", "baseball", "medicine", "politics"]
        rows = [
            {"clean_text": f"sample_{label}_{index}", "category": label}
            for label in labels
            for index in range(50)
        ]
        df = pd.DataFrame(rows)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
        self.assertEqual(len(df), len(X_train) + len(X_val) + len(X_test))
        self.assertEqual(len(X_train), 128)
        self.assertEqual(len(X_val), 32)
        self.assertEqual(len(X_test), 40)
        self.assertFalse(set(X_train.index) & set(X_val.index))
        self.assertFalse(set(X_train.index) & set(X_test.index))
        self.assertFalse(set(X_val.index) & set(X_test.index))
        for target in (y_train, y_val, y_test):
            self.assertEqual(set(target.unique()), set(labels))
        self.assertTrue(y_train.index.equals(X_train.index))
        self.assertTrue(y_val.index.equals(X_val.index))
        self.assertTrue(y_test.index.equals(X_test.index))

    def test_duplicate_cleaned_text_is_dropped(self):
        df = pd.DataFrame({
            "text": [
                "Medical research and science progress!",
                "Medical research and science progress?",
                "Baseball team scores a home run today.",
            ],
            "category": ["science", "science", "sports"],
        })
        cleaned = prepare_text_data(df)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["clean_text"].nunique(), 2)

    def test_insufficient_data_has_clear_error(self):
        df = pd.DataFrame({
            "clean_text": [f"text_{i}" for i in range(8)],
            "category": ["A", "A", "B", "B", "C", "C", "D", "D"],
        })
        with self.assertRaisesRegex(ValueError, "Not enough examples"):
            split_dataset(df)

    def test_download_failure_does_not_silently_fallback(self):
        with patch(
            "src.data_loader.fetch_20newsgroups",
            side_effect=OSError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "dataset is unavailable"):
                load_news_dataset()


if __name__ == "__main__":
    unittest.main()
