import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = ROOT / "results" / "history.json"
METRICS_FILE = ROOT / "results" / "validation_metrics.json"


class TestRecordedTrainingResults(unittest.TestCase):

    def test_history_file_exists(self):
        self.assertTrue(HISTORY_FILE.exists())

    def test_validation_metrics_file_exists(self):
        self.assertTrue(METRICS_FILE.exists())

    def test_history_contains_five_epochs(self):
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

        self.assertEqual(len(history["accuracy"]), 5)
        self.assertEqual(len(history["loss"]), 5)
        self.assertEqual(len(history["val_accuracy"]), 5)
        self.assertEqual(len(history["val_loss"]), 5)

    def test_history_series_have_equal_lengths(self):
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

        lengths = {
            len(history["accuracy"]),
            len(history["loss"]),
            len(history["val_accuracy"]),
            len(history["val_loss"]),
        }

        self.assertEqual(len(lengths), 1)

    def test_accuracy_values_are_valid(self):
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

        for value in history["accuracy"] + history["val_accuracy"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_loss_values_are_non_negative(self):
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

        for value in history["loss"] + history["val_loss"]:
            self.assertGreaterEqual(value, 0.0)

    def test_saved_metrics_match_best_validation_loss(self):
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))

        best_index = history["val_loss"].index(min(history["val_loss"]))

        self.assertAlmostEqual(
            metrics["loss"],
            history["val_loss"][best_index],
            places=10,
        )

        self.assertAlmostEqual(
            metrics["accuracy"],
            history["val_accuracy"][best_index],
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
