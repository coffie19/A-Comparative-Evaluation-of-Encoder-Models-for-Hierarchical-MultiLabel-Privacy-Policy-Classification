import unittest

import numpy as np

from evaluation import compute_multilabel_classification_metrics


class MultilabelMetricTests(unittest.TestCase):
    def test_macro_f1_and_mean_labelwise_balanced_accuracy(self):
        y_true = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
        y_pred = np.array([[1, 1], [0, 0], [1, 0], [0, 0]])

        metrics = compute_multilabel_classification_metrics(
            y_true, y_pred, ["A", "B"]
        )

        self.assertEqual(metrics["primary_metric"], "macro_f1")
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 12)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.625)
        self.assertEqual(metrics["balanced_accuracy_labels_defined"], 2)

    def test_all_negative_predictions_have_half_balanced_accuracy(self):
        y_true = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
        y_pred = np.zeros_like(y_true)

        metrics = compute_multilabel_classification_metrics(
            y_true, y_pred, ["A", "B"]
        )

        self.assertEqual(metrics["macro_f1"], 0.0)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_multilabel_classification_metrics(
                np.zeros((2, 2)), np.zeros((2, 1)), ["A", "B"]
            )


if __name__ == "__main__":
    unittest.main()
