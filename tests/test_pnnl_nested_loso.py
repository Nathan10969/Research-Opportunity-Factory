import unittest

import numpy as np

from pnnl_pilot.nested_loso import (
    coordinate_features,
    elastic_l2_equivalent_ridge_alpha,
    sample_balanced_regression_metrics,
    select_config_inner_logo,
)


class NestedLosoTests(unittest.TestCase):
    def test_zero_l1_elastic_endpoint_uses_equivalent_ridge_penalty(self):
        self.assertAlmostEqual(elastic_l2_equivalent_ridge_alpha(0.25, 80), 20.0)

    def test_coordinate_features_are_normalized_by_source_dimensions(self):
        rows = [
            {"sample_id": "S1", "center_u_px": "25", "center_v_px": "75"},
            {"sample_id": "S2", "center_u_px": "30", "center_v_px": "20"},
        ]
        dimensions = {"S1": (100, 100), "S2": (60, 40)}
        np.testing.assert_allclose(
            coordinate_features(rows, dimensions),
            np.asarray([[0.25, 0.75], [0.50, 0.50]], dtype=np.float32),
        )

    def test_inner_selection_holds_out_whole_groups(self):
        x = np.arange(18, dtype=np.float32).reshape(9, 2)
        y = np.arange(9, dtype=np.float32)
        groups = np.asarray(["A"] * 3 + ["B"] * 3 + ["C"] * 3)
        seen = []

        def evaluate(config, train_index, valid_index):
            self.assertFalse(set(groups[train_index]) & set(groups[valid_index]))
            self.assertEqual(len(set(groups[valid_index])), 1)
            seen.append((config["bias"], groups[valid_index][0]))
            return y[valid_index] + config["bias"]

        selected, scores = select_config_inner_logo(
            x,
            y,
            groups,
            [{"bias": 2.0}, {"bias": 0.0}],
            evaluate,
        )
        self.assertEqual(selected, {"bias": 0.0})
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(seen), 6)

    def test_sample_balanced_metrics_weight_samples_equally(self):
        y_true = np.asarray([0.0, 0.0, 10.0])
        y_pred = np.asarray([2.0, 2.0, 10.0])
        groups = np.asarray(["large", "large", "small"])
        metrics = sample_balanced_regression_metrics(y_true, y_pred, groups)
        self.assertAlmostEqual(metrics["sample_balanced_mae"], 1.0)
        self.assertAlmostEqual(metrics["sample_balanced_rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
