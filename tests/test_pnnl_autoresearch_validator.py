import unittest

from pnnl_pilot.autoresearch_validator import validate_autoresearch_evidence


class AutoresearchValidatorTests(unittest.TestCase):
    def test_blocks_when_gate_a_has_not_passed(self):
        result = validate_autoresearch_evidence(
            {"status": "partial"},
            None,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "blocked_by_gate_a")

    def test_rejects_outer_test_selection_and_unfair_budget(self):
        result = validate_autoresearch_evidence(
            {"status": "passed"},
            {
                "inner_group_cv": True,
                "equal_trial_budget": False,
                "outer_test_used_for_selection": True,
                "outer_sample_overlap": 0,
                "per_fold_results_present": True,
                "per_sample_results_present": True,
                "per_seed_results_present": True,
                "configuration_provenance_present": True,
            },
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "failed_validation")
        self.assertIn("equal_trial_budget", result["failed_checks"])
        self.assertIn("outer_test_used_for_selection", result["failed_checks"])

    def test_accepts_only_complete_leakage_safe_evidence(self):
        result = validate_autoresearch_evidence(
            {"status": "passed"},
            {
                "inner_group_cv": True,
                "equal_trial_budget": True,
                "outer_test_used_for_selection": False,
                "outer_sample_overlap": 0,
                "per_fold_results_present": True,
                "per_sample_results_present": True,
                "per_seed_results_present": True,
                "configuration_provenance_present": True,
            },
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["failed_checks"], [])


if __name__ == "__main__":
    unittest.main()
