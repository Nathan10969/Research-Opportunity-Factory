from __future__ import annotations

from typing import Any, Mapping


def validate_autoresearch_evidence(
    gate_a_report: Mapping[str, Any],
    final_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate that an autoresearch result respected the frozen PNNL protocol."""
    if gate_a_report.get("status") != "passed":
        return {
            "status": "blocked_by_gate_a",
            "passed": False,
            "failed_checks": ["gate_a_status"],
        }

    if final_report is None:
        return {
            "status": "missing_final_report",
            "passed": False,
            "failed_checks": ["final_report_present"],
        }

    checks = {
        "inner_group_cv": final_report.get("inner_group_cv") is True,
        "equal_trial_budget": final_report.get("equal_trial_budget") is True,
        "outer_test_used_for_selection": (
            final_report.get("outer_test_used_for_selection") is False
        ),
        "outer_sample_overlap": final_report.get("outer_sample_overlap") == 0,
        "per_fold_results_present": (
            final_report.get("per_fold_results_present") is True
        ),
        "per_sample_results_present": (
            final_report.get("per_sample_results_present") is True
        ),
        "per_seed_results_present": (
            final_report.get("per_seed_results_present") is True
        ),
        "configuration_provenance_present": (
            final_report.get("configuration_provenance_present") is True
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed else "failed_validation",
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }
