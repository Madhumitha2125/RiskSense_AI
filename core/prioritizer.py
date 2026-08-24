"""
RiskSense AI - Prioritizer

Generates optimized regression test suites by selecting
highest-risk tests and calculating coverage metrics.
"""

import pandas as pd

import config


def generate_regression_plan(risk_results, data, reduction_pct):
    """
    Generate a reduced regression suite by selecting highest-risk tests.

    Args:
        risk_results: DataFrame from risk_engine.calculate_risk_scores()
        data: dict with all datasets (for defect coverage calculation)
        reduction_pct: float, e.g. 0.15 for 15% reduction

    Returns:
        dict with:
            - recommended_tests: DataFrame of selected tests
            - total_tests: int
            - recommended_count: int
            - reduced_count: int
            - reduction_pct: float
            - defect_coverage: dict with coverage metrics
    """
    total = len(risk_results)
    recommended_count = total - int(total * reduction_pct)

    # Select top N tests by risk score (already sorted descending)
    sorted_results = risk_results.sort_values("risk_score", ascending=False)
    recommended = sorted_results.head(recommended_count).copy()
    excluded = sorted_results.tail(total - recommended_count)

    # Calculate defect coverage
    defect_coverage = _calculate_defect_coverage(
        recommended, excluded, data["defects"]
    )

    return {
        "recommended_tests": recommended,
        "excluded_tests": excluded,
        "total_tests": total,
        "recommended_count": recommended_count,
        "reduced_count": total - recommended_count,
        "reduction_pct": reduction_pct,
        "defect_coverage": defect_coverage,
    }


def _calculate_defect_coverage(recommended, excluded, defects):
    """
    Calculate what percentage of historical high/critical defects
    are covered by the recommended suite.
    """
    recommended_test_ids = set(recommended["test_id"])
    recommended_req_ids = set(recommended["req_id"])

    # All high/critical severity defects
    high_crit_defects = defects[defects["severity"].isin(["Critical", "High"])]
    total_high_crit = len(high_crit_defects)

    if total_high_crit == 0:
        return {
            "total_high_critical_defects": 0,
            "covered_high_critical_defects": 0,
            "coverage_pct": 100.0,
            "total_defects": len(defects),
            "covered_defects": 0,
            "total_coverage_pct": 100.0,
        }

    # Defects covered = defects whose test_id is in the recommended suite
    covered = high_crit_defects[
        high_crit_defects["test_id"].isin(recommended_test_ids)
    ]
    covered_count = len(covered)
    coverage_pct = round((covered_count / total_high_crit) * 100, 1)

    # Total defect coverage
    all_covered = defects[defects["test_id"].isin(recommended_test_ids)]
    total_coverage_pct = round((len(all_covered) / max(len(defects), 1)) * 100, 1)

    return {
        "total_high_critical_defects": total_high_crit,
        "covered_high_critical_defects": covered_count,
        "coverage_pct": coverage_pct,
        "total_defects": len(defects),
        "covered_defects": len(all_covered),
        "total_coverage_pct": total_coverage_pct,
    }


def get_regression_suite_csv(recommended_tests):
    """
    Convert recommended tests to CSV string for download.

    Args:
        recommended_tests: DataFrame of selected tests

    Returns:
        str: CSV content
    """
    export_cols = [
        "rank", "test_id", "test_name", "module", "test_type",
        "priority", "risk_score", "risk_level",
        "defect_count", "failure_rate", "change_count",
    ]
    available_cols = [c for c in export_cols if c in recommended_tests.columns]
    return recommended_tests[available_cols].to_csv(index=False)
