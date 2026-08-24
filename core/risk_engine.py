"""
RiskSense AI - Risk Engine

Deterministic weighted risk scoring for test cases.
No ML, no LLM involvement in score calculation.

Final Score = weighted sum of 5 sub-scores (each 0-100):
  - Historical Defect Risk:    25%
  - Historical Failure Risk:   20%
  - Current Change Impact:     25%
  - Business Criticality:      15%
  - Safety/Regulatory Impact:  15%
"""

import numpy as np
import pandas as pd

import config


def calculate_risk_scores(data):
    """
    Calculate risk scores for all test cases.

    Args:
        data: dict with keys: test_cases, test_executions, defects,
              release_changes, requirements

    Returns:
        DataFrame with test cases and their risk scores, sub-scores,
        and risk levels.
    """
    test_cases = data["test_cases"].copy()
    executions = data["test_executions"]
    defects = data["defects"]
    changes = data["release_changes"]

    # Pre-compute aggregations
    defect_stats = _compute_defect_stats(defects)
    failure_stats = _compute_failure_stats(executions)
    change_stats = _compute_change_stats(changes)

    # Calculate sub-scores for each test case
    scores = []
    for _, tc in test_cases.iterrows():
        test_id = tc["test_id"]
        req_id = tc["req_id"]
        module = tc["module"]

        # Sub-score 1: Historical Defect Risk (0-100)
        defect_score = _calc_defect_score(test_id, req_id, module, defect_stats)

        # Sub-score 2: Historical Failure Risk (0-100)
        failure_score = _calc_failure_score(test_id, failure_stats)

        # Sub-score 3: Current Change Impact (0-100)
        change_score = _calc_change_impact_score(req_id, module, change_stats)

        # Sub-score 4: Business Criticality (0-100)
        biz_score = config.MODULE_BUSINESS_CRITICALITY.get(module, 50)

        # Sub-score 5: Safety/Regulatory Impact (0-100)
        safety_score = config.MODULE_SAFETY_REGULATORY.get(module, 50)

        # Weighted final score
        final_score = (
            config.RISK_WEIGHTS["historical_defect"] * defect_score
            + config.RISK_WEIGHTS["historical_failure"] * failure_score
            + config.RISK_WEIGHTS["change_impact"] * change_score
            + config.RISK_WEIGHTS["business_criticality"] * biz_score
            + config.RISK_WEIGHTS["safety_regulatory"] * safety_score
        )

        # Clamp to 0-100
        final_score = float(np.clip(final_score, 0, 100))

        # Determine risk level
        risk_level = _get_risk_level(final_score)

        scores.append({
            "test_id": test_id,
            "req_id": req_id,
            "module": module,
            "test_name": tc["test_name"],
            "test_type": tc["test_type"],
            "priority": tc["priority"],
            "complexity": tc["complexity"],
            "historical_defect_score": round(defect_score, 2),
            "historical_failure_score": round(failure_score, 2),
            "change_impact_score": round(change_score, 2),
            "business_criticality_score": round(biz_score, 2),
            "safety_regulatory_score": round(safety_score, 2),
            "risk_score": round(final_score, 2),
            "risk_level": risk_level,
        })

    result = pd.DataFrame(scores)

    # Add additional context columns
    result = _add_context_columns(result, defect_stats, failure_stats, change_stats)

    # Sort by risk score descending
    result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result


def _compute_defect_stats(defects):
    """Pre-compute defect statistics per test and per requirement."""
    stats = {}

    # Defects per test
    if len(defects) > 0:
        test_defects = defects.groupby("test_id").agg(
            defect_count=("defect_id", "count"),
            critical_count=("severity", lambda x: (x == "Critical").sum()),
            high_count=("severity", lambda x: (x == "High").sum()),
        ).to_dict("index")
        stats["by_test"] = test_defects

        # Defects per requirement
        req_defects = defects.groupby("req_id").agg(
            defect_count=("defect_id", "count"),
        ).to_dict("index")
        stats["by_req"] = req_defects

        # Defects per module
        mod_defects = defects.groupby("module").agg(
            defect_count=("defect_id", "count"),
        ).to_dict("index")
        stats["by_module"] = mod_defects

        # Global max for normalization
        stats["max_test_defects"] = max(
            (d["defect_count"] for d in test_defects.values()), default=1
        )
        stats["max_req_defects"] = max(
            (d["defect_count"] for d in req_defects.values()), default=1
        )
    else:
        stats["by_test"] = {}
        stats["by_req"] = {}
        stats["by_module"] = {}
        stats["max_test_defects"] = 1
        stats["max_req_defects"] = 1

    return stats


def _compute_failure_stats(executions):
    """Pre-compute failure rates per test case."""
    stats = {}

    if len(executions) > 0:
        test_exec = executions.groupby("test_id").agg(
            total=("exec_id", "count"),
            fail_count=("status", lambda x: (x == "Fail").sum()),
        )
        test_exec["failure_rate"] = test_exec["fail_count"] / test_exec["total"]
        stats["by_test"] = test_exec.to_dict("index")
    else:
        stats["by_test"] = {}

    return stats


def _compute_change_stats(changes):
    """Pre-compute change impact statistics for R5."""
    stats = {}

    r5_changes = changes[changes["release"] == config.CURRENT_RELEASE]

    if len(r5_changes) > 0:
        # Changes per requirement
        req_changes = r5_changes.groupby("req_id").agg(
            change_count=("change_id", "count"),
            has_bug_fix=("change_type", lambda x: ("Bug Fix" in x.values)),
            has_new_feature=("change_type", lambda x: ("New Feature" in x.values)),
        ).to_dict("index")
        stats["by_req"] = req_changes

        # Changes per module
        mod_changes = r5_changes.groupby("module").agg(
            change_count=("change_id", "count"),
        ).to_dict("index")
        stats["by_module"] = mod_changes

        stats["max_req_changes"] = max(
            (d["change_count"] for d in req_changes.values()), default=1
        )
    else:
        stats["by_req"] = {}
        stats["by_module"] = {}
        stats["max_req_changes"] = 1

    return stats


def _calc_defect_score(test_id, req_id, module, defect_stats):
    """
    Calculate historical defect risk score (0-100).

    Considers defects directly on this test, its requirement, and its module.
    Weighted: direct test defects (50%), requirement defects (30%), module defects (20%).
    """
    by_test = defect_stats["by_test"]
    by_req = defect_stats["by_req"]
    by_module = defect_stats["by_module"]

    # Test-level defects (normalized)
    test_info = by_test.get(test_id, {"defect_count": 0, "critical_count": 0, "high_count": 0})
    test_count = test_info["defect_count"]
    # Severity bonus: critical and high defects weigh more
    severity_multiplier = 1.0 + (test_info["critical_count"] * 0.5 + test_info["high_count"] * 0.25)
    test_normalized = min((test_count * severity_multiplier) / max(defect_stats["max_test_defects"], 1), 1.0)

    # Requirement-level defects (normalized)
    req_count = by_req.get(req_id, {"defect_count": 0})["defect_count"]
    req_normalized = min(req_count / max(defect_stats["max_req_defects"], 1), 1.0)

    # Module-level defects (normalized against average)
    mod_count = by_module.get(module, {"defect_count": 0})["defect_count"]
    avg_mod = sum(d["defect_count"] for d in by_module.values()) / max(len(by_module), 1)
    mod_normalized = min(mod_count / max(avg_mod * 2, 1), 1.0)

    score = (test_normalized * 0.50 + req_normalized * 0.30 + mod_normalized * 0.20) * 100
    return min(score, 100)


def _calc_failure_score(test_id, failure_stats):
    """
    Calculate historical failure risk score (0-100).

    Based on failure rate of the test across historical executions.
    """
    by_test = failure_stats["by_test"]
    info = by_test.get(test_id, {"failure_rate": 0.0, "total": 0})
    failure_rate = info["failure_rate"]

    # Scale: 0% failure → 0, 50%+ failure → 100
    score = min(failure_rate / 0.50, 1.0) * 100
    return min(score, 100)


def _calc_change_impact_score(req_id, module, change_stats):
    """
    Calculate current change impact score (0-100).

    Based on R5 changes touching this test's requirement and module.
    Bug fixes and new features have higher impact.
    """
    by_req = change_stats["by_req"]
    by_module = change_stats["by_module"]
    max_changes = change_stats.get("max_req_changes", 1)

    # Requirement-level changes
    req_info = by_req.get(req_id, {"change_count": 0, "has_bug_fix": False, "has_new_feature": False})
    req_change_count = req_info["change_count"]

    # Change type multiplier
    type_multiplier = 1.0
    if req_info.get("has_bug_fix", False):
        type_multiplier += 0.3
    if req_info.get("has_new_feature", False):
        type_multiplier += 0.2

    req_normalized = min((req_change_count * type_multiplier) / max(max_changes, 1), 1.0)

    # Module-level change density
    mod_count = by_module.get(module, {"change_count": 0})["change_count"]
    avg_mod = sum(d["change_count"] for d in by_module.values()) / max(len(by_module), 1)
    mod_normalized = min(mod_count / max(avg_mod * 2, 1), 1.0)

    # Combine: requirement changes (70%) + module change density (30%)
    score = (req_normalized * 0.70 + mod_normalized * 0.30) * 100

    # If no changes at all, score is 0
    if req_change_count == 0 and mod_count == 0:
        score = 0

    return min(score, 100)


def _get_risk_level(score):
    """Determine risk level from score."""
    for level, (low, high) in config.RISK_LEVELS.items():
        if low <= score <= high:
            return level
    return "Low"


def _add_context_columns(result, defect_stats, failure_stats, change_stats):
    """Add human-readable context columns to results."""
    result["defect_count"] = result["test_id"].apply(
        lambda tid: defect_stats["by_test"].get(tid, {"defect_count": 0})["defect_count"]
    )
    result["failure_rate"] = result["test_id"].apply(
        lambda tid: round(failure_stats["by_test"].get(tid, {"failure_rate": 0.0})["failure_rate"] * 100, 1)
    )
    result["change_count"] = result["req_id"].apply(
        lambda rid: change_stats["by_req"].get(rid, {"change_count": 0})["change_count"]
    )

    return result
