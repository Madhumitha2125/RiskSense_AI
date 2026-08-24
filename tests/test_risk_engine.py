"""
RiskSense AI - Validation Tests

Validates data generation, risk calculations, heatmap,
regression plans, and Ollama integration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.data_loader import load_all_data
from core.risk_engine import calculate_risk_scores
from core.prioritizer import generate_regression_plan
from core.evaluator import compute_summary, compute_heatmap_data
from ai.ollama_service import check_ollama_status, generate_risk_explanation


def test_data_loading():
    """Test 1: Data loading and validation."""
    print("=" * 60)
    print("TEST 1: Data Loading & Validation")
    print("=" * 60)

    data = load_all_data()

    counts = {
        "releases": len(data["releases"]),
        "requirements": len(data["requirements"]),
        "test_cases": len(data["test_cases"]),
        "test_executions": len(data["test_executions"]),
        "defects": len(data["defects"]),
        "release_changes": len(data["release_changes"]),
    }

    for name, count in counts.items():
        print(f"  {name}: {count} rows")

    assert counts["releases"] == 5, f"Expected 5 releases, got {counts['releases']}"
    assert counts["requirements"] == config.TARGET_REQUIREMENTS, \
        f"Expected {config.TARGET_REQUIREMENTS} requirements, got {counts['requirements']}"
    assert counts["test_cases"] == config.TARGET_TEST_CASES, \
        f"Expected {config.TARGET_TEST_CASES} test cases, got {counts['test_cases']}"

    # Verify all 12 modules are represented
    modules_in_data = set(data["requirements"]["module"])
    assert modules_in_data == set(config.MODULES), \
        f"Missing modules: {set(config.MODULES) - modules_in_data}"

    # Verify R5 changes
    r5_changes = data["release_changes"][data["release_changes"]["release"] == "R5"]
    assert len(r5_changes) == len(data["release_changes"]), \
        "All release changes should be for R5"

    print("  âœ… Data loading & validation PASSED")
    return data


def test_relationships(data):
    """Test 2: Referential integrity."""
    print("\n" + "=" * 60)
    print("TEST 2: Relationship Integrity")
    print("=" * 60)

    req_ids = set(data["requirements"]["req_id"])
    test_ids = set(data["test_cases"]["test_id"])

    # Test cases â†’ Requirements
    tc_req_ids = set(data["test_cases"]["req_id"])
    invalid = tc_req_ids - req_ids
    assert len(invalid) == 0, f"Invalid test_caseâ†’requirement refs: {invalid}"
    print("  âœ… test_cases â†’ requirements: OK")

    # Executions â†’ Test cases
    exec_test_ids = set(data["test_executions"]["test_id"])
    invalid = exec_test_ids - test_ids
    assert len(invalid) == 0, f"Invalid executionâ†’test refs: {invalid}"
    print("  âœ… test_executions â†’ test_cases: OK")

    # Defects â†’ Test cases
    def_test_ids = set(data["defects"]["test_id"])
    invalid = def_test_ids - test_ids
    assert len(invalid) == 0, f"Invalid defectâ†’test refs: {invalid}"
    print("  âœ… defects â†’ test_cases: OK")

    # Defects â†’ Requirements
    def_req_ids = set(data["defects"]["req_id"])
    invalid = def_req_ids - req_ids
    assert len(invalid) == 0, f"Invalid defectâ†’requirement refs: {invalid}"
    print("  âœ… defects â†’ requirements: OK")

    # Changes â†’ Requirements
    chg_req_ids = set(data["release_changes"]["req_id"])
    invalid = chg_req_ids - req_ids
    assert len(invalid) == 0, f"Invalid changeâ†’requirement refs: {invalid}"
    print("  âœ… release_changes â†’ requirements: OK")

    print("  âœ… Relationship integrity PASSED")


def test_risk_calculations(data):
    """Test 3: Risk scoring."""
    print("\n" + "=" * 60)
    print("TEST 3: Risk Calculations")
    print("=" * 60)

    risk_results = calculate_risk_scores(data)

    assert len(risk_results) == config.TARGET_TEST_CASES, \
        f"Expected {config.TARGET_TEST_CASES} scored tests, got {len(risk_results)}"

    # All scores in 0-100
    assert risk_results["risk_score"].min() >= 0, "Score below 0 found"
    assert risk_results["risk_score"].max() <= 100, "Score above 100 found"

    # All risk levels valid
    valid_levels = set(config.RISK_LEVEL_ORDER)
    actual_levels = set(risk_results["risk_level"])
    assert actual_levels.issubset(valid_levels), \
        f"Invalid risk levels: {actual_levels - valid_levels}"

    # Sub-scores in 0-100
    for col in ["historical_defect_score", "historical_failure_score",
                "change_impact_score", "business_criticality_score",
                "safety_regulatory_score"]:
        assert risk_results[col].min() >= 0, f"{col} below 0"
        assert risk_results[col].max() <= 100, f"{col} above 100"

    # Non-uniform distribution: at least 2 risk levels should have tests
    level_counts = risk_results["risk_level"].value_counts()
    assert len(level_counts) >= 2, "Risk distribution too uniform"

    print(f"  Score range: {risk_results['risk_score'].min():.1f} - {risk_results['risk_score'].max():.1f}")
    print(f"  Average score: {risk_results['risk_score'].mean():.1f}")
    print(f"  Risk levels: {dict(level_counts)}")
    print("  âœ… Risk calculations PASSED")

    return risk_results


def test_heatmap(risk_results):
    """Test 4: Heatmap generation."""
    print("\n" + "=" * 60)
    print("TEST 4: Heatmap Generation")
    print("=" * 60)

    heatmap = compute_heatmap_data(risk_results)

    # Should have columns for risk levels
    for level in config.RISK_LEVEL_ORDER:
        assert level in heatmap.columns, f"Missing column: {level}"

    # All 12 modules should be rows
    assert len(heatmap) == len(config.MODULES), \
        f"Expected {len(config.MODULES)} modules, got {len(heatmap)}"

    # Total should equal total test cases
    total = heatmap.values.sum()
    assert total == config.TARGET_TEST_CASES, \
        f"Heatmap total {total} != {config.TARGET_TEST_CASES}"

    print(f"  Shape: {heatmap.shape}")
    print(f"  Total cells: {total}")
    print("  âœ… Heatmap generation PASSED")


def test_regression_plans(risk_results, data):
    """Test 5-6: Regression plan at 15% and 20%."""
    print("\n" + "=" * 60)
    print("TEST 5-6: Regression Plans")
    print("=" * 60)

    for label, pct in config.REDUCTION_LEVELS.items():
        plan = generate_regression_plan(risk_results, data, pct)

        expected_recommended = config.TARGET_TEST_CASES - int(config.TARGET_TEST_CASES * pct)

        assert plan["total_tests"] == config.TARGET_TEST_CASES
        assert plan["recommended_count"] == expected_recommended, \
            f"{label}: expected {expected_recommended}, got {plan['recommended_count']}"
        assert plan["reduced_count"] == config.TARGET_TEST_CASES - expected_recommended

        # Coverage should be > 50% (risk-based selection should retain high-risk defects)
        coverage = plan["defect_coverage"]["coverage_pct"]
        assert coverage > 50, \
            f"{label}: coverage {coverage}% too low for risk-based selection"

        print(f"  {label} reduction:")
        print(f"    Recommended: {plan['recommended_count']} tests")
        print(f"    Reduced: {plan['reduced_count']} tests")
        print(f"    High/Critical defect coverage: {coverage}%")

    print("  âœ… Regression plans PASSED")


def test_ollama():
    """Test 7-8: Ollama integration and fallback."""
    print("\n" + "=" * 60)
    print("TEST 7-8: Ollama Integration & Fallback")
    print("=" * 60)

    status = check_ollama_status()
    print(f"  Status: {status['message']}")

    # Test explanation generation (should work regardless of Ollama status)
    test_info = {"test_id": "TC-001", "test_name": "Test Case 1", "module": "Emergency Department"}
    risk_factors = {
        "risk_score": 85.0,
        "risk_level": "Critical",
        "historical_defect_score": 75.0,
        "historical_failure_score": 60.0,
        "change_impact_score": 90.0,
        "business_criticality_score": 95.0,
        "safety_regulatory_score": 98.0,
        "defect_count": 5,
        "failure_rate": 30.0,
        "change_count": 3,
    }

    explanation = generate_risk_explanation(test_info, risk_factors)
    assert len(explanation) > 0, "Empty explanation generated"

    print(f"  Explanation length: {len(explanation)} chars")
    print(f"  Preview: {explanation[:100]}...")

    if status["available"]:
        print("  âœ… Ollama integration PASSED (LLM response)")
    else:
        print("  âœ… Ollama fallback PASSED (template response)")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("  RiskSense AI â€” Validation Suite")
    print("=" * 60)

    # Test 1: Data loading
    data = test_data_loading()

    # Test 2: Relationships
    test_relationships(data)

    # Test 3: Risk calculations
    risk_results = test_risk_calculations(data)

    # Test 4: Heatmap
    test_heatmap(risk_results)

    # Test 5-6: Regression plans
    test_regression_plans(risk_results, data)

    # Test 7-8: Ollama
    test_ollama()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED âœ…")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

