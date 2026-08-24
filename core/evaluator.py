"""
RiskSense AI - Evaluator

Computes summary statistics, effort reduction metrics,
and dynamic heatmap cross-tabulations.
"""

import pandas as pd

import config


def compute_summary(risk_results):
    """
    Compute summary statistics from risk results.

    Args:
        risk_results: DataFrame from risk_engine.calculate_risk_scores()

    Returns:
        dict with summary metrics
    """
    total = len(risk_results)

    # Count by risk level
    level_counts = {}
    for level in config.RISK_LEVEL_ORDER:
        level_counts[level] = int((risk_results["risk_level"] == level).sum())

    # Module-level aggregation
    module_stats = (
        risk_results.groupby("module")
        .agg(
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            test_count=("test_id", "count"),
            critical_count=("risk_level", lambda x: (x == "Critical").sum()),
            high_count=("risk_level", lambda x: (x == "High").sum()),
            medium_count=("risk_level", lambda x: (x == "Medium").sum()),
            low_count=("risk_level", lambda x: (x == "Low").sum()),
        )
        .round(2)
    )

    # Ensure all 12 modules exist
    for mod in config.MODULES:
        if mod not in module_stats.index:
            module_stats.loc[mod] = {
                "avg_risk": 0.0,
                "max_risk": 0.0,
                "test_count": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            }

    # Sort by avg_risk descending
    module_stats = module_stats.sort_values("avg_risk", ascending=False)

    # Average risk score
    avg_risk = round(risk_results["risk_score"].mean(), 2) if total > 0 else 0.0
    max_risk = round(risk_results["risk_score"].max(), 2) if total > 0 else 0.0
    min_risk = round(risk_results["risk_score"].min(), 2) if total > 0 else 0.0

    return {
        "total_tests": total,
        "level_counts": level_counts,
        "critical_count": level_counts.get("Critical", 0),
        "high_count": level_counts.get("High", 0),
        "medium_count": level_counts.get("Medium", 0),
        "low_count": level_counts.get("Low", 0),
        "avg_risk_score": avg_risk,
        "max_risk_score": max_risk,
        "min_risk_score": min_risk,
        "module_stats": module_stats,
    }


def compute_effort_reduction(risk_results, reduction_pct=0.15):
    """
    Estimate effort reduction from risk-based prioritization.

    Args:
        risk_results: DataFrame from risk_engine
        reduction_pct: float, default 0.15

    Returns:
        dict with effort metrics
    """
    total = len(risk_results)
    recommended = total - int(total * reduction_pct)
    reduced = total - recommended

    return {
        "total_tests": total,
        "recommended_tests": recommended,
        "reduced_tests": reduced,
        "reduction_pct": round(reduction_pct * 100, 1),
    }


def compute_heatmap_data(risk_results):
    """
    Compute dynamic heatmap data: module (Y) x risk level (X) -> test count.

    X-axis: Low / Medium / High / Critical
    Y-axis: 12 hospital modules (ordered with high/critical concentration at top)
    Cell: Number of test cases

    Args:
        risk_results: DataFrame with 'module' and 'risk_level' columns

    Returns:
        DataFrame suitable for heatmap visualization (rows = 12 modules, cols = 4 risk levels)
    """
    if len(risk_results) > 0:
        heatmap = pd.crosstab(
            risk_results["module"],
            risk_results["risk_level"],
        )
    else:
        heatmap = pd.DataFrame()

    # Ensure all 12 modules exist as rows
    for mod in config.MODULES:
        if mod not in heatmap.index:
            heatmap.loc[mod] = 0

    # Ensure all 4 risk levels exist as columns
    for level in config.HEATMAP_X_ORDER:
        if level not in heatmap.columns:
            heatmap[level] = 0

    # Fill any NaNs with 0
    heatmap = heatmap.fillna(0).astype(int)

    # Order X-axis: Low -> Medium -> High -> Critical
    heatmap = heatmap[config.HEATMAP_X_ORDER]

    # Order Y-axis by concentration of Critical + High tests descending
    # (Higher weighted Critical so high-risk modules rank at top)
    heatmap["_sort_key"] = heatmap["Critical"] * 2 + heatmap["High"]
    heatmap = heatmap.sort_values("_sort_key", ascending=False)
    heatmap = heatmap.drop(columns=["_sort_key"])

    return heatmap
