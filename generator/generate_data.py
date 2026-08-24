"""
RiskSense AI - Synthetic Data Generator

Generates reproducible synthetic hospital testing data with realistic
relationships and non-uniform risk distribution across modules.

Usage:
    python generator/generate_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_releases():
    """Generate 5 releases. R5 is the current release."""
    data = {
        "release_id": ["R1", "R2", "R3", "R4", "R5"],
        "release_name": [
            "Foundation Release",
            "Core Services Update",
            "Integration Release",
            "Stability Release",
            "Current Release",
        ],
        "release_date": [
            "2024-01-15",
            "2024-04-20",
            "2024-07-10",
            "2024-10-05",
            "2025-01-20",
        ],
        "status": ["Completed", "Completed", "Completed", "Completed", "In Progress"],
    }
    return pd.DataFrame(data)


def generate_requirements(rng):
    """Generate 120 requirements distributed across modules and releases."""
    requirements = []
    req_id = 1

    # Distribute requirements across modules (10 per module)
    for module in config.MODULES:
        count = config.TARGET_REQUIREMENTS // len(config.MODULES)
        for _ in range(count):
            # Assign to releases with slight bias toward earlier releases
            release = rng.choice(config.RELEASES, p=[0.25, 0.25, 0.20, 0.15, 0.15])

            # Higher-risk modules get more critical/high priority requirements
            weight = config.MODULE_DEFECT_WEIGHT[module]
            if weight >= 0.12:
                priority = rng.choice(
                    config.REQUIREMENT_PRIORITIES, p=[0.30, 0.35, 0.25, 0.10]
                )
            elif weight >= 0.08:
                priority = rng.choice(
                    config.REQUIREMENT_PRIORITIES, p=[0.15, 0.30, 0.35, 0.20]
                )
            else:
                priority = rng.choice(
                    config.REQUIREMENT_PRIORITIES, p=[0.10, 0.20, 0.35, 0.35]
                )

            req_type = rng.choice(config.REQUIREMENT_TYPES, p=[0.40, 0.15, 0.15, 0.15, 0.15])

            requirements.append({
                "req_id": f"REQ-{req_id:03d}",
                "module": module,
                "requirement_name": f"{module} - Requirement {req_id}",
                "priority": priority,
                "type": req_type,
                "release": release,
            })
            req_id += 1

    return pd.DataFrame(requirements)


def generate_test_cases(rng, requirements_df):
    """Generate 240 test cases linked to requirements (2 per requirement)."""
    test_cases = []
    test_id = 1

    for _, req in requirements_df.iterrows():
        # 2 test cases per requirement
        for t in range(2):
            test_type = rng.choice(config.TEST_TYPES, p=[0.35, 0.20, 0.15, 0.10, 0.20])

            # Priority correlates with requirement priority
            req_prio_idx = config.REQUIREMENT_PRIORITIES.index(req["priority"])
            prio_probs = [0.1, 0.2, 0.4, 0.3]
            # Shift probability mass toward matching priority
            prio_probs[req_prio_idx] = max(prio_probs[req_prio_idx], 0.40)
            total = sum(prio_probs)
            prio_probs = [p / total for p in prio_probs]
            priority = rng.choice(config.TEST_PRIORITIES, p=prio_probs)

            complexity = rng.choice(config.TEST_COMPLEXITIES, p=[0.25, 0.50, 0.25])

            test_cases.append({
                "test_id": f"TC-{test_id:03d}",
                "req_id": req["req_id"],
                "module": req["module"],
                "test_name": f"{req['module']} - Test {test_id} ({'A' if t == 0 else 'B'})",
                "test_type": test_type,
                "priority": priority,
                "complexity": complexity,
                "estimated_duration_min": int(rng.choice([5, 10, 15, 20, 30, 45, 60])),
            })
            test_id += 1

    return pd.DataFrame(test_cases)


def generate_test_executions(rng, test_cases_df):
    """Generate ~5000 test executions with module-dependent failure rates."""
    executions = []
    exec_id = 1

    # Each test case is executed across multiple releases
    for _, tc in test_cases_df.iterrows():
        module = tc["module"]
        defect_weight = config.MODULE_DEFECT_WEIGHT[module]

        # Base failure rate derived from module defect weight
        base_fail_rate = min(0.05 + defect_weight * 2.5, 0.45)

        # Number of executions per test: ~20 on average, varies
        num_executions = int(rng.normal(loc=config.TARGET_EXECUTIONS / config.TARGET_TEST_CASES, scale=5))
        num_executions = max(5, min(35, num_executions))

        for _ in range(num_executions):
            release = rng.choice(config.RELEASES[:4], p=[0.30, 0.30, 0.25, 0.15])

            # Failure probability varies by release (earlier releases more buggy)
            release_factor = {"R1": 1.3, "R2": 1.1, "R3": 0.9, "R4": 0.8}
            fail_rate = min(base_fail_rate * release_factor[release], 0.50)
            blocked_rate = 0.05

            rand_val = rng.random()
            if rand_val < fail_rate:
                status = "Fail"
            elif rand_val < fail_rate + blocked_rate:
                status = "Blocked"
            else:
                status = "Pass"

            duration = max(1, int(rng.normal(loc=tc["estimated_duration_min"], scale=5)))

            executions.append({
                "exec_id": f"EX-{exec_id:05d}",
                "test_id": tc["test_id"],
                "release": release,
                "status": status,
                "duration_min": duration,
                "execution_date": _random_date(rng, release),
            })
            exec_id += 1

    return pd.DataFrame(executions)


def generate_defects(rng, test_executions_df, test_cases_df):
    """Generate ~600 defects from failed test executions."""
    failed_execs = test_executions_df[test_executions_df["status"] == "Fail"].copy()

    # Merge to get module info
    failed_execs = failed_execs.merge(
        test_cases_df[["test_id", "module", "req_id"]], on="test_id", how="left"
    )

    # Sample ~600 defects from failures
    num_defects = min(config.TARGET_DEFECTS, len(failed_execs))
    sampled = failed_execs.sample(n=num_defects, random_state=config.RANDOM_SEED, replace=False)

    defects = []
    defect_id = 1
    for _, row in sampled.iterrows():
        module = row["module"]
        defect_weight = config.MODULE_DEFECT_WEIGHT[module]

        # Higher-risk modules produce more severe defects
        if defect_weight >= 0.12:
            severity = rng.choice(config.DEFECT_SEVERITIES, p=[0.25, 0.35, 0.30, 0.10])
        elif defect_weight >= 0.08:
            severity = rng.choice(config.DEFECT_SEVERITIES, p=[0.10, 0.30, 0.40, 0.20])
        else:
            severity = rng.choice(config.DEFECT_SEVERITIES, p=[0.05, 0.15, 0.40, 0.40])

        defects.append({
            "defect_id": f"DEF-{defect_id:03d}",
            "test_id": row["test_id"],
            "req_id": row["req_id"],
            "module": module,
            "severity": severity,
            "release": row["release"],
            "description": f"Defect in {module} during {row['release']} testing",
            "status": rng.choice(["Open", "Closed", "Resolved"], p=[0.15, 0.55, 0.30]),
        })
        defect_id += 1

    return pd.DataFrame(defects)


def generate_release_changes(rng, requirements_df):
    """Generate ~250 release changes for R5 (current release)."""
    # Filter requirements — changes reference existing requirements
    all_req_ids = requirements_df["req_id"].tolist()

    changes = []
    change_id = 1

    # Distribute changes with bias toward high-risk modules
    modules_weighted = []
    weights = []
    for module in config.MODULES:
        modules_weighted.append(module)
        weights.append(config.MODULE_DEFECT_WEIGHT[module])
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    for _ in range(config.TARGET_RELEASE_CHANGES):
        module = rng.choice(modules_weighted, p=weights)

        # Pick a requirement from this module
        module_reqs = requirements_df[requirements_df["module"] == module]["req_id"].tolist()
        if module_reqs:
            req_id = rng.choice(module_reqs)
        else:
            req_id = rng.choice(all_req_ids)

        change_type = rng.choice(config.CHANGE_TYPES, p=[0.20, 0.30, 0.25, 0.15, 0.10])

        descriptions = {
            "New Feature": f"New functionality added to {module}",
            "Bug Fix": f"Critical bug fix in {module} workflow",
            "Enhancement": f"Performance improvement in {module}",
            "Configuration": f"Configuration update for {module}",
            "Refactoring": f"Code refactoring in {module} module",
        }

        changes.append({
            "change_id": f"CHG-{change_id:03d}",
            "req_id": req_id,
            "module": module,
            "change_type": change_type,
            "release": config.CURRENT_RELEASE,
            "description": descriptions[change_type],
            "impact_area": rng.choice(["Core Logic", "Data Flow", "UI", "Integration", "Security"]),
        })
        change_id += 1

    return pd.DataFrame(changes)


def _random_date(rng, release):
    """Generate a random date within a release's timeframe."""
    date_ranges = {
        "R1": ("2024-01-15", "2024-04-19"),
        "R2": ("2024-04-20", "2024-07-09"),
        "R3": ("2024-07-10", "2024-10-04"),
        "R4": ("2024-10-05", "2025-01-19"),
    }
    start, end = date_ranges.get(release, ("2025-01-20", "2025-03-31"))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    delta = (end_ts - start_ts).days
    random_days = rng.integers(0, max(delta, 1))
    return str((start_ts + pd.Timedelta(days=int(random_days))).date())


def generate_all():
    """Generate all synthetic datasets and save to data/ directory."""
    rng = np.random.default_rng(config.RANDOM_SEED)

    os.makedirs(config.DATA_DIR, exist_ok=True)

    print("Generating releases...")
    releases_df = generate_releases()
    releases_df.to_csv(config.RELEASES_FILE, index=False)
    print(f"  -> {len(releases_df)} releases")

    print("Generating requirements...")
    requirements_df = generate_requirements(rng)
    requirements_df.to_csv(config.REQUIREMENTS_FILE, index=False)
    print(f"  -> {len(requirements_df)} requirements")

    print("Generating test cases...")
    test_cases_df = generate_test_cases(rng, requirements_df)
    test_cases_df.to_csv(config.TEST_CASES_FILE, index=False)
    print(f"  -> {len(test_cases_df)} test cases")

    print("Generating test executions...")
    executions_df = generate_test_executions(rng, test_cases_df)
    executions_df.to_csv(config.TEST_EXECUTIONS_FILE, index=False)
    print(f"  -> {len(executions_df)} test executions")

    print("Generating defects...")
    defects_df = generate_defects(rng, executions_df, test_cases_df)
    defects_df.to_csv(config.DEFECTS_FILE, index=False)
    print(f"  -> {len(defects_df)} defects")

    print("Generating release changes...")
    changes_df = generate_release_changes(rng, requirements_df)
    changes_df.to_csv(config.RELEASE_CHANGES_FILE, index=False)
    print(f"  -> {len(changes_df)} release changes")

    print("\nAll datasets generated successfully.")
    return {
        "releases": releases_df,
        "requirements": requirements_df,
        "test_cases": test_cases_df,
        "test_executions": executions_df,
        "defects": defects_df,
        "release_changes": changes_df,
    }


if __name__ == "__main__":
    generate_all()
