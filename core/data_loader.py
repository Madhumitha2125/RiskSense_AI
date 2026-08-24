"""
RiskSense AI - Data Loader & Validator

Loads CSV datasets and validates data integrity including
duplicate IDs, missing values, invalid references, and
invalid categorical values.
"""

import os

import pandas as pd

import config


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def load_all_data():
    """
    Load all 6 CSV datasets and run validation.

    Returns:
        dict: Dictionary with keys: releases, requirements, test_cases,
              test_executions, defects, release_changes
    Raises:
        DataValidationError: If any validation check fails.
        FileNotFoundError: If any data file is missing.
    """
    data_files = {
        "releases": config.RELEASES_FILE,
        "requirements": config.REQUIREMENTS_FILE,
        "test_cases": config.TEST_CASES_FILE,
        "test_executions": config.TEST_EXECUTIONS_FILE,
        "defects": config.DEFECTS_FILE,
        "release_changes": config.RELEASE_CHANGES_FILE,
    }

    # Check all files exist
    for name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file '{path}' not found. "
                f"Run 'python generator/generate_data.py' to generate datasets."
            )

    # Load all CSVs
    data = {}
    for name, path in data_files.items():
        data[name] = pd.read_csv(path)

    # Run validations
    _validate_data(data)

    return data


def _validate_data(data):
    """Run all validation checks on loaded data."""
    errors = []

    # 1. Check for duplicate IDs
    _check_duplicates(data["releases"], "release_id", "releases", errors)
    _check_duplicates(data["requirements"], "req_id", "requirements", errors)
    _check_duplicates(data["test_cases"], "test_id", "test_cases", errors)
    _check_duplicates(data["test_executions"], "exec_id", "test_executions", errors)
    _check_duplicates(data["defects"], "defect_id", "defects", errors)
    _check_duplicates(data["release_changes"], "change_id", "release_changes", errors)

    # 2. Check for missing values in key columns
    _check_missing(data["requirements"], ["req_id", "module", "priority"], "requirements", errors)
    _check_missing(data["test_cases"], ["test_id", "req_id", "module"], "test_cases", errors)
    _check_missing(data["test_executions"], ["exec_id", "test_id", "status"], "test_executions", errors)
    _check_missing(data["defects"], ["defect_id", "test_id", "module", "severity"], "defects", errors)
    _check_missing(data["release_changes"], ["change_id", "req_id", "module"], "release_changes", errors)

    # 3. Check invalid references
    valid_req_ids = set(data["requirements"]["req_id"])
    valid_test_ids = set(data["test_cases"]["test_id"])
    valid_releases = set(config.RELEASES)

    # test_cases.req_id → requirements.req_id
    invalid_refs = set(data["test_cases"]["req_id"]) - valid_req_ids
    if invalid_refs:
        errors.append(f"test_cases: invalid req_id references: {invalid_refs}")

    # test_executions.test_id → test_cases.test_id
    invalid_refs = set(data["test_executions"]["test_id"]) - valid_test_ids
    if invalid_refs:
        errors.append(f"test_executions: invalid test_id references: {invalid_refs}")

    # defects.test_id → test_cases.test_id
    invalid_refs = set(data["defects"]["test_id"]) - valid_test_ids
    if invalid_refs:
        errors.append(f"defects: invalid test_id references: {invalid_refs}")

    # defects.req_id → requirements.req_id
    invalid_refs = set(data["defects"]["req_id"]) - valid_req_ids
    if invalid_refs:
        errors.append(f"defects: invalid req_id references: {invalid_refs}")

    # release_changes.req_id → requirements.req_id
    invalid_refs = set(data["release_changes"]["req_id"]) - valid_req_ids
    if invalid_refs:
        errors.append(f"release_changes: invalid req_id references: {invalid_refs}")

    # 4. Check invalid modules
    valid_modules = set(config.MODULES)
    for dataset_name in ["requirements", "test_cases", "defects", "release_changes"]:
        if "module" in data[dataset_name].columns:
            invalid_mods = set(data[dataset_name]["module"]) - valid_modules
            if invalid_mods:
                errors.append(f"{dataset_name}: invalid modules: {invalid_mods}")

    # 5. Check invalid releases
    for dataset_name in ["requirements", "test_executions", "defects", "release_changes"]:
        if "release" in data[dataset_name].columns:
            invalid_rels = set(data[dataset_name]["release"]) - valid_releases
            if invalid_rels:
                errors.append(f"{dataset_name}: invalid releases: {invalid_rels}")

    # 6. Check invalid categorical values
    _check_values(data["test_executions"], "status", config.EXECUTION_STATUSES, "test_executions", errors)
    _check_values(data["defects"], "severity", config.DEFECT_SEVERITIES, "defects", errors)
    _check_values(data["test_cases"], "test_type", config.TEST_TYPES, "test_cases", errors)
    _check_values(data["test_cases"], "priority", config.TEST_PRIORITIES, "test_cases", errors)
    _check_values(data["release_changes"], "change_type", config.CHANGE_TYPES, "release_changes", errors)

    if errors:
        raise DataValidationError(
            "Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def _check_duplicates(df, id_col, dataset_name, errors):
    """Check for duplicate IDs in a DataFrame."""
    if id_col in df.columns:
        dupes = df[df.duplicated(subset=[id_col], keep=False)]
        if len(dupes) > 0:
            errors.append(f"{dataset_name}: {len(dupes)} duplicate {id_col} values")


def _check_missing(df, columns, dataset_name, errors):
    """Check for missing values in key columns."""
    for col in columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                errors.append(f"{dataset_name}: {missing} missing values in '{col}'")


def _check_values(df, col, valid_values, dataset_name, errors):
    """Check for invalid categorical values."""
    if col in df.columns:
        invalid = set(df[col].dropna()) - set(valid_values)
        if invalid:
            errors.append(f"{dataset_name}: invalid {col} values: {invalid}")
