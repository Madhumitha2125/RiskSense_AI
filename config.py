"""
RiskSense AI - Configuration

Centralized configuration for the application.
All constants, weights, thresholds, and module definitions.
"""

# --------------------------------------------------
# Application
# --------------------------------------------------
APP_NAME = "RiskSense AI"
APP_VERSION = "0.1.0"
APP_SUBTITLE = "Risk-Based Regression Testing"

# --------------------------------------------------
# Random Seed (reproducibility)
# --------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------
# Data Paths
# --------------------------------------------------
DATA_DIR = "data"
RELEASES_FILE = f"{DATA_DIR}/releases.csv"
REQUIREMENTS_FILE = f"{DATA_DIR}/requirements.csv"
TEST_CASES_FILE = f"{DATA_DIR}/test_cases.csv"
TEST_EXECUTIONS_FILE = f"{DATA_DIR}/test_executions.csv"
DEFECTS_FILE = f"{DATA_DIR}/defects.csv"
RELEASE_CHANGES_FILE = f"{DATA_DIR}/release_changes.csv"

# --------------------------------------------------
# Releases
# --------------------------------------------------
RELEASES = ["R1", "R2", "R3", "R4", "R5"]
CURRENT_RELEASE = "R5"

# --------------------------------------------------
# Hospital Modules (12)
# --------------------------------------------------
MODULES = [
    "Patient Registration",
    "Appointment Management",
    "Doctor Management",
    "Emergency Department",
    "Inpatient Management",
    "Pharmacy",
    "Laboratory",
    "Radiology",
    "Billing",
    "Insurance",
    "Medical Records",
    "User Access Management",
]

# Business Criticality scores per module (0-100)
MODULE_BUSINESS_CRITICALITY = {
    "Patient Registration": 70,
    "Appointment Management": 55,
    "Doctor Management": 50,
    "Emergency Department": 95,
    "Inpatient Management": 80,
    "Pharmacy": 90,
    "Laboratory": 75,
    "Radiology": 65,
    "Billing": 72,
    "Insurance": 60,
    "Medical Records": 68,
    "User Access Management": 45,
}

# Safety / Regulatory Impact scores per module (0-100)
MODULE_SAFETY_REGULATORY = {
    "Patient Registration": 60,
    "Appointment Management": 35,
    "Doctor Management": 40,
    "Emergency Department": 98,
    "Inpatient Management": 85,
    "Pharmacy": 95,
    "Laboratory": 88,
    "Radiology": 80,
    "Billing": 50,
    "Insurance": 55,
    "Medical Records": 75,
    "User Access Management": 65,
}

# Module inherent defect-proneness weights (for data generation)
MODULE_DEFECT_WEIGHT = {
    "Patient Registration": 0.12,
    "Appointment Management": 0.07,
    "Doctor Management": 0.05,
    "Emergency Department": 0.15,
    "Inpatient Management": 0.10,
    "Pharmacy": 0.13,
    "Laboratory": 0.09,
    "Radiology": 0.05,
    "Billing": 0.11,
    "Insurance": 0.06,
    "Medical Records": 0.04,
    "User Access Management": 0.03,
}

# --------------------------------------------------
# Risk Scoring Weights
# --------------------------------------------------
RISK_WEIGHTS = {
    "historical_defect": 0.25,
    "historical_failure": 0.20,
    "change_impact": 0.25,
    "business_criticality": 0.15,
    "safety_regulatory": 0.15,
}

# --------------------------------------------------
# Risk Level Thresholds
# --------------------------------------------------
RISK_LEVELS = {
    "Critical": (80, 100),
    "High": (60, 79),
    "Medium": (40, 59),
    "Low": (0, 39),
}

RISK_LEVEL_ORDER = ["Critical", "High", "Medium", "Low"]
HEATMAP_X_ORDER = ["Low", "Medium", "High", "Critical"]

RISK_COLORS = {
    "Critical": "#DC2626",
    "High": "#F97316",
    "Medium": "#EAB308",
    "Low": "#22C55E",
}

# --------------------------------------------------
# Data Generation Targets
# --------------------------------------------------
TARGET_REQUIREMENTS = 120
TARGET_TEST_CASES = 240
TARGET_EXECUTIONS = 5000
TARGET_DEFECTS = 600
TARGET_RELEASE_CHANGES = 250

# --------------------------------------------------
# Categorical Values
# --------------------------------------------------
REQUIREMENT_TYPES = ["Functional", "Non-Functional", "Security", "Performance", "Compliance"]
REQUIREMENT_PRIORITIES = ["Critical", "High", "Medium", "Low"]

TEST_TYPES = ["Functional", "Integration", "Security", "Performance", "Regression"]
TEST_PRIORITIES = ["Critical", "High", "Medium", "Low"]
TEST_COMPLEXITIES = ["High", "Medium", "Low"]

EXECUTION_STATUSES = ["Pass", "Fail", "Blocked"]

DEFECT_SEVERITIES = ["Critical", "High", "Medium", "Low"]

CHANGE_TYPES = ["New Feature", "Bug Fix", "Enhancement", "Configuration", "Refactoring"]

# --------------------------------------------------
# Regression Plan Reduction Levels
# --------------------------------------------------
REDUCTION_LEVELS = {
    "15%": 0.15,
    "20%": 0.20,
}

# --------------------------------------------------
# Ollama Configuration
# --------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_CONNECT_TIMEOUT = 5
OLLAMA_READ_TIMEOUT = 20

# --------------------------------------------------
# UI Configuration
# --------------------------------------------------
PAGES = ["Dashboard", "Risk Analysis", "Risk Heatmap", "Regression Plan"]
TOP_N_RISKS = 10
