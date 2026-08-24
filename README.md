# RiskSense AI - Risk-Based Regression Testing POC

**UC-006: RiskSense AI – Risk-Based Testing**

An intelligent risk-based test optimization platform that analyzes synthetic hospital requirements, historical defects, test execution history, and release changes (R5) to identify and prioritize high-risk regression scenarios.

---

## 🎯 Business Objective

Demonstrate **15–20% regression effort reduction** through deterministic risk-based prioritization while preserving **>97% historical high/critical defect coverage**.

- **No ML / Black-box models**: Pure deterministic weighted scoring.
- **Local AI Explanations**: Integrates with local Ollama (`llama3.2`) via REST API for semantic explanations and change impact insights (with seamless offline fallback).
- **Hospital Scope**: 12 critical hospital modules (Emergency Department, Pharmacy, Laboratory, Inpatient Management, Billing, etc.).

---

## 🏗️ Architecture & Project Structure

```
RiskSense_AI/
├── app.py                          # Streamlit multi-page dashboard
├── config.py                       # Centralized configuration & weights
├── pyproject.toml                  # UV project configuration
├── uv.lock                        # Locked dependencies
├── .gitignore                      # Git ignore rules
│
├── data/                           # Reproducible synthetic datasets
│   ├── releases.csv                (5 releases: R1-R4 completed, R5 current)
│   ├── requirements.csv            (120 requirements across 12 modules)
│   ├── test_cases.csv              (240 test cases, 2 per requirement)
│   ├── test_executions.csv         (4,913 historical test executions)
│   ├── defects.csv                 (600 historical defects)
│   └── release_changes.csv         (250 changes for release R5)
│
├── core/
│   ├── data_loader.py              # CSV loader & referential integrity validator
│   ├── risk_engine.py              # Deterministic weighted risk scoring engine
│   ├── prioritizer.py              # Regression plan optimizer & coverage calculator
│   └── evaluator.py                # Aggregations, metrics, and dynamic heatmap matrix
│
├── ai/
│   └── ollama_service.py           # Local Ollama REST client & fallback template engine
│
├── visualization/
│   └── charts.py                   # Plotly charts (Risk Distribution, Heatmap, Top Risks)
│
├── generator/
│   └── generate_data.py            # Reproducible synthetic data generator (seed=42)
│
└── tests/
    └── test_risk_engine.py         # Automated test suite (8 test cases)
```

---

## ⚖️ Deterministic Risk Engine Formula

Every test case is assigned a deterministic risk score (0–100):

$$\text{Risk Score} = 0.25 \times D + 0.20 \times F + 0.25 \times C + 0.15 \times B + 0.15 \times S$$

| Risk Factor | Weight | Description |
|---|---|---|
| **Historical Defect Risk ($D$)** | 25% | Test, requirement, and module defect density & severity |
| **Historical Failure Risk ($F$)** | 20% | Historical execution failure rate |
| **Current Change Impact ($C$)** | 25% | R5 changes touching the requirement & module |
| **Business Criticality ($B$)** | 15% | Module operational criticality score |
| **Safety / Regulatory Impact ($S$)** | 15% | Clinical safety and regulatory compliance rating |

### Risk Classification Thresholds
- **Critical Risk**: 80 – 100
- **High Risk**: 60 – 79
- **Medium Risk**: 40 – 59
- **Low Risk**: 0 – 39

---

## 📊 Streamlit Dashboard Pages

1. **Dashboard**: Executive summary, KPI cards, risk distribution donut chart, module overview stacked bar chart, and top 10 highest-risk test cases.
2. **Risk Analysis**: Interactive test case table with filters (Module, Risk Level, Test Type, Priority, Search), 5-factor score breakdown, and on-demand Ollama AI explanation generator.
3. **Risk Heatmap**: Dynamic 12 Module × 4 Risk Level (`Low → Medium → High → Critical`) matrix with interactive filters and concentration insights.
4. **Regression Plan**: Configurable reduction targets (**15%** / **20%**), defect coverage metrics, and one-click CSV export of the recommended regression suite.

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.13+
- [UV package manager](https://github.com/astral-sh/uv)
- (Optional) [Ollama](https://ollama.com/) running with `llama3.2`

### Installation & Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd RiskSense_AI

# 2. Sync dependencies
uv sync

# 3. Generate synthetic data (if needed)
uv run python generator/generate_data.py

# 4. Run automated validation tests
uv run python tests/test_risk_engine.py

# 5. Launch the Streamlit dashboard
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.
