"""
RiskSense AI - Main Streamlit Application

UC-006: Risk-Based Regression Testing POC
"""

import streamlit as st
import pandas as pd

import config
from core.data_loader import load_all_data, DataValidationError
from core.risk_engine import calculate_risk_scores
from core.prioritizer import generate_regression_plan, get_regression_suite_csv
from core.evaluator import compute_summary, compute_effort_reduction, compute_heatmap_data
from ai.ollama_service import check_ollama_status, generate_risk_explanation
from visualization.charts import (
    risk_distribution_chart,
    module_risk_overview_chart,
    risk_heatmap_chart,
    top_risks_chart,
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title=f"{config.APP_NAME} - {config.APP_SUBTITLE}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom Styles (Dark Modern Theme)
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .metric-critical .metric-value { color: #DC2626; }
    .metric-high .metric-value { color: #F97316; }
    .metric-medium .metric-value { color: #EAB308; }
    .metric-low .metric-value { color: #22C55E; }
    .metric-info .metric-value { color: #38BDF8; }

    .risk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .risk-critical { background: #DC2626; color: #FFFFFF; }
    .risk-high { background: #F97316; color: #FFFFFF; }
    .risk-medium { background: #EAB308; color: #0F172A; }
    .risk-low { background: #22C55E; color: #0F172A; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #334155;
    }

    .ollama-status {
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .ollama-online { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .ollama-offline { background: #450A0A; color: #FCA5A5; border: 1px solid #991B1B; }

    .insight-box {
        background: #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        color: #E2E8F0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Cached Data Loading & Risk Calculation
# --------------------------------------------------
@st.cache_data
def load_and_compute():
    """
    Load all CSV datasets and compute base risk scores.
    Single cached function to prevent serialization/hashing issues.
    """
    data = load_all_data()
    risk_results = calculate_risk_scores(data)
    summary = compute_summary(risk_results)
    return data, risk_results, summary


@st.cache_data(ttl=60)
def cached_ollama_status():
    """Check Ollama status with a 60s cache to keep UI responsive."""
    return check_ollama_status()


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def metric_card(label, value, css_class=""):
    """Render a styled metric card."""
    st.markdown(
        f"""<div class="metric-card {css_class}">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ RiskSense AI")
    st.markdown(f"**{config.APP_SUBTITLE}**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        config.PAGES,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Ollama status indicator
    ollama_status = cached_ollama_status()
    if ollama_status["available"] and ollama_status["model_available"]:
        st.markdown(
            '<div class="ollama-status ollama-online">🟢 Ollama Online (llama3.2)</div>',
            unsafe_allow_html=True,
        )
    elif ollama_status["available"]:
        st.markdown(
            '<div class="ollama-status ollama-offline">🟡 Ollama Online (Model missing)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ollama-status ollama-offline">🔴 Ollama Offline (Fallback active)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption(f"RiskSense AI POC v{config.APP_VERSION}")


# --------------------------------------------------
# Load Data
# --------------------------------------------------
try:
    data, risk_results, summary = load_and_compute()
except FileNotFoundError as e:
    st.error(f"Data files not found: {str(e)}")
    st.info("Run: `uv run python generator/generate_data.py` to generate datasets.")
    st.stop()
except DataValidationError as e:
    st.error(f"Data Validation Error:\n\n{str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    st.stop()


# ==================================================
# PAGE 1: Dashboard
# ==================================================
if page == "Dashboard":
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("Risk-based test optimization summary for current release **R5**.")

    # KPI Metrics Row
    effort = compute_effort_reduction(risk_results, 0.15)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Tests", summary["total_tests"])
    with col2:
        metric_card("Critical Risk", summary["critical_count"], "metric-critical")
    with col3:
        metric_card("High Risk", summary["high_count"], "metric-high")
    with col4:
        metric_card("Recommended Suite", effort["recommended_tests"], "metric-info")
    with col5:
        metric_card("Effort Reduction", f"{effort['reduction_pct']}%", "metric-low")

    st.markdown("")

    # Visualizations Row
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-header">Risk Level Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_distribution_chart(risk_results), use_container_width=True)
    with col_right:
        st.markdown('<div class="section-header">Module Risk Overview</div>', unsafe_allow_html=True)
        st.plotly_chart(module_risk_overview_chart(risk_results), use_container_width=True)

    # Top 10 High-Risk Tests
    st.markdown('<div class="section-header">Top 10 High-Risk Test Cases</div>', unsafe_allow_html=True)
    st.plotly_chart(top_risks_chart(risk_results, config.TOP_N_RISKS), use_container_width=True)

    top10_table = risk_results.head(10)[
        ["rank", "test_id", "test_name", "module", "risk_score", "risk_level",
         "defect_count", "failure_rate", "change_count"]
    ].copy()
    top10_table.columns = [
        "Rank", "Test ID", "Test Name", "Module", "Risk Score", "Risk Level",
        "Defects", "Fail Rate (%)", "Changes"
    ]
    st.dataframe(top10_table, use_container_width=True, hide_index=True)


# ==================================================
# PAGE 2: Risk Analysis
# ==================================================
elif page == "Risk Analysis":
    st.markdown("## 🔍 Test Case Risk Analysis")
    st.markdown("Detailed breakdown of deterministic weighted risk factors per test case.")

    # Filters Row
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        mod_filter = st.selectbox("Filter Module", ["All"] + config.MODULES, key="ra_mod")
    with f2:
        lvl_filter = st.selectbox("Filter Risk Level", ["All"] + config.RISK_LEVEL_ORDER, key="ra_lvl")
    with f3:
        typ_filter = st.selectbox("Filter Test Type", ["All"] + config.TEST_TYPES, key="ra_typ")
    with f4:
        prio_filter = st.selectbox("Filter Priority", ["All"] + config.TEST_PRIORITIES, key="ra_prio")

    search_query = st.text_input("🔎 Search by Test ID or Name", key="ra_search")

    # Filter data
    filtered = risk_results.copy()
    if mod_filter != "All":
        filtered = filtered[filtered["module"] == mod_filter]
    if lvl_filter != "All":
        filtered = filtered[filtered["risk_level"] == lvl_filter]
    if typ_filter != "All":
        filtered = filtered[filtered["test_type"] == typ_filter]
    if prio_filter != "All":
        filtered = filtered[filtered["priority"] == prio_filter]
    if search_query:
        q = search_query.lower()
        filtered = filtered[
            filtered["test_name"].str.lower().str.contains(q, na=False)
            | filtered["test_id"].str.lower().str.contains(q, na=False)
        ]

    st.markdown(f"**Showing {len(filtered)} of {len(risk_results)} test cases**")

    # Table of filtered results
    table_cols = [
        "rank", "test_id", "test_name", "module", "risk_score", "risk_level",
        "defect_count", "failure_rate", "change_count",
    ]
    disp_df = filtered[table_cols].copy()
    disp_df.columns = [
        "Rank", "Test ID", "Test Name", "Module", "Risk Score", "Risk Level",
        "Defects", "Fail Rate (%)", "Changes",
    ]
    st.dataframe(disp_df, use_container_width=True, hide_index=True, height=350)

    # Detailed test case inspection
    st.markdown('<div class="section-header">Test Case Detail & AI Explanation</div>', unsafe_allow_html=True)

    if len(filtered) > 0:
        test_ids = filtered["test_id"].tolist()
        selected_id = st.selectbox("Select Test Case for Detailed Analysis", test_ids, key="ra_sel_test")

        selected = filtered[filtered["test_id"] == selected_id].iloc[0]

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown(f"### {selected['test_id']}: {selected['test_name']}")
            st.markdown(f"**Module:** {selected['module']} &nbsp;|&nbsp; **Type:** {selected['test_type']} &nbsp;|&nbsp; **Priority:** {selected['priority']}")
            st.markdown(f"**Risk Score:** `{selected['risk_score']}/100` ({selected['risk_level']} Risk)")

            factor_data = pd.DataFrame({
                "Risk Factor": [
                    "Historical Defect Risk (25%)",
                    "Historical Failure Risk (20%)",
                    "Current Change Impact (25%)",
                    "Business Criticality (15%)",
                    "Safety/Regulatory Impact (15%)",
                ],
                "Sub-Score": [
                    selected["historical_defect_score"],
                    selected["historical_failure_score"],
                    selected["change_impact_score"],
                    selected["business_criticality_score"],
                    selected["safety_regulatory_score"],
                ],
            })
            st.dataframe(factor_data, use_container_width=True, hide_index=True)

        with d_col2:
            st.markdown("### AI Risk Explanation")
            st.caption("Interprets deterministic metrics and R5 release context using local Ollama llama3.2.")

            if st.button("🤖 Generate AI Explanation", key="btn_ai_explain"):
                with st.spinner("Analyzing risk factors..."):
                    t_info = {
                        "test_id": selected["test_id"],
                        "test_name": selected["test_name"],
                        "module": selected["module"],
                    }
                    r_factors = {
                        "risk_score": selected["risk_score"],
                        "risk_level": selected["risk_level"],
                        "historical_defect_score": selected["historical_defect_score"],
                        "historical_failure_score": selected["historical_failure_score"],
                        "change_impact_score": selected["change_impact_score"],
                        "business_criticality_score": selected["business_criticality_score"],
                        "safety_regulatory_score": selected["safety_regulatory_score"],
                        "defect_count": selected["defect_count"],
                        "failure_rate": selected["failure_rate"],
                        "change_count": selected["change_count"],
                    }
                    explanation = generate_risk_explanation(t_info, r_factors)
                    st.info(explanation)
    else:
        st.info("No test cases match the selected filter criteria.")


# ==================================================
# PAGE 3: Risk Heatmap (DYNAMIC)
# ==================================================
elif page == "Risk Heatmap":
    st.markdown("## 🗺️ Module Risk Heatmap")
    st.markdown(
        "Dynamic risk matrix displaying test case concentrations across **12 Hospital Modules** (Y-axis) "
        "and **Risk Levels: Low → Medium → High → Critical** (X-axis)."
    )

    # Dynamic Heatmap Controls
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        hm_type_filter = st.selectbox(
            "Filter by Test Type",
            ["All Types"] + config.TEST_TYPES,
            key="hm_type",
        )
    with h_col2:
        hm_prio_filter = st.selectbox(
            "Filter by Priority",
            ["All Priorities"] + config.TEST_PRIORITIES,
            key="hm_prio",
        )
    with h_col3:
        hm_module_filter = st.multiselect(
            "Focus Specific Modules",
            config.MODULES,
            default=[],
            key="hm_modules",
            help="Leave empty to include all 12 modules.",
        )

    # Apply dynamic filtering to risk_results
    hm_filtered = risk_results.copy()
    if hm_type_filter != "All Types":
        hm_filtered = hm_filtered[hm_filtered["test_type"] == hm_type_filter]
    if hm_prio_filter != "All Priorities":
        hm_filtered = hm_filtered[hm_filtered["priority"] == hm_prio_filter]
    if hm_module_filter:
        hm_filtered = hm_filtered[hm_filtered["module"].isin(hm_module_filter)]

    # Compute dynamic heatmap data from filtered dataset
    dynamic_heatmap = compute_heatmap_data(hm_filtered)

    # If specific modules selected, filter rows accordingly
    if hm_module_filter:
        dynamic_heatmap = dynamic_heatmap.loc[dynamic_heatmap.index.isin(hm_module_filter)]

    # Render Heatmap Chart
    st.plotly_chart(risk_heatmap_chart(dynamic_heatmap), use_container_width=True)

    # Insights on concentration
    top_risk_modules = dynamic_heatmap["Critical"] + dynamic_heatmap["High"]
    highest_module = top_risk_modules.idxmax() if len(top_risk_modules) > 0 else "None"
    highest_count = top_risk_modules.max() if len(top_risk_modules) > 0 else 0

    st.markdown(
        f"""<div class="insight-box">
            <b>📌 Risk Concentration Insight:</b><br>
            <b>{highest_module}</b> has the highest concentration of high-risk scenarios with
            <b>{highest_count}</b> Critical/High test cases in the current view.
            Prioritizing regression cycles on top-ranking rows yields maximum defect prevention.
        </div>""",
        unsafe_allow_html=True,
    )

    # Detailed Module Breakdown Table
    st.markdown('<div class="section-header">Module Risk Breakdown</div>', unsafe_allow_html=True)
    table_data = []
    for mod in dynamic_heatmap.index:
        row = dynamic_heatmap.loc[mod]
        total_m = int(row.sum())
        crit = int(row.get("Critical", 0))
        high = int(row.get("High", 0))
        med = int(row.get("Medium", 0))
        low = int(row.get("Low", 0))
        high_crit_pct = round(((crit + high) / total_m) * 100, 1) if total_m > 0 else 0.0

        table_data.append({
            "Hospital Module": mod,
            "Total Tests": total_m,
            "Critical": crit,
            "High": high,
            "Medium": med,
            "Low": low,
            "% High/Critical": f"{high_crit_pct}%",
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ==================================================
# PAGE 4: Regression Plan
# ==================================================
elif page == "Regression Plan":
    st.markdown("## 📋 Optimized Regression Plan")
    st.markdown(
        "Demonstrates risk-based test reduction while preserving maximum historical defect coverage."
    )

    # Reduction Selector
    red_col1, red_col2 = st.columns([1, 2])
    with red_col1:
        reduction_option = st.radio(
            "Select Regression Effort Reduction Target",
            list(config.REDUCTION_LEVELS.keys()),
            horizontal=True,
            key="rp_reduction",
        )
    reduction_pct = config.REDUCTION_LEVELS[reduction_option]

    # Generate Plan
    plan = generate_regression_plan(risk_results, data, reduction_pct)
    coverage = plan["defect_coverage"]

    # Summary KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Total Tests", plan["total_tests"])
    with k2:
        metric_card("Recommended", plan["recommended_count"], "metric-info")
    with k3:
        metric_card("Tests Reduced", plan["reduced_count"], "metric-low")
    with k4:
        metric_card("Effort Saved", f"{plan['reduction_pct'] * 100:.0f}%", "metric-low")
    with k5:
        metric_card("Defect Coverage", f"{coverage['coverage_pct']}%", "metric-info")

    st.markdown("")

    # Coverage Analysis Insights
    st.markdown(
        f"""<div class="insight-box">
            <b>🎯 Risk-Based Prioritization Effectiveness:</b><br>
            By reducing regression test suite size by <b>{plan['reduction_pct'] * 100:.0f}%</b>
            (removing <b>{plan['reduced_count']}</b> low-risk tests), the recommended suite still covers
            <b>{coverage['covered_high_critical_defects']} of {coverage['total_high_critical_defects']}</b>
            historical High & Critical defects (<b>{coverage['coverage_pct']}%</b> coverage).
        </div>""",
        unsafe_allow_html=True,
    )

    # Recommended Suite Table
    st.markdown('<div class="section-header">Recommended Regression Test Suite</div>', unsafe_allow_html=True)
    rec_table = plan["recommended_tests"][
        ["rank", "test_id", "test_name", "module", "risk_score", "risk_level",
         "defect_count", "failure_rate", "change_count"]
    ].copy()
    rec_table.columns = [
        "Rank", "Test ID", "Test Name", "Module", "Risk Score", "Risk Level",
        "Defects", "Fail Rate (%)", "Changes"
    ]
    st.dataframe(rec_table, use_container_width=True, hide_index=True, height=380)

    # CSV Download Button
    csv_payload = get_regression_suite_csv(plan["recommended_tests"])
    st.download_button(
        label=f"📥 Download Recommended Suite CSV ({reduction_option} Reduction)",
        data=csv_payload,
        file_name=f"risksense_regression_suite_{reduction_option.replace('%', 'pct')}.csv",
        mime="text/csv",
        key="btn_download_plan",
    )
