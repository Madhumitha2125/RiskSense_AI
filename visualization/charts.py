"""
RiskSense AI - Visualization Charts

Plotly-based charts for the Streamlit dashboard.
Professional, modern dark-themed styling.
"""

import plotly.graph_objects as go
import pandas as pd

import config


# --------------------------------------------------
# Color Scheme & Theming
# --------------------------------------------------
RISK_COLOR_MAP = config.RISK_COLORS
CHART_BG = "rgba(0,0,0,0)"
CHART_FONT = dict(family="Inter, sans-serif", size=12, color="#E2E8F0")
CHART_TEMPLATE = "plotly_dark"


def risk_distribution_chart(risk_results):
    """
    Donut chart showing distribution of tests across risk levels.
    """
    counts = []
    for level in config.RISK_LEVEL_ORDER:
        counts.append({
            "Risk Level": level,
            "Count": int((risk_results["risk_level"] == level).sum()),
        })
    df = pd.DataFrame(counts)

    fig = go.Figure(
        data=[go.Pie(
            labels=df["Risk Level"],
            values=df["Count"],
            hole=0.55,
            marker=dict(colors=[RISK_COLOR_MAP[level] for level in df["Risk Level"]]),
            textinfo="label+value",
            textfont=dict(size=13, color="#FFFFFF"),
            hovertemplate="<b>%{label}</b><br>Tests: %{value}<br>%{percent}<extra></extra>",
        )]
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=30, b=40, l=20, r=20),
        height=350,
    )

    return fig


def module_risk_overview_chart(risk_results):
    """
    Horizontal stacked bar chart showing risk distribution per module.
    """
    module_risk = pd.crosstab(risk_results["module"], risk_results["risk_level"])

    # Ensure all modules & levels present
    for mod in config.MODULES:
        if mod not in module_risk.index:
            module_risk.loc[mod] = 0

    for level in config.RISK_LEVEL_ORDER:
        if level not in module_risk.columns:
            module_risk[level] = 0

    # Sort by total high-risk (Critical + High) ascending so top modules appear at top of chart
    module_risk["_sort"] = module_risk["Critical"] * 2 + module_risk["High"]
    module_risk = module_risk.sort_values("_sort", ascending=True)
    module_risk = module_risk.drop(columns=["_sort"])

    fig = go.Figure()

    for level in reversed(config.RISK_LEVEL_ORDER):
        fig.add_trace(go.Bar(
            name=level,
            y=module_risk.index.tolist(),
            x=module_risk[level].tolist(),
            orientation="h",
            marker_color=RISK_COLOR_MAP[level],
            hovertemplate=f"<b>%{{y}}</b><br>{level}: %{{x}} tests<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(title="Number of Test Cases"),
        yaxis=dict(title=""),
        margin=dict(t=30, b=50, l=180, r=20),
        height=450,
    )

    return fig


def risk_heatmap_chart(heatmap_data):
    """
    Dynamic Heatmap:
    Y-axis: 12 hospital modules
    X-axis: Low / Medium / High / Critical
    Cell: Number of test cases
    """
    modules_y = heatmap_data.index.tolist()
    risk_levels_x = heatmap_data.columns.tolist()
    z_matrix = heatmap_data.values

    # Determine max value for colorscale scaling
    max_val = int(z_matrix.max()) if z_matrix.size > 0 and z_matrix.max() > 0 else 1

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=risk_levels_x,
            y=modules_y,
            colorscale=[
                [0.0, "#0F172A"],
                [0.2, "#1E293B"],
                [0.4, "#166534"],
                [0.6, "#854D0E"],
                [0.8, "#C2410C"],
                [1.0, "#DC2626"],
            ],
            zmin=0,
            zmax=max(max_val, 5),
            text=z_matrix,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF", family="Inter, sans-serif"),
            hovertemplate=(
                "<b>Module:</b> %{y}<br>"
                "<b>Risk Level:</b> %{x}<br>"
                "<b>Test Cases:</b> %{z}<extra></extra>"
            ),
            showscale=True,
            colorbar=dict(
                title=dict(text="Test Count", font=dict(size=12, color="#E2E8F0")),
                tickfont=dict(size=11, color="#94A3B8"),
            ),
        )
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        xaxis=dict(
            title=dict(text="Risk Level", font=dict(size=13, color="#E2E8F0")),
            side="bottom",
            tickfont=dict(size=12, color="#E2E8F0"),
        ),
        yaxis=dict(
            title=dict(text="Hospital Module", font=dict(size=13, color="#E2E8F0")),
            autorange="reversed",  # Shows highest risk module (row 0) at the top
            tickfont=dict(size=12, color="#E2E8F0"),
        ),
        margin=dict(t=40, b=60, l=190, r=30),
        height=520,
    )

    return fig


def top_risks_chart(risk_results, n=10):
    """
    Horizontal bar chart of top N highest-risk tests.
    """
    top = risk_results.head(n).copy()
    top = top.sort_values("risk_score", ascending=True)

    colors = [RISK_COLOR_MAP.get(level, "#6B7280") for level in top["risk_level"]]

    fig = go.Figure(
        data=[go.Bar(
            y=top["test_id"] + " (" + top["module"] + ")",
            x=top["risk_score"],
            orientation="h",
            marker_color=colors,
            text=top["risk_score"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Risk Score: %{x:.1f}<extra></extra>"
            ),
        )]
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        xaxis=dict(title="Risk Score", range=[0, 105]),
        yaxis=dict(title=""),
        margin=dict(t=30, b=50, l=230, r=50),
        height=400,
    )

    return fig
