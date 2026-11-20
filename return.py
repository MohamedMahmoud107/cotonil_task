
# ============================================================
# COTTONIL - RETURNS ANALYSIS DASHBOARD
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cottonil | Returns Analysis",
    page_icon="📦",
    layout="wide"
)

DATA_DIR = Path(r"D:\HVIA\cotonil\data")

ORDERS_PATH = DATA_DIR / "sales_orders.csv"
RETURNS_PATH = DATA_DIR / "sales_returns.csv"

CUTOFF_DATE = pd.Timestamp("2026-10-18")

COLORS = px.colors.qualitative.Set2


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    orders = pd.read_csv(ORDERS_PATH)
    returns = pd.read_csv(RETURNS_PATH)

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        errors="coerce"
    )

    returns["return_date"] = pd.to_datetime(
        returns["return_date"],
        errors="coerce"
    )

    orders["paid_total_EGP"] = pd.to_numeric(
        orders["paid_total_EGP"],
        errors="coerce"
    ).fillna(0)

    returns["refund_amount_EGP"] = pd.to_numeric(
        returns["refund_amount_EGP"],
        errors="coerce"
    ).fillna(0)

    returns["return_quantity"] = pd.to_numeric(
        returns["return_quantity"],
        errors="coerce"
    ).fillna(0)

    return orders, returns


df_orders, df_returns = load_data()


# ============================================================
# 3. DATA PREPARATION
# ============================================================

# One row per order is maintained in df_orders.
# Returns remain at return-line level in df_returns.

df_or_re = df_returns.merge(
    df_orders[
        [
            "sales_order_id",
            "order_date",
            "branch_id",
            "customer_id",
            "sales_channel",
            "payment_method",
            "items_count",
            "paid_total_EGP"
        ]
    ],
    on="sales_order_id",
    how="left",
    validate="many_to_one"
)

df_or_re["reason"] = (
    df_or_re["reason"]
    .fillna("Unknown")
    .astype(str)
)

df_or_re["sales_channel"] = (
    df_or_re["sales_channel"]
    .fillna("Unknown")
    .astype(str)
)

# Remove invalid dates
df_orders = df_orders.dropna(subset=["order_date"])
df_returns = df_returns.dropna(subset=["return_date"])

df_or_re = df_returns.merge(
    df_orders[
        [
            "sales_order_id",
            "order_date",
            "branch_id",
            "customer_id",
            "sales_channel",
            "payment_method",
            "items_count",
            "paid_total_EGP"
        ]
    ],
    on="sales_order_id",
    how="left",
    validate="many_to_one"
)

df_or_re["reason"] = (
    df_or_re["reason"].fillna("Unknown")
)

df_or_re["sales_channel"] = (
    df_or_re["sales_channel"].fillna("Unknown")
)

# Available years
available_years = sorted(
    df_orders["order_date"].dt.year.dropna().unique(),
    reverse=True
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_period_data(year, month=None, quarter=None, cutoff=None):

    """
    Returns orders and returns belonging to a specific period.

    Orders are filtered using order_date.
    Returns are filtered using return_date.

    If cutoff is supplied, both datasets are restricted
    to that date.
    """

    orders = df_orders[
        df_orders["order_date"].dt.year == year
    ].copy()

    returns = df_or_re[
        df_or_re["return_date"].dt.year == year
    ].copy()

    if month is not None:

        orders = orders[
            orders["order_date"].dt.month == month
        ]

        returns = returns[
            returns["return_date"].dt.month == month
        ]

    if quarter is not None:

        orders = orders[
            orders["order_date"].dt.quarter == quarter
        ]

        returns = returns[
            returns["return_date"].dt.quarter == quarter
        ]

    if cutoff is not None:

        orders = orders[
            orders["order_date"] <= cutoff
        ]

        returns = returns[
            returns["return_date"] <= cutoff
        ]

    return orders, returns


def calculate_return_ratio(orders, returns):

    """
    Returned Orders / Total Orders

    Each order is counted once regardless of
    the number of returned lines.
    """

    total_orders = orders["sales_order_id"].nunique()

    if total_orders == 0:
        return 0

    returned_ids = set(
        returns["sales_order_id"].dropna().unique()
    )

    order_ids = set(
        orders["sales_order_id"].dropna().unique()
    )

    returned_orders = len(
        returned_ids.intersection(order_ids)
    )

    return returned_orders / total_orders * 100


def calculate_refund(returns):

    return returns["refund_amount_EGP"].sum()


def format_egp(value):

    return f"{value:,.2f} EGP"


def get_top_sku(returns):

    sku_returns = (
        returns.groupby("sku_id")["return_quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    if sku_returns.empty:
        return "N/A", 0

    return sku_returns.index[0], sku_returns.iloc[0]


def get_top_branch(returns):

    branch_returns = (
        returns.groupby("branch_id")["refund_amount_EGP"]
        .sum()
        .sort_values(ascending=False)
    )

    if branch_returns.empty:
        return "N/A", 0

    return branch_returns.index[0], branch_returns.iloc[0]


def apply_plot_style(fig, height=400):

    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(
            x=0.5,
            xanchor="center"
        ),
        margin=dict(
            l=20,
            r=20,
            t=65,
            b=20
        ),
        legend_title_text=""
    )

    return fig


# ============================================================
# 5. DASHBOARD HEADER
# ============================================================

st.title("📦 Cottonil | Returns Analysis")

st.caption(
    "Analyze product returns, refund amounts, "
    "return reasons and channel performance."
)

st.divider()


# ============================================================
# 6. PART ONE - YEARLY ANALYSIS
# ============================================================

st.header("1. Yearly Returns Analysis")

selected_year = st.selectbox(
    "Select Year",
    options=available_years
)

# 2026 is restricted to October 18.
# Historical years use their complete available data.

yearly_cutoff = (
    CUTOFF_DATE
    if selected_year == 2026
    else None
)

if yearly_cutoff is not None:

    st.info(
        f"Showing data till {yearly_cutoff.strftime('%d/%m/%Y')}"
    )

orders_year, returns_year = get_period_data(
    selected_year,
    cutoff=yearly_cutoff
)


# ============================================================
# 6.1 KPI CALCULATIONS
# ============================================================

returned_ratio = calculate_return_ratio(
    orders_year,
    returns_year
)

total_refund = calculate_refund(
    returns_year
)

top_sku, top_sku_quantity = get_top_sku(
    returns_year
)

# Previous year comparison
previous_year = selected_year - 1

orders_previous, returns_previous = get_period_data(
    previous_year,
    cutoff=(
        pd.Timestamp(
            year=previous_year,
            month=10,
            day=18
        )
        if selected_year == 2026
        else None
    )
)

previous_refund = calculate_refund(
    returns_previous
)

refund_difference = total_refund - previous_refund

if previous_refund != 0:

    refund_change = (
        refund_difference / previous_refund
    ) * 100

else:

    refund_change = 0


# ============================================================
# 6.2 KPI CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Returned Orders Ratio",
        f"{returned_ratio:.2f}%"
    )

with col2:

    st.metric(
        "Most Returned SKU",
        str(top_sku),
        help=(
            f"Total returned quantity: "
            f"{top_sku_quantity:,.0f}"
        )
    )

    st.caption(
        f"Returned Quantity: {top_sku_quantity:,.0f}"
    )

with col3:

    st.metric(
        "Total Refund Amount",
        format_egp(total_refund),
        delta=(
            f"{refund_change:+.2f}% vs previous year"
            if previous_refund != 0
            else None
        ),
        delta_color="inverse"
    )

    st.caption(
        f"Previous Year: {format_egp(previous_refund)}"
    )


st.divider()


# ============================================================
# 6.3 RETURN REASONS DONUT CHART
# ============================================================

st.subheader("Return Reasons Distribution")

if not returns_year.empty:

    reason_distribution = (
        returns_year.groupby("reason")
        ["refund_amount_EGP"]
        .sum()
        .reset_index()
    )

    # Distribution by number of return lines.
    reason_counts = (
        returns_year["reason"]
        .value_counts()
        .reset_index()
    )

    reason_counts.columns = [
        "reason",
        "count"
    ]

    fig_reason = px.pie(
        reason_counts,
        names="reason",
        values="count",
        hole=0.55,
        title="Return Reasons Distribution",
        color_discrete_sequence=COLORS
    )

    fig_reason.update_traces(
        textinfo="percent+label",
        hovertemplate=(
            "%{label}<br>"
            "Return Lines: %{value}<br>"
            "Percentage: %{percent}"
            "<extra></extra>"
        )
    )

    apply_plot_style(fig_reason, 400)

    st.plotly_chart(
        fig_reason,
        use_container_width=True
    )

else:

    st.info("No return data available for this year.")


# ============================================================
# 6.4 MONTHLY RETURN ORDERS
# ============================================================

st.subheader("Monthly Returned Orders")

if not returns_year.empty:

    monthly_returns = (
        returns_year.groupby(
            returns_year["return_date"].dt.month
        )["sales_order_id"]
        .nunique()
        .reset_index()
    )

    monthly_returns.columns = [
        "Month",
        "Returned Orders"
    ]

    month_names = {
        i: pd.Timestamp(2026, i, 1).strftime("%b")
        for i in range(1, 13)
    }

    monthly_returns["Month Name"] = (
        monthly_returns["Month"].map(month_names)
    )

    fig_monthly = px.bar(
        monthly_returns,
        x="Month Name",
        y="Returned Orders",
        text="Returned Orders",
        title="Monthly Returned Orders",
        color_discrete_sequence=[COLORS[0]]
    )

    fig_monthly.update_traces(
        textposition="outside"
    )

    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Returned Orders"
    )

    apply_plot_style(fig_monthly, 400)

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

else:

    st.info("No monthly return data available.")


st.divider()


# ============================================================
# 7. PART TWO - GENERAL ANALYSIS
# ============================================================

st.header("2. General Returns Analysis")

st.caption(
    "Historical analysis across all available years."
)


# ============================================================
# 7.1 YEARLY RETURN RATIO
# Cutoff: October 18 for every year.
# ============================================================

st.subheader("Yearly Returned Orders Ratio")

yearly_ratio_data = []

for year in sorted(available_years):

    cutoff = pd.Timestamp(
        year=int(year),
        month=10,
        day=18
    )

    orders, returns = get_period_data(
        int(year),
        cutoff=cutoff
    )

    ratio = calculate_return_ratio(
        orders,
        returns
    )

    yearly_ratio_data.append({
        "Year": int(year),
        "Returned Ratio (%)": ratio
    })


yearly_ratio_df = pd.DataFrame(
    yearly_ratio_data
)

if not yearly_ratio_df.empty:

    fig_yearly_ratio = px.line(
        yearly_ratio_df,
        x="Year",
        y="Returned Ratio (%)",
        markers=True,
        text="Returned Ratio (%)",
        title="Yearly Returned Orders Ratio",
        color_discrete_sequence=[COLORS[0]]
    )

    fig_yearly_ratio.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="top center",
        line=dict(width=3)
    )

    fig_yearly_ratio.update_layout(
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        yaxis_title="Returned Orders Ratio (%)"
    )

    apply_plot_style(fig_yearly_ratio, 450)

    st.plotly_chart(
        fig_yearly_ratio,
        use_container_width=True
    )


st.divider()


# ============================================================
# 7.2 QUARTERLY ANALYSIS
# Full available data - NO October 18 cutoff.
# ============================================================

st.subheader("Quarterly Refund & Return Ratio Analysis")

quarterly_data = []

for year in sorted(available_years):

    for quarter in range(1, 5):

        orders, returns = get_period_data(
            int(year),
            quarter=quarter
        )

        refund = calculate_refund(
            returns
        )

        ratio = calculate_return_ratio(
            orders,
            returns
        )

        quarterly_data.append({

            "Year": int(year),

            "Quarter": f"Q{quarter}",

            "Period": f"{year} Q{quarter}",

            "Refund Amount": refund,

            "Returned Ratio (%)": ratio

        })


quarterly_df = pd.DataFrame(
    quarterly_data
)

if not quarterly_df.empty:

    # Keep quarters with actual orders or returns.
    quarterly_df = quarterly_df[
        (quarterly_df["Refund Amount"] > 0)
        | (quarterly_df["Returned Ratio (%)"] > 0)
    ].copy()

    fig_quarterly = go.Figure()

    # Refund bars
    fig_quarterly.add_trace(
        go.Bar(
            x=quarterly_df["Period"],
            y=quarterly_df["Refund Amount"],
            name="Total Refund Amount",
            marker_color=COLORS[0],
            yaxis="y",
            hovertemplate=(
                "%{x}<br>"
                "Refund: %{y:,.2f} EGP"
                "<extra></extra>"
            )
        )
    )

    # Returned ratio line
    fig_quarterly.add_trace(
        go.Scatter(
            x=quarterly_df["Period"],
            y=quarterly_df["Returned Ratio (%)"],
            name="Returned Orders Ratio",
            mode="lines+markers",
            line=dict(
                color=COLORS[1],
                width=3
            ),
            yaxis="y2",
            hovertemplate=(
                "%{x}<br>"
                "Returned Ratio: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig_quarterly.update_layout(

        title=dict(
            text="Quarterly Refund Amount & Returned Orders Ratio",
            x=0.5,
            xanchor="center"
        ),

        template="plotly_white",

        height=550,

        barmode="group",

        xaxis=dict(
            title="Quarter",
            tickangle=-45
        ),

        yaxis=dict(
            title="Refund Amount (EGP)",
            tickformat=",.0f"
        ),

        yaxis2=dict(
            title="Returned Orders Ratio (%)",
            overlaying="y",
            side="right",
            ticksuffix="%"
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        margin=dict(
            l=30,
            r=30,
            t=100,
            b=80
        )
    )

    st.plotly_chart(
        fig_quarterly,
        use_container_width=True
    )

    st.caption(
        "Quarterly analysis uses all available data. "
        "The latest year's Q4 is treated as Q4 even if incomplete."
    )


st.divider()


# ============================================================
# 7.3 MOST AFFECTED BRANCH PER YEAR
# Ranking: Total Refund Amount.
# ============================================================

st.subheader("Most Affected Branch per Year")

branch_yearly_data = []

for year in sorted(available_years):

    returns = df_or_re[
        df_or_re["return_date"].dt.year == year
    ].copy()

    # Aggregate refund amount per branch
    branch_refunds = (
        returns.groupby("branch_id")
        .agg(
            Total_Refund=("refund_amount_EGP", "sum"),
            Returned_Orders=("sales_order_id", "nunique")
        )
        .reset_index()
        .sort_values(
            "Total_Refund",
            ascending=False
        )
    )

    if not branch_refunds.empty:

        top_branch = branch_refunds.iloc[0]

        branch_yearly_data.append({

            "Year": int(year),

            "Branch ID": top_branch["branch_id"],

            "Total Refund (EGP)": top_branch["Total_Refund"],

            "Returned Orders": top_branch["Returned_Orders"]

        })


branch_yearly_df = pd.DataFrame(
    branch_yearly_data
)

if not branch_yearly_df.empty:

    st.dataframe(
        branch_yearly_df.style.format({
            "Total Refund (EGP)": "{:,.2f}",
            "Returned Orders": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig_branch = px.bar(
        branch_yearly_df,
        x="Year",
        y="Total Refund (EGP)",
        color="Branch ID",
        text="Branch ID",
        title="Highest Refund Branch per Year",
        color_discrete_sequence=COLORS
    )

    fig_branch.update_traces(
        textposition="outside"
    )

    apply_plot_style(fig_branch, 400)

    st.plotly_chart(
        fig_branch,
        use_container_width=True
    )

else:

    st.info("No branch return data available.")


st.divider()


# ============================================================
# 7.4 RETURN REASONS BY SALES CHANNEL
# 100% STACKED BAR CHART
# ============================================================

st.subheader("Return Reasons Distribution by Sales Channel")

if not df_or_re.empty:

    channel_returns = pd.crosstab(
        df_or_re["sales_channel"],
        df_or_re["reason"],
        normalize="index"
    ) * 100

    fig_channel = px.bar(
        channel_returns,
        x=channel_returns.index,
        y=channel_returns.columns,
        title="Return Reasons by Sales Channel",
        barmode="stack",
        labels={
            "x": "Sales Channel",
            "value": "Percentage (%)",
            "variable": "Return Reason"
        },
        color_discrete_sequence=COLORS
    )

    fig_channel.update_layout(
        template="plotly_white",
        title=dict(
            x=0.5,
            xanchor="center"
        ),
        yaxis=dict(
            title="Percentage (%)",
            range=[0, 100],
            ticksuffix="%"
        ),
        xaxis_title="Sales Channel",
        legend_title="Return Reason",
        height=500
    )

    fig_channel.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="inside",
        hovertemplate=(
            "%{x}<br>"
            "%{fullData.name}: %{y:.2f}%"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        fig_channel,
        use_container_width=True
    )

else:

    st.info("No channel return data available.")


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "Cottonil | Returns Analysis Dashboard"
)

st.caption(
    "Data Source: Sales Orders & Sales Returns"
)
