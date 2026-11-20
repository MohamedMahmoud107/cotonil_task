

from pathlib import Path

import pandas as pd
import numpy as np

import plotly.express as px

import streamlit as st
import joblib
import datetime
from datetime import date
# =========================================================

# App Configuration

# =========================================================

st.set_page_config(

    page_title="COTTONIL Business Dashboard",

    page_icon="📊",

    layout="wide",

)

DATA_DIR = Path(__file__).resolve().parent / "data"

YEARLY_PATH = DATA_DIR / "yearly_profit_gross_sales.csv"

BRANCHES_PATH = DATA_DIR / "branches.csv"

ORDERS_PATH = DATA_DIR / "sales_order_items.csv"
PRODUCTS_PATH = DATA_DIR / "products_catalog.csv"

SALES_ORDERS_PATH = DATA_DIR / "sales_orders.csv"

EMPLOYEES_PATH = DATA_DIR / "employees.csv"

EXPENSES_PATH = DATA_DIR / "operating_expenses.csv"

PERFORMANCE_PATH = Path(__file__).resolve().parent / "branch_performance.csv"

# =========================================================

# Data Loading

# =========================================================

@st.cache_data

def load_csv(path):

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    return df

def validate_columns(df, required_columns, file_name):

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:

        st.error(

            f"Missing columns in {file_name}: "

            f"{', '.join(sorted(missing_columns))}"

        )

        st.stop()

# =========================================================

# Overview Page

# =========================================================

def overview_page():

    """Executive dashboard; all date windows are anchored to the latest data date."""

    st.title("📊 Executive Dashboard")

    st.caption("COTTONIL | Executive business performance")

    required_files = {

        "branches": BRANCHES_PATH,

        "sales items": ORDERS_PATH,
        "products": PRODUCTS_PATH,

        "sales orders": SALES_ORDERS_PATH,

        "employees": EMPLOYEES_PATH,

        "operating expenses": EXPENSES_PATH,

    }

    loaded = {}

    for key, path in required_files.items():

        try:

            loaded[key] = load_csv(path)

        except FileNotFoundError:

            st.error(f"Required file not found: {path}")

            return

    branches = loaded["branches"].copy()

    items = loaded["sales items"].copy()
    products = loaded["products"].copy()

    sales_orders = loaded["sales orders"].copy()

    employees = loaded["employees"].copy()

    expenses = loaded["operating expenses"].copy()

    validate_columns(branches, {"branch_id", "branch_name_ar", "governorate_ar"}, BRANCHES_PATH.name)

    validate_columns(items, {"order_date", "branch_id", "sales_order_id", "net_line_egp", "product_id", "margin_egp"}, ORDERS_PATH.name)
    validate_columns(products, {"product_id", "product_name_ar"}, PRODUCTS_PATH.name)

    validate_columns(sales_orders, {"order_date", "sales_order_id", "net_before_tax_egp", "items_count"}, SALES_ORDERS_PATH.name)

    validate_columns(employees, {"employee_id", "employee_name_ar", "position_en", "salary_base_egp_2026"}, EMPLOYEES_PATH.name)

    validate_columns(expenses, {"expense_date", "amount_egp"}, EXPENSES_PATH.name)

    for frame, col in [(items, "order_date"), (sales_orders, "order_date"), (expenses, "expense_date")]:

        frame[col] = pd.to_datetime(frame[col], errors="coerce")

    for frame, col in [(items, "net_line_egp"), (items, "margin_egp"), (sales_orders, "net_before_tax_egp"), (sales_orders, "items_count"), (expenses, "amount_egp"), (employees, "salary_base_egp_2026")]:

        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    for frame in (branches, items, employees):

        if "branch_id" in frame.columns:

            frame["branch_id"] = frame["branch_id"].astype("string").str.strip()

    max_date = items["order_date"].max()

    if pd.isna(max_date):

        st.warning("No valid sales dates were found in sales_order_items.csv.")

        return

    as_of = max_date.normalize()

    st.caption(f"Data as of: **{as_of:%d %B %Y}** (maximum sales date in the dataset)")

    mode = st.radio("Comparison view", ["YoY", "Quarter-by-quarter", "Current Quarter"], horizontal=True)

    if mode == "YoY":

        period_start = pd.Timestamp(as_of.year, 1, 1)

        period_end = as_of

        previous_start = period_start - pd.DateOffset(years=1)

        previous_end = period_end - pd.DateOffset(years=1)

        period_label = f"YTD {as_of.year} vs same period {as_of.year - 1}"

    elif mode == "Current Quarter":

        period_start = as_of - pd.DateOffset(months=3) + pd.Timedelta(days=1)

        period_end = as_of

        previous_end = period_start - pd.Timedelta(days=1)

        previous_start = previous_end - pd.DateOffset(months=3) + pd.Timedelta(days=1)

        period_label = "Rolling 3 months vs preceding 3 months"

    else:

        # Calendar-quarter comparison: show each quarter's revenue and QoQ change.

        quarter_start_month = ((as_of.month - 1) // 3) * 3 + 1

        period_start = pd.Timestamp(as_of.year, quarter_start_month, 1)

        period_end = as_of

        previous_start = period_start - pd.DateOffset(months=3)

        previous_end = period_start - pd.Timedelta(days=1)

        period_label = "Calendar quarters (each quarter vs previous quarter)"

    def sum_between(frame, date_col, value_col, start, end):

        mask = frame[date_col].between(pd.Timestamp(start), pd.Timestamp(end))

        return float(frame.loc[mask, value_col].fillna(0).sum())

    revenue_now = sum_between(items, "order_date", "net_line_egp", period_start, period_end)

    revenue_prev = sum_between(items, "order_date", "net_line_egp", previous_start, previous_end)

    expense_now = sum_between(expenses, "expense_date", "amount_egp", period_start, period_end)

    expense_prev = sum_between(expenses, "expense_date", "amount_egp", previous_start, previous_end)

    order_now = sales_orders[sales_orders["order_date"].between(period_start, period_end)]

    avg_basket = float(order_now["net_before_tax_egp"].mean()) if not order_now.empty else 0.0

    efficiency = revenue_now / expense_now if expense_now > 0 else None

    revenue_change = ((revenue_now - revenue_prev) / abs(revenue_prev) * 100) if revenue_prev else None

    st.subheader(f"Executive KPIs — {period_label}")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Branches", f"{branches['branch_id'].nunique():,}")

    k2.metric("Total Revenue", f"EGP {revenue_now:,.0f}", f"{revenue_change:+.1f}%" if revenue_change is not None else None)

    k3.metric("Average Basket Size", f"EGP {avg_basket:,.0f}")

    k4.metric("Operational Efficiency", f"{efficiency:.2f}x" if efficiency is not None else "N/A", help="Revenue divided by operating expenses; higher is better.")

    st.caption("Average Basket Size = average net-before-tax value per sales order. Operational Efficiency = revenue ÷ operating expenses.")

    # -----------------------------------------------------
    # Product profit contribution — Top 5 + Others
    # -----------------------------------------------------
    st.divider()
    st.subheader("Product Profit Contribution")
    st.caption(
        f"Profit contribution for the selected comparison window: "
        f"{period_start:%d %b %Y} – {period_end:%d %b %Y}. "
        "Profit is measured using margin_EGP from sales_order_items.csv."
    )

    product_items = items.loc[
        items["order_date"].between(period_start, period_end)
    ].copy()

    product_items["product_id"] = (
        product_items["product_id"].astype("string").str.strip()
    )
    product_catalog = products[["product_id", "product_name_ar"]].copy()
    product_catalog["product_id"] = (
        product_catalog["product_id"].astype("string").str.strip()
    )
    product_catalog = product_catalog.drop_duplicates("product_id")

    product_items["margin_egp"] = pd.to_numeric(
        product_items["margin_egp"], errors="coerce"
    ).fillna(0)

    product_summary = (
        product_items.groupby("product_id", dropna=False, as_index=False)
        ["margin_egp"].sum()
        .merge(product_catalog, on="product_id", how="left")
    )
    product_summary["product_label"] = product_summary.apply(
        lambda row: (
            str(row["product_name_ar"])
            if pd.notna(row["product_name_ar"])
            and str(row["product_name_ar"]).strip()
            else f"Unknown product ({row['product_id']})"
        ),
        axis=1,
    )
    product_summary = product_summary.sort_values(
        "margin_egp", ascending=False
    )

    top_products = product_summary.head(5)[
        ["product_label", "margin_egp"]
    ].copy()
    other_profit = float(
        product_summary.iloc[5:]["margin_egp"].sum()
    )

    if other_profit > 0:
        top_products = pd.concat(
            [
                top_products,
                pd.DataFrame(
                    [{
                        "product_label": "Others",
                        "margin_egp": other_profit,
                    }]
                ),
            ],
            ignore_index=True,
        )

    if top_products.empty:
        st.info("No product profit data is available for this period.")
    elif (top_products["margin_egp"] < 0).any():
        st.warning(
            "The selected period contains products with negative margins. "
            "A donut chart cannot represent negative profit contributions "
            "as positive shares, so the product breakdown is shown as a table."
        )
        st.dataframe(
            top_products.rename(
                columns={
                    "product_label": "Product",
                    "margin_egp": "Profit Margin (EGP)",
                }
            ).style.format({"Profit Margin (EGP)": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    elif top_products["margin_egp"].sum() <= 0:
        st.info("There is no positive profit margin to display as a share.")
    else:
        fig_products = px.pie(
            top_products,
            names="product_label",
            values="margin_egp",
            hole=0.55,
            template="plotly_white",
            title="Top 5 Products + Others — Share of Profit Margin",
        )
        fig_products.update_traces(
            textposition="inside",
            textinfo="label+percent",
            hovertemplate=(
                "Product: %{label}<br>"
                "Profit Margin: %{value:,.2f} EGP<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )
        fig_products.update_layout(
            height=440,
            margin=dict(l=10, r=10, t=55, b=10),
            legend_title_text="Product",
        )
        st.plotly_chart(fig_products, use_container_width=True)

    st.divider()

    st.subheader("Branch footprint by governorate")

    governorate_counts = (branches.dropna(subset=["governorate_ar", "branch_id"])

        .groupby("governorate_ar", as_index=False)["branch_id"].nunique()

        .rename(columns={"branch_id": "branch_count"}))

    # Approximate governorate centroids (latitude, longitude); unknown areas remain visible in the table.

    geo = {

        "القاهرة": (30.0444, 31.2357), "الجيزة": (30.0131, 31.2089), "الإسكندرية": (31.2001, 29.9187),

        "القليوبية": (30.3292, 31.2165), "الشرقية": (30.5877, 31.5020), "الدقهلية": (31.0364, 31.3807),

        "الغربية": (30.8754, 31.0335), "المنوفية": (30.5972, 30.9876), "البحيرة": (30.8481, 30.3436),

        "كفر الشيخ": (31.1107, 30.9388), "دمياط": (31.4175, 31.8144), "بورسعيد": (31.2653, 32.3019),

        "الإسماعيلية": (30.5965, 32.2715), "السويس": (29.9668, 32.5498), "الفيوم": (29.3084, 30.8428),

        "بني سويف": (29.0661, 31.0994), "المنيا": (28.1099, 30.7503), "أسيوط": (27.1783, 31.1859),

        "سوهاج": (26.5591, 31.6957), "قنا": (26.1551, 32.7160), "الأقصر": (25.6872, 32.6396),

        "أسوان": (24.0889, 32.8998), "البحر الأحمر": (27.2579, 33.8116), "مطروح": (31.3543, 27.2373),

        "شمال سيناء": (30.4560, 33.8),

    }

    # Correct the Sinai coordinate separately to avoid relying on unavailable geocoding services.

    geo["شمال سيناء"] = (30.4560, 33.8)

    geo["جنوب سيناء"] = (28.5, 33.9)

    mapped = governorate_counts.copy()

    mapped["lat"] = mapped["governorate_ar"].map(lambda x: geo.get(str(x), (None, None))[0])

    mapped["lon"] = mapped["governorate_ar"].map(lambda x: geo.get(str(x), (None, None))[1])

    mapped = mapped.dropna(subset=["lat", "lon"])

    if not mapped.empty:

        fig_map = px.scatter_geo(mapped, lat="lat", lon="lon", size="branch_count", color="branch_count",

            hover_name="governorate_ar", hover_data={"branch_count": True, "lat": False, "lon": False},

            scope="africa", center={"lat": 26.8, "lon": 30.8}, template="plotly_white",

            title="Branch count by governorate (bubble size = branches)")

        fig_map.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="#999")

        fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0))

        st.plotly_chart(fig_map, use_container_width=True)

    else:

        st.info("Governorate names did not match the built-in map locations. Branch counts are shown below.")

    st.dataframe(governorate_counts.sort_values("branch_count", ascending=False), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Revenue by governorate")

    branch_revenue = items[items["order_date"].between(period_start, period_end)].merge(

        branches[["branch_id", "governorate_ar"]].drop_duplicates(), on="branch_id", how="left")

    gov_revenue = (branch_revenue.groupby("governorate_ar", dropna=False, as_index=False)["net_line_egp"].sum()

                   .sort_values("net_line_egp", ascending=False))

    top5 = gov_revenue.head(5).copy()

    others = gov_revenue.iloc[5:]["net_line_egp"].sum()

    if others > 0:

        top5 = pd.concat([top5, pd.DataFrame([{"governorate_ar": "Others", "net_line_egp": others}])], ignore_index=True)

    if not top5.empty:

        fig_gov = px.bar(top5, x="governorate_ar", y="net_line_egp", text_auto=".2s", template="plotly_white",

                         labels={"governorate_ar": "Governorate", "net_line_egp": "Revenue (EGP)"})

        fig_gov.update_layout(height=420, xaxis_title="", yaxis_title="Revenue (EGP)")

        st.plotly_chart(fig_gov, use_container_width=True)

    else:

        st.info("No revenue rows are available for this period.")

    st.divider()

    st.subheader("Selected branch — revenue growth")

    branch_options = branches.dropna(subset=["branch_id"]).drop_duplicates("branch_id").copy()

    branch_options["display"] = branch_options["branch_name_ar"].fillna(branch_options["branch_id"]).astype(str)

    if not branch_options.empty:

        chosen = st.selectbox("Select a branch", branch_options["branch_id"].tolist(),

                              format_func=lambda x: branch_options.set_index("branch_id").loc[x, "display"])

        branch_items = items[items["branch_id"] == chosen].copy()

        branch_items["month"] = branch_items["order_date"].dt.to_period("M").dt.to_timestamp()

        branch_monthly = branch_items.groupby("month", as_index=False)["net_line_egp"].sum()

        current_year = branch_monthly[branch_monthly["month"].between(pd.Timestamp(as_of.year, 1, 1), as_of)].copy()

        previous_year = branch_monthly[branch_monthly["month"].between(pd.Timestamp(as_of.year-1, 1, 1), as_of-pd.DateOffset(years=1))].copy()

        current_year["Series"] = str(as_of.year)

        previous_year["month"] = previous_year["month"] + pd.DateOffset(years=1)

        previous_year["Series"] = str(as_of.year-1)

        trend = pd.concat([current_year, previous_year], ignore_index=True)

        if not trend.empty:

            fig_branch = px.line(trend, x="month", y="net_line_egp", color="Series", markers=True,

                                 template="plotly_white", labels={"month": "Month", "net_line_egp": "Revenue (EGP)"})

            fig_branch.update_layout(height=420, yaxis_title="Revenue (EGP)")

            st.plotly_chart(fig_branch, use_container_width=True)

        else:

            st.info("No branch revenue history is available for the selected branch.")

    if mode == "Quarter-by-quarter":

        st.divider()

        st.subheader("Quarterly revenue — quarter over quarter")

        q = items.dropna(subset=["order_date"]).copy()

        q["quarter"] = q["order_date"].dt.to_period("Q").astype(str)

        quarterly = q.groupby("quarter", as_index=False)["net_line_egp"].sum().sort_values("quarter")

        quarterly["qoq_change_pct"] = quarterly["net_line_egp"].pct_change() * 100

        fig_q = px.bar(quarterly, x="quarter", y="net_line_egp", hover_data={"qoq_change_pct": ":.1f"},

                       template="plotly_white", labels={"quarter": "Quarter", "net_line_egp": "Revenue (EGP)"})

        fig_q.update_layout(height=400)

        st.plotly_chart(fig_q, use_container_width=True)

        st.dataframe(quarterly, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Employee analysis by job")

    employees["position_en"] = employees["position_en"].fillna("Unspecified").astype(str).str.strip()

    employees["employee_name_ar"] = employees["employee_name_ar"].fillna("Unknown").astype(str).str.strip()

    job_stats = employees.groupby("position_en").agg(

        employee_count=("employee_id", "nunique"),

        min_salary=("salary_base_egp_2026", "min"),

        max_salary=("salary_base_egp_2026", "max"),

    ).reset_index().rename(columns={"position_en": "Job", "employee_count": "Employee Count", "min_salary": "Minimum Salary (EGP)", "max_salary": "Maximum Salary (EGP)"})

    max_salary_by_job = employees.groupby("position_en")["salary_base_egp_2026"].transform("max")

    highest_paid = employees[employees["salary_base_egp_2026"].eq(max_salary_by_job) & employees["salary_base_egp_2026"].notna()][["position_en", "employee_name_ar", "salary_base_egp_2026"]].drop_duplicates()

    highest_paid = highest_paid.rename(columns={"position_en": "Job", "employee_name_ar": "Employee with Maximum Salary", "salary_base_egp_2026": "Maximum Salary (EGP)"})

    ec1, ec2 = st.columns([1.2, 1])

    with ec1:

        st.markdown("**Employees and salary range by job**")

        st.dataframe(job_stats.style.format({"Minimum Salary (EGP)": "{:,.0f}", "Maximum Salary (EGP)": "{:,.0f}"}), use_container_width=True, hide_index=True)

    with ec2:

        st.markdown("**Highest-paid employee(s) in each job**")

        st.dataframe(highest_paid.style.format({"Maximum Salary (EGP)": "{:,.0f}"}), use_container_width=True, hide_index=True)

# =========================================================

# Branches Page

# =========================================================


def branches_page():
    """Branch distribution and quarterly branch performance analysis."""
    import re
    import numpy as np
    import plotly.graph_objects as go

    st.title("🏢 Branch Performance")
    st.caption("Branch distribution, quarterly overview and branch rankings")

    # -----------------------------------------------------
    # Load and prepare the quarterly ranking dataset
    # -----------------------------------------------------
    data_path = Path(__file__).resolve().parent / "data" / "branch_ranks.csv"

    @st.cache_data
    def load_branch_ranking_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def parse_branch_quarter(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, pd.Period):
            return value.asfreq("Q")

        quarter_text = str(value).strip().upper()
        match = re.search(r"(\d{4})\s*[-_/ ]?\s*Q([1-4])", quarter_text)
        if match:
            return pd.Period(
                f"{match.group(1)}Q{match.group(2)}", freq="Q"
            )
        try:
            return pd.Period(quarter_text, freq="Q")
        except Exception:
            return pd.NaT

    def prepare_branch_ranking_data(data: pd.DataFrame) -> pd.DataFrame:
        required_columns = [
            "branch_id",
            "branch_name_ar",
            "governorate_ar",
            "Quarter Year",
            "Total_Orders",
            "Returned_Orders",
            "Completed_Orders",
            "Net_Profit",
            "amount_EGP",
        ]
        missing = [
            column for column in required_columns
            if column not in data.columns
        ]
        if missing:
            raise ValueError(
                "Missing required columns in data/branch_ranks.csv: "
                + ", ".join(missing)
            )

        data = data.copy()
        data["Quarter"] = data["Quarter Year"].apply(parse_branch_quarter)
        data = data.dropna(subset=["Quarter", "branch_id"])

        numeric_columns = [
            "Total_Orders",
            "Returned_Orders",
            "Completed_Orders",
            "Net_Profit",
            "amount_EGP",
        ]
        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            ).fillna(0)

        data["Completed_Ratio"] = np.where(
            data["Total_Orders"] > 0,
            data["Completed_Orders"] / data["Total_Orders"],
            np.nan,
        )

        # Rank within each quarter:
        # Net Profit descending, Completed Ratio descending,
        # then Amount EGP ascending.
        data = data.sort_values(
            ["Quarter", "Net_Profit", "Completed_Ratio", "amount_EGP"],
            ascending=[True, False, False, True],
            na_position="last",
        )
        data["Branch_Rank"] = (
            data.groupby("Quarter", observed=True).cumcount() + 1
        )
        return data

    if not data_path.exists():
        st.error(
            "Could not find data/branch_ranks.csv. "
            "Place branch_ranks.csv inside the data folder."
        )
        return

    try:
        ranking_df = prepare_branch_ranking_data(
            load_branch_ranking_data(str(data_path))
        )
    except Exception as error:
        st.error(f"Could not load or prepare branch ranking data: {error}")
        return

    if ranking_df.empty:
        st.warning("The branch ranking dataset has no valid quarterly records.")
        return

    # -----------------------------------------------------
    # Preserve the original Branches page map and donut chart
    # -----------------------------------------------------
    try:
        branches = load_csv(BRANCHES_PATH)
        validate_columns(
            branches,
            {"branch_id", "branch_name_ar", "governorate_ar"},
            BRANCHES_PATH.name,
        )
    except FileNotFoundError as error:
        st.error(f"Required file not found: {error.filename}")
        return

    branches["branch_id"] = (
        branches["branch_id"].astype("string").str.strip()
    )
    branches["governorate_ar"] = (
        branches["governorate_ar"].astype("string").str.strip()
    )

    governorate_counts = (
        branches.dropna(subset=["governorate_ar", "branch_id"])
        .groupby("governorate_ar", as_index=False)["branch_id"]
        .nunique()
        .rename(columns={"branch_id": "branch_count"})
        .sort_values("branch_count", ascending=False)
    )

    st.subheader("Branch Distribution Across Governorates")

    if governorate_counts.empty:
        st.info("No governorate branch data is available.")
    else:
        map_col, share_col = st.columns([1.35, 1])

        with map_col:
            st.markdown("**Branch locations by governorate**")
            branch_map = folium.Map(
                location=[26.8, 30.8],
                zoom_start=5,
                tiles="OpenStreetMap",
                control_scale=True,
            )

            max_branch_count = max(
                int(governorate_counts["branch_count"].max()), 1
            )
            missing_map_locations = []

            for _, row in governorate_counts.iterrows():
                governorate_name = str(row["governorate_ar"])
                branch_count = int(row["branch_count"])
                coordinates = get_coordinates_safe(governorate_name)

                if coordinates is None:
                    missing_map_locations.append(governorate_name)
                    continue

                intensity = branch_count / max_branch_count
                folium.CircleMarker(
                    location=coordinates,
                    radius=6 + 11 * intensity,
                    color="#7A1F2B",
                    weight=1,
                    fill=True,
                    fill_color="#7A1F2B",
                    fill_opacity=0.35 + 0.55 * intensity,
                    tooltip=f"{governorate_name}: {branch_count} branches",
                    popup=folium.Popup(
                        f"<b>{governorate_name}</b><br>"
                        f"Branches: {branch_count}",
                        max_width=220,
                    ),
                ).add_to(branch_map)

            st_folium(
                branch_map,
                use_container_width=True,
                height=430,
                key="branches_governorate_distribution_map",
            )

            if missing_map_locations:
                st.caption(
                    "Map coordinates unavailable for: "
                    + "، ".join(sorted(set(missing_map_locations)))
                )

        with share_col:
            st.markdown("**Branch share by governorate**")
            share_data = governorate_counts.head(5)[
                ["governorate_ar", "branch_count"]
            ].copy()
            other_count = int(
                governorate_counts.iloc[5:]["branch_count"].sum()
            )

            if other_count > 0:
                share_data = pd.concat(
                    [
                        share_data,
                        pd.DataFrame([{
                            "governorate_ar": "Others",
                            "branch_count": other_count,
                        }]),
                    ],
                    ignore_index=True,
                )

            fig_branch_share = px.pie(
                share_data,
                names="governorate_ar",
                values="branch_count",
                hole=0.48,
                labels={
                    "governorate_ar": "Governorate",
                    "branch_count": "Branches",
                },
                template="plotly_white",
            )
            fig_branch_share.update_traces(
                textposition="inside",
                textinfo="label+percent",
                hovertemplate=(
                    "Governorate: %{label}<br>"
                    "Branches: %{value:,}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            )
            fig_branch_share.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=25, b=10),
            )
            st.plotly_chart(fig_branch_share, use_container_width=True)

    # -----------------------------------------------------
    # Formatting helpers and quarterly summary (from file 2)
    # -----------------------------------------------------
    def format_branch_egp(value):
        return f"{value:,.0f} EGP"

    def branch_quarter_label(period):
        return f"{period.year} Q{period.quarter}"

    quarterly = (
        ranking_df.groupby("Quarter", observed=True)
        .agg(
            Total_Branches=("branch_id", "nunique"),
            Total_Orders=("Total_Orders", "sum"),
            Returned_Orders=("Returned_Orders", "sum"),
            Completed_Orders=("Completed_Orders", "sum"),
            Net_Profit=("Net_Profit", "sum"),
            Amount_EGP=("amount_EGP", "sum"),
        )
        .reset_index()
        .sort_values("Quarter")
    )

    available_quarters = sorted(ranking_df["Quarter"].dropna().unique())
    current_quarter = available_quarters[-1]
    last_quarter = current_quarter - 1

    current_row = quarterly.loc[quarterly["Quarter"] == current_quarter]
    last_row = quarterly.loc[quarterly["Quarter"] == last_quarter]

    current_branches = (
        int(current_row["Total_Branches"].iloc[0])
        if not current_row.empty else 0
    )
    last_branches = (
        int(last_row["Total_Branches"].iloc[0])
        if not last_row.empty else 0
    )
    current_profit = (
        float(current_row["Net_Profit"].iloc[0])
        if not current_row.empty else 0.0
    )
    last_profit = (
        float(last_row["Net_Profit"].iloc[0])
        if not last_row.empty else 0.0
    )

    # -----------------------------------------------------
    # 1. Overall Performance
    # -----------------------------------------------------
    st.divider()
    st.header("1. Overall Performance")
    st.caption(
        f"Latest available quarter: {branch_quarter_label(current_quarter)} "
        f"| Previous quarter: {branch_quarter_label(last_quarter)}"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Branches — Latest Quarter",
        f"{current_branches:,}",
        help="Distinct branches with records in the latest available quarter.",
    )
    k2.metric(
        "Branches — Previous Quarter",
        f"{last_branches:,}",
        help="Distinct branches with records in the previous quarter.",
    )
    k3.metric(
        "Net Profit — Current Quarter",
        format_branch_egp(current_profit),
        delta=(
            f"{current_profit - last_profit:,.0f} EGP vs previous quarter"
            if not last_row.empty else None
        ),
    )
    k4.metric(
        "Net Profit — Previous Quarter",
        format_branch_egp(last_profit),
    )

    st.subheader("Company Quarterly Trends")

    # Chart 1: Net Profit bars + Amount EGP line.
    fig_profit_amount = go.Figure()
    fig_profit_amount.add_trace(
        go.Bar(
            x=quarterly["Quarter"].astype(str),
            y=quarterly["Net_Profit"],
            name="Net Profit",
            marker_color="#2878B5",
            yaxis="y",
            hovertemplate=(
                "Quarter: %{x}<br>"
                "Net Profit: %{y:,.0f} EGP<extra></extra>"
            ),
        )
    )
    fig_profit_amount.add_trace(
        go.Scatter(
            x=quarterly["Quarter"].astype(str),
            y=quarterly["Amount_EGP"],
            name="Amount EGP",
            mode="lines+markers",
            line=dict(color="#E58A2B", width=3),
            marker=dict(size=7),
            yaxis="y2",
            hovertemplate=(
                "Quarter: %{x}<br>Amount: %{y:,.0f} EGP<extra></extra>"
            ),
        )
    )
    fig_profit_amount.update_layout(
        title="Quarterly Net Profit vs Amount EGP",
        xaxis_title="Quarter",
        yaxis=dict(title="Net Profit (EGP)", separatethousands=True),
        yaxis2=dict(
            title="Amount EGP",
            overlaying="y",
            side="right",
            separatethousands=True,
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=70),
    )
    st.plotly_chart(fig_profit_amount, use_container_width=True)

    # Chart 2: order composition by quarter.
    fig_orders = go.Figure()
    fig_orders.add_trace(
        go.Bar(
            x=quarterly["Quarter"].astype(str),
            y=quarterly["Completed_Orders"],
            name="Completed Orders",
            marker_color="#27966F",
        )
    )
    fig_orders.add_trace(
        go.Bar(
            x=quarterly["Quarter"].astype(str),
            y=quarterly["Returned_Orders"],
            name="Returned Orders",
            marker_color="#C95C54",
        )
    )
    fig_orders.update_layout(
        title="Quarterly Completed vs Returned Orders",
        xaxis_title="Quarter",
        yaxis_title="Number of Orders",
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=60, b=70),
    )
    st.plotly_chart(fig_orders, use_container_width=True)

    # -----------------------------------------------------
    # 2. Branch Performance Analysis
    # -----------------------------------------------------
    st.divider()
    st.header("2. Branch Performance Analysis")

    quarter_options = sorted(available_quarters, reverse=True)
    selected_quarter = st.selectbox(
        "Select a quarter",
        options=quarter_options,
        format_func=branch_quarter_label,
        index=0,
        key="branch_performance_quarter_selector",
    )

    quarter_df = ranking_df.loc[
        ranking_df["Quarter"] == selected_quarter
    ].copy()
    quarter_df = quarter_df.sort_values("Branch_Rank")

    top_five = quarter_df.nsmallest(5, "Branch_Rank").copy()
    bottom_five = (
        quarter_df.nlargest(5, "Branch_Rank")
        .sort_values("Branch_Rank", ascending=False)
        .copy()
    )

    def show_ranking_table(data: pd.DataFrame, title: str):
        st.subheader(title)
        if data.empty:
            st.info("No branches available for this quarter.")
            return

        display = data[
            [
                "Branch_Rank",
                "branch_name_ar",
                "governorate_ar",
                "Total_Orders",
                "Completed_Orders",
                "Completed_Ratio",
                "Net_Profit",
                "amount_EGP",
            ]
        ].copy()

        display = display.rename(
            columns={
                "Branch_Rank": "Rank",
                "branch_name_ar": "اسم الفرع",
                "governorate_ar": "المحافظة",
                "Total_Orders": "Total Orders",
                "Completed_Orders": "Completed Orders",
                "Completed_Ratio": "Completed Ratio",
                "Net_Profit": "Net Profit (EGP)",
                "amount_EGP": "Amount EGP",
            }
        )
        display["Completed Ratio"] = display["Completed Ratio"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    def show_branch_analysis(
        ranked_data: pd.DataFrame,
        section_key: str,
        section_title: str,
    ):
        st.subheader(section_title)
        if ranked_data.empty:
            st.info("No branches available for this selection.")
            return

        branch_options = ranked_data["branch_id"].tolist()
        branch_name_lookup = (
            ranked_data.set_index("branch_id")["branch_name_ar"].to_dict()
        )

        selected_branch_id = st.selectbox(
            "Choose a branch",
            options=branch_options,
            format_func=lambda branch_id: (
                f"{branch_name_lookup.get(branch_id, branch_id)} — {branch_id}"
            ),
            key=f"{section_key}_branch_selector",
        )

        branch_history = (
            ranking_df.loc[
                ranking_df["branch_id"] == selected_branch_id
            ]
            .sort_values("Quarter")
            .copy()
        )

        if branch_history.empty:
            st.info("No historical records found for this branch.")
            return

        branch_name = branch_history["branch_name_ar"].iloc[0]
        governorate = branch_history["governorate_ar"].iloc[0]
        first_quarter = branch_history["Quarter"].min()

        st.caption(
            f"{branch_name} | {governorate} | "
            f"First available quarter: {branch_quarter_label(first_quarter)}"
        )

        # Line chart 1: completed orders ratio over time.
        ratio_fig = px.line(
            branch_history,
            x=branch_history["Quarter"].astype(str),
            y="Completed_Ratio",
            markers=True,
            title="Completed Orders Ratio by Quarter",
            labels={
                "x": "Quarter",
                "Completed_Ratio": "Completed Orders Ratio",
            },
        )
        ratio_fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        ratio_fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=30),
        )
        st.plotly_chart(ratio_fig, use_container_width=True)

        # Net Profit bars + Amount EGP line for this branch.
        branch_fig = go.Figure()
        branch_fig.add_trace(
            go.Bar(
                x=branch_history["Quarter"].astype(str),
                y=branch_history["Net_Profit"],
                name="Net Profit",
                marker_color="#2878B5",
                yaxis="y",
            )
        )
        branch_fig.add_trace(
            go.Scatter(
                x=branch_history["Quarter"].astype(str),
                y=branch_history["amount_EGP"],
                name="Amount EGP",
                mode="lines+markers",
                line=dict(color="#E58A2B", width=3),
                yaxis="y2",
            )
        )
        branch_fig.update_layout(
            title="Quarterly Net Profit vs Amount EGP",
            xaxis_title="Quarter",
            yaxis=dict(title="Net Profit (EGP)", separatethousands=True),
            yaxis2=dict(
                title="Amount EGP",
                overlaying="y",
                side="right",
                separatethousands=True,
            ),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=20, r=20, t=60, b=70),
        )
        st.plotly_chart(branch_fig, use_container_width=True)

    def show_ranking_governorate_map(
        data: pd.DataFrame, title: str, key: str
    ):
        st.subheader(title)

        governorate_coordinates = {
            "القاهرة": (30.0444, 31.2357),
            "الجيزة": (30.0131, 31.2089),
            "القليوبية": (30.3292, 31.2168),
            "الإسكندرية": (31.2001, 29.9187),
            "البحيرة": (30.8481, 30.3436),
            "مطروح": (31.3543, 27.2373),
            "دمياط": (31.4175, 31.8144),
            "الدقهلية": (31.0409, 31.3785),
            "الشرقية": (30.5877, 31.5020),
            "الغربية": (30.8754, 31.0335),
            "المنوفية": (30.5972, 30.9876),
            "كفر الشيخ": (31.1107, 30.9388),
            "الإسماعيلية": (30.5965, 32.2715),
            "بورسعيد": (31.2653, 32.3019),
            "السويس": (29.9668, 32.5498),
            "شمال سيناء": (30.2824, 33.6176),
            "جنوب سيناء": (28.2410, 33.6220),
            "الفيوم": (29.3084, 30.8428),
            "بني سويف": (29.0661, 31.0994),
            "المنيا": (28.1099, 30.7503),
            "أسيوط": (27.1809, 31.1837),
            "سوهاج": (26.5591, 31.6957),
            "قنا": (26.1551, 32.7160),
            "الأقصر": (25.6872, 32.6396),
            "أسوان": (24.0889, 32.8998),
            "البحر الأحمر": (27.2579, 33.8116),
            "الوادي الجديد": (25.4514, 30.5460),
        }

        if data.empty:
            st.info("No locations available for this ranking.")
            return

        map_data = (
            data.groupby("governorate_ar", dropna=False)
            .agg(Branch_Count=("branch_id", "nunique"))
            .reset_index()
        )

        branch_map = folium.Map(
            location=[26.8, 30.8],
            zoom_start=5,
            tiles="OpenStreetMap",
            control_scale=True,
        )
        missing_locations = []
        max_count = max(int(map_data["Branch_Count"].max()), 1)

        for _, row in map_data.iterrows():
            governorate_name = str(row["governorate_ar"])
            count = int(row["Branch_Count"])
            coordinates = governorate_coordinates.get(governorate_name)

            if coordinates is None:
                missing_locations.append(governorate_name)
                continue

            intensity = count / max_count
            folium.CircleMarker(
                location=coordinates,
                radius=7 + (5 * intensity),
                color="#7A1F2B",
                weight=1,
                fill=True,
                fill_color="#7A1F2B",
                fill_opacity=0.35 + (0.6 * intensity),
                tooltip=f"{governorate_name}: {count} branch(es)",
                popup=folium.Popup(
                    f"<b>{governorate_name}</b><br>Branches: {count}",
                    max_width=220,
                ),
            ).add_to(branch_map)

        st_folium(
            branch_map,
            use_container_width=True,
            height=430,
            key=key,
        )

        if missing_locations:
            st.caption(
                "Map coordinates are not configured for: "
                + "، ".join(sorted(set(missing_locations)))
            )

    # Top 5 section
    st.divider()
    show_ranking_table(top_five, "Top 5 Branches")
    show_branch_analysis(top_five, "top", "Top 5 Branch Analysis")
    show_ranking_governorate_map(
        top_five, "Top 5 Branches by Governorate", "top_ranking_map"
    )

    # Bottom 5 section
    st.divider()
    show_ranking_table(bottom_five, "Bottom 5 Branches")
    show_branch_analysis(bottom_five, "bottom", "Bottom 5 Branch Analysis")
    show_ranking_governorate_map(
        bottom_five, "Bottom 5 Branches by Governorate", "bottom_ranking_map"
    )

    # -----------------------------------------------------
    # Data notes
    # -----------------------------------------------------
    with st.expander("Data and ranking notes"):
        st.markdown(
            """
            - The dashboard uses the latest quarter available in
              `data/branch_ranks.csv` as the latest/current reporting quarter.
            - A branch is counted as active in a quarter when it has a row for
              that quarter in the dataset.
            - Branch ranking is calculated independently for each quarter:
              Net Profit descending, Completed Orders Ratio descending, then
              Amount EGP ascending.
            - The original Branch Distribution map uses governorate-level
              coordinates and shows branch concentration, not exact branch
              addresses.
            - The ranking maps show approximate governorate-centre locations.
            """
        )

from pathlib import Path

import math

import pandas as pd

import streamlit as st

import plotly.graph_objects as go

import folium

from folium.plugins import MarkerCluster

from streamlit_folium import st_folium

from egygeo import get_coordinates

# =========================================================

# FILE PATHS

# =========================================================

BASE_DIR = Path(__file__).resolve().parent

BRANCH_PERFORMANCE_PATH = BASE_DIR / "branch_performance.csv"

FACTORIES_PATH = BASE_DIR / "data" / "factories.csv"

WAREHOUSES_PATH = BASE_DIR / "data" / "warehouses.csv"

# =========================================================

# HELPER FUNCTIONS

# =========================================================

def read_csv_clean(path):

    """Read CSV and normalize column names."""

    if not path.exists():

        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    df.columns = df.columns.str.strip().str.lower()

    return df

def get_coordinates_safe(governorate):

    """Return approximate governorate coordinates."""

    try:

        coords = get_coordinates(str(governorate))

        if coords is None or len(coords) < 2:

            return None

        lat = float(coords[0])

        lon = float(coords[1])

        if -90 <= lat <= 90 and -180 <= lon <= 180:

            return lat, lon

    except Exception:

        pass

    return None

def haversine_km(lat1, lon1, lat2, lon2):

    """Calculate straight-line distance in kilometers."""

    earth_radius = 6371

    lat1, lon1, lat2, lon2 = map(

        math.radians,

        [lat1, lon1, lat2, lon2],

    )

    dlat = lat2 - lat1

    dlon = lon2 - lon1

    a = (

        math.sin(dlat / 2) ** 2

        + math.cos(lat1)

        * math.cos(lat2)

        * math.sin(dlon / 2) ** 2

    )

    return 2 * earth_radius * math.asin(math.sqrt(a))

def add_locations(df, governorate_column="governorate_ar"):

    """Use existing coordinates when available; otherwise resolve governorates."""

    df = df.copy()

    if "latitude" not in df.columns:

        df["latitude"] = pd.NA

    if "longitude" not in df.columns:

        df["longitude"] = pd.NA

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    for idx, row in df.iterrows():

        lat = row["latitude"]

        lon = row["longitude"]

        if pd.notna(lat) and pd.notna(lon):

            if -90 <= lat <= 90 and -180 <= lon <= 180:

                continue

        coords = get_coordinates_safe(row[governorate_column])

        if coords is not None:

            df.at[idx, "latitude"] = coords[0]

            df.at[idx, "longitude"] = coords[1]

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    return df.dropna(subset=["latitude", "longitude"])

# =========================================================

# PAGE

# =========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


# ============================================================
# SALES PERFORMANCE INTELLIGENCE
# ============================================================

def sales_performance_intelligence_page(data_dir="data"):

    st.title("Sales Performance Intelligence")
    st.caption(
        "Analyze sales, profitability, product performance, "
        "and geographic contribution."
    )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    data_dir = Path(data_dir)

    required_files = {
        "orders": "sales_orders.csv",
        "items": "sales_order_items.csv",
        "returns": "sales_returns.csv",
        "products": "products_catalog.csv",
        "branches": "branches.csv",
    }

    try:
        data = {
            key: pd.read_csv(data_dir / filename)
            for key, filename in required_files.items()
        }

    except FileNotFoundError as e:
        st.error(f"Missing data file: {e.filename}")
        return

    orders = data["orders"]
    items = data["items"]
    returns = data["returns"]
    products = data["products"]
    branches = data["branches"]

    # --------------------------------------------------------
    # 2. DATA PREPARATION
    # --------------------------------------------------------

    items["order_date"] = pd.to_datetime(
        items["order_date"], errors="coerce"
    )

    items["quantity"] = pd.to_numeric(
        items["quantity"], errors="coerce"
    ).fillna(0)

    items["net_line_EGP"] = pd.to_numeric(
        items["net_line_EGP"], errors="coerce"
    ).fillna(0)

    items["margin_EGP"] = pd.to_numeric(
        items["margin_EGP"], errors="coerce"
    ).fillna(0)

    returns["return_quantity"] = pd.to_numeric(
        returns["return_quantity"], errors="coerce"
    ).fillna(0)

    returns["refund_amount_EGP"] = pd.to_numeric(
        returns["refund_amount_EGP"], errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # 3. PROCESS RETURNS
    # --------------------------------------------------------

    # Aggregate returns by sales line to avoid duplicate rows.

    returns_by_line = (
        returns.groupby("sales_line_id", as_index=False)
        .agg(
            returned_quantity=("return_quantity", "sum"),
            refund_amount=("refund_amount_EGP", "sum")
        )
    )

    items = items.merge(
        returns_by_line,
        on="sales_line_id",
        how="left"
    )

    items["returned_quantity"] = (
        items["returned_quantity"].fillna(0)
    )

    items["refund_amount"] = (
        items["refund_amount"].fillna(0)
    )

    # Remaining quantity after returns.

    items["sold_quantity"] = (
        items["quantity"] - items["returned_quantity"]
    ).clip(lower=0)

    # Calculate returned margin proportionally.

    items["unit_margin"] = np.where(
        items["quantity"] > 0,
        items["margin_EGP"] / items["quantity"],
        0
    )

    items["returned_margin"] = (
        items["unit_margin"] * items["returned_quantity"]
    )

    # Net sales and profit.

    items["net_sales"] = (
        items["net_line_EGP"] - items["refund_amount"]
    )

    items["net_profit"] = (
        items["margin_EGP"] - items["returned_margin"]
    )

    # --------------------------------------------------------
    # 4. MERGE PRODUCT AND BRANCH INFORMATION
    # --------------------------------------------------------

    product_info = products[
        ["product_id", "product_name_ar", "category_en"]
    ].drop_duplicates("product_id")

    items = items.merge(
        product_info,
        on="product_id",
        how="left"
    )

    items["product_name_ar"] = (
        items["product_name_ar"].fillna("Unknown Product")
    )

    items["category_en"] = (
        items["category_en"].fillna("Other")
    )

    branch_info = branches[
        ["branch_id", "governorate_ar"]
    ].drop_duplicates("branch_id")

    items = items.merge(
        branch_info,
        on="branch_id",
        how="left"
    )

    items["governorate_ar"] = (
        items["governorate_ar"].fillna("Unknown")
    )

    # --------------------------------------------------------
    # 5. PERIOD SELECTION
    # --------------------------------------------------------

    today = pd.Timestamp.today().normalize()

    period_option = st.radio(
        "Analysis Period",
        ["Last Quarter", "Last 12 Months"],
        horizontal=True
    )

    if period_option == "Last Quarter":

        # Current quarter starts at its first month.
        current_quarter_start = today.to_period("Q").start_time

        # Last completed quarter.
        period_end = current_quarter_start

        period_start = (
            period_end - pd.DateOffset(months=3)
        )

        previous_end = period_start

        previous_start = (
            previous_end - pd.DateOffset(months=3)
        )

    else:

        # Rolling 12 months including current date.
        period_end = today + pd.Timedelta(days=1)

        period_start = (
            today - pd.DateOffset(months=12)
        )

        previous_end = period_start

        previous_start = (
            previous_end - pd.DateOffset(months=12)
        )

    # --------------------------------------------------------
    # 6. FILTER PERIODS
    # --------------------------------------------------------

    current = items[
        (items["order_date"] >= period_start) &
        (items["order_date"] < period_end)
    ].copy()

    previous = items[
        (items["order_date"] >= previous_start) &
        (items["order_date"] < previous_end)
    ].copy()

    if current.empty:
        st.warning("No sales data available for the selected period.")
        return

    # --------------------------------------------------------
    # 7. KPI CALCULATIONS
    # --------------------------------------------------------

    def calculate_kpis(df):

        total_sales = df["net_sales"].sum()

        total_profit = df["net_profit"].sum()

        total_items = df["sold_quantity"].sum()

        # Calculate basket value from net sales per order.
        order_sales = (
            df.groupby("sales_order_id")["net_sales"].sum()
        )

        # Exclude fully refunded / zero-value orders.
        valid_orders = order_sales[order_sales > 0]

        number_of_orders = len(valid_orders)

        average_basket = (
            valid_orders.mean()
            if number_of_orders > 0 else 0
        )

        return {
            "sales": total_sales,
            "profit": total_profit,
            "items": total_items,
            "orders": number_of_orders,
            "basket": average_basket
        }

    current_kpis = calculate_kpis(current)
    previous_kpis = calculate_kpis(previous)

    def percentage_change(current_value, previous_value):

        if previous_value == 0:
            return None

        return (
            (current_value - previous_value)
            / abs(previous_value)
        ) * 100

    def format_delta(current_value, previous_value):

        change = percentage_change(
            current_value, previous_value
        )

        if change is None:
            return "N/A"

        return f"{change:+.2f}%"

    # --------------------------------------------------------
    # 8. KPI DASHBOARD
    # --------------------------------------------------------

    st.subheader("Sales Overview")

    st.caption(
        f"Current Period: {period_start:%d %b %Y} "
        f"— {(period_end - pd.Timedelta(days=1)):%d %b %Y}"
    )

    st.caption(
        f"Comparison Period: {previous_start:%d %b %Y} "
        f"— {(previous_end - pd.Timedelta(days=1)):%d %b %Y}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Sales",
            f"{current_kpis['sales']:,.0f} EGP",
            format_delta(
                current_kpis["sales"],
                previous_kpis["sales"]
            )
        )

    with col2:
        st.metric(
            "Items Sold",
            f"{current_kpis['items']:,.0f}",
            format_delta(
                current_kpis["items"],
                previous_kpis["items"]
            )
        )

    with col3:
        st.metric(
            "Average Basket Value",
            f"{current_kpis['basket']:,.0f} EGP",
            format_delta(
                current_kpis["basket"],
                previous_kpis["basket"]
            )
        )

    with col4:
        st.metric(
            "Total Profit",
            f"{current_kpis['profit']:,.0f} EGP",
            format_delta(
                current_kpis["profit"],
                previous_kpis["profit"]
            )
        )

    st.divider()

    # --------------------------------------------------------
    # 9. SALES BY CATEGORY
    # --------------------------------------------------------

    st.subheader("Sales by Category")

    category_profit = (
        current.groupby("category_en")["net_profit"]
        .sum()
        .sort_values(ascending=False)
    )

    top_categories = category_profit.head(4)

    other_profit = category_profit.iloc[4:].sum()

    if other_profit != 0:
        category_chart_data = pd.concat([
            top_categories,
            pd.Series({"Other": other_profit})
        ])
    else:
        category_chart_data = top_categories

    category_df = (
        category_chart_data.rename("Profit").reset_index()
    )

    category_df.columns = ["Category", "Profit"]

    category_df["Profit Share %"] = (
        category_df["Profit"] /
        category_df["Profit"].sum() * 100
        if category_df["Profit"].sum() != 0 else 0
    )

    fig_category = px.pie(
        category_df,
        names="Category",
        values="Profit",
        hole=0.55,
        title="Profit Contribution by Category",
        custom_data=["Profit Share %"]
    )

    fig_category.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Profit: %{value:,.0f} EGP<br>"
            "Contribution: %{customdata[0]:.2f}%"
            "<extra></extra>"
        )
    )

    fig_category.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=20, r=20)
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # 10. PROFIT BY GOVERNORATE
    # --------------------------------------------------------

    st.subheader("Profit by Governorate")

    governorate_profit = (
        current.groupby("governorate_ar")["net_profit"]
        .sum()
        .sort_values(ascending=False)
    )

    top_governorates = governorate_profit.head(5)

    other_governorates = governorate_profit.iloc[5:].sum()

    if other_governorates != 0:
        governorate_chart_data = pd.concat([
            top_governorates,
            pd.Series({"Other": other_governorates})
        ])
    else:
        governorate_chart_data = top_governorates

    governorate_df = (
        governorate_chart_data.rename("Profit").reset_index()
    )

    governorate_df.columns = ["Governorate", "Profit"]

    governorate_df["Profit Share %"] = (
        governorate_df["Profit"] /
        governorate_df["Profit"].sum() * 100
        if governorate_df["Profit"].sum() != 0 else 0
    )

    fig_governorate = px.bar(
        governorate_df.sort_values("Profit"),
        x="Profit",
        y="Governorate",
        orientation="h",
        text="Profit",
        title="Profit Contribution by Governorate",
        custom_data=["Profit Share %"]
    )

    fig_governorate.update_traces(
        texttemplate="%{x:,.0f}",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Profit: %{x:,.0f} EGP<br>"
            "Contribution: %{customdata[0]:.2f}%"
            "<extra></extra>"
        )
    )

    fig_governorate.update_layout(
        height=420,
        xaxis_title="Profit (EGP)",
        yaxis_title="",
        showlegend=False
    )

    st.plotly_chart(
        fig_governorate,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # 11. MONTHLY SALES & PROFIT PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Monthly Sales & Profit Performance")

    monthly = (
        current.assign(
            month=current["order_date"].dt.to_period("M")
        )
        .groupby("month")
        .agg(
            Profit=("net_profit", "sum"),
            Items_Sold=("sold_quantity", "sum"),
            Sales=("net_sales", "sum")
        )
    )

    # Include months without sales as zero.
    all_months = pd.period_range(
        start=period_start.to_period("M"),
        end=(period_end - pd.Timedelta(days=1)).to_period("M"),
        freq="M"
    )

    monthly = monthly.reindex(all_months, fill_value=0)

    monthly.index = monthly.index.astype(str)

    monthly = monthly.reset_index()
    monthly.columns = ["Month", "Profit", "Items_Sold", "Sales"]

    fig_monthly = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    # Items sold columns.
    fig_monthly.add_trace(
        go.Bar(
            x=monthly["Month"],
            y=monthly["Items_Sold"],
            name="Items Sold",
            opacity=0.45
        ),
        secondary_y=False
    )

    # Profit line.
    fig_monthly.add_trace(
        go.Scatter(
            x=monthly["Month"],
            y=monthly["Profit"],
            name="Profit",
            mode="lines+markers",
            line=dict(width=3)
        ),
        secondary_y=True
    )

    fig_monthly.update_layout(
        title="Monthly Profit vs Items Sold",
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12
        )
    )

    fig_monthly.update_xaxes(
        title_text="Month"
    )

    fig_monthly.update_yaxes(
        title_text="Items Sold",
        secondary_y=False
    )

    fig_monthly.update_yaxes(
        title_text="Profit (EGP)",
        secondary_y=True
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # 12. TOP 5 BEST-SELLING PRODUCTS
    # --------------------------------------------------------

    st.subheader("Top 5 Best-Selling Products")

    product_performance = (
        current.groupby(
            ["product_id", "product_name_ar"],
            as_index=False
        )
        .agg(
            Items_Sold=("sold_quantity", "sum"),
            Profit=("net_profit", "sum"),
            Sales=("net_sales", "sum")
        )
    )

    product_performance = (
        product_performance[
            product_performance["Items_Sold"] > 0
        ]
        .sort_values("Items_Sold", ascending=False)
        .head(5)
        .copy()
    )

    # Calculate contribution to total items sold.
    total_items_sold = current["sold_quantity"].sum()

    product_performance["Sales Share %"] = (
        product_performance["Items_Sold"]
        / total_items_sold * 100
        if total_items_sold > 0 else 0
    )

    # Find the best-selling month for each product.
    monthly_product = (
        current.assign(
            month=current["order_date"].dt.to_period("M")
        )
        .groupby(
            ["product_id", "month"]
        )["sold_quantity"]
        .sum()
        .reset_index()
    )

    best_months = (
        monthly_product.loc[
            monthly_product.groupby("product_id")[
                "sold_quantity"
            ].idxmax()
        ][["product_id", "month"]]
        .copy()
    )

    best_months["Best Month"] = (
        best_months["month"].astype(str)
    )

    product_performance = product_performance.merge(
        best_months[["product_id", "Best Month"]],
        on="product_id",
        how="left"
    )

    # --------------------------------------------------------
    # 13. DISPLAY PRODUCT TABLE
    # --------------------------------------------------------

    display_products = product_performance[
        [
            "product_name_ar",
            "Items_Sold",
            "Profit",
            "Best Month",
            "Sales Share %"
        ]
    ].copy()

    display_products.columns = [
        "Product",
        "Items Sold",
        "Profit (EGP)",
        "Best Selling Month",
        "Sales Share (%)"
    ]

    display_products["Items Sold"] = (
        display_products["Items Sold"].map(
            lambda x: f"{x:,.0f}"
        )
    )

    display_products["Profit (EGP)"] = (
        display_products["Profit (EGP)"].map(
            lambda x: f"{x:,.2f}"
        )
    )

    display_products["Sales Share (%)"] = (
        display_products["Sales Share (%)"].map(
            lambda x: f"{x:.2f}%"
        )
    )

    st.dataframe(
        display_products,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 14. ADDITIONAL PRODUCT INSIGHTS
    # --------------------------------------------------------

    st.subheader("Product Performance Details")

    if not product_performance.empty:

        selected_product = st.selectbox(
            "Select Product",
            product_performance["product_name_ar"].tolist()
        )

        selected_id = product_performance.loc[
            product_performance["product_name_ar"]
            == selected_product,
            "product_id"
        ].iloc[0]

        product_monthly = monthly_product[
            monthly_product["product_id"] == selected_id
        ].copy()

        product_monthly["Month"] = (
            product_monthly["month"].astype(str)
        )

        fig_product = px.bar(
            product_monthly,
            x="Month",
            y="sold_quantity",
            title=f"Monthly Sales: {selected_product}",
            labels={
                "sold_quantity": "Items Sold",
                "Month": "Month"
            },
            text="sold_quantity"
        )

        fig_product.update_layout(
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig_product,
            use_container_width=True
        )

    st.divider()

    st.caption(
        "Sales Performance Intelligence | "
        "Data source: Cottonil Business Database"
    )




def warehouses_factory_recommendations_page():

    st.title("Warehouses and Factory Recommendations")

    # =====================================================
    # 1. PROFIT SHARE BY GOVERNORATE
    # =====================================================

    st.header("Profit Share by Governorate")

    try:
        branches = read_csv_clean(BRANCH_PERFORMANCE_PATH)

        required = {"profit", "governorate_ar"}

        if not required.issubset(branches.columns):
            st.error(
                "branch_performance.csv must contain "
                "'profit' and 'governorate_ar'."
            )
            return

        branches["profit"] = pd.to_numeric(
            branches["profit"], errors="coerce"
        )

        branches["governorate_ar"] = (
            branches["governorate_ar"].astype("string").str.strip()
        )

        branches = branches.dropna(
            subset=["profit", "governorate_ar"]
        )

        profit_by_governorate = (
            branches.groupby("governorate_ar", as_index=False)["profit"]
            .sum()
            .sort_values("profit", ascending=False)
        )

        total_profit = profit_by_governorate["profit"].sum()

        if profit_by_governorate.empty or total_profit <= 0:
            st.warning(
                "Cannot calculate profit share: total profit is zero "
                "or negative."
            )

        else:
            top_7 = profit_by_governorate.head(7).copy()
            others = profit_by_governorate.iloc[7:]

            chart_data = top_7.copy()

            if not others.empty:
                chart_data = pd.concat(
                    [
                        chart_data,
                        pd.DataFrame({
                            "governorate_ar": ["Other"],
                            "profit": [others["profit"].sum()]
                        })
                    ],
                    ignore_index=True
                )

            chart_data["share"] = (
                chart_data["profit"] / total_profit * 100
            )

            fig = go.Figure()

            for _, row in chart_data.iterrows():

                share = row["share"]

                fig.add_trace(
                    go.Bar(
                        name=str(row["governorate_ar"]),
                        x=[share],
                        y=["Profit Share"],
                        orientation="h",
                        text=f"{share:.1f}%" if share >= 5 else "",
                        textposition="inside",
                        hovertemplate=(
                            f'<b>{row["governorate_ar"]}</b><br>'
                            f'Profit: {row["profit"]:,.2f} EGP<br>'
                            f'Share: {share:.2f}%'
                            "<extra></extra>"
                        )
                    )
                )

            fig.update_layout(
                barmode="stack",
                title="Profit Share by Governorate",
                xaxis={
                    "title": "Share of Total Profit",
                    "range": [0, 100],
                    "ticksuffix": "%"
                },
                yaxis={"title": ""},
                height=240,
                legend={
                    "title": "Governorate",
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.45,
                    "xanchor": "left",
                    "x": 0
                },
                margin={"l": 20, "r": 20, "t": 60, "b": 110}
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "The chart shows the top 7 governorates by profit. "
                "All remaining governorates are grouped under Other."
            )

    except Exception as e:
        st.error(f"Could not load branch performance data: {e}")
        return

    st.divider()

    # =====================================================
    # 2. FACTORIES MAP
    # =====================================================

    st.header("Factories Map")

    try:
        factories = read_csv_clean(FACTORIES_PATH)

        if "governorate_ar" not in factories.columns:
            st.error("factories.csv must contain 'governorate_ar'.")
            return

        factories["governorate_ar"] = (
            factories["governorate_ar"].astype("string").str.strip()
        )

        factories = factories.dropna(
            subset=["governorate_ar"]
        )

        factories = add_locations(factories)

        if factories.empty:
            st.warning("No factory locations could be resolved.")

        else:
            factory_map = folium.Map(
                location=[26.8, 30.8],
                zoom_start=6,
                tiles="OpenStreetMap"
            )

            factory_cluster = MarkerCluster(
                name="Factories"
            ).add_to(factory_map)

            for _, factory in factories.iterrows():

                factory_name = factory.get(
                    "factory_name_ar",
                    factory.get("factory_id", "Factory")
                )

                factory_id = factory.get("factory_id", "N/A")
                governorate = factory["governorate_ar"]
                production = factory.get("production_focus", "N/A")

                popup_html = (
                    f"<b>Factory:</b> {factory_name}<br>"
                    f"<b>ID:</b> {factory_id}<br>"
                    f"<b>Governorate:</b> {governorate}<br>"
                    f"<b>Production:</b> {production}"
                )

                folium.Marker(
                    location=[
                        factory["latitude"],
                        factory["longitude"]
                    ],
                    tooltip=str(factory_name),
                    popup=folium.Popup(
                        popup_html,
                        max_width=350
                    ),
                    icon=folium.Icon(
                        color="red",
                        icon="industry",
                        prefix="fa"
                    )
                ).add_to(factory_cluster)

            st_folium(
                factory_map,
                height=500,
                use_container_width=True,
                key="factories_only_map"
            )

    except Exception as e:
        st.error(f"Could not load factory data: {e}")

    st.divider()

    # =====================================================
    # 3. WAREHOUSES MAP AND DISTANCE RANKING
    # =====================================================

    st.header("Warehouses Distance Recommendations")

    try:
        warehouses = read_csv_clean(WAREHOUSES_PATH)

        if "governorate_ar" not in warehouses.columns:
            st.error("warehouses.csv must contain 'governorate_ar'.")
            return

        warehouses["governorate_ar"] = (
            warehouses["governorate_ar"].astype("string").str.strip()
        )

        warehouses = warehouses.dropna(
            subset=["governorate_ar"]
        ).copy()

        warehouses = add_locations(warehouses)

        if warehouses.empty:
            st.warning(
                "No warehouse locations could be resolved. "
                "Check governorate names and coordinates."
            )

        else:
            warehouse_name_column = next(
                (
                    col for col in [
                        "warehouse_name_ar",
                        "warehouse_name",
                        "warehouse_id",
                        "name",
                        "id"
                    ]
                    if col in warehouses.columns
                ),
                None
            )

            if warehouse_name_column is None:
                warehouses["_warehouse_label"] = [
                    f"Warehouse {i + 1}"
                    for i in range(len(warehouses))
                ]
                warehouse_name_column = "_warehouse_label"

            egypt_governorates = [
                "القاهرة",
                "الجيزة",
                "الإسكندرية",
                "القليوبية",
                "البحيرة",
                "مطروح",
                "دمياط",
                "الدقهلية",
                "كفر الشيخ",
                "الغربية",
                "المنوفية",
                "الشرقية",
                "الإسماعيلية",
                "بورسعيد",
                "السويس",
                "شمال سيناء",
                "جنوب سيناء",
                "بني سويف",
                "الفيوم",
                "المنيا",
                "أسيوط",
                "سوهاج",
                "قنا",
                "الأقصر",
                "أسوان",
                "البحر الأحمر",
                "الوادي الجديد"
            ]

            selected_governorate = st.selectbox(
                "Select a governorate",
                options=egypt_governorates,
                index=None,
                placeholder="Choose any governorate in Egypt",
                key="warehouse_distance_governorate"
            )

            show_lines = st.checkbox(
                "Show distance lines to warehouses",
                value=False,
                key="warehouse_distance_lines"
            )

            warehouse_map = folium.Map(
                location=[26.8, 30.8],
                zoom_start=6,
                tiles="OpenStreetMap"
            )

            warehouse_cluster = MarkerCluster(
                name="Warehouses"
            ).add_to(warehouse_map)

            for _, warehouse in warehouses.iterrows():

                warehouse_name = warehouse[warehouse_name_column]

                folium.Marker(
                    location=[
                        warehouse["latitude"],
                        warehouse["longitude"]
                    ],
                    tooltip=str(warehouse_name),
                    popup=folium.Popup(
                        (
                            f"<b>Warehouse:</b> {warehouse_name}<br>"
                            f"<b>Governorate:</b> "
                            f'{warehouse["governorate_ar"]}'
                        ),
                        max_width=300
                    ),
                    icon=folium.Icon(
                        color="blue",
                        icon="home",
                        prefix="fa"
                    )
                ).add_to(warehouse_cluster)

            if selected_governorate is not None:

                selected_coords = get_coordinates_safe(
                    selected_governorate
                )

                if selected_coords is None:
                    st.warning(
                        "Could not find coordinates for the selected "
                        "governorate. Check the governorate name."
                    )

                else:
                    selected_lat, selected_lon = selected_coords

                    folium.Marker(
                        location=[selected_lat, selected_lon],
                        tooltip=f"Selected: {selected_governorate}",
                        popup=f"Starting point: {selected_governorate}",
                        icon=folium.Icon(
                            color="green",
                            icon="star",
                            prefix="fa"
                        )
                    ).add_to(warehouse_map)

                    warehouses["distance_km"] = warehouses.apply(
                        lambda row: haversine_km(
                            selected_lat,
                            selected_lon,
                            row["latitude"],
                            row["longitude"]
                        ),
                        axis=1
                    )

                    warehouses = warehouses.sort_values(
                        ["distance_km", warehouse_name_column],
                        ascending=[True, True]
                    ).reset_index(drop=True)

                    warehouses["rank"] = range(
                        1, len(warehouses) + 1
                    )

                    if show_lines:
                        for _, warehouse in warehouses.iterrows():

                            folium.PolyLine(
                                locations=[
                                    [selected_lat, selected_lon],
                                    [
                                        warehouse["latitude"],
                                        warehouse["longitude"]
                                    ]
                                ],
                                color="blue",
                                weight=2,
                                opacity=0.65,
                                dash_array="5, 8"
                            ).add_to(warehouse_map)

                    warehouse_map.location = [
                        selected_lat,
                        selected_lon
                    ]

            st_folium(
                warehouse_map,
                height=500,
                use_container_width=True,
                key="warehouse_locations_map"
            )

            if (
                selected_governorate is not None
                and "distance_km" in warehouses.columns
            ):

                st.subheader(
                    f"Warehouse Ranking from {selected_governorate}"
                )

                nearest = warehouses.iloc[0]

                col1, col2 = st.columns(2)

                col1.metric(
                    "Nearest Warehouse",
                    str(nearest[warehouse_name_column])
                )

                col2.metric(
                    "Approx. Distance",
                    f'{nearest["distance_km"]:.2f} km'
                )

                result = warehouses[
                    [
                        "rank",
                        warehouse_name_column,
                        "governorate_ar",
                        "distance_km"
                    ]
                ].rename(
                    columns={
                        "rank": "Rank",
                        warehouse_name_column: "Warehouse",
                        "governorate_ar": "Warehouse Governorate",
                        "distance_km": "Approx. Distance (km)"
                    }
                )

                result["Approx. Distance (km)"] = (
                    result["Approx. Distance (km)"].round(2)
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as e:
        st.error(f"Could not load warehouse data: {e}")

    st.divider()

    # =====================================================
    # 4. REGIONAL TRANSPORT COST ANALYSIS (2026)
    # =====================================================

    st.header("Regional Transport Cost Analysis")

    try:
        import branca.colormap as cm
        from egygeo import get_boundary, list_governorates

        inventory_path = BASE_DIR / "data" / "inventory_transfers.csv"

        inventory = read_csv_clean(inventory_path)
        branches_data = read_csv_clean(BRANCHES_PATH)
        warehouses_data = read_csv_clean(WAREHOUSES_PATH)

        # -------------------------------------------------
        # 4.1 Validate columns
        # -------------------------------------------------

        required_inventory = {
            "destination_branch_id",
            "source_warehouse_id",
            "transfer_date"
        }

        if not required_inventory.issubset(inventory.columns):
            missing = required_inventory - set(inventory.columns)
            st.error(
                "inventory_transfers.csv is missing columns: "
                + ", ".join(sorted(missing))
            )
            return

        # Support both transport_cost_EGP and transport_cost_egp
        cost_column = next(
            (
                col for col in [
                    "transport_cost_EGP",
                    "transport_cost_egp"
                ]
                if col in inventory.columns
            ),
            None
        )

        if cost_column is None:
            st.error(
                "Could not find transport cost column. "
                "Expected transport_cost_EGP or transport_cost_egp."
            )
            return

        if "branch_id" not in branches_data.columns:
            st.error("branches.csv must contain branch_id.")
            return

        if "warehouse_id" not in warehouses_data.columns:
            st.error("warehouses.csv must contain warehouse_id.")
            return

        # Find the Arabic governorate column in branches.csv
        branch_governorate_column = next(
            (
                col for col in [
                    "branch_governorate_ar",
                    "governorate_ar",
                    "branch_governorate"
                ]
                if col in branches_data.columns
            ),
            None
        )

        if branch_governorate_column is None:
            st.error(
                "branches.csv must contain an Arabic governorate column. "
                f"Available columns: {list(branches_data.columns)}"
            )
            return

        # -------------------------------------------------
        # 4.2 Prepare IDs and avoid merge suffix conflicts
        # -------------------------------------------------

        inventory["destination_branch_id"] = (
            inventory["destination_branch_id"]
            .astype("string")
            .str.strip()
        )

        inventory["source_warehouse_id"] = (
            inventory["source_warehouse_id"]
            .astype("string")
            .str.strip()
        )

        branches_data["branch_id"] = (
            branches_data["branch_id"]
            .astype("string")
            .str.strip()
        )

        warehouses_data["warehouse_id"] = (
            warehouses_data["warehouse_id"]
            .astype("string")
            .str.strip()
        )

        branches_data = branches_data.rename(
            columns={
                branch_governorate_column:
                    "_destination_branch_governorate_ar"
            }
        )

        inventory = inventory.rename(
            columns={
                cost_column: "_transport_cost_egp"
            }
        )

        # -------------------------------------------------
        # 4.3 Merge the three datasets
        # -------------------------------------------------

        inventory_branches = pd.merge(
            inventory,
            branches_data,
            left_on="destination_branch_id",
            right_on="branch_id",
            how="left",
            suffixes=("_inv", "_branch")
        )

        inventory_branch_warehouses = pd.merge(
            inventory_branches,
            warehouses_data,
            left_on="source_warehouse_id",
            right_on="warehouse_id",
            how="left",
            suffixes=("_inv_branch", "_warehouse")
        )

        transport = inventory_branch_warehouses.copy()

        # -------------------------------------------------
        # 4.4 Filter 2026 records
        # -------------------------------------------------

        transport["transfer_date"] = pd.to_datetime(
            transport["transfer_date"],
            dayfirst=True,
            errors="coerce"
        )

        transport["_transport_cost_egp"] = pd.to_numeric(
            transport["_transport_cost_egp"],
            errors="coerce"
        )

        transport_2026 = transport[
            transport["transfer_date"].dt.year == 2026
        ].copy()

        if transport_2026.empty:
            st.warning("No transfer records found for 2026.")

        else:
            latest_date = transport_2026["transfer_date"].max()

            st.caption(
                f"Data until: {latest_date:%d-%m-%Y} | "
                f"Total 2026 transactions: {len(transport_2026):,}"
            )

            # -------------------------------------------------
            # 4.5 Arabic to English governorate mapping
            # -------------------------------------------------

            arabic_to_english = {
                "القاهرة": "Cairo",
                "الجيزة": "Giza",
                "الإسكندرية": "Alexandria",
                "القليوبية": "Qalyubia",
                "البحيرة": "Beheira",
                "مطروح": "Matrouh",
                "دمياط": "Damietta",
                "الدقهلية": "Dakahlia",
                "كفر الشيخ": "Kafr El Sheikh",
                "الغربية": "Gharbia",
                "المنوفية": "Monufia",
                "الشرقية": "Sharqia",
                "الإسماعيلية": "Ismailia",
                "بورسعيد": "Port Said",
                "السويس": "Suez",
                "شمال سيناء": "North Sinai",
                "جنوب سيناء": "South Sinai",
                "بني سويف": "Beni Suef",
                "الفيوم": "Fayoum",
                "المنيا": "Minya",
                "أسيوط": "Asyut",
                "سوهاج": "Sohag",
                "قنا": "Qena",
                "الأقصر": "Luxor",
                "أسوان": "Aswan",
                "البحر الأحمر": "Red Sea",
                "الوادي الجديد": "New Valley"
            }

            # -------------------------------------------------
            # 4.6 Canonical governorate names
            # -------------------------------------------------

            def get_canonical_from_english(name):

                if pd.isna(name):
                    return None

                feature = get_boundary(
                    str(name).strip(),
                    as_feature=True
                )

                if feature is None:
                    return str(name).strip()

                return feature["properties"]["Subdivision_en"]

            def get_canonical_name(name):

                if pd.isna(name):
                    return None

                normalized_name = str(name).strip()

                english_name = arabic_to_english.get(
                    normalized_name
                )

                if english_name is None:
                    return None

                return get_canonical_from_english(english_name)

            governorate_column = (
                "_destination_branch_governorate_ar"
            )

            transport_2026["branch_governorate_key"] = (
                transport_2026[governorate_column]
                .apply(get_canonical_name)
            )

            # -------------------------------------------------
            # 4.7 Define regions
            # -------------------------------------------------

            region_members = {
                "Cairo_Giza": [
                    "Cairo",
                    "Giza"
                ],

                "Delta": [
                    "Dakahlia",
                    "Sharqia",
                    "Gharbia",
                    "Monufia",
                    "Qalyubia"
                ],

                "Alex_Beheira": [
                    "Alexandria",
                    "Beheira"
                ],

                "Channel Cities": [
                    "Port Said",
                    "Suez",
                    "Ismailia"
                ],

                "Coastal Arm": [
                    "Matrouh",
                    "Kafr El Sheikh",
                    "Damietta"
                ],

                "Sinai": [
                    "North Sinai",
                    "South Sinai"
                ],

                "Red Sea": [
                    "Red Sea"
                ],

                "Upper Egypt": [
                    "Beni Suef",
                    "Fayoum",
                    "Minya",
                    "Asyut",
                    "Sohag",
                    "Qena",
                    "Luxor",
                    "Aswan",
                    "New Valley"
                ]
            }

            # IMPORTANT:
            # Region members are already English.
            # Therefore use get_canonical_from_english(),
            # NOT get_canonical_name().

            governorate_to_region = {}

            for region_name, governorates in region_members.items():

                for governorate in governorates:

                    canonical = get_canonical_from_english(
                        governorate
                    )

                    if canonical is not None:
                        governorate_to_region[canonical] = region_name

            # -------------------------------------------------
            # 4.8 Assign regions to transactions
            # -------------------------------------------------

            transport_2026["region"] = (
                transport_2026["branch_governorate_key"]
                .map(governorate_to_region)
            )

            unassigned = sorted(
                transport_2026.loc[
                    transport_2026["region"].isna(),
                    governorate_column
                ].dropna().astype(str).unique()
            )

            if unassigned:
                st.warning(
                    "Unassigned governorates: "
                    + ", ".join(unassigned)
                )

            # -------------------------------------------------
            # 4.9 Filter valid transactions
            # -------------------------------------------------

            analysis_data = transport_2026[
                transport_2026["region"].notna()
                & transport_2026["_transport_cost_egp"].notna()
            ].copy()

            if analysis_data.empty:

                st.warning(
                    "No assigned 2026 transactions have valid "
                    "transport costs."
                )

            else:

                # -------------------------------------------------
                # 4.10 Aggregate transport cost by region
                # -------------------------------------------------

                region_data = (
                    analysis_data.groupby("region")
                    .agg(
                        total_transport_cost=(
                            "_transport_cost_egp",
                            "sum"
                        ),
                        transaction_count=(
                            "_transport_cost_egp",
                            "count"
                        )
                    )
                    .reset_index()
                )

                region_data["average_cost_per_transaction"] = (
                    region_data["total_transport_cost"]
                    / region_data["transaction_count"]
                )

                region_data = region_data.sort_values(
                    "average_cost_per_transaction",
                    ascending=False
                )

                st.subheader("Average Transport Cost by Region")

                # -------------------------------------------------
                # 4.11 Plotly chart
                # -------------------------------------------------

                fig = px.bar(
                    region_data,
                    x="region",
                    y="average_cost_per_transaction",
                    text="average_cost_per_transaction",
                    hover_data={
                        "total_transport_cost": ":,.2f",
                        "transaction_count": ":,",
                        "average_cost_per_transaction": ":,.2f"
                    },
                    title=(
                        "Average Transport Cost per Transaction by Region (2026)"
                    ),
                    labels={
                        "region": "Region",
                        "average_cost_per_transaction": (
                            "Average Cost per Transaction (EGP)"
                        ),
                        "total_transport_cost": (
                            "Total Transport Cost (EGP)"
                        ),
                        "transaction_count": "Transactions"
                    },
                    color="average_cost_per_transaction",
                    color_continuous_scale="YlOrRd"
                )

                fig.update_traces(
                    texttemplate="%{text:,.2f} EGP",
                    textposition="outside"
                )

                fig.update_layout(
                    xaxis_title="Region",
                    yaxis_title="Average Cost per Transaction (EGP)",
                    coloraxis_showscale=False,
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    region_data,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------------------
                # 4.12 Prepare governorate GeoJSON
                # -------------------------------------------------

                region_cost_lookup = (
                    region_data.set_index("region")
                    ["average_cost_per_transaction"]
                    .to_dict()
                )

                features = []

                for governorate in list_governorates():

                    feature = get_boundary(
                        governorate["english"],
                        as_feature=True
                    )

                    if feature is None:
                        continue

                    governorate_name = (
                        feature["properties"]["Subdivision_en"]
                    )

                    region_name = governorate_to_region.get(
                        governorate_name
                    )

                    average_cost = (
                        region_cost_lookup.get(region_name)
                        if region_name is not None
                        else None
                    )

                    feature["properties"]["region"] = (
                        region_name or "Unassigned"
                    )

                    feature["properties"]["average_cost"] = (
                        average_cost
                    )

                    features.append(feature)

                egypt_geojson = {
                    "type": "FeatureCollection",
                    "features": features
                }

                # -------------------------------------------------
                # 4.13 Create color scale
                # -------------------------------------------------

                min_cost = float(
                    region_data[
                        "average_cost_per_transaction"
                    ].min()
                )

                max_cost = float(
                    region_data[
                        "average_cost_per_transaction"
                    ].max()
                )

                if min_cost == max_cost:
                    max_cost = min_cost + 1

                cost_colormap = cm.LinearColormap(
                    colors=[
                        "#FFF0F0",
                        "#FFB3B3",
                        "#FF6666",
                        "#CC0000",
                        "#800000"
                    ],
                    vmin=min_cost,
                    vmax=max_cost,
                    caption=(
                        "Average Transport Cost per Transaction (EGP)"
                    )
                )

                # -------------------------------------------------
                # 4.14 Map styling
                # -------------------------------------------------

                def transport_cost_style(feature):

                    average_cost = (
                        feature["properties"]["average_cost"]
                    )

                    if average_cost is None:
                        fill_color = "#D3D3D3"
                    else:
                        fill_color = cost_colormap(average_cost)

                    return {
                        "fillColor": fill_color,
                        "color": "#FFFFFF",
                        "weight": 1.2,
                        "fillOpacity": 0.8
                    }

                # -------------------------------------------------
                # 4.15 Create map
                # -------------------------------------------------

                transport_map = folium.Map(
                    location=[26.8, 30.8],
                    zoom_start=6,
                    tiles="OpenStreetMap"
                )

                folium.GeoJson(
                    egypt_geojson,
                    style_function=transport_cost_style,
                    tooltip=folium.GeoJsonTooltip(
                        fields=[
                            "Subdivision_en",
                            "region",
                            "average_cost"
                        ],
                        aliases=[
                            "Governorate:",
                            "Region:",
                            "Average Cost per Transaction (EGP):"
                        ],
                        localize=True,
                        sticky=True
                    )
                ).add_to(transport_map)

                cost_colormap.add_to(transport_map)

                st.subheader(
                    "Transport Cost Map by Governorate"
                )

                st_folium(
                    transport_map,
                    height=550,
                    use_container_width=True,
                    key="regional_transport_cost_map"
                )

    except FileNotFoundError as e:
        st.error(f"Could not load transport cost data: {e}")

    except Exception as e:
        st.error(
            f"Could not create regional transport cost analysis: {e}"
        )

    st.divider()

    # =====================================================
    # 5. BUSINESS RECOMMENDATIONS
    # =====================================================

    st.header("Business Recommendations")

    st.markdown(
        """
        ### 1. Manufacturing Strategy

        Our manufacturing facilities are concentrated in the Nile Delta,
        supporting our cotton-based textile operations and benefiting from
        the region's established agricultural and industrial ecosystem.

        ### 2. Distribution Strategy

        Our branch network is concentrated in areas with strong sales
        activity, helping us serve existing demand. However, our presence
        in Upper Egypt remains limited, creating an opportunity to expand
        our geographic coverage.

        ### 3. Recommended Expansion: Minya

        We recommend considering Minya as a strategic distribution hub
        for Upper Egypt. Its location could improve access to nearby
        governorates, expand our customer reach, and support more efficient
        distribution.

        Before implementation, validate this recommendation using
        branch-level sales, customer demand, transportation costs,
        forecast demand, and the availability of suitable locations.
        """
    )

    st.caption(
        "Distances are approximate straight-line distances, not road "
        "travel distances. If only governorate names are available, "
        "locations represent approximate governorate centers."
    )

def customer_segmentation_page():

    import streamlit as st

    import pandas as pd

    import joblib

    import plotly.express as px

    from rfm_utils import calculate_rfm

    st.title("Customer Segmentation")

    st.markdown(

        "Upload new customer transactions to predict their RFM segments."

    )

    # ------------------------------------------

    # Segment Mapping

    # ------------------------------------------

    segment_names = {

        0: 'Lost / Inactive Customers',

        1: 'High-Value Loyal Customers',

        2: 'Regular Customers',

        3: 'VIP / Premium Customers',

        4: 'Potential Loyal Customers'

    }

    # ------------------------------------------

    # Load Trained Pipeline

    # ------------------------------------------

    @st.cache_resource

    def load_pipeline():

        return joblib.load('rfm_segmentation_pipeline.joblib')

    # ------------------------------------------

    # Upload CSV

    # ------------------------------------------

    uploaded_file = st.file_uploader(

        "Upload New Transactions CSV",

        type=["csv"],

        key="segmentation_upload"

    )

    if uploaded_file is None:

        st.info("Upload a transactions CSV file to start.")

        return

    new_transactions = pd.read_csv(uploaded_file)

    st.write(f"**Uploaded Transactions:** {len(new_transactions):,}")

    with st.expander("Preview Uploaded Data"):

        st.dataframe(new_transactions.head(), use_container_width=True)

    # ------------------------------------------

    # Run Segmentation

    # ------------------------------------------

    if st.button("Run Customer Segmentation", type="primary"):

        try:

            with st.spinner("Processing customer transactions..."):

                loaded_pipeline = load_pipeline()

                # Calculate RFM

                new_rfm = calculate_rfm(new_transactions)

                # Predict clusters

                predictions = loaded_pipeline.predict(new_rfm)

                # Generate results

                results = new_rfm.copy()

                results['cluster'] = predictions

                results['segment'] = results['cluster'].map(

                    segment_names

                )

                # Segment distribution

                segment_counts = (

                    results['segment']

                    .value_counts()

                    .reindex(segment_names.values(), fill_value=0)

                    .rename_axis('Segment')

                    .reset_index(name='Customers')

                )

                segment_counts['Percentage'] = (

                    segment_counts['Customers'] /

                    segment_counts['Customers'].sum()

                ) * 100

                # Save results in session state

                st.session_state['segmentation_results'] = results

                st.session_state['segment_counts'] = segment_counts

        except Exception as e:

            st.error(f"Segmentation Error: {e}")

    # ------------------------------------------

    # Display Results

    # ------------------------------------------

    if 'segmentation_results' in st.session_state:

        results = st.session_state['segmentation_results']

        segment_counts = st.session_state['segment_counts']

        st.divider()

        st.subheader("Customer Segment Distribution")

        col1, col2 = st.columns([2, 1])

        with col1:

            fig = px.pie(

                segment_counts,

                names='Segment',

                values='Customers',

                hole=0.4,

                title='Customer Distribution by Segment',

                color_discrete_sequence=px.colors.qualitative.Set2

            )

            fig.update_traces(

                textposition='inside',

                textinfo='percent+label'

            )

            fig.update_layout(height=500)

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        with col2:

            st.markdown("### Segment Summary")

            st.dataframe(

                segment_counts.style.format({

                    'Customers': '{:,.0f}',

                    'Percentage': '{:.2f}%'

                }),

                use_container_width=True,

                hide_index=True

            )

            st.metric(

                "Total Customers",

                f"{len(results):,}"

            )

            st.metric(

                "Identified Segments",

                results['segment'].nunique()

            )

        # --------------------------------------

        # Results Table

        # --------------------------------------

        st.subheader("Customer Segmentation Results")

        st.dataframe(

            results,

            use_container_width=True,

            hide_index=True

        )

        # --------------------------------------

        # Download Results

        # --------------------------------------

        csv = results.to_csv(index=True).encode('utf-8-sig')

        st.download_button(

            "Download Segmentation Results",

            data=csv,

            file_name="customer_segmentation_results.csv",

            mime="text/csv",

            type="primary"

        )




# ============================================================
# INVENTORY & WAREHOUSE INTELLIGENCE
# ============================================================




def inventory_warehouse_intelligence_page(data_dir="data"):

    from pathlib import Path
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px

    # ========================================================
    # 1. PAGE CONFIGURATION & DATA LOADING
    # ========================================================

    st.title("Inventory & Warehouse Intelligence")

    st.caption(
        "Monitor inventory health, warehouse capacity, "
        "stock movements, procurement and replenishment."
    )

    data_dir = Path(data_dir)

    required_files = {
        "stock": "monthly_stock.csv",
        "warehouses": "warehouses.csv",
        "transfers": "inventory_transfers.csv",
        "transfer_items": "inventory_transfer_items.csv",
        "purchase_orders": "purchase_orders.csv",
        "purchase_items": "purchase_order_items.csv",
        "suppliers": "suppliers.csv",
        "products": "products_catalog.csv",
        "variants": "sku_variants.csv",
        "sales_items": "sales_order_items.csv",
        "returns": "sales_returns.csv"
    }

    try:
        data = {
            key: pd.read_csv(data_dir / filename)
            for key, filename in required_files.items()
        }

    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}")
        return

    stock = data["stock"].copy()
    warehouses = data["warehouses"].copy()
    transfers = data["transfers"].copy()
    transfer_items = data["transfer_items"].copy()
    purchase_orders = data["purchase_orders"].copy()
    purchase_items = data["purchase_items"].copy()
    suppliers = data["suppliers"].copy()
    products = data["products"].copy()
    variants = data["variants"].copy()
    sales_items = data["sales_items"].copy()
    returns = data["returns"].copy()

    # ========================================================
    # 2. DATA CLEANING
    # ========================================================

    data_frames = {
        "stock": stock,
        "warehouses": warehouses,
        "transfers": transfers,
        "transfer_items": transfer_items,
        "purchase_orders": purchase_orders,
        "purchase_items": purchase_items,
        "suppliers": suppliers,
        "products": products,
        "variants": variants,
        "sales_items": sales_items,
        "returns": returns
    }

    numeric_columns = {
        "stock": [
            "closing_units",
            "closing_stock_value_EGP",
            "stockout_risk_score"
        ],
        "warehouses": ["capacity_units"],
        "transfers": ["transport_cost_EGP"],
        "transfer_items": ["quantity_units"],
        "purchase_orders": ["po_subtotal_EGP"],
        "purchase_items": [
            "quantity",
            "unit_cost_EGP",
            "line_cost_EGP"
        ],
        "sales_items": [
            "quantity",
            "net_line_EGP",
            "margin_EGP"
        ],
        "returns": [
            "return_quantity",
            "refund_amount_EGP"
        ]
    }

    for key, columns in numeric_columns.items():

        for column in columns:

            if column in data_frames[key].columns:

                data_frames[key][column] = pd.to_numeric(
                    data_frames[key][column],
                    errors="coerce"
                ).fillna(0)

    # Normalize IDs.
    id_columns = {
        "stock": ["stock_location_id"],
        "warehouses": ["warehouse_id"],
        "transfers": [
            "transfer_id",
            "source_warehouse_id",
            "destination_branch_id"
        ],
        "transfer_items": ["transfer_id", "sku_id"],
        "purchase_orders": [
            "po_id",
            "supplier_id",
            "receiving_warehouse_id"
        ],
        "purchase_items": ["po_id", "sku_id"],
        "suppliers": ["supplier_id"],
        "products": ["product_id"],
        "variants": ["sku_id", "product_id"],
        "sales_items": [
            "sales_line_id",
            "sales_order_id",
            "product_id",
            "sku_id"
        ],
        "returns": ["sales_line_id"]
    }

    for key, columns in id_columns.items():

        for column in columns:

            if column in data_frames[key].columns:

                data_frames[key][column] = (
                    data_frames[key][column]
                    .astype("string")
                    .str.strip()
                )

    # ========================================================
    # 3. DATE PREPARATION
    # ========================================================

    stock["year_month"] = pd.to_datetime(
        stock["year_month"].astype(str),
        errors="coerce"
    ).dt.to_period("M")

    transfers["transfer_date"] = pd.to_datetime(
        transfers["transfer_date"],
        errors="coerce"
    )

    purchase_orders["po_date"] = pd.to_datetime(
        purchase_orders["po_date"],
        errors="coerce"
    )

    purchase_items["delivery_date"] = pd.to_datetime(
        purchase_items["delivery_date"],
        errors="coerce"
    )

    sales_items["order_date"] = pd.to_datetime(
        sales_items["order_date"],
        errors="coerce"
    )

    # ========================================================
    # 4. INVENTORY SNAPSHOT
    # ========================================================

    st.subheader("Inventory Overview")

    if stock["year_month"].dropna().empty:

        st.warning("No valid inventory snapshots found.")
        return

    latest_month = stock["year_month"].max()

    latest_stock = stock[
        stock["year_month"] == latest_month
    ].copy()

    latest_stock["stock_location_type"] = (
        latest_stock["stock_location_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    total_stock_units = latest_stock[
        "closing_units"
    ].sum()

    total_stock_value = latest_stock[
        "closing_stock_value_EGP"
    ].sum()

    average_risk = (
        latest_stock["stockout_risk_score"].mean()
        if not latest_stock.empty else 0
    )

    category_count = latest_stock[
        "product_category"
    ].nunique()

    # Previous available snapshot.
    previous_months = (
        stock.loc[
            stock["year_month"] < latest_month,
            "year_month"
        ]
        .dropna()
        .unique()
    )

    if len(previous_months):

        previous_month = max(previous_months)

        previous_stock = stock[
            stock["year_month"] == previous_month
        ]

        previous_value = previous_stock[
            "closing_stock_value_EGP"
        ].sum()

        value_change = (
            (total_stock_value - previous_value)
            / abs(previous_value) * 100
            if previous_value != 0 else None
        )

    else:

        value_change = None

    # ========================================================
    # 5. WAREHOUSE CAPACITY
    # ========================================================

    warehouse_stock = latest_stock[
        latest_stock["stock_location_type"].isin(
            ["warehouse", "warehouses", "wh"]
        )
    ].copy()

    warehouse_summary = (
        warehouse_stock.groupby(
            "stock_location_id",
            as_index=False
        )
        .agg(
            Stock_Units=("closing_units", "sum"),
            Stock_Value=(
                "closing_stock_value_EGP", "sum"
            )
        )
        .rename(
            columns={
                "stock_location_id": "warehouse_id"
            }
        )
    )

    warehouse_summary = warehouses.merge(
        warehouse_summary,
        on="warehouse_id",
        how="left"
    )

    warehouse_summary[
        ["Stock_Units", "Stock_Value"]
    ] = warehouse_summary[
        ["Stock_Units", "Stock_Value"]
    ].fillna(0)

    warehouse_summary["Utilization %"] = np.where(
        warehouse_summary["capacity_units"] > 0,
        warehouse_summary["Stock_Units"]
        / warehouse_summary["capacity_units"] * 100,
        np.nan
    )

    warehouse_summary["Available Capacity"] = (
        warehouse_summary["capacity_units"]
        - warehouse_summary["Stock_Units"]
    ).clip(lower=0)

    total_capacity = warehouse_summary[
        "capacity_units"
    ].sum()

    total_warehouse_units = warehouse_summary[
        "Stock_Units"
    ].sum()

    utilization = (
        total_warehouse_units / total_capacity * 100
        if total_capacity > 0 else 0
    )

    # ========================================================
    # 6. KPI CARDS
    # ========================================================

    st.caption(
        f"Latest inventory snapshot: {latest_month}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Stock Value",
        f"{total_stock_value:,.0f} EGP",
        (
            f"{value_change:+.2f}%"
            if value_change is not None else None
        )
    )

    c2.metric(
        "Total Stock Units",
        f"{total_stock_units:,.0f}"
    )

    c3.metric(
        "Average Stockout Risk",
        f"{average_risk:.2f}"
    )

    c4.metric(
        "Warehouse Utilization",
        f"{utilization:.1f}%"
    )

    st.caption(
        f"Tracked Categories: {category_count:,} | "
        f"Warehouses: {len(warehouses):,}"
    )

    st.divider()

    # ========================================================
    # 7. STOCK HEALTH & RISK
    # ========================================================

    st.subheader("Stock Health & Risk Analysis")

    category_risk = (
        latest_stock.groupby("product_category")
        .agg(
            Stock_Units=("closing_units", "sum"),
            Stock_Value=(
                "closing_stock_value_EGP", "sum"
            ),
            Risk=("stockout_risk_score", "mean")
        )
        .reset_index()
        .sort_values("Risk", ascending=False)
    )

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:

        fig_risk = px.bar(
            category_risk.head(10).sort_values("Risk"),
            x="Risk",
            y="product_category",
            orientation="h",
            color="Risk",
            title="Top Categories by Stockout Risk",
            text="Risk",
            color_continuous_scale="Reds"
        )

        fig_risk.update_layout(
            height=420,
            yaxis_title="",
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True,
            key="inventory_stock_risk"
        )

    with risk_col2:

        category_value = (
            latest_stock.groupby("product_category")
            ["closing_stock_value_EGP"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_value = px.pie(
            category_value.head(6),
            names="product_category",
            values="closing_stock_value_EGP",
            hole=0.5,
            title="Inventory Value by Category"
        )

        fig_value.update_layout(height=420)

        st.plotly_chart(
            fig_value,
            use_container_width=True,
            key="inventory_category_value"
        )

    fig_risk_scatter = px.scatter(
        category_risk,
        x="Stock_Units",
        y="Risk",
        size="Stock_Value",
        color="product_category",
        hover_name="product_category",
        title="Stock Quantity vs Stockout Risk",
        labels={
            "Stock_Units": "Closing Units",
            "Risk": "Average Risk Score"
        }
    )

    fig_risk_scatter.update_layout(height=430)

    st.plotly_chart(
        fig_risk_scatter,
        use_container_width=True,
        key="inventory_risk_scatter"
    )

    st.divider()

    # ========================================================
    # 8. STOCK COVERAGE & REPLENISHMENT
    # ========================================================

    st.subheader("Stock Coverage & Replenishment")

    returns_by_line = (
        returns.groupby("sales_line_id", as_index=False)
        ["return_quantity"].sum()
    )

    sales_items = sales_items.merge(
        returns_by_line,
        on="sales_line_id",
        how="left"
    )

    sales_items["return_quantity"] = (
        sales_items["return_quantity"].fillna(0)
    )

    sales_items["sold_quantity"] = (
        sales_items["quantity"]
        - sales_items["return_quantity"]
    ).clip(lower=0)

    product_categories = products[
        ["product_id", "category_en"]
    ].drop_duplicates("product_id")

    sales_items = sales_items.merge(
        product_categories,
        on="product_id",
        how="left"
    )

    sales_items["category_en"] = (
        sales_items["category_en"].fillna("Other")
    )

    # Trailing 3 calendar months.
    today = pd.Timestamp.today().normalize()

    demand_start = (
        today.to_period("M") - 2
    ).start_time

    demand_end = today + pd.Timedelta(days=1)

    recent_sales = sales_items[
        (sales_items["order_date"] >= demand_start)
        & (sales_items["order_date"] < demand_end)
    ].copy()

    monthly_demand = (
        recent_sales.groupby("category_en")
        ["sold_quantity"].sum() / 3
    )

    category_stock = (
        latest_stock.groupby("product_category")
        ["closing_units"].sum()
    )

    coverage_df = pd.DataFrame({
        "Category": category_stock.index,
        "Current Stock": category_stock.values
    })

    coverage_df["Average Monthly Demand"] = (
        coverage_df["Category"]
        .map(monthly_demand)
        .fillna(0)
    )

    coverage_df["Stock Coverage (Months)"] = np.where(
        coverage_df["Average Monthly Demand"] > 0,
        coverage_df["Current Stock"]
        / coverage_df["Average Monthly Demand"],
        np.nan
    )

    def coverage_status(row):

        coverage = row["Stock Coverage (Months)"]

        if pd.isna(coverage):
            return "No Recent Demand"

        if coverage < 1:
            return "Critical"

        if coverage < 2:
            return "Low"

        if coverage > 6:
            return "Overstock"

        return "Healthy"

    coverage_df["Status"] = coverage_df.apply(
        coverage_status,
        axis=1
    )

    coverage_df = coverage_df.sort_values(
        "Stock Coverage (Months)",
        na_position="first"
    )

    coverage_col1, coverage_col2 = st.columns(2)

    with coverage_col1:

        status_counts = (
            coverage_df["Status"]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "Status",
            "Categories"
        ]

        fig_coverage_status = px.pie(
            status_counts,
            names="Status",
            values="Categories",
            hole=0.5,
            title="Category Stock Health"
        )

        st.plotly_chart(
            fig_coverage_status,
            use_container_width=True,
            key="inventory_coverage_status"
        )

    with coverage_col2:

        fig_coverage = px.bar(
            coverage_df,
            x="Stock Coverage (Months)",
            y="Category",
            orientation="h",
            color="Status",
            title="Stock Coverage by Category",
            labels={
                "Stock Coverage (Months)": "Months of Coverage"
            }
        )

        fig_coverage.update_layout(height=420)

        st.plotly_chart(
            fig_coverage,
            use_container_width=True,
            key="inventory_coverage_chart"
        )

    st.dataframe(
        coverage_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Coverage uses the latest stock snapshot and average "
        "net demand over the latest three calendar months. "
        "Categories without recent demand are not assigned "
        "an infinite coverage value."
    )

    st.divider()

    # ========================================================
    # 9. WAREHOUSE PERFORMANCE
    # ========================================================

    st.subheader("Warehouse Performance")

    if not warehouse_summary.empty:

        warehouse_col1, warehouse_col2 = st.columns(2)

        with warehouse_col1:

            fig_utilization = px.bar(
                warehouse_summary.sort_values(
                    "Utilization %",
                    ascending=True
                ),
                x="Utilization %",
                y="warehouse_label",
                orientation="h",
                color="Utilization %",
                title="Warehouse Capacity Utilization",
                text="Utilization %",
                color_continuous_scale="Blues"
            )

            fig_utilization.add_vline(
                x=100,
                line_dash="dash",
                annotation_text="100% Capacity"
            )

            fig_utilization.update_layout(
                height=430,
                yaxis_title="",
                coloraxis_showscale=False
            )

            st.plotly_chart(
                fig_utilization,
                use_container_width=True,
                key="warehouse_utilization"
            )

        with warehouse_col2:

            fig_warehouse_value = px.bar(
                warehouse_summary.sort_values(
                    "Stock_Value",
                    ascending=True
                ),
                x="Stock_Value",
                y="warehouse_label",
                orientation="h",
                title="Stock Value by Warehouse",
                text="Stock_Value"
            )

            fig_warehouse_value.update_layout(
                height=430,
                yaxis_title=""
            )

            st.plotly_chart(
                fig_warehouse_value,
                use_container_width=True,
                key="warehouse_stock_value"
            )

        warehouse_display = warehouse_summary[
            [
                "warehouse_label",
                "governorate_ar",
                "capacity_units",
                "Stock_Units",
                "Available Capacity",
                "Utilization %",
                "Stock_Value"
            ]
        ].copy()

        warehouse_display.columns = [
            "Warehouse",
            "Governorate",
            "Capacity",
            "Stock Units",
            "Available Capacity",
            "Utilization %",
            "Stock Value (EGP)"
        ]

        st.dataframe(
            warehouse_display,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No warehouse stock records matched the warehouse "
            "IDs and stock location types."
        )

    st.divider()

    # ========================================================
    # 10. INVENTORY MOVEMENT & TRANSFERS
    # ========================================================

    st.subheader("Inventory Movement & Transfers")

    transfer_quantities = (
        transfer_items.groupby("transfer_id")
        ["quantity_units"]
        .sum()
        .rename("Transferred Units")
        .reset_index()
    )

    transfer_analysis = transfers.merge(
        transfer_quantities,
        on="transfer_id",
        how="left"
    )

    transfer_analysis["Transferred Units"] = (
        transfer_analysis["Transferred Units"].fillna(0)
    )

    total_transfers = (
        transfer_analysis["transfer_id"].nunique()
    )

    total_transferred_units = (
        transfer_analysis["Transferred Units"].sum()
    )

    total_transport_cost = (
        transfer_analysis["transport_cost_EGP"].sum()
    )

    average_cost_per_unit = (
        total_transport_cost / total_transferred_units
        if total_transferred_units > 0 else 0
    )

    transfer_k1, transfer_k2, transfer_k3, transfer_k4 = (
        st.columns(4)
    )

    transfer_k1.metric(
        "Total Transfers",
        f"{total_transfers:,}"
    )

    transfer_k2.metric(
        "Transferred Units",
        f"{total_transferred_units:,.0f}"
    )

    transfer_k3.metric(
        "Transportation Cost",
        f"{total_transport_cost:,.0f} EGP"
    )

    transfer_k4.metric(
        "Cost per Unit",
        f"{average_cost_per_unit:,.2f} EGP"
    )

    transfer_col1, transfer_col2 = st.columns(2)

    with transfer_col1:

        status_data = (
            transfer_analysis["transfer_status"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        status_data.columns = [
            "Status",
            "Transfers"
        ]

        fig_transfer_status = px.pie(
            status_data,
            names="Status",
            values="Transfers",
            hole=0.5,
            title="Transfer Status Distribution"
        )

        st.plotly_chart(
            fig_transfer_status,
            use_container_width=True,
            key="transfer_status_distribution"
        )

    with transfer_col2:

        monthly_transfers = (
            transfer_analysis.dropna(
                subset=["transfer_date"]
            )
            .assign(
                Month=lambda x: (
                    x["transfer_date"]
                    .dt.to_period("M")
                    .astype(str)
                )
            )
            .groupby("Month")
            .agg(
                Transfers=("transfer_id", "nunique"),
                Units=("Transferred Units", "sum"),
                Cost=("transport_cost_EGP", "sum")
            )
            .reset_index()
        )

        fig_transfer_trend = px.line(
            monthly_transfers,
            x="Month",
            y=["Transfers", "Units"],
            markers=True,
            title="Monthly Transfer Activity"
        )

        st.plotly_chart(
            fig_transfer_trend,
            use_container_width=True,
            key="monthly_transfer_activity"
        )

    destination_summary = (
        transfer_analysis.groupby("destination_branch_id")
        .agg(
            Transfers=("transfer_id", "nunique"),
            Units=("Transferred Units", "sum"),
            Transport_Cost=("transport_cost_EGP", "sum")
        )
        .reset_index()
        .sort_values("Units", ascending=False)
        .head(10)
    )

    fig_destinations = px.bar(
        destination_summary,
        x="destination_branch_id",
        y="Units",
        color="Transport_Cost",
        title="Top 10 Transfer Destinations",
        labels={
            "destination_branch_id": "Destination Branch",
            "Units": "Transferred Units",
            "Transport_Cost": "Transport Cost (EGP)"
        }
    )

    st.plotly_chart(
        fig_destinations,
        use_container_width=True,
        key="transfer_destinations"
    )

    st.divider()

    # ========================================================
    # 11. PROCUREMENT INTELLIGENCE
    # ========================================================

    st.subheader("Procurement Intelligence")

    purchase_order_analysis = purchase_orders.merge(
        suppliers[
            [
                "supplier_id",
                "supplier_name_ar",
                "material_type"
            ]
        ].drop_duplicates("supplier_id"),
        on="supplier_id",
        how="left"
    )

    total_purchase_orders = (
        purchase_orders["po_id"].nunique()
    )

    total_procurement_value = (
        purchase_orders["po_subtotal_EGP"].sum()
    )

    total_purchase_quantity = (
        purchase_items["quantity"].sum()
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Purchase Orders",
        f"{total_purchase_orders:,}"
    )

    p2.metric(
        "Procurement Value",
        f"{total_procurement_value:,.0f} EGP"
    )

    p3.metric(
        "Purchased Units",
        f"{total_purchase_quantity:,.0f}"
    )

    procurement_col1, procurement_col2 = st.columns(2)

    with procurement_col1:

        po_status = (
            purchase_orders["po_status"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        po_status.columns = [
            "Status",
            "Orders"
        ]

        fig_po_status = px.pie(
            po_status,
            names="Status",
            values="Orders",
            hole=0.5,
            title="Purchase Order Status"
        )

        st.plotly_chart(
            fig_po_status,
            use_container_width=True,
            key="purchase_order_status"
        )

    with procurement_col2:

        supplier_spending = (
            purchase_order_analysis.groupby(
                "supplier_name_ar"
            )["po_subtotal_EGP"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig_supplier = px.bar(
            supplier_spending.sort_values(
                "po_subtotal_EGP"
            ),
            x="po_subtotal_EGP",
            y="supplier_name_ar",
            orientation="h",
            title="Top Suppliers by Procurement Value"
        )

        fig_supplier.update_layout(
            height=420,
            yaxis_title=""
        )

        st.plotly_chart(
            fig_supplier,
            use_container_width=True,
            key="supplier_procurement_value"
        )

    # ========================================================
    # 12. PROCUREMENT LEAD TIME
    # ========================================================

    purchase_lead = purchase_items.merge(
        purchase_orders[
            ["po_id", "po_date", "supplier_id"]
        ].drop_duplicates("po_id"),
        on="po_id",
        how="left"
    )

    purchase_lead["Lead Time Days"] = (
        purchase_lead["delivery_date"]
        - purchase_lead["po_date"]
    ).dt.days

    valid_lead = purchase_lead[
        purchase_lead["Lead Time Days"].ge(0)
    ]

    if not valid_lead.empty:

        supplier_lead = (
            valid_lead.groupby("supplier_id")
            ["Lead Time Days"]
            .mean()
            .reset_index()
        )

        supplier_lead = supplier_lead.merge(
            suppliers[
                [
                    "supplier_id",
                    "supplier_name_ar"
                ]
            ].drop_duplicates("supplier_id"),
            on="supplier_id",
            how="left"
        )

        fig_lead = px.bar(
            supplier_lead.sort_values("Lead Time Days"),
            x="Lead Time Days",
            y="supplier_name_ar",
            orientation="h",
            title="Average Supplier Delivery Lead Time",
            text="Lead Time Days"
        )

        fig_lead.update_layout(
            height=400,
            yaxis_title=""
        )

        st.plotly_chart(
            fig_lead,
            use_container_width=True,
            key="supplier_delivery_lead_time"
        )

    st.divider()

    # ========================================================
    # 13. INVENTORY OPTIMIZATION & RECOMMENDATIONS
    # ========================================================

    st.subheader("Inventory Optimization & Recommendations")

    recommendations = []

    # Category stock coverage recommendations.
    for _, row in coverage_df.iterrows():

        category = row["Category"]
        status = row["Status"]

        if status == "Critical":

            recommendations.append({
                "Priority": "High",
                "Area": category,
                "Issue": "Critical Stock Coverage",
                "Recommendation": (
                    "Prioritize replenishment and review "
                    "available stock across locations."
                )
            })

        elif status == "Low":

            recommendations.append({
                "Priority": "Medium",
                "Area": category,
                "Issue": "Low Stock Coverage",
                "Recommendation": (
                    "Review upcoming demand and purchase "
                    "order delivery schedules."
                )
            })

        elif status == "Overstock":

            recommendations.append({
                "Priority": "Medium",
                "Area": category,
                "Issue": "Potential Overstock",
                "Recommendation": (
                    "Review future purchasing quantities and "
                    "consider redistributing existing stock."
                )
            })

    # Warehouse capacity recommendations.
    for _, row in warehouse_summary.iterrows():

        warehouse_name = row["warehouse_label"]
        warehouse_util = row["Utilization %"]

        if (
            pd.notna(warehouse_util)
            and warehouse_util >= 90
        ):

            recommendations.append({
                "Priority": "High",
                "Area": warehouse_name,
                "Issue": "High Warehouse Utilization",
                "Recommendation": (
                    "Review warehouse capacity and "
                    "redistribution opportunities."
                )
            })

    if recommendations:

        recommendations_df = pd.DataFrame(
            recommendations
        )

        recommendations_df["_priority_order"] = (
            recommendations_df["Priority"]
            .map({
                "High": 0,
                "Medium": 1
            })
        )

        recommendations_df = (
            recommendations_df
            .sort_values("_priority_order")
            .drop(columns="_priority_order")
        )

        st.dataframe(
            recommendations_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No inventory recommendations were triggered "
            "by the current rules."
        )

    st.divider()

    # ========================================================
    # 14. RAW INVENTORY DETAILS
    # ========================================================

    with st.expander("View Latest Inventory Snapshot"):

        st.dataframe(
            latest_stock.sort_values(
                "closing_stock_value_EGP",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("View Transfer Details"):

        st.dataframe(
            transfer_analysis,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Inventory & Warehouse Intelligence | "
        "Cottonil Business Analytics"
    )

@st.cache_resource
def load_campaign_model():

    artifact = joblib.load(
        Path("artifacts/campaign_orders_model.joblib")
    )

    return artifact


@st.cache_data
def load_campaign_history():

    return pd.read_csv(
        Path("artifacts/campaign_history.csv")
    )


# ============================================================
# 2. PAGE FUNCTION
# ============================================================

def campaign_recommendation_page():

    st.title("Marketing Campaign Recommendation")

    st.caption(
        "Compare marketing channels and estimate attributed orders "
        "before launching a campaign."
    )

    try:
        artifact = load_campaign_model()
        history = load_campaign_history()

    except FileNotFoundError:
        st.error(
            "Campaign model artifacts were not found. "
            "Run the modeling cell first."
        )
        return

    model = artifact["model"]
    channels = artifact["channels"]
    governorate_map = artifact["governorate_map"]

    global_median_budget = artifact["global_median_budget"]

    # --------------------------------------------------------
    # 3. USER INPUTS
    # --------------------------------------------------------

    st.header("Campaign Configuration")

    col1, col2 = st.columns(2)

    available_governorates = sorted(
        history["governorate_ar"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    available_audiences = artifact["target_audiences"]

    with col1:

        selected_governorate_ar = st.selectbox(
            "Governorate",
            available_governorates,
            key="campaign_governorate",
        )

        selected_audience = st.selectbox(
            "Target Audience",
            available_audiences,
            key="campaign_audience",
        )

    with col2:

        campaign_dates = st.date_input(
            "Campaign Start and End Date",
            value=(date.today(), date.today()),
            key="campaign_dates",
        )

    if not isinstance(campaign_dates, (tuple, list)) or len(campaign_dates) != 2:
        st.info("Please select both campaign start and end dates.")
        return

    start_date, end_date = campaign_dates

    if end_date < start_date:
        st.error("End date must be after or equal to start date.")
        return

    duration_days = (end_date - start_date).days + 1

    selected_governorate = (
        governorate_map.get(
            selected_governorate_ar,
            selected_governorate_ar,
        )
    )

    # --------------------------------------------------------
    # 4. MATCH HISTORICAL CAMPAIGNS
    # --------------------------------------------------------

    matching_history = history[
        (
            history["governorate_ar"].astype(str)
            == str(selected_governorate_ar)
        )
        &
        (
            history["target_audience"].astype(str)
            == str(selected_audience)
        )
    ].copy()

    st.divider()

    st.header("Historical Performance")

    if matching_history.empty:

        st.warning(
            "No historical campaigns were found for this exact "
            "governorate and target audience combination."
        )

        recommended_budget = global_median_budget

        st.info(
            "A historical benchmark for this specific audience "
            "and governorate is unavailable. The overall median "
            "campaign budget will be used only as a fallback "
            "in case you do not enter a budget."
        )

        historical_top5 = pd.DataFrame()

    else:

        matching_history["cost_per_attributed_order"] = (
            matching_history["cost_per_attributed_order"]
            .replace([np.inf, -np.inf], np.nan)
        )

        # Top 5 successful campaigns by attributed orders
        historical_top5 = (
            matching_history
            .sort_values(
                "orders_attributed",
                ascending=False,
            )
            .head(5)
            .copy()
        )

        # Starting budget based on top 5 campaign budgets
        recommended_budget = float(
            historical_top5["budget_EGP"].mean()
        )

        average_orders = matching_history[
            "orders_attributed"
        ].mean()

        average_cost_per_order = matching_history[
            "cost_per_attributed_order"
        ].mean()

        total_campaigns = len(matching_history)

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Historical Campaigns",
            f"{total_campaigns:,}",
        )

        k2.metric(
            "Average Attributed Orders",
            f"{average_orders:,.1f}",
        )

        k3.metric(
            "Average Cost per Attributed Order",
            (
                f"{average_cost_per_order:,.2f} EGP"
                if pd.notna(average_cost_per_order)
                else "N/A"
            ),
        )

        st.subheader("Historical Channel Performance")

        channel_history = (
            matching_history
            .groupby("channel", as_index=False)
            .agg(
                campaigns_count=("campaign_id", "count"),
                avg_attributed_orders=(
                    "orders_attributed",
                    "mean",
                ),
                avg_budget=("budget_EGP", "mean"),
                avg_spend=("spend_EGP", "mean"),
                avg_cost_per_order=(
                    "cost_per_attributed_order",
                    "mean",
                ),
            )
            .sort_values(
                "avg_attributed_orders",
                ascending=False,
            )
        )

        st.dataframe(
            channel_history.style.format({
                "avg_attributed_orders": "{:,.2f}",
                "avg_budget": "{:,.2f}",
                "avg_spend": "{:,.2f}",
                "avg_cost_per_order": "{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Top 5 Successful Historical Campaigns")

        top5_display = historical_top5[
            [
                "campaign_id",
                "channel",
                "budget_EGP",
                "spend_EGP",
                "orders_attributed",
                "cost_per_attributed_order",
            ]
        ].copy()

        st.dataframe(
            top5_display.style.format({
                "budget_EGP": "{:,.2f}",
                "spend_EGP": "{:,.2f}",
                "orders_attributed": "{:,.0f}",
                "cost_per_attributed_order": "{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # 5. BUDGET INPUT
    # --------------------------------------------------------

    st.divider()

    st.header("Campaign Budget")

    use_custom_budget = st.checkbox(
        "I want to specify a campaign budget",
        value=False,
        key="use_custom_campaign_budget",
    )

    if use_custom_budget:

        campaign_budget = st.number_input(
            "Campaign Budget (EGP)",
            min_value=0.0,
            value=float(max(recommended_budget, 0)),
            step=1000.0,
            key="custom_campaign_budget",
        )

    else:

        campaign_budget = float(recommended_budget)

        st.metric(
            "Recommended Starting Budget",
            f"{campaign_budget:,.2f} EGP",
        )

        if not matching_history.empty:
            st.caption(
                "Calculated as the average budget of the top 5 "
                "historical campaigns for this governorate "
                "and target audience."
            )
        else:
            st.caption(
                "Fallback to the overall median campaign budget "
                "because no matching historical campaigns exist."
            )

    # --------------------------------------------------------
    # 6. PREPARE INFERENCE INPUT
    # --------------------------------------------------------

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    # These are the same date-derived features used in training.
    duration_days = (end_timestamp - start_timestamp).days + 1

    start_month = start_timestamp.month
    start_dayofweek = start_timestamp.dayofweek
    end_month = end_timestamp.month

    # --------------------------------------------------------
    # 7. PREDICT ALL CHANNELS
    # --------------------------------------------------------

    st.divider()

    st.header("Channel Recommendations")

    if st.button(
        "Generate Recommendations",
        type="primary",
        use_container_width=True,
        key="generate_campaign_recommendations",
    ):

        prediction_rows = []

        for channel in channels:

            input_data = pd.DataFrame([{
                "channel": channel,
                "target_audience": selected_audience,
                "governorate": selected_governorate,
                "budget_EGP": campaign_budget,
                "duration_days": duration_days,
                "start_month": start_month,
                "start_dayofweek": start_dayofweek,
                "end_month": end_month,
            }])

            # Pipeline prediction
            predicted_orders = float(
                model.predict(input_data)[0]
            )

            # Tree-level predictions for an estimated range
            fitted_preprocessor = model.named_steps["preprocessor"]
            fitted_forest = model.named_steps["regressor"]

            transformed_input = fitted_preprocessor.transform(
                input_data
            )

            tree_predictions = np.array([
                tree.predict(transformed_input)[0]
                for tree in fitted_forest.estimators_
            ])

            lower_bound = float(
                np.percentile(tree_predictions, 10)
            )

            upper_bound = float(
                np.percentile(tree_predictions, 90)
            )

            prediction_rows.append({
                "Channel": channel,
                "Predicted Orders": max(0, predicted_orders),
                "Lower Range": max(0, lower_bound),
                "Upper Range": max(0, upper_bound),
            })

        results = pd.DataFrame(prediction_rows)

        results = results.sort_values(
            "Predicted Orders",
            ascending=False,
        ).reset_index(drop=True)

        results.insert(
            0,
            "Rank",
            range(1, len(results) + 1),
        )

        # ----------------------------------------------------
        # 8. BEST CHANNEL
        # ----------------------------------------------------

        best_channel = results.iloc[0]

        st.success(
            f"Recommended Channel: {best_channel['Channel']}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Recommended Channel",
            str(best_channel["Channel"]),
        )

        c2.metric(
            "Expected Attributed Orders",
            f"{best_channel['Predicted Orders']:.1f}",
        )

        c3.metric(
            "Predicted Range",
            (
                f"{best_channel['Lower Range']:.1f}"
                f" - "
                f"{best_channel['Upper Range']:.1f}"
            ),
        )

        # ----------------------------------------------------
        # 9. ALL CHANNELS RESULTS
        # ----------------------------------------------------

        st.subheader("All Channels Comparison")

        display_results = results.copy()

        for col in [
            "Predicted Orders",
            "Lower Range",
            "Upper Range",
        ]:
            display_results[col] = display_results[col].round(1)

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # 10. VISUALIZATION
        # ----------------------------------------------------

        fig = px.bar(
            results,
            x="Channel",
            y="Predicted Orders",
            error_y=(
                results["Upper Range"]
                - results["Predicted Orders"]
            ),
            error_y_minus=(
                results["Predicted Orders"]
                - results["Lower Range"]
            ),
            text="Predicted Orders",
            title="Predicted Attributed Orders by Channel",
            labels={
                "Predicted Orders": "Expected Attributed Orders",
                "Channel": "Marketing Channel",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )

        fig.update_layout(
            height=450,
            xaxis_title="Marketing Channel",
            yaxis_title="Expected Attributed Orders",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="campaign_channel_predictions",
        )

        # ----------------------------------------------------
        # 11. CAMPAIGN SUMMARY
        # ----------------------------------------------------

        st.subheader("Campaign Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "Governorate",
                "Target Audience",
                "Start Date",
                "End Date",
                "Duration",
                "Campaign Budget",
                "Recommended Channel",
                "Expected Attributed Orders",
            ],
            "Value": [
                selected_governorate_ar,
                selected_audience,
                str(start_date),
                str(end_date),
                f"{duration_days} days",
                f"{campaign_budget:,.2f} EGP",
                str(best_channel["Channel"]),
                f"{best_channel['Predicted Orders']:.1f}",
            ],
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )


def returns_analysis_page():

    from pathlib import Path

    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    # ============================================================
    # CONFIGURATION
    # ============================================================

    DATA_DIR = Path(r"D:\HVIA\cotonil\data")

    ORDERS_PATH = DATA_DIR / "sales_orders.csv"
    RETURNS_PATH = DATA_DIR / "sales_returns.csv"

    CUTOFF_DATE = pd.Timestamp("2026-10-18")

    COLORS = px.colors.qualitative.Set2

    # ============================================================
    # LOAD DATA
    # ============================================================

    @st.cache_data
    def load_returns_data():

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

    df_orders, df_returns = load_returns_data()

    # ============================================================
    # DATA PREPARATION
    # ============================================================

    df_orders = df_orders.dropna(
        subset=["order_date"]
    ).copy()

    df_returns = df_returns.dropna(
        subset=["return_date"]
    ).copy()

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

    available_years = sorted(
        df_orders["order_date"].dt.year.dropna().unique(),
        reverse=True
    )

    if not available_years:
        st.warning("No valid orders data available.")
        return

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================

    def get_period_data(
        year,
        month=None,
        quarter=None,
        cutoff=None
    ):

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
    # DASHBOARD HEADER
    # ============================================================

    st.title("📦 Cottonil | Returns Analysis")

    st.caption(
        "Analyze product returns, refund amounts, "
        "return reasons and channel performance."
    )

    st.divider()

    # ============================================================
    # 1. YEARLY RETURNS ANALYSIS
    # ============================================================

    st.header("1. Yearly Returns Analysis")

    selected_year = st.selectbox(
        "Select Year",
        options=available_years,
        key="returns_selected_year"
    )

    yearly_cutoff = (
        CUTOFF_DATE
        if selected_year == 2026
        else None
    )

    if yearly_cutoff is not None:

        st.info(
            f"Showing data till "
            f"{yearly_cutoff.strftime('%d/%m/%Y')}"
        )

    orders_year, returns_year = get_period_data(
        selected_year,
        cutoff=yearly_cutoff
    )

    # ============================================================
    # 1.1 KPI CALCULATIONS
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

    refund_change = (
        refund_difference / previous_refund * 100
        if previous_refund != 0
        else 0
    )

    # ============================================================
    # 1.2 KPI CARDS
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
    # 1.3 RETURN REASONS DONUT CHART
    # ============================================================

    st.subheader("Return Reasons Distribution")

    if not returns_year.empty:

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
    # 1.4 MONTHLY RETURN ORDERS
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
    # 2. GENERAL RETURNS ANALYSIS
    # ============================================================

    st.header("2. General Returns Analysis")

    st.caption(
        "Historical analysis across all available years."
    )

    # ============================================================
    # 2.1 YEARLY RETURN RATIO
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

        apply_plot_style(
            fig_yearly_ratio,
            450
        )

        st.plotly_chart(
            fig_yearly_ratio,
            use_container_width=True
        )

    st.divider()

    # ============================================================
    # 2.2 QUARTERLY ANALYSIS
    # ============================================================

    st.subheader(
        "Quarterly Refund & Return Ratio Analysis"
    )

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

        quarterly_df = quarterly_df[
            (quarterly_df["Refund Amount"] > 0)
            | (quarterly_df["Returned Ratio (%)"] > 0)
        ].copy()

        if not quarterly_df.empty:

            fig_quarterly = go.Figure()

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
                    text=(
                        "Quarterly Refund Amount & "
                        "Returned Orders Ratio"
                    ),
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

        else:

            st.info("No quarterly return data available.")

    st.caption(
        "Quarterly analysis uses all available data. "
        "The latest year's Q4 is treated as Q4 even if incomplete."
    )

    st.divider()

    # ============================================================
    # 2.3 MOST AFFECTED BRANCH PER YEAR
    # ============================================================

    st.subheader("Most Affected Branch per Year")

    branch_yearly_data = []

    for year in sorted(available_years):

        returns = df_or_re[
            df_or_re["return_date"].dt.year == year
        ].copy()

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

        apply_plot_style(
            fig_branch,
            400
        )

        st.plotly_chart(
            fig_branch,
            use_container_width=True
        )

    else:

        st.info("No branch return data available.")

    st.divider()

    # ============================================================
    # 2.4 RETURN REASONS BY SALES CHANNEL
    # ============================================================

    st.subheader(
        "Return Reasons Distribution by Sales Channel"
    )

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
    # FOOTER
    # ============================================================

    st.divider()

    st.caption(
        "Cottonil | Returns Analysis Dashboard"
    )

    st.caption(
        "Data Source: Sales Orders & Sales Returns"
    )


# =========================================================

# Streamlit Navigation

# =========================================================



pages = [

    st.Page(
        overview_page,
        title="Executive",
        icon="📊",
        default=True,
    ),

    st.Page(
        sales_performance_intelligence_page,
        title="Sales Performance Intelligence",
        icon="📈",
    ),

    st.Page(
        inventory_warehouse_intelligence_page,
        title="Inventory & Warehouse Intelligence",
        icon="📦",
    ),

    st.Page(
        returns_analysis_page,
        title="Returns Analysis",
        icon="🔄",
    ),

    st.Page(
        branches_page,
        title="Branches",
        icon="🏢",
    ),

    st.Page(
        warehouses_factory_recommendations_page,
        title="Warehouses and Factory Recommendations",
        icon="🏭",
    ),

    st.Page(
        customer_segmentation_page,
        title="Customer Segmentation",
        icon="👥",
    ),

    st.Page(
        campaign_recommendation_page,
        title="Campaign Recommendation",
        icon="📣",
    )

]

navigation = st.navigation(pages)
navigation.run()
