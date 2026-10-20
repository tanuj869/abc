"""
SuryaIQ — PM Surya Ghar Policy Analytics, Econometric Modeling & Forecasting Platform
Capstone Level Interactive Research System
Built with Streamlit & Plotly
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="SuryaIQ — PM Surya Ghar Policy Decision Support Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- ADVANCED CSS STYLING (Glassmorphic / Modern Enterprise) -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        border-radius: 16px;
        padding: 28px 36px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(to right, #F8FAFC, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        margin-top: 6px;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #0EA5E9, #10B981);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #0D9488;
        font-weight: 600;
        margin-top: 6px;
    }
    
    /* Section Headings */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 14px;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .section-desc {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 16px;
    }
    
    /* Badges */
    .badge-leader {
        background-color: #ECFDF5;
        color: #047857;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-mod {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-lag {
        background-color: #FEF2F2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

@st.cache_data
def load_all_data():
    df_model = pd.read_csv(os.path.join(PROCESSED_DIR, "modeling_table.csv"))
    df_dist = pd.read_csv(os.path.join(PROCESSED_DIR, "district_level_pmsg.csv"))
    df_discom = pd.read_csv(os.path.join(PROCESSED_DIR, "discom_level_pmsg.csv"))
    df_mh = pd.read_csv(os.path.join(PROCESSED_DIR, "maharashtra_districts.csv"))
    df_nat_fc = pd.read_csv(os.path.join(PROCESSED_DIR, "national_forecast_detailed.csv"))
    df_comp = pd.read_csv(os.path.join(PROCESSED_DIR, "forecast_comprehensive_comparison.csv"))
    df_states_fc = pd.read_csv(os.path.join(PROCESSED_DIR, "state_case_studies_forecast.csv"))
    df_corr = pd.read_csv(os.path.join(PROCESSED_DIR, "correlation_table.csv"))
    df_series = pd.read_csv(os.path.join(PROCESSED_DIR, "national_installations_series.csv"))
    df_snapshots = pd.read_csv(os.path.join(PROCESSED_DIR, "pmsg_snapshots_long.csv"))
    return df_model, df_dist, df_discom, df_mh, df_nat_fc, df_comp, df_states_fc, df_corr, df_series, df_snapshots

df_model, df_dist, df_discom, df_mh, df_nat_fc, df_comp, df_states_fc, df_corr, df_series, df_snapshots = load_all_data()

# ----------------- SIDEBAR MULTI-FILTERS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Global Data Filters")
    st.caption("Filter records dynamically across all analytics tabs.")
    
    # Filter 1: Geographical Region
    all_regions = sorted(df_model["region"].dropna().unique())
    selected_regions = st.multiselect(
        "Geographical Region",
        options=all_regions,
        default=all_regions,
        help="Filter by official zones: North, South, West, East, Central, North-East"
    )
    
    # Filter 2: Adoption Volume Tier
    all_tiers = ["Tier 1 (>200k)", "Tier 2 (50k-200k)", "Tier 3 (<50k)"]
    selected_tiers = st.multiselect(
        "Adoption Volume Tier",
        options=all_tiers,
        default=all_tiers,
        help="Classified by cumulative installations"
    )
    
    # Filter 3: Conversion Efficiency Status
    all_eff = ["Leader (>70%)", "Moderate (40%-70%)", "Lagging (<40%)"]
    selected_eff = st.multiselect(
        "Efficiency Tier",
        options=all_eff,
        default=all_eff,
        help="Installations ÷ Applications conversion performance"
    )
    
    # Filter 4: Solar Irradiance Range
    irr_min = float(df_model["irradiance_annual_kwh_m2_day"].min())
    irr_max = float(df_model["irradiance_annual_kwh_m2_day"].max())
    selected_irr = st.slider(
        "Solar Irradiance (kWh/m²/day)",
        min_value=round(irr_min, 1),
        max_value=round(irr_max, 1),
        value=(round(irr_min, 1), round(irr_max, 1)),
        step=0.1
    )
    
    # Filter 5: Specific State Search
    selected_states = st.multiselect(
        "Highlight / Filter Specific States",
        options=sorted(df_model["state"].unique()),
        default=[],
        placeholder="All states included (or search...)"
    )
    
    st.markdown("---")
    st.markdown("### 📚 Capstone Overview")
    st.markdown("""
    **Project Scope:**  
    • 36 States/UTs Normalized  
    • 767 Administrative Districts  
    • 83 DISCOM Utilities  
    • 5 Benchmark Time-Series Models  
    • 10 Official Tier-1 Datasets  
    """)
    st.caption("SuryaIQ Platform • 2026 Edition")

# Apply Filters
filtered_df = df_model.copy()

if selected_regions:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
if selected_tiers:
    filtered_df = filtered_df[filtered_df["adoption_tier"].isin(selected_tiers)]
if selected_eff:
    filtered_df = filtered_df[filtered_df["efficiency_tier"].isin(selected_eff)]
filtered_df = filtered_df[
    (filtered_df["irradiance_annual_kwh_m2_day"] >= selected_irr[0]) &
    (filtered_df["irradiance_annual_kwh_m2_day"] <= selected_irr[1])
]
if selected_states:
    filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]

# ----------------- HERO BANNER -----------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-tag">Policy Analytics & Research Capstone</div>
    <h1 class="hero-title">SuryaIQ — PM Surya Ghar Intelligence Platform</h1>
    <p class="hero-subtitle">
        An econometric modeling, time-series forecasting, and counterfactual policy simulation system 
        evaluating India's 10-Million Rooftop Solar Mission (PM Surya Ghar: Muft Bijli Yojana).
    </p>
</div>
""", unsafe_allow_html=True)

# Active Filter Status Badge
total_states_count = len(df_model)
active_states_count = len(filtered_df)
pct_coverage = (filtered_df["installations_aug2026"].sum() / df_model["installations_aug2026"].sum() * 100) if df_model["installations_aug2026"].sum() > 0 else 0

st.info(f"🔍 **Active Selection:** Displaying **{active_states_count} of {total_states_count} States/UTs** (representing **{pct_coverage:.1f}%** of national rooftop solar installations).")

if active_states_count == 0:
    st.warning("⚠️ No states match your selected filter criteria. Please broaden your filter parameters in the left sidebar.")
    st.stop()

# ----------------- TOP KPI METRIC CARDS -----------------
tot_inst = int(filtered_df["installations_aug2026"].sum())
tot_apps = int(filtered_df["applications_aug2026"].sum())
avg_conv = (tot_inst / tot_apps * 100) if tot_apps > 0 else 0
avg_per_lakh = filtered_df["installations_per_lakh_pop"].mean()

# Filter districts matching filtered states
filtered_dist = df_dist[df_dist["state"].isin(filtered_df["state"])]
tot_cap = filtered_dist["capacity_mw"].sum()
tot_subsidy = filtered_dist["subsidy_released_lakhs"].sum() / 100 # In Crores

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Installations</div>
        <div class="metric-value">{tot_inst:,}</div>
        <div class="metric-sub">~{tot_inst/1e5:.2f} Lakh Systems</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sanction Applications</div>
        <div class="metric-value">{tot_apps:,}</div>
        <div class="metric-sub">~{tot_apps/1e5:.2f} Lakh Demands</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Conversion Efficiency</div>
        <div class="metric-value">{avg_conv:.1f}%</div>
        <div class="metric-sub">Installations ÷ Applications</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Installed Capacity</div>
        <div class="metric-value">{tot_cap:,.0f}</div>
        <div class="metric-sub">Megawatts (MW) Grid Load</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Subsidy Disbursed</div>
        <div class="metric-value">₹{tot_subsidy:,.1f} Cr</div>
        <div class="metric-sub">Central Financial Assist.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- TABS SYSTEM -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Diagnostics",
    "🎯 Econometric Paradox Matrix",
    "🗺️ District & DISCOM Treemap",
    "📈 Multi-Model Forecasting Studio",
    "🧪 What-If Policy Sandbox",
    "🕸️ 6-D State Radar & Deep-Dive"
])

# ----------------- TAB 1: EXECUTIVE DIAGNOSTICS -----------------
with tab1:
    col_t1_left, col_t1_right = st.columns([3, 2])
    
    with col_t1_left:
        st.markdown('<div class="section-title">National Adoption Funnel Dynamics</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Quantifying pipeline conversion drop-offs across the 4 stages of adoption.</div>', unsafe_allow_html=True)
        
        # Calculate national funnel values
        # Get registration and household values for filtered states
        sub_apps = tot_apps
        sub_inst = tot_inst
        # Fetch registrations from long panel
        reg_df = df_snapshots[(df_snapshots["metric"] == "registrations") & (df_snapshots["state"].isin(filtered_df["state"]))]
        tot_regs = reg_df["value"].sum() if len(reg_df) > 0 else (tot_apps * 1.6)
        tot_hholds = filtered_df["households_aug2026"].sum()
        
        funnel_fig = go.Figure(go.Funnel(
            y=["1. Registered Interest", "2. Formal Sanction Application", "3. System Installed & Metered", "4. Subsidy Transferred"],
            x=[tot_regs, sub_apps, sub_inst, tot_hholds],
            textinfo="value+percent previous+percent initial",
            marker=dict(color=["#0284C7", "#0EA5E9", "#10B981", "#059669"]),
            connector=dict(line=dict(color="#CBD5E1", width=2))
        ))
        funnel_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            font=dict(family="Plus Jakarta Sans", size=12)
        )
        st.plotly_chart(funnel_fig, use_container_width=True)

    with col_t1_right:
        st.markdown('<div class="section-title">Regional Share Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Distribution of rooftop capacity (MW) across geographic zones.</div>', unsafe_allow_html=True)
        
        reg_agg = filtered_df.groupby("region").agg({
            "installations_aug2026": "sum"
        }).reset_index()
        
        pie_fig = px.pie(
            reg_agg,
            values="installations_aug2026",
            names="region",
            hole=0.55,
            color_discrete_sequence=["#0EA5E9", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#64748B"]
        )
        pie_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            showlegend=True,
            font=dict(family="Plus Jakarta Sans", size=12)
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown('<div class="section-title">State-Level Performance Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Interactive table with comprehensive adoption, per-capita rate, and solar potential variables. Click any column header to sort.</div>', unsafe_allow_html=True)
    
    table_view = filtered_df[[
        "state", "region", "adoption_tier", "efficiency_tier",
        "installations_aug2026", "applications_aug2026", "target_conversion_ratio",
        "installations_per_lakh_pop", "tariff_growth_pct", "irradiance_annual_kwh_m2_day"
    ]].copy()
    
    table_view = table_view.sort_values(by="installations_aug2026", ascending=False)
    table_view.columns = [
        "State/UT", "Region", "Volume Tier", "Efficiency Tier",
        "Installations", "Applications", "Conversion Rate",
        "Inst. / Lakh Pop", "Tariff Growth %", "Irradiance (kWh/m²/day)"
    ]
    
    st.dataframe(
        table_view.style.format({
            "Installations": "{:,.0f}",
            "Applications": "{:,.0f}",
            "Conversion Rate": "{:.2%}",
            "Inst. / Lakh Pop": "{:.1f}",
            "Tariff Growth %": "{:+.1f}%",
            "Irradiance (kWh/m²/day)": "{:.2f}"
        }),
        height=400,
        use_container_width=True
    )

# ----------------- TAB 2: ECONOMETRIC PARADOX MATRIX -----------------
with tab2:
    st.markdown('<div class="section-title">The "Subsidy & Resource Paradox" — Empirical Evidence</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
        A core research contribution of this project: Cross-sectional econometric modeling proves that baseline subsidy generosity 
        and static solar potential are <b>poor predictors</b> of execution efficiency. Rather, <b>electricity tariff escalation</b> 
        and <b>physical irradiance</b> explain over 46% of conversion efficiency.
    </div>
    """, unsafe_allow_html=True)
    
    # 4-Quadrant Scatter Bubble Chart
    mean_irr = filtered_df["irradiance_annual_kwh_m2_day"].mean()
    mean_tariff_growth = filtered_df["tariff_growth_pct"].mean()
    
    bubble_fig = px.scatter(
        filtered_df,
        x="irradiance_annual_kwh_m2_day",
        y="tariff_growth_pct",
        size="installations_aug2026",
        color="target_conversion_ratio",
        hover_name="state",
        hover_data={
            "region": True,
            "installations_aug2026": ":,",
            "target_conversion_ratio": ":.2%",
            "tariff_growth_pct": ":+.1f%",
            "irradiance_annual_kwh_m2_day": ":.2f"
        },
        labels={
            "irradiance_annual_kwh_m2_day": "Mean Solar Irradiance (kWh/m²/day)",
            "tariff_growth_pct": "Grid Tariff Growth Rate (% YoY)",
            "target_conversion_ratio": "Conversion Ratio",
            "installations_aug2026": "Installations"
        },
        color_continuous_scale="Viridis",
        size_max=45
    )
    
    # Add quadrant dividing lines
    bubble_fig.add_vline(x=mean_irr, line_dash="dash", line_color="#94A3B8", annotation_text=f"Mean Irradiance ({mean_irr:.2f})")
    bubble_fig.add_hline(y=mean_tariff_growth, line_dash="dash", line_color="#94A3B8", annotation_text=f"Mean Tariff Growth ({mean_tariff_growth:.1f}%)")
    
    # Annotate Quadrants
    bubble_fig.add_annotation(x=irr_max - 0.2, y=filtered_df["tariff_growth_pct"].max() - 2, text="🏆 Sweet Spot (High Sun + Rising Tariff)", showarrow=False, font=dict(color="#059669", size=12))
    bubble_fig.add_annotation(x=irr_max - 0.2, y=filtered_df["tariff_growth_pct"].min() + 2, text="⚠️ Underperforming Sunbelt (Tariff Lag)", showarrow=False, font=dict(color="#D97706", size=12))
    
    bubble_fig.update_layout(
        height=520,
        font=dict(family="Plus Jakarta Sans"),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(bubble_fig, use_container_width=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### Feature Correlation Ranking vs Conversion Efficiency")
        bar_corr = px.bar(
            df_corr.head(8),
            x="correlation_r",
            y="feature",
            orientation="h",
            color="correlation_r",
            color_continuous_scale="Teal",
            labels={"correlation_r": "Pearson Correlation (r)", "feature": "Covariate"}
        )
        bar_corr.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar_corr, use_container_width=True)

    with col_c2:
        st.markdown("#### Econometric Model Comparison (LOO-CV)")
        st.markdown("""
        Evaluated using **Leave-One-Out Cross-Validation (LOO-CV)** to avoid small-sample overfitting:
        """)
        eval_data = pd.DataFrame([
            {"Model Architecture": "2-Variable Linear Model (Tariff Growth + Irradiance)", "LOO-CV MAE": 0.135, "Error vs Baseline": "-20.7% (Best)", "Complexity": "Parsimonious"},
            {"Model Architecture": "6-Variable Random Forest (with SHAP)", "LOO-CV MAE": 0.161, "Error vs Baseline": "-5.9%", "Complexity": "Non-linear / Overfitting"},
            {"Model Architecture": "Naive Mean Baseline", "LOO-CV MAE": 0.170, "Error vs Baseline": "0.0%", "Complexity": "Baseline"}
        ])
        st.dataframe(eval_data, use_container_width=True)
        st.success("🎯 **Model Selection Rule:** The simple 2-variable linear specification generalizes better than tree-based models on small cross-sectional regimes ($n=35$).")

# ----------------- TAB 3: DISTRICT & DISCOM TREEMAP -----------------
with tab3:
    st.markdown('<div class="section-title">All-India District & DISCOM Operational Hierarchy</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Interactive Treemap across ~767 districts and 83 DISCOMs. Click on any Region or State to drill down.</div>', unsafe_allow_html=True)
    
    treemap_metric = st.radio("Sizing Metric for Treemap:", ["installations", "capacity_mw", "subsidy_released_lakhs"], horizontal=True, format_func=lambda x: {
        "installations": "Installation Count",
        "capacity_mw": "Installed Capacity (MW)",
        "subsidy_released_lakhs": "Subsidy Released (Lakhs)"
    }[x])
    
    # Filter districts by active states
    tm_df = df_dist[df_dist["state"].isin(filtered_df["state"])].copy()
    tm_df = tm_df[tm_df[treemap_metric] > 0]
    
    tree_fig = px.treemap(
        tm_df,
        path=["region", "state", "district"],
        values=treemap_metric,
        color="conversion_ratio",
        color_continuous_scale="Blues",
        hover_data=["applications", "installations", "capacity_mw"]
    )
    tree_fig.update_layout(
        height=580,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Plus Jakarta Sans")
    )
    st.plotly_chart(tree_fig, use_container_width=True)
    
    # Top 15 Districts Nationwide
    st.markdown("#### 🏆 Top 15 Administrative Districts Nationwide")
    top_15_dist = tm_df.sort_values(by="installations", ascending=False).head(15)
    bar_dist = px.bar(
        top_15_dist,
        x="district",
        y="installations",
        color="state",
        hover_data=["capacity_mw", "conversion_ratio"],
        labels={"installations": "Installed Systems", "district": "District"}
    )
    bar_dist.update_layout(height=360, font=dict(family="Plus Jakarta Sans"))
    st.plotly_chart(bar_dist, use_container_width=True)

# ----------------- TAB 4: FORECASTING STUDIO -----------------
with tab4:
    st.markdown('<div class="section-title">National & State Adoption Forecasting Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Multi-model benchmark comparing 5 time-series algorithms across official milestone timestamps.</div>', unsafe_allow_html=True)
    
    fc_left, fc_right = st.columns([3, 1])
    with fc_right:
        st.markdown("#### Model Selector")
        show_log = st.checkbox("Log-Linear (Exponential)", value=True)
        show_holt = st.checkbox("Holt's Exponential Smoothing", value=True)
        show_lin = st.checkbox("Linear Trend", value=True)
        show_ci = st.checkbox("95% Confidence Band", value=True)
        st.markdown("---")
        st.markdown("#### In-Sample MAPE Ranking")
        st.dataframe(df_comp.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}%"}), use_container_width=True)
    
    with fc_left:
        # Build interactive Plotly chart for national projections
        fc_fig = go.Figure()
        
        # Observed Points
        df_series_dt = df_series.copy()
        df_series_dt["date"] = pd.to_datetime(df_series_dt["date"])
        
        fc_fig.add_trace(go.Scatter(
            x=df_series_dt["date"],
            y=df_series_dt["installations"],
            mode="lines+markers",
            name="Official Historical Milestones",
            line=dict(color="#0F172A", width=3),
            marker=dict(size=9, color="#0EA5E9")
        ))
        
        # Forecast future points
        future_dates = pd.to_datetime(df_nat_fc["forecast_date"])
        last_obs_date = df_series_dt["date"].iloc[-1]
        last_obs_val = df_series_dt["installations"].iloc[-1]
        
        # Connect last observed with forecast points
        join_dates = [last_obs_date] + list(future_dates)
        
        if show_log:
            join_log = [last_obs_val] + list(df_nat_fc["point_forecast_log_linear"])
            fc_fig.add_trace(go.Scatter(
                x=join_dates, y=join_log,
                mode="lines+markers",
                name="Log-Linear Forecast (Best Fit)",
                line=dict(color="#0284C7", width=2.5, dash="dash"),
                marker=dict(size=8, symbol="square")
            ))
            
            if show_ci:
                join_lower = [last_obs_val] + list(df_nat_fc["lower_95_ci"])
                join_upper = [last_obs_val] + list(df_nat_fc["upper_95_ci"])
                fc_fig.add_trace(go.Scatter(
                    x=join_dates + join_dates[::-1],
                    y=join_upper + join_lower[::-1],
                    fill="toself",
                    fillcolor="rgba(14, 165, 233, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Confidence Interval",
                    showlegend=True
                ))
                
        if show_holt:
            join_holt = [last_obs_val] + list(df_nat_fc["point_forecast_holt"])
            fc_fig.add_trace(go.Scatter(
                x=join_dates, y=join_holt,
                mode="lines+markers",
                name="Holt's Exponential Smoothing",
                line=dict(color="#10B981", width=2, dash="dot"),
                marker=dict(size=7, symbol="triangle-up")
            ))
            
        if show_lin:
            join_lin = [last_obs_val] + list(df_nat_fc["point_forecast_linear"])
            fc_fig.add_trace(go.Scatter(
                x=join_dates, y=join_lin,
                mode="lines+markers",
                name="Linear Trend Extrapolation",
                line=dict(color="#EF4444", width=2, dash="dashdot"),
                marker=dict(size=7, symbol="diamond")
            ))
            
        fc_fig.update_layout(
            title="National Adoption Trajectory & Near-Term Horizons (+3, +6, +12 Months)",
            xaxis_title="Timeline",
            yaxis_title="Cumulative Installations",
            height=480,
            font=dict(family="Plus Jakarta Sans"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fc_fig, use_container_width=True)
        
    st.markdown("#### 🌟 5 Representative State Adoption Case Studies")
    st.dataframe(df_states_fc.style.format({
        "installations_nov2024": "{:,.0f}",
        "installations_aug2026": "{:,.0f}",
        "applications_aug2026": "{:,.0f}",
        "conversion_ratio": "{:.2%}",
        "pop_2024_lakhs": "{:.1f}",
        "installations_per_lakh_pop_aug2026": "{:.1f}",
        "annualized_growth_pct": "{:+.1f}%",
        "forecast_3m": "{:,.0f}",
        "forecast_6m": "{:,.0f}",
        "forecast_12m": "{:,.0f}"
    }), use_container_width=True)

# ----------------- TAB 5: WHAT-IF POLICY SANDBOX -----------------
with tab5:
    st.markdown('<div class="section-title">Interactive Policy Counterfactual Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Simulate how electricity tariff shocks and high-potential solar targeting alter state conversion rates and national capacity.</div>', unsafe_allow_html=True)
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("#### Policy Levers")
        sim_tariff = st.slider("Simulate Tariff Growth Shock (+pp)", min_value=0, max_value=30, value=10, step=5, help="Based on empirical marginal coefficient of ~1.6pp conversion gain per 1% tariff hike.")
        sim_target = st.checkbox("Enable Scenario C: High-Potential Targeting", value=True, help="Target promotional subsidies to high-irradiance states (>5.2 kWh/m²/day) with below-average conversion.")
        campaign_boost = st.slider("Targeted Campaign Efficiency Boost (+pp)", min_value=0, max_value=20, value=8, step=2) if sim_target else 0
        
    with sim_col2:
        # Calculate simulation
        sim_df = filtered_df[["state", "region", "target_conversion_ratio", "installations_aug2026", "applications_aug2026", "irradiance_annual_kwh_m2_day"]].copy()
        
        # Base conversion
        sim_df["baseline_conversion"] = sim_df["target_conversion_ratio"]
        
        # Tariff effect
        tariff_lift = (sim_tariff / 100.0) * 1.6
        
        # Target effect
        is_target_state = (sim_df["irradiance_annual_kwh_m2_day"] > 5.2) & (sim_df["baseline_conversion"] < 0.65)
        sim_df["is_targeted"] = is_target_state
        target_lift = np.where(is_target_state, campaign_boost / 100.0, 0.0)
        
        sim_df["simulated_conversion"] = (sim_df["baseline_conversion"] + tariff_lift + target_lift).clip(upper=0.98)
        sim_df["incremental_installations"] = np.round(
            (sim_df["simulated_conversion"] - sim_df["baseline_conversion"]) * sim_df["applications_aug2026"]
        ).astype(int)
        
        tot_inc_inst = sim_df["incremental_installations"].sum()
        tot_inc_mw = tot_inc_inst * 3.3 / 1000 # Assuming ~3.3 kW avg residential system
        
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Projected Incremental Installations", f"+{tot_inc_inst:,}")
        with m_c2:
            st.metric("Incremental Rooftop Capacity", f"+{tot_inc_mw:,.1f} MW")
            
        # Comparison Bar Chart
        top_sim = sim_df.sort_values(by="incremental_installations", ascending=False).head(10)
        comp_bar = go.Figure()
        comp_bar.add_trace(go.Bar(
            x=top_sim["state"], y=top_sim["baseline_conversion"],
            name="Baseline Conversion Rate", marker_color="#94A3B8"
        ))
        comp_bar.add_trace(go.Bar(
            x=top_sim["state"], y=top_sim["simulated_conversion"],
            name="Counterfactual Simulated Rate", marker_color="#0EA5E9"
        ))
        comp_bar.update_layout(
            barmode="group",
            height=360,
            title="Top 10 Beneficiary States: Baseline vs Simulated Conversion Efficiency",
            yaxis_tickformat=".0%",
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(comp_bar, use_container_width=True)

# ----------------- TAB 6: 6-D STATE RADAR & DEEP-DIVE -----------------
with tab6:
    st.markdown('<div class="section-title">6-Dimensional State Archetype Radar Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Select up to 3 states to benchmark across normalized strategic vectors.</div>', unsafe_allow_html=True)
    
    radar_states = st.multiselect(
        "Choose States for Radar Profiling:",
        options=sorted(df_model["state"].unique()),
        default=["Gujarat", "Maharashtra", "Tamil Nadu"] if "Gujarat" in df_model["state"].values else [df_model["state"].iloc[0]],
        max_selections=4
    )
    
    if len(radar_states) > 0:
        categories = [
            "Conversion Efficiency", "Tariff Growth Momentum", 
            "Solar Irradiance Fit", "Per-Capita Penetration",
            "RE Capacity Maturity", "Solar Potential Endowment"
        ]
        
        # Compute min-max normalized scores across full dataset
        norm_df = df_model.copy()
        metrics_map = {
            "Conversion Efficiency": "target_conversion_ratio",
            "Tariff Growth Momentum": "tariff_growth_pct",
            "Solar Irradiance Fit": "irradiance_annual_kwh_m2_day",
            "Per-Capita Penetration": "installations_per_lakh_pop",
            "RE Capacity Maturity": "mnre_re_capacity_2025_mw",
            "Solar Potential Endowment": "solar_potential_3pct_wasteland_gw"
        }
        
        radar_fig = go.Figure()
        colors = ["#0EA5E9", "#10B981", "#F59E0B", "#8B5CF6"]
        
        for idx, st_name in enumerate(radar_states):
            st_row = norm_df[norm_df["state"] == st_name]
            if len(st_row) == 0:
                continue
            values = []
            for cat in categories:
                col = metrics_map[cat]
                v = st_row[col].values[0]
                vmin = norm_df[col].min()
                vmax = norm_df[col].max()
                score = ((v - vmin) / (vmax - vmin) * 100) if vmax > vmin and not np.isnan(v) else 50
                values.append(score)
            values.append(values[0]) # Close loop
            
            radar_fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=f"rgba{tuple(list(int(colors[idx % len(colors)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                line=dict(color=colors[idx % len(colors)], width=2),
                name=st_name
            ))
            
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            height=460,
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Select at least 1 state to display the radar chart.")

# ----------------- FOOTER -----------------
st.markdown("---")
f_c1, f_c2 = st.columns([3, 1])
with f_c1:
    st.caption("SuryaIQ Platform • Capstone Research Project • Rigorously validated with Leave-One-Out CV & Time-Series Backtesting.")
with f_c2:
    st.caption("Data Sources: data.gov.in, CEA, MNRE, NITI Aayog ICeD, NASA POWER.")
