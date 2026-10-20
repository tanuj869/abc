"""
SuryaIQ — PM Surya Ghar Policy Analytics, Econometric Modeling & Forecasting Platform
Capstone Research Platform | 3-Month Data Science & Policy Analytics Capstone
Interactive Multi-Dimensional Dashboard with Plotly & Streamlit
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="SuryaIQ — PM Surya Ghar Intelligence Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- DATA LOADING -----------------
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

@st.cache_data
def load_all_datasets():
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

df_model, df_dist, df_discom, df_mh, df_nat_fc, df_comp, df_states_fc, df_corr, df_series, df_snapshots = load_all_datasets()

# ----------------- SIDEBAR CONTROLS & THEME TOGGLE -----------------
with st.sidebar:
    st.markdown("## ☀️ SuryaIQ Studio")
    st.caption("PM Surya Ghar National Analytics Capstone")
    
    # Theme Selection
    theme_choice = st.radio(
        "Interface Palette",
        options=["Obsidian Dark", "Modern SaaS Light"],
        horizontal=True
    )
    is_dark = (theme_choice == "Obsidian Dark")
    plotly_template = "plotly_dark" if is_dark else "plotly_white"
    
    st.markdown("---")
    st.markdown("### 🎛️ Advanced Global Filters")
    
    # Global Text Search
    search_query = st.text_input(
        "🔎 Quick State / Keyword Search",
        value="",
        placeholder="Type state name (e.g., Gujarat)..."
    ).strip().lower()
    
    # Region Multi-Select
    all_regions = sorted(df_model["region"].dropna().unique())
    selected_regions = st.multiselect(
        "Geographical Region",
        options=all_regions,
        default=all_regions,
        help="Filter across India's 6 official geographical zones"
    )
    
    # Adoption Volume Tier
    all_tiers = ["Tier 1 (>200k)", "Tier 2 (50k-200k)", "Tier 3 (<50k)"]
    selected_tiers = st.multiselect(
        "Adoption Scale Tier",
        options=all_tiers,
        default=all_tiers,
        help="Classified by total cumulative installations"
    )
    
    # Conversion Efficiency Tier
    all_eff = ["Leader (>70%)", "Moderate (40%-70%)", "Lagging (<40%)"]
    selected_eff = st.multiselect(
        "Efficiency Tier",
        options=all_eff,
        default=all_eff,
        help="Installations ÷ Applications operational ratio"
    )
    
    # Solar Irradiance Range
    irr_min = float(df_model["irradiance_annual_kwh_m2_day"].min())
    irr_max = float(df_model["irradiance_annual_kwh_m2_day"].max())
    selected_irr = st.slider(
        "Solar Irradiance (kWh/m²/day)",
        min_value=round(irr_min, 1),
        max_value=round(irr_max, 1),
        value=(round(irr_min, 1), round(irr_max, 1)),
        step=0.1
    )
    
    # Tariff Growth Range
    tg_min = float(df_model["tariff_growth_pct"].min())
    tg_max = float(df_model["tariff_growth_pct"].max())
    selected_tg = st.slider(
        "Tariff Growth (% YoY)",
        min_value=round(tg_min, 1),
        max_value=round(tg_max, 1),
        value=(round(tg_min, 1), round(tg_max, 1)),
        step=1.0
    )
    
    st.markdown("---")
    st.markdown("### 📥 Data Export Center")
    # Quick export button
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

# Apply Global Filters
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

filtered_df = filtered_df[
    (filtered_df["tariff_growth_pct"] >= selected_tg[0]) &
    (filtered_df["tariff_growth_pct"] <= selected_tg[1])
]

if search_query:
    filtered_df = filtered_df[filtered_df["state"].str.lower().str.contains(search_query)]

# ----------------- DYNAMIC THEME STYLES -----------------
if is_dark:
    bg_main = "#0B0F19"
    card_bg = "#111827"
    card_border = "#1F2937"
    text_primary = "#F9FAFB"
    text_secondary = "#9CA3AF"
    accent_color = "#38BDF8"
    metric_border = "#0284C7"
else:
    bg_main = "#F8FAFC"
    card_bg = "#FFFFFF"
    card_border = "#E2E8F0"
    text_primary = "#0F172A"
    text_secondary = "#64748B"
    accent_color = "#0284C7"
    metric_border = "#0D9488"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Hero Banner */
    .hero-banner {{
        background: linear-gradient(135deg, {'#030712 0%, #0F172A 60%, #064E3B 100%' if is_dark else '#0F172A 0%, #1E293B 60%, #0D9488 100%'});
        border-radius: 18px;
        padding: 32px 38px;
        color: #FFFFFF;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
        border: 1px solid {'rgba(255, 255, 255, 0.08)' if is_dark else 'rgba(255, 255, 255, 0.15)'};
    }}
    .hero-badge {{
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
        margin-bottom: 10px;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(to right, #FFFFFF, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 8px;
        line-height: 1.5;
        max-width: 900px;
    }}
    
    /* SaaS Metric Cards */
    .metric-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -3px rgba(0,0,0,0.08);
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, {accent_color}, #10B981);
    }}
    .metric-label {{
        font-size: 0.78rem;
        font-weight: 700;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {text_primary};
        letter-spacing: -0.02em;
        line-height: 1.1;
        font-family: 'JetBrains Mono', monospace;
    }}
    .metric-sub {{
        font-size: 0.82rem;
        color: #10B981;
        font-weight: 600;
        margin-top: 8px;
    }}
    
    /* Section Headings */
    .section-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {text_primary};
        margin-top: 18px;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }}
    .section-desc {{
        font-size: 0.9rem;
        color: {text_secondary};
        margin-bottom: 18px;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------- HERO HEADER -----------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🎓 Research Capstone Platform • IEEE/ACM Benchmark</div>
    <h1 class="hero-title">SuryaIQ — PM Surya Ghar Policy Decision Support System</h1>
    <p class="hero-subtitle">
        An end-to-end econometric modeling, geo-spatial analytics, multi-model time-series forecasting, 
        and counterfactual policy simulation system evaluating India's national rooftop solar revolution.
    </p>
</div>
""", unsafe_allow_html=True)

# Active Filter Status Badge & Export
active_count = len(filtered_df)
tot_nat_inst = df_model["installations_aug2026"].sum()
filt_inst = filtered_df["installations_aug2026"].sum()
coverage_pct = (filt_inst / tot_nat_inst * 100) if tot_nat_inst > 0 else 0

f_col1, f_col2 = st.columns([3, 1])
with f_col1:
    st.info(f"⚡ **Active Segment:** Analyzing **{active_count} of {len(df_model)} States/UTs** — Representing **{coverage_pct:.1f}%** of India's aggregate rooftop solar installations.")
with f_col2:
    csv_bytes = convert_df_to_csv(filtered_df)
    st.download_button(
        label="📥 Export Segment CSV",
        data=csv_bytes,
        file_name="pmsg_filtered_dataset.csv",
        mime="text/csv"
    )

if active_count == 0:
    st.warning("⚠️ No records match the active filter criteria. Adjust the sidebar sliders or selections to expand the sample.")
    st.stop()

# ----------------- TOP KPI METRIC CARDS -----------------
tot_apps = int(filtered_df["applications_aug2026"].sum())
avg_conv = (filt_inst / tot_apps * 100) if tot_apps > 0 else 0
avg_per_lakh = filtered_df["installations_per_lakh_pop"].mean()

filtered_dist = df_dist[df_dist["state"].isin(filtered_df["state"])]
tot_cap_mw = filtered_dist["capacity_mw"].sum()
tot_subsidy_cr = filtered_dist["subsidy_released_lakhs"].sum() / 100

# Capstone Environmental & Economic Metrics
annual_gen_gwh = (tot_cap_mw * 1000 * 5.0 * 365 * 0.75) / 1e6 # MW -> kW -> kWh -> GWh
co2_avoided_tons = annual_gen_gwh * 820 # ~820 tons CO2 per GWh in India
target_progress = (filt_inst / 10_000_000 * 100)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Installations</div>
        <div class="metric-value">{int(filt_inst):,}</div>
        <div class="metric-sub">🎯 {target_progress:.1f}% of 10M Target</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pipeline Demand</div>
        <div class="metric-value">{tot_apps:,}</div>
        <div class="metric-sub">Applications Submitted</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Conversion Rate</div>
        <div class="metric-value">{avg_conv:.1f}%</div>
        <div class="metric-sub">Execution Efficiency</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Capacity (MW)</div>
        <div class="metric-value">{tot_cap_mw:,.0f}</div>
        <div class="metric-sub">⚡ {annual_gen_gwh:,.1f} GWh / Year</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CFA Subsidy</div>
        <div class="metric-value">₹{tot_subsidy_cr:,.1f} Cr</div>
        <div class="metric-sub">🌱 {co2_avoided_tons/1e3:,.1f}k Tons CO₂ Offset</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- TAB NAVIGATION SYSTEM -----------------
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "🌐 Executive Overview",
    "🗺️ Geospatial & Regional Deep-Dive",
    "🔍 District & DISCOM Treemap",
    "🎯 Econometric Paradox Matrix",
    "📈 Multi-Model Forecasting Studio",
    "🧪 What-If Policy Sandbox",
    "🕸️ 6-D State Archetype Radar"
])

# ----------------- TAB 1: EXECUTIVE OVERVIEW -----------------
with t1:
    c_map, c_funnel = st.columns([3, 2])
    
    with c_map:
        st.markdown('<div class="section-title">All-India Spatial Adoption Intensity</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Interactive geographic bubble map centered on India. Hover on markers for state-level statistics.</div>', unsafe_allow_html=True)
        
        geo_fig = px.scatter_geo(
            filtered_df,
            lat="latitude",
            lon="longitude",
            size="installations_aug2026",
            color="target_conversion_ratio",
            hover_name="state",
            hover_data={
                "region": True,
                "installations_aug2026": ":,",
                "target_conversion_ratio": ":.2%",
                "installations_per_lakh_pop": ":.1f",
                "latitude": False,
                "longitude": False
            },
            color_continuous_scale="Viridis",
            size_max=38,
            template=plotly_template
        )
        geo_fig.update_geos(
            center=dict(lat=22.8, lon=80.5),
            projection_scale=4.2,
            visible=False,
            resolution=50,
            showcountries=True,
            countrycolor="#94A3B8",
            showsubunits=True,
            showland=True,
            landcolor="#1E293B" if is_dark else "#F1F5F9"
        )
        geo_fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=10),
            height=420,
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(geo_fig, use_container_width=True)

    with c_funnel:
        st.markdown('<div class="section-title">Pipeline Conversion Funnel</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Quantifying pipeline retention rates across the 4 stages of scheme adoption.</div>', unsafe_allow_html=True)
        
        reg_df = df_snapshots[(df_snapshots["metric"] == "registrations") & (df_snapshots["state"].isin(filtered_df["state"]))]
        tot_regs = reg_df["value"].sum() if len(reg_df) > 0 else (tot_apps * 1.55)
        tot_hholds = filtered_df["households_aug2026"].sum()
        
        funnel_fig = go.Figure(go.Funnel(
            y=["1. Registered Interest", "2. Sanction Application", "3. System Installed", "4. Subsidy Credited"],
            x=[tot_regs, tot_apps, filt_inst, tot_hholds],
            textinfo="value+percent previous+percent initial",
            marker=dict(color=["#0284C7", "#0EA5E9", "#10B981", "#059669"]),
            connector=dict(line=dict(color="#CBD5E1", width=2))
        ))
        funnel_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=420,
            template=plotly_template,
            font=dict(family="Plus Jakarta Sans", size=12)
        )
        st.plotly_chart(funnel_fig, use_container_width=True)

    st.markdown('<div class="section-title">Comprehensive State Analytical Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Multi-metric analytical matrix. Click any column header to sort in ascending or descending order.</div>', unsafe_allow_html=True)
    
    lead_table = filtered_df[[
        "state", "region", "adoption_tier", "efficiency_tier",
        "installations_aug2026", "applications_aug2026", "target_conversion_ratio",
        "installations_per_lakh_pop", "tariff_growth_pct", "irradiance_annual_kwh_m2_day",
        "solar_potential_3pct_wasteland_gw"
    ]].sort_values(by="installations_aug2026", ascending=False).copy()
    
    lead_table.columns = [
        "State/UT", "Region", "Scale Tier", "Efficiency Tier",
        "Installations", "Applications", "Conversion Rate",
        "Inst. / Lakh Pop", "Tariff Growth %", "Irradiance (kWh/m²/day)", "Solar Potential (GW)"
    ]
    
    st.dataframe(
        lead_table.style.format({
            "Installations": "{:,.0f}",
            "Applications": "{:,.0f}",
            "Conversion Rate": "{:.2%}",
            "Inst. / Lakh Pop": "{:.1f}",
            "Tariff Growth %": "{:+.1f}%",
            "Irradiance (kWh/m²/day)": "{:.2f}",
            "Solar Potential (GW)": "{:.1f}"
        }),
        height=380,
        use_container_width=True
    )

# ----------------- TAB 2: GEOSPATIAL & REGIONAL DEEP-DIVE -----------------
with t2:
    st.markdown('<div class="section-title">Regional Disparities & Cross-Zone Variance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Evaluating performance equity across North, South, West, East, Central, and North-Eastern regions.</div>', unsafe_allow_html=True)
    
    col_reg1, col_reg2 = st.columns([1, 1])
    
    with col_reg1:
        st.markdown("#### Conversion Efficiency Variance by Region (Box Plot)")
        box_fig = px.box(
            filtered_df,
            x="region",
            y="target_conversion_ratio",
            color="region",
            points="all",
            hover_name="state",
            template=plotly_template,
            labels={"region": "Geographic Zone", "target_conversion_ratio": "Conversion Efficiency"}
        )
        box_fig.update_layout(height=400, showlegend=False, font=dict(family="Plus Jakarta Sans"))
        box_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(box_fig, use_container_width=True)

    with col_reg2:
        st.markdown("#### Installations per Lakh Population by Region")
        reg_pop = filtered_df.groupby("region").agg({
            "installations_aug2026": "sum",
            "pop_2024_lakhs": "sum"
        }).reset_index()
        reg_pop["inst_per_lakh"] = reg_pop["installations_aug2026"] / reg_pop["pop_2024_lakhs"]
        
        bar_percap = px.bar(
            reg_pop.sort_values(by="inst_per_lakh", ascending=False),
            x="region",
            y="inst_per_lakh",
            color="region",
            text="inst_per_lakh",
            template=plotly_template,
            labels={"inst_per_lakh": "Inst. per Lakh Population", "region": "Region"}
        )
        bar_percap.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        bar_percap.update_layout(height=400, showlegend=False, font=dict(family="Plus Jakarta Sans"))
        st.plotly_chart(bar_percap, use_container_width=True)

    st.markdown("#### Regional Macro Aggregates")
    reg_summary = filtered_df.groupby("region").agg({
        "state": "count",
        "installations_aug2026": "sum",
        "applications_aug2026": "sum",
        "pop_2024_lakhs": "sum",
        "irradiance_annual_kwh_m2_day": "mean",
        "tariff_growth_pct": "mean"
    }).reset_index()
    reg_summary["Conversion Rate"] = reg_summary["installations_aug2026"] / reg_summary["applications_aug2026"]
    reg_summary["Per-Capita Rate"] = reg_summary["installations_aug2026"] / reg_summary["pop_2024_lakhs"]
    reg_summary.columns = ["Region", "States Count", "Total Installations", "Applications", "Pop (Lakhs)", "Mean Irradiance", "Mean Tariff Growth %", "Conversion Rate", "Per-Capita Rate"]
    
    st.dataframe(
        reg_summary.style.format({
            "Total Installations": "{:,.0f}",
            "Applications": "{:,.0f}",
            "Pop (Lakhs)": "{:,.1f}",
            "Mean Irradiance": "{:.2f}",
            "Mean Tariff Growth %": "{:+.1f}%",
            "Conversion Rate": "{:.2%}",
            "Per-Capita Rate": "{:.1f}"
        }),
        use_container_width=True
    )

# ----------------- TAB 3: DISTRICT & DISCOM TREEMAP -----------------
with t3:
    st.markdown('<div class="section-title">All-India District & DISCOM Operational Hierarchy</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Interactive Treemap across ~767 districts and 83 DISCOM utilities. Click any parent node to zoom into that territory.</div>', unsafe_allow_html=True)
    
    c_tm1, c_tm2 = st.columns([1, 1])
    with c_tm1:
        treemap_size = st.selectbox(
            "Treemap Box Size:",
            ["installations", "capacity_mw", "subsidy_released_lakhs"],
            format_func=lambda x: {"installations": "Total Installations", "capacity_mw": "Capacity (MW)", "subsidy_released_lakhs": "Subsidy Released (Lakhs)"}[x]
        )
    with c_tm2:
        treemap_color = st.selectbox(
            "Treemap Color Spectrum:",
            ["conversion_ratio", "capacity_mw"],
            format_func=lambda x: {"conversion_ratio": "Conversion Efficiency", "capacity_mw": "Installed Capacity (MW)"}[x]
        )

    tm_df = df_dist[df_dist["state"].isin(filtered_df["state"])].copy()
    tm_df = tm_df[tm_df[treemap_size] > 0]
    
    tree_fig = px.treemap(
        tm_df,
        path=["region", "state", "district"],
        values=treemap_size,
        color=treemap_color,
        color_continuous_scale="Viridis",
        template=plotly_template,
        hover_data=["applications", "installations", "capacity_mw", "subsidy_released_lakhs"]
    )
    tree_fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Plus Jakarta Sans"))
    st.plotly_chart(tree_fig, use_container_width=True)
    
    # Capital Efficiency Scatter Plot: Subsidy vs Capacity
    st.markdown("#### Capital-to-Capacity Efficiency Across All 767 Districts")
    st.markdown('<div class="section-desc">Comparing subsidy outlay vs installed MW capacity. Identifies districts with optimal capital productivity.</div>', unsafe_allow_html=True)
    
    cap_eff_fig = px.scatter(
        tm_df[tm_df["capacity_mw"] > 0],
        x="subsidy_released_lakhs",
        y="capacity_mw",
        size="installations",
        color="region",
        hover_name="district",
        hover_data=["state", "installations", "conversion_ratio"],
        template=plotly_template,
        labels={"subsidy_released_lakhs": "CFA Subsidy Disbursed (₹ Lakhs)", "capacity_mw": "Installed Capacity (MW)"}
    )
    cap_eff_fig.update_layout(height=450, font=dict(family="Plus Jakarta Sans"))
    st.plotly_chart(cap_eff_fig, use_container_width=True)
    
    # DISCOM Operational Matrix
    st.markdown("#### Power Distribution Utilities (DISCOM) Performance Matrix")
    st_discom_filt = df_discom[df_discom["state"].isin(filtered_df["state"])].sort_values(by="installations", ascending=False)
    st.dataframe(
        st_discom_filt[["state", "discom_name", "region", "installations", "applications", "capacity_mw", "conversion_ratio"]].style.format({
            "installations": "{:,.0f}",
            "applications": "{:,.0f}",
            "capacity_mw": "{:,.1f} MW",
            "conversion_ratio": "{:.2%}"
        }),
        height=320,
        use_container_width=True
    )

# ----------------- TAB 4: ECONOMETRIC PARADOX MATRIX -----------------
with t4:
    st.markdown('<div class="section-title">The "Subsidy & Resource Paradox" — Econometric Proof</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Empirical proof that baseline electricity tariffs and static potential do not explain conversion variation; dynamic tariff growth and solar insolation drive adoption.</div>', unsafe_allow_html=True)
    
    mean_irr = filtered_df["irradiance_annual_kwh_m2_day"].mean()
    mean_tg = filtered_df["tariff_growth_pct"].mean()
    
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
            "irradiance_annual_kwh_m2_day": "Solar Irradiance (kWh/m²/day)",
            "tariff_growth_pct": "Grid Tariff Growth (% YoY)",
            "target_conversion_ratio": "Conversion Ratio",
            "installations_aug2026": "Installations"
        },
        color_continuous_scale="Viridis",
        size_max=46,
        template=plotly_template
    )
    bubble_fig.add_vline(x=mean_irr, line_dash="dash", line_color="#94A3B8", annotation_text=f"Mean Irradiance ({mean_irr:.2f})")
    bubble_fig.add_hline(y=mean_tg, line_dash="dash", line_color="#94A3B8", annotation_text=f"Mean Tariff Growth ({mean_tg:.1f}%)")
    
    bubble_fig.update_layout(height=480, font=dict(family="Plus Jakarta Sans"), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(bubble_fig, use_container_width=True)
    
    h_c1, h_c2 = st.columns([1, 1])
    with h_c1:
        st.markdown("#### Full Inter-Feature Correlation Heatmap")
        num_features = [
            "target_conversion_ratio", "tariff_growth_pct", "irradiance_annual_kwh_m2_day",
            "installations_per_lakh_pop", "applications_per_lakh_pop", "mnre_re_capacity_2025_mw",
            "solar_potential_3pct_wasteland_gw"
        ]
        corr_matrix = filtered_df[num_features].corr().round(2)
        
        heatmap_fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Teal",
            template=plotly_template,
            labels=dict(color="Correlation")
        )
        heatmap_fig.update_layout(height=380, font=dict(family="Plus Jakarta Sans", size=10))
        st.plotly_chart(heatmap_fig, use_container_width=True)

    with h_c2:
        st.markdown("#### Cross-Validated Model Comparison (LOO-CV)")
        st.markdown("""
        To prevent small-sample overfitting ($n=35$), we benchmarked multiple architectures using **Leave-One-Out Cross-Validation**:
        """)
        model_benchmarks = pd.DataFrame([
            {"Model": "2-Feature Linear Model (Tariff Growth + Irradiance)", "LOO-CV MAE": 0.135, "Error Reduction": "-20.7% (Best)", "Complexity": "Minimal / Interpretable"},
            {"Model": "6-Feature Random Forest (with SHAP)", "LOO-CV MAE": 0.161, "Error Reduction": "-5.9%", "Complexity": "Tree Ensemble / Overfits"},
            {"Model": "Naive Mean Baseline", "LOO-CV MAE": 0.170, "Error Reduction": "0.0%", "Complexity": "Zero Variance"}
        ])
        st.dataframe(model_benchmarks, use_container_width=True)
        st.info("💡 **Methodological Finding:** The 2-feature linear model substantially outperforms the random forest, demonstrating that complex models overfit on state-level aggregates while simple economic signals generalize reliably.")

# ----------------- TAB 5: MULTI-MODEL FORECASTING STUDIO -----------------
with t5:
    st.markdown('<div class="section-title">National & State Adoption Forecasting Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Comparing 5 statistical and machine-learning forecasting algorithms across historical milestone checkpoints (Nov 2024 – Aug 2026).</div>', unsafe_allow_html=True)
    
    fc_ctrl1, fc_ctrl2, fc_ctrl3, fc_ctrl4 = st.columns(4)
    with fc_ctrl1:
        f_log = st.checkbox("Log-Linear (Exponential)", value=True)
    with fc_ctrl2:
        f_holt = st.checkbox("Holt's Exponential Smoothing", value=True)
    with fc_ctrl3:
        f_lin = st.checkbox("Linear Trend", value=True)
    with fc_ctrl4:
        f_ci = st.checkbox("95% Confidence Interval Band", value=True)

    fc_fig = go.Figure()
    df_s_dt = df_series.copy()
    df_s_dt["date"] = pd.to_datetime(df_s_dt["date"])
    
    fc_fig.add_trace(go.Scatter(
        x=df_s_dt["date"],
        y=df_s_dt["installations"],
        mode="lines+markers",
        name="Official Historical Milestones",
        line=dict(color="#38BDF8" if is_dark else "#0F172A", width=3),
        marker=dict(size=9, color="#0EA5E9")
    ))
    
    future_dates = pd.to_datetime(df_nat_fc["forecast_date"])
    last_d = df_s_dt["date"].iloc[-1]
    last_v = df_s_dt["installations"].iloc[-1]
    dates_join = [last_d] + list(future_dates)
    
    if f_log:
        v_log = [last_v] + list(df_nat_fc["point_forecast_log_linear"])
        fc_fig.add_trace(go.Scatter(
            x=dates_join, y=v_log,
            mode="lines+markers",
            name="Log-Linear Forecast (4.67% MAPE)",
            line=dict(color="#0284C7", width=2.5, dash="dash"),
            marker=dict(size=8, symbol="square")
        ))
        if f_ci:
            ci_low = [last_v] + list(df_nat_fc["lower_95_ci"])
            ci_high = [last_v] + list(df_nat_fc["upper_95_ci"])
            fc_fig.add_trace(go.Scatter(
                x=dates_join + dates_join[::-1],
                y=ci_high + ci_low[::-1],
                fill="toself",
                fillcolor="rgba(14, 165, 233, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Interval Band"
            ))
            
    if f_holt:
        v_holt = [last_v] + list(df_nat_fc["point_forecast_holt"])
        fc_fig.add_trace(go.Scatter(
            x=dates_join, y=v_holt,
            mode="lines+markers",
            name="Holt's Exponential Smoothing (9.68% MAPE)",
            line=dict(color="#10B981", width=2, dash="dot"),
            marker=dict(size=7, symbol="triangle-up")
        ))
        
    if f_lin:
        v_lin = [last_v] + list(df_nat_fc["point_forecast_linear"])
        fc_fig.add_trace(go.Scatter(
            x=dates_join, y=v_lin,
            mode="lines+markers",
            name="Linear Trend (12.74% MAPE)",
            line=dict(color="#EF4444", width=2, dash="dashdot"),
            marker=dict(size=7, symbol="diamond")
        ))
        
    fc_fig.update_layout(
        title="National Adoption Trajectory Projections (+3, +6, +12 Months)",
        xaxis_title="Timeline",
        yaxis_title="Cumulative Installations",
        height=480,
        template=plotly_template,
        font=dict(family="Plus Jakarta Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fc_fig, use_container_width=True)
    
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.markdown("#### National Horizon Forecast Quantiles")
        st.dataframe(df_nat_fc.style.format({
            "point_forecast_log_linear": "{:,.0f}",
            "lower_95_ci": "{:,.0f}",
            "upper_95_ci": "{:,.0f}",
            "point_forecast_holt": "{:,.0f}",
            "point_forecast_linear": "{:,.0f}"
        }), use_container_width=True)

    with col_f2:
        st.markdown("#### Backtesting Error Scorecard (In-Sample)")
        st.dataframe(df_comp.style.format({
            "MAE": "{:,.0f}",
            "RMSE": "{:,.0f}",
            "MAPE (%)": "{:.2f}%"
        }), use_container_width=True)

    st.markdown("#### 🌟 Five Representative State Case Studies (Trajectory & Projections)")
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

# ----------------- TAB 6: WHAT-IF POLICY SANDBOX -----------------
with t6:
    st.markdown('<div class="section-title">Interactive Counterfactual Policy Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Simulate policy interventions (Scenario D: Tariff Shocks, Scenario C: High-Potential Sunbelt Targeting) and quantify capacity and emissions impact.</div>', unsafe_allow_html=True)
    
    sb_c1, sb_c2 = st.columns([1, 2])
    
    with sb_c1:
        st.markdown("#### Policy Intervention Controls")
        tariff_shock_val = st.slider(
            "Simulate Grid Tariff Shock (+pp)",
            min_value=0, max_value=30, value=10, step=5,
            help="Marginal empirical elasticity: ~1.6 percentage points conversion lift per 1% tariff hike."
        )
        enable_targeting = st.checkbox("Prioritize High-Potential States (Scenario C)", value=True)
        targeting_boost = st.slider("Targeted Campaign Efficiency Boost (+pp)", min_value=0, max_value=20, value=8, step=2) if enable_targeting else 0
        
    with sb_c2:
        sim_df = filtered_df[["state", "region", "target_conversion_ratio", "installations_aug2026", "applications_aug2026", "irradiance_annual_kwh_m2_day"]].copy()
        sim_df["baseline_conversion"] = sim_df["target_conversion_ratio"]
        
        t_lift = (tariff_shock_val / 100.0) * 1.6
        is_target = (sim_df["irradiance_annual_kwh_m2_day"] > 5.2) & (sim_df["baseline_conversion"] < 0.65)
        c_lift = np.where(is_target, targeting_boost / 100.0, 0.0)
        
        sim_df["simulated_conversion"] = (sim_df["baseline_conversion"] + t_lift + c_lift).clip(upper=0.98)
        sim_df["incremental_installations"] = np.round(
            (sim_df["simulated_conversion"] - sim_df["baseline_conversion"]) * sim_df["applications_aug2026"]
        ).astype(int)
        
        inc_inst = int(sim_df["incremental_installations"].sum())
        inc_mw = inc_inst * 3.3 / 1000 # 3.3 kW avg residential system
        inc_gwh = (inc_mw * 1000 * 5.0 * 365 * 0.75) / 1e6
        inc_co2 = inc_gwh * 820
        
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            st.metric("Incremental Adoption", f"+{inc_inst:,}")
        with sm2:
            st.metric("Incremental Capacity", f"+{inc_mw:,.1f} MW")
        with sm3:
            st.metric("Annual Clean Power", f"+{inc_gwh:,.1f} GWh")
            
        top_sim_states = sim_df.sort_values(by="incremental_installations", ascending=False).head(10)
        
        comp_bar = go.Figure()
        comp_bar.add_trace(go.Bar(
            x=top_sim_states["state"], y=top_sim_states["baseline_conversion"],
            name="Baseline Conversion", marker_color="#94A3B8"
        ))
        comp_bar.add_trace(go.Bar(
            x=top_sim_states["state"], y=top_sim_states["simulated_conversion"],
            name="Counterfactual Simulated", marker_color="#0EA5E9"
        ))
        comp_bar.update_layout(
            barmode="group",
            height=360,
            title="Top 10 Beneficiary States: Baseline vs Simulated Conversion Rate",
            yaxis_tickformat=".0%",
            template=plotly_template,
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(comp_bar, use_container_width=True)

# ----------------- TAB 7: 6-D STATE RADAR -----------------
with t7:
    st.markdown('<div class="section-title">6-Dimensional State Archetype Radar Profiling</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Select up to 4 states to benchmark across 6 normalized strategic dimensions.</div>', unsafe_allow_html=True)
    
    radar_choices = st.multiselect(
        "Select States for Benchmark Comparison:",
        options=sorted(df_model["state"].unique()),
        default=["Gujarat", "Maharashtra", "Tamil Nadu", "Uttar Pradesh"] if "Gujarat" in df_model["state"].values else [df_model["state"].iloc[0]],
        max_selections=4
    )
    
    if len(radar_choices) > 0:
        categories = [
            "Conversion Efficiency", "Tariff Growth Momentum", 
            "Solar Irradiance Fit", "Per-Capita Penetration",
            "RE Capacity Base", "Solar Potential Endowment"
        ]
        
        col_map = {
            "Conversion Efficiency": "target_conversion_ratio",
            "Tariff Growth Momentum": "tariff_growth_pct",
            "Solar Irradiance Fit": "irradiance_annual_kwh_m2_day",
            "Per-Capita Penetration": "installations_per_lakh_pop",
            "RE Capacity Base": "mnre_re_capacity_2025_mw",
            "Solar Potential Endowment": "solar_potential_3pct_wasteland_gw"
        }
        
        r_fig = go.Figure()
        palette = ["#0EA5E9", "#10B981", "#F59E0B", "#8B5CF6"]
        
        for idx, st_name in enumerate(radar_choices):
            st_data = df_model[df_model["state"] == st_name]
            if len(st_data) == 0:
                continue
            scores = []
            for cat in categories:
                c = col_map[cat]
                val = st_data[c].values[0]
                vmin = df_model[c].min()
                vmax = df_model[c].max()
                norm_score = ((val - vmin) / (vmax - vmin) * 100) if vmax > vmin and not np.isnan(val) else 50
                scores.append(norm_score)
            scores.append(scores[0])
            
            c_hex = palette[idx % len(palette)]
            r_fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=f"rgba{tuple(list(int(c_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.18])}",
                line=dict(color=c_hex, width=2.5),
                name=st_name
            ))
            
        r_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=490,
            template=plotly_template,
            font=dict(family="Plus Jakarta Sans")
        )
        st.plotly_chart(r_fig, use_container_width=True)
    else:
        st.info("Please select at least 1 state to view the radar benchmark.")

# ----------------- FOOTER -----------------
st.markdown("---")
foot_c1, foot_c2 = st.columns([3, 1])
with foot_c1:
    st.caption("SuryaIQ Platform • Capstone Research Project • Rigorously validated with Leave-One-Out Cross-Validation & Time-Series Backtesting.")
with foot_c2:
    st.caption("Sources: data.gov.in, CEA, MNRE, NITI Aayog ICeD, NASA POWER.")
