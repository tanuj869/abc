# PM Surya Ghar: Muft Bijli Yojana — Data Dictionary & Dataset Register

**Project:** SuryaIQ / PM Surya Ghar Policy Analytics & Forecasting Platform  
**Guideline Compliance:** Step 1 & Step 2 Deliverable (Section 4 & Section 10)  
**Last Updated:** October 2026  

---

## 1. Dataset Register (Raw Tier-1 Sources)

All raw source files reside in [`data/raw/`](file:///d:/abc/data/raw/). These are official government repositories, parliamentary records, and scientific climatology models.

| # | File Name | Source / Portal | Grain | Date / Session | Description |
|---|---|---|---|---|---|
| 1 | `Registrations__applications__households_benefited.xlsx` | data.gov.in / OGD Platform | State/UT | Rajya Sabha Session 267 (AU 931, 11-Feb-2025) | Official counts of registrations, applications, and households benefited across 36 states/UTs. |
| 2 | `Installations.xlsx` | data.gov.in / OGD Platform | State/UT | Rajya Sabha Session 266 (AU 945, 03-Dec-2024) | State-wise installation counts under PM Surya Ghar. |
| 3 | `District_and_discom_info-PM_surya_ghar.xlsx` | National Rooftop Solar Portal (`pmsuryaghar.gov.in`) | District & DISCOM | 23-Aug-2026 Snapshot | Live administrative data covering ~800 districts and ~119 DISCOMs: applications, installations, capacity (MW), and subsidy released. |
| 4 | `3__District-wise_beneficiaries__Maharashtra__11-12-2024.xlsx` | data.gov.in / OGD Platform | District (MH) | Rajya Sabha Session 266 (AU 2545, 11-Dec-2024) | District-level Central Financial Assistance (CFA) beneficiaries for Maharashtra (36 districts). |
| 5 | `PMSG_Tariff_-_2024.xlsx` | Central Electricity Authority (CEA) | Utility / Slabs | March 2024 Edition | Average rates and duties/taxes for Domestic (1KW, 2KW, 4KW, 6KW, 8KW) and Commercial (2KW). |
| 6 | `PMSG_Tariff_-_2025.xlsx` | Central Electricity Authority (CEA) | Utility / Slabs | March 2025 Edition | Updated domestic tariff books reflecting revision cycles across DISCOMs and states. |
| 7 | `PMSG_ToD_Regime_by_State.xlsx` | Forum of Regulators / State ERCs | State/UT | FY 2024-25 | Assessment of Time-of-Day (ToD) tariff implementation across consumer categories. |
| 8 | `Solar_potential.xlsx` | NITI Aayog India Climate & Energy Dashboard (ICeD) | State/UT | Baseline assessment | State-wise technical solar potential estimated at 3% and 6.69% wasteland availability (GW). |
| 9 | `States_nasa.xlsx` | NASA POWER Climatology (Renewable Energy) | State/UT | 2001–2020 20-Year Baseline | Monthly and annual average daily solar insolation ($kWh/m^2/day$) at centroid coordinates. |
| 10 | `state-wise_RE_and_non-RE_installed_capacity_at_two_points.xlsx` | Ministry of New & Renewable Energy (MNRE) | State/UT | 31-Mar-2018 vs 31-Mar-2025 | Progress in grid-connected Renewable Energy (RE) and Non-RE capacity (MW). |

---

## 2. Processed & Derived Tables Register

All processed tables reside in [`data/processed/`](file:///d:/abc/data/processed/) and are generated programmatically by the notebooks.

| File Name | Generating Script | Grain | Records | Description |
|---|---|---|---|---|
| `pmsg_snapshots_long.csv` | `01_panel_build.ipynb` | State × Snapshot × Metric | 284 | Long-format historical administrative tracking panel. |
| `state_covariates_wide.csv` | `01_panel_build.ipynb` | State/UT | 36 | Wide matrix of structural, physical, policy, and demographic covariates. |
| `modeling_table.csv` | `02_eda.ipynb` | State/UT (Cross-section) | 35 | Regression and ML analysis table joining August 2026 snapshot with covariates. |
| `correlation_table.csv` | `02_eda.ipynb` | Feature level | 13 | Pearson correlation ($r$) and sample size ($n$) vs conversion ratio. |
| `national_installations_series.csv` | `04_forecasting.ipynb` | National Milestone | 6 | Time-series of cumulative installations from official statements (Nov 2024 – Aug 2026). |
| `installations_forecast.csv` | `04_forecasting.ipynb` | Forecast Horizon | 3 | Point forecasts and 95% prediction intervals at +3, +6, and +12 months. |
| `forecast_method_comparison.csv`| `04_forecasting.ipynb` | Model | 3 | In-sample MAPE comparison of extrapolation approaches. |
| `scenario_a_base_case.csv` | `05_scenario_simulation.ipynb`| State/UT | 28 | Baseline predicted conversion ratio vs actuals. |
| `scenario_c_targeting.csv` | `05_scenario_simulation.ipynb`| State/UT | 4 | High-potential, below-median conversion states targeted for campaign intervention. |
| `scenario_d_tariff_shock.csv` | `05_scenario_simulation.ipynb`| State/UT | 28 | Simulated conversion ratios under +10pp and +15pp tariff growth shocks. |
| `scenario_e_combined_partial.csv` | `05_scenario_simulation.ipynb`| State/UT | 4 | Priority target states with simulated tariff acceleration. |
| `utility_level_tariff.csv` | `01_panel_build.ipynb` | Utility × Year | 10 | Tariff values for sub-state utility providers (e.g. Torrent Power, BEST). |

---

## 3. Detailed Data Dictionary (Field Definitions)

### 3.1 Primary Outcome & Performance Metrics

| Column Name | Source File | Data Type | Units / Scale | Definition & Business Logic |
|---|---|---|---|---|
| `state` | Canonical Standard | String | Categorical | Normalized state or union territory name (from [`src/state_mapping.py`](file:///d:/abc/src/state_mapping.py)). |
| `registrations` | `Registrations...xlsx` | Integer | Count | Cumulative consumer registrations submitted on the National Portal. |
| `applications` | `District_and_discom_info...xlsx` | Integer | Count | Formal applications submitted with rooftop sanction requests. |
| `installations` | `District_and_discom_info...xlsx` | Integer | Count | Successfully completed and commissioned rooftop solar systems. |
| `households_benefited` | `Registrations...xlsx` | Integer | Count | Households that have received central financial assistance (subsidy) credit. |
| `target_conversion_ratio` | Derived | Float (0.0 – 1.0) | Ratio | $\frac{\text{installations}}{\text{applications}}$. The primary dependent variable measuring operational execution efficiency. |
| `installations_per_lakh_pop` | Derived | Float | Ratio | $\frac{\text{installations}}{\text{Population (in Lakhs)}}$. Normalized per-capita adoption volume. |
| `applications_per_lakh_pop` | Derived | Float | Ratio | $\frac{\text{applications}}{\text{Population (in Lakhs)}}$. Normalized per-capita consumer interest. |
| `capacity_mw` | `District_and_discom_info...xlsx` | Float | Megawatts (MW) | Total aggregate capacity of installed rooftop systems. |
| `subsidy_released_lakhs` | `District_and_discom_info...xlsx` | Float | ₹ Lakhs | Total central financial assistance disbursed to consumers in that state. |

### 3.2 External Covariates

| Column Name | Source | Data Type | Units | Definition |
|---|---|---|---|---|
| `tariff_2024_paise_kwh` | CEA Tariff Book 2024 | Float | Paise/kWh | State average domestic electricity tariff (2 kW consumption slab, including duties/taxes). |
| `tariff_2025_paise_kwh` | CEA Tariff Book 2025 | Float | Paise/kWh | Updated state average domestic electricity tariff for 2025. |
| `tariff_growth_pct` | Derived | Float | Percentage (%) | $\frac{\text{tariff\_2025} - \text{tariff\_2024}}{\text{tariff\_2024}} \times 100$. Measure of escalating grid electricity cost. |
| `irradiance_annual_kwh_m2_day`| NASA POWER | Float | $kWh/m^2/day$ | 20-year climatological mean daily solar insolation incident on a horizontal surface. |
| `solar_potential_gw` | NITI Aayog ICeD | Float | Gigawatts (GW) | Estimated technical solar potential at 3% wasteland utilization. |
| `solar_potential_6pct_gw` | NITI Aayog ICeD | Float | Gigawatts (GW) | Estimated technical solar potential at 6.69% wasteland utilization. |
| `mnre_re_capacity_2018_mw` | MNRE Physical Progress | Float | Megawatts (MW) | Total grid-connected renewable capacity as of March 31, 2018. |
| `mnre_re_capacity_2025_mw` | MNRE Physical Progress | Float | Megawatts (MW) | Total grid-connected renewable capacity as of March 31, 2025. |
| `re_capacity_growth_pct` | Derived | Float | Percentage (%) | Percentage expansion in state RE capacity between 2018 and 2025. |
| `tod_domestic_applicability` | SERC / FoR Reports | Categorical | Binary / Category | Whether Time-of-Day tariff pricing is applicable to domestic rooftop prosumers. |
| `pop_2024_lakhs` | Census / MoHFW Projections | Float | Lakhs ($10^5$) | State population projection for 2024 based on National Commission on Population. |

---

## 4. Normalization Rules & Data Integrity

1. **State Normalization**: All raw text matches are passed through `normalize_state()` in [`src/state_mapping.py`](file:///d:/abc/src/state_mapping.py). 
2. **Double-Counting Prevention**: Sub-state utility rows (e.g. Torrent Power Ahmedabad, BEST Mumbai) are prefixed with `__UTILITY_` and excluded from state totals.
3. **Missingness Handling**: Missing values in covariates (e.g., small UTs lacking tariff records) are preserved as `NaN` rather than zero-filled to prevent bias during regression fitting.
4. **Non-Causal Interpretation**: Model outputs and scenario simulations represent statistical predictions under fitted assumptions, not counterfactual causal claims.
