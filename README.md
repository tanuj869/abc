# PM Surya Ghar — Data Science Project (SuryaIQ)

A comprehensive data science, econometric modeling, and policy forecasting platform analyzing India's **PM Surya Ghar: Muft Bijli Yojana**, built in strict compliance with the project guidelines.

---

## 1. Quick Setup & Dashboard Launch

```bash
# 1. Activate virtual environment
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Interactive Policy Dashboard
streamlit run dashboard/app.py --server.port=8502
```
The interactive dashboard will be live at `http://localhost:8502`.

---

## 2. Project Architecture

```
pmsg-project/
├── data/
│   ├── raw/                       10 verified Tier-1 official source files
│   └── processed/                  Master analytical tables, forecasts, & figures
│       ├── modeling_table.csv      Cross-sectional dataset with per-capita metrics
│       ├── district_level_pmsg.csv All-India district progress (~767 districts)
│       ├── discom_level_pmsg.csv   DISCOM operational records (~83 utilities)
│       ├── maharashtra_districts.csv Maharashtra CFA beneficiaries drill-down
│       ├── national_forecast_detailed.csv 3/6/12-month projections with 95% CIs
│       └── forecast_comprehensive_comparison.csv Model evaluation (MAE, RMSE, MAPE)
├── dashboard/
│   └── app.py                     Interactive Streamlit policy analytics platform
├── notebooks/
│   ├── 01_panel_build.ipynb       Ingests raw/, builds canonical panel
│   ├── 02_eda.ipynb               Exploratory analysis, per-capita metrics, correlation
│   ├── 03_modeling.ipynb          Regression vs Random Forest, SHAP explainability
│   ├── 04_forecasting.ipynb       National time-series projections
│   └── 05_scenario_simulation.ipynb Counterfactual policy simulation (tariff shocks)
├── src/
│   ├── state_mapping.py           Canonical 36-state normalization dictionary
│   ├── population.py              2024 state population projections (MoHFW)
│   ├── build_extended_panel.py    District & demographic ingestion pipeline
│   ├── rebuild_all_panels.py      Master synchronization & merged-cell resolution
│   └── forecasting_engine.py      Multi-model forecasting & 5-state case studies
├── DATA_DICTIONARY.md             Formal data dictionary & dataset register
├── PMSG_Research_Report.docx      4-6 page research-style draft paper
└── README.md
```

---

## 3. Key Findings & Empirical Takeaways

1. **The "Subsidy Paradox" Confirmed**:
   * Cross-sectional regression and SHAP analyses prove that baseline electricity tariff levels and technical solar potential are among the *weakest* predictors of rooftop adoption efficiency across states.
   * Instead, **recent electricity tariff growth** ($r = 0.47$) and **mean solar irradiance** ($r = 0.56$) dominate adoption conversion.
2. **Model Parsimony (Linear vs Random Forest)**:
   * A 2-variable linear model (tariff growth + irradiance) generalizes better under Leave-One-Out Cross-Validation ($\text{MAE} = 0.135$, a $20.7\%$ improvement over naive baseline) than a 6-feature Random Forest ($\text{MAE} = 0.161$).
3. **Data Integrity Resolution**:
   * Resolved an Excel merged-cell omission in raw administrative records that previously truncated multi-DISCOM states (e.g. Maharashtra, Gujarat, UP). With multi-DISCOM aggregation, national August 2026 installations correctly tally to **4.47 Million**, exactly aligning with official national milestones.

---

## 4. Forecasting Benchmarks (`04_forecasting.ipynb` & `src/forecasting_engine.py`)

Benchmarked across 6 historical milestones (Nov 2024 – Aug 2026) using chronological backtesting:

| Model | MAE | RMSE | In-Sample MAPE (%) |
|---|---|---|---|
| **Log-Linear (Exponential Growth)** | **94,683** | **103,572** | **4.67%** |
| Holt's Linear Exponential Smoothing | 232,568 | 298,418 | 9.68% |
| Linear Trend Extrapolation | 324,563 | 376,476 | 12.74% |
| ARIMA(1, 1, 0) | 534,490 | 734,870 | 19.30% |
| Naive Baseline | 715,610 | 873,153 | 30.37% |

### National Projections (95% Confidence Intervals)
* **+3 Months (Nov 2026)**: ~5.24 Million (4.62M – 5.95M)
* **+6 Months (Feb 2027)**: ~6.84 Million (6.03M – 7.75M)
* **+12 Months (Aug 2027)**: ~11.61 Million (10.24M – 13.17M)

---

## 5. Five Representative State Case Studies

| State | Nov 2024 Inst. | Aug 2026 Inst. | Conversion % | Inst. / Lakh Pop | +6M Projection | +12M Projection |
|---|---|---|---|---|---|---|
| **Gujarat** | 240,683 | 811,533 | 60.6% | 1,119.7 | 1,090,602 | 1,477,720 |
| **Maharashtra** | 126,344 | 746,100 | 50.4% | 586.7 | 1,235,270 | 2,074,000 |
| **Uttar Pradesh** | 63,124 | 774,506 | 52.8% | 325.1 | 1,654,743 | 3,610,726 |
| **Rajasthan** | 35,468 | 301,362 | 56.6% | 367.8 | 663,529 | 1,493,318 |
| **Tamil Nadu** | 39,211 | 92,186 | 85.9% | 119.8 | 141,955 | 221,231 |

---

## 6. Policy Scenario Simulation (`05_scenario_simulation.ipynb`)

* **Scenario A (Base Case)**: Evaluates observed vs model-fitted conversion rates.
* **Scenario C (High-Potential Targeting)**: Isolates 4 prime high-irradiance, below-median conversion states (Gujarat, Karnataka, Chhattisgarh, Andhra Pradesh) for priority marketing intervention.
* **Scenario D (Tariff Shock)**: Simulates adoption response under $+10\text{pp}$ and $+15\text{pp}$ grid tariff acceleration, indicating a projected $+16\text{pp}$ conversion lift under model assumptions.
