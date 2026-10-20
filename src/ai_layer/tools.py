"""
LangChain Tool Bindings for PM Surya Ghar Core Data Assets & Models:
Exposes data science pipelines to LangGraph / LangChain agents as type-safe tools.
"""

import os
import sys

# Ensure workspace root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from src.ai_layer.vectorless_rag import rag_engine

PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")

def _load_data():
    model_df = pd.read_csv(os.path.join(PROCESSED_DIR, "modeling_table.csv"))
    states_fc = pd.read_csv(os.path.join(PROCESSED_DIR, "state_case_studies_forecast.csv"))
    nat_fc = pd.read_csv(os.path.join(PROCESSED_DIR, "national_forecast_detailed.csv"))
    return model_df, states_fc, nat_fc

@tool
def get_state_metrics(state_name: str) -> Dict[str, Any]:
    """Retrieve detailed operational, economic, and solar metrics for a specific Indian State or UT."""
    model_df, _, _ = _load_data()
    row = model_df[model_df["state"].str.lower() == state_name.strip().lower()]
    if len(row) == 0:
        # Try substring match
        row = model_df[model_df["state"].str.lower().str.contains(state_name.strip().lower())]
    if len(row) == 0:
        return {"error": f"State '{state_name}' not found in canonical 35-state panel."}
    
    r = row.iloc[0]
    return {
        "state": r["state"],
        "region": r["region"],
        "installations_aug2026": int(r["installations_aug2026"]),
        "applications_aug2026": int(r["applications_aug2026"]),
        "conversion_ratio": round(float(r["target_conversion_ratio"]), 4),
        "installations_per_lakh_pop": round(float(r["installations_per_lakh_pop"]), 1),
        "tariff_growth_pct": round(float(r["tariff_growth_pct"]), 1),
        "irradiance_annual_kwh_m2_day": round(float(r["irradiance_annual_kwh_m2_day"]), 2),
        "solar_potential_3pct_gw": round(float(r["solar_potential_3pct_wasteland_gw"]), 1) if not np.isnan(r["solar_potential_3pct_wasteland_gw"]) else "N/A",
        "adoption_tier": r["adoption_tier"],
        "efficiency_tier": r["efficiency_tier"]
    }

@tool
def compare_states(state_a: str, state_b: str) -> Dict[str, Any]:
    """Perform side-by-side comparative analysis between two Indian States or UTs."""
    m_a = get_state_metrics.invoke({"state_name": state_a})
    m_b = get_state_metrics.invoke({"state_name": state_b})
    if "error" in m_a:
        return m_a
    if "error" in m_b:
        return m_b
    return {
        "comparison": {
            "metric": ["Installations", "Conversion Efficiency", "Per-Capita Rate", "Tariff Growth %", "Solar Irradiance", "Region"],
            state_a: [
                f"{m_a['installations_aug2026']:,}",
                f"{m_a['conversion_ratio']*100:.1f}%",
                f"{m_a['installations_per_lakh_pop']:.1f}",
                f"{m_a['tariff_growth_pct']:+.1f}%",
                f"{m_a['irradiance_annual_kwh_m2_day']} kWh/m²/day",
                m_a["region"]
            ],
            state_b: [
                f"{m_b['installations_aug2026']:,}",
                f"{m_b['conversion_ratio']*100:.1f}%",
                f"{m_b['installations_per_lakh_pop']:.1f}",
                f"{m_b['tariff_growth_pct']:+.1f}%",
                f"{m_b['irradiance_annual_kwh_m2_day']} kWh/m²/day",
                m_b["region"]
            ]
        },
        "higher_conversion_state": state_a if m_a['conversion_ratio'] > m_b['conversion_ratio'] else state_b,
        "higher_volume_state": state_a if m_a['installations_aug2026'] > m_b['installations_aug2026'] else state_b
    }

@tool
def get_adoption_forecast(scope: Optional[str] = "national") -> Dict[str, Any]:
    """Retrieve official time-series adoption projections (+3, +6, +12 months) and 95% confidence intervals."""
    _, states_fc, nat_fc = _load_data()
    scope_clean = (scope or "national").strip().lower()
    
    if scope_clean in ["national", "india", "country", "all", "none"]:
        return {
            "scope": "National (All-India)",
            "benchmark_model": "Log-Linear Growth Model (4.67% In-Sample MAPE)",
            "horizons": [
                {
                    "horizon": "+3 Months (Nov 2026)",
                    "point_forecast": int(nat_fc.loc[0, "point_forecast_log_linear"]),
                    "confidence_interval_95": f"{int(nat_fc.loc[0, 'lower_95_ci']):,} - {int(nat_fc.loc[0, 'upper_95_ci']):,}"
                },
                {
                    "horizon": "+6 Months (Feb 2027)",
                    "point_forecast": int(nat_fc.loc[1, "point_forecast_log_linear"]),
                    "confidence_interval_95": f"{int(nat_fc.loc[1, 'lower_95_ci']):,} - {int(nat_fc.loc[1, 'upper_95_ci']):,}"
                },
                {
                    "horizon": "+12 Months (Aug 2027)",
                    "point_forecast": int(nat_fc.loc[2, "point_forecast_log_linear"]),
                    "confidence_interval_95": f"{int(nat_fc.loc[2, 'lower_95_ci']):,} - {int(nat_fc.loc[2, 'upper_95_ci']):,}"
                }
            ]
        }
    else:
        # Check if state is in 5-state case study table
        row = states_fc[states_fc["state"].str.lower().str.contains(scope_clean)]
        if len(row) > 0:
            r = row.iloc[0]
            return {
                "scope": r["state"],
                "installations_aug2026": int(r["installations_aug2026"]),
                "annualized_growth_rate": f"{r['annualized_growth_pct']:+.1f}%",
                "projections": {
                    "+3 Months": int(r["forecast_3m"]),
                    "+6 Months": int(r["forecast_6m"]),
                    "+12 Months": int(r["forecast_12m"])
                }
            }
        return {
            "scope": scope,
            "message": f"Granular state-level curve fitting is currently calibrated for the 5 benchmark states: Gujarat, Maharashtra, Tamil Nadu, Uttar Pradesh, Rajasthan."
        }

@tool
def run_scenario_simulation(tariff_shock_pct: float = 0.0, subsidy_change_pct: float = 0.0, target_high_potential: bool = True) -> Dict[str, Any]:
    """Execute econometric counterfactual simulation modeling adoption lift or contraction under electricity tariff shocks, subsidy cuts/increases, and sunbelt targeting."""
    model_df, _, _ = _load_data()
    
    # Empirical elasticities derived from cross-sectional econometric modeling
    tariff_lift = (tariff_shock_pct / 100.0) * 1.6
    subsidy_lift = (subsidy_change_pct / 100.0) * 0.45
    
    sim_df = model_df.copy()
    is_target = (sim_df["irradiance_annual_kwh_m2_day"] > 5.2) & (sim_df["target_conversion_ratio"] < 0.65)
    target_lift = np.where(is_target & target_high_potential & (tariff_shock_pct > 0), 0.08, 0.0)
    
    total_delta = tariff_lift + subsidy_lift + target_lift
    sim_df["simulated_conv"] = (sim_df["target_conversion_ratio"] + total_delta).clip(lower=0.05, upper=0.98)
    sim_df["incremental_inst"] = (
        (sim_df["simulated_conv"] - sim_df["target_conversion_ratio"]) * sim_df["applications_aug2026"]
    ).fillna(0).astype(int)
    
    tot_inc = int(sim_df["incremental_inst"].sum())
    inc_mw = round(tot_inc * 3.3 / 1000, 1)
    
    # Payback period calculation
    base_payback = 4.2
    est_payback = round(base_payback * (1.0 - (subsidy_change_pct / 100.0) * 0.7) / (1.0 + (tariff_shock_pct / 100.0) * 0.6), 1)
    
    if tot_inc >= 0:
        top_states = sim_df.sort_values(by="incremental_inst", ascending=False)[["state", "incremental_inst", "simulated_conv"]].head(5)
    else:
        top_states = sim_df.sort_values(by="incremental_inst", ascending=True)[["state", "incremental_inst", "simulated_conv"]].head(5)
        
    return {
        "inputs": {
            "tariff_shock_applied": f"{tariff_shock_pct:+.1f}pp",
            "subsidy_change_applied": f"{subsidy_change_pct:+.1f}%",
            "high_potential_targeting": target_high_potential
        },
        "national_impact": {
            "incremental_installations": tot_inc,
            "incremental_capacity_mw": inc_mw,
            "estimated_annual_clean_power_gwh": round((inc_mw * 1000 * 5.0 * 365 * 0.75) / 1e6, 1),
            "estimated_payback_years": est_payback,
            "baseline_payback_years": base_payback
        },
        "top_affected_states": [
            {"state": r["state"], "incremental_units": int(r["incremental_inst"]), "new_conversion_ratio": f"{r['simulated_conv']*100:.1f}%"}
            for _, r in top_states.iterrows()
        ]
    }

@tool
def search_policy_documents(query: str) -> Dict[str, Any]:
    """Retrieve grounded passages from official PM Surya Ghar policy guidelines, data dictionaries, and research notes using Vectorless RAG."""
    results = rag_engine.search(query, top_k=3)
    return {
        "query": query,
        "retrieved_passages": results
    }

ALL_TOOLS = [get_state_metrics, compare_states, get_adoption_forecast, run_scenario_simulation, search_policy_documents]
