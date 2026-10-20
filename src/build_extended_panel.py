"""
Extended panel builder:
1. Enhances state_covariates_wide.csv with 2024 population (in Lakhs).
2. Cleans and exports district_level_pmsg.csv and discom_level_pmsg.csv from District_and_discom_info-PM_surya_ghar.xlsx.
3. Cleans and exports maharashtra_districts.csv from 3__District-wise_beneficiaries__Maharashtra__11-12-2024.xlsx.
4. Updates modeling_table.csv with per-capita normalization (installations_per_lakh_pop, applications_per_lakh_pop).
"""

import sys
import os
import openpyxl
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("."))
from src.state_mapping import normalize_state, CANONICAL_STATES, get_state_region
from src.population import STATE_POPULATION_2024_LAKHS

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def run_pipeline():
    print("--- 1. Updating state_covariates_wide.csv with Population ---")
    cov_path = os.path.join(PROCESSED_DIR, "state_covariates_wide.csv")
    cov_df = pd.read_csv(cov_path)
    cov_df["pop_2024_lakhs"] = cov_df["state"].map(STATE_POPULATION_2024_LAKHS)
    cov_df.to_csv(cov_path, index=False)
    print(f"Updated {cov_path} with pop_2024_lakhs. Non-null count: {cov_df['pop_2024_lakhs'].notnull().sum()}/36")

    print("\n--- 2. Ingesting District & DISCOM Data ---")
    dist_file = os.path.join(RAW_DIR, "District_and_discom_info-PM_surya_ghar.xlsx")
    
    # Process District Sheet
    df_dist_raw = pd.read_excel(dist_file, sheet_name="District info")
    # Forward-fill merged State cells
    df_dist_raw["State"] = df_dist_raw["State"].ffill()
    df_dist_raw = df_dist_raw.dropna(subset=["District"])
    df_dist_raw["canonical_state"] = df_dist_raw["State"].apply(normalize_state)
    # Filter out drop rows if any
    df_dist_raw = df_dist_raw[~df_dist_raw["canonical_state"].str.startswith("__DROP")]
    
    # Rename columns to snake_case
    dist_cols = {
        "canonical_state": "state",
        "District": "district",
        "Application (no)": "applications",
        "Installation (no)": "installations",
        "Household covered (no)": "households_covered",
        "Installation capacity (MW)": "capacity_mw",
        "Subsidy released (lakhs)": "subsidy_released_lakhs"
    }
    df_dist_clean = df_dist_raw[list(dist_cols.keys())].rename(columns=dist_cols)
    for col in ["applications", "installations", "households_covered", "capacity_mw", "subsidy_released_lakhs"]:
        df_dist_clean[col] = pd.to_numeric(df_dist_clean[col], errors="coerce").fillna(0)
    
    df_dist_clean["conversion_ratio"] = np.where(
        df_dist_clean["applications"] > 0,
        df_dist_clean["installations"] / df_dist_clean["applications"],
        0.0
    )
    df_dist_clean["region"] = df_dist_clean["state"].apply(get_state_region)
    dist_out = os.path.join(PROCESSED_DIR, "district_level_pmsg.csv")
    df_dist_clean.to_csv(dist_out, index=False)
    print(f"Saved {dist_out} with {len(df_dist_clean)} districts across {df_dist_clean['state'].nunique()} states.")

    # Process DISCOM Sheet
    df_discom_raw = pd.read_excel(dist_file, sheet_name="Discom info")
    df_discom_raw["State"] = df_discom_raw["State"].ffill()
    df_discom_raw = df_discom_raw.dropna(subset=["Discom name"])
    df_discom_raw["canonical_state"] = df_discom_raw["State"].apply(normalize_state)
    df_discom_raw = df_discom_raw[~df_discom_raw["canonical_state"].str.startswith("__DROP")]

    discom_cols = {
        "canonical_state": "state",
        "Discom name": "discom_name",
        "Application (no)": "applications",
        "Installation (no)": "installations",
        "Household covered (no)": "households_covered",
        "Installation capacity (MW)": "capacity_mw",
        "Subsidy released (lakhs)": "subsidy_released_lakhs"
    }
    df_discom_clean = df_discom_raw[list(discom_cols.keys())].rename(columns=discom_cols)
    for col in ["applications", "installations", "households_covered", "capacity_mw", "subsidy_released_lakhs"]:
        df_discom_clean[col] = pd.to_numeric(df_discom_clean[col], errors="coerce").fillna(0)

    df_discom_clean["conversion_ratio"] = np.where(
        df_discom_clean["applications"] > 0,
        df_discom_clean["installations"] / df_discom_clean["applications"],
        0.0
    )
    df_discom_clean["region"] = df_discom_clean["state"].apply(get_state_region)
    discom_out = os.path.join(PROCESSED_DIR, "discom_level_pmsg.csv")
    df_discom_clean.to_csv(discom_out, index=False)
    print(f"Saved {discom_out} with {len(df_discom_clean)} DISCOMs across {df_discom_clean['state'].nunique()} states.")

    print("\n--- 3. Ingesting Maharashtra District Dataset ---")
    mh_file = os.path.join(RAW_DIR, "3__District-wise_beneficiaries__Maharashtra__11-12-2024.xlsx")
    df_mh_raw = pd.read_excel(mh_file)
    df_mh_raw = df_mh_raw.dropna(subset=["District"])
    df_mh_raw = df_mh_raw[~df_mh_raw["District"].astype(str).str.contains("Total", case=False)]
    df_mh_clean = pd.DataFrame({
        "district": df_mh_raw["District"].str.strip(),
        "cfa_beneficiaries_dec2024": pd.to_numeric(
            df_mh_raw["No of Beneficiaries that have received Central Financial Assistance (CFA)"], 
            errors="coerce"
        ).fillna(0)
    })
    mh_out = os.path.join(PROCESSED_DIR, "maharashtra_districts.csv")
    df_mh_clean.to_csv(mh_out, index=False)
    print(f"Saved {mh_out} with {len(df_mh_clean)} Maharashtra districts.")

    print("\n--- 4. Updating modeling_table.csv with Per-Capita Metrics ---")
    model_path = os.path.join(PROCESSED_DIR, "modeling_table.csv")
    model_df = pd.read_csv(model_path)
    model_df["pop_2024_lakhs"] = model_df["state"].map(STATE_POPULATION_2024_LAKHS)
    model_df["installations_per_lakh_pop"] = model_df["installations_aug2026"] / model_df["pop_2024_lakhs"]
    model_df["applications_per_lakh_pop"] = model_df["applications_aug2026"] / model_df["pop_2024_lakhs"]
    model_df.to_csv(model_path, index=False)
    print(f"Updated {model_path} with installations_per_lakh_pop and applications_per_lakh_pop.")

if __name__ == "__main__":
    run_pipeline()
