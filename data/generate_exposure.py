"""
Generate a synthetic but realistic exposure file for a Florida coastal
homeowners portfolio and save it as OED v4 Location + Account CSVs.

All random draws come from a single Generator seeded at SEED so the output
is fully reproducible across machines and Python versions.

Outputs
-------
data/oed/location.csv  -- one row per risk location (OED Location file)
data/oed/account.csv   -- one row for the single account/policy (OED Account file)
"""

import os
import numpy as np
import pandas as pd

from model_config import load_exposure_cfg
_ecfg = load_exposure_cfg()

# ---------------------------------------------------------------------------
# Parameters -- loaded from config/exposure.yaml
# ---------------------------------------------------------------------------
N_LOCATIONS      = _ecfg.n_locations
SEED             = _ecfg.seed
TARGET_TIV       = _ecfg.target_tiv
COUNTIES         = _ecfg.counties
COUNTY_WEIGHTS   = np.array(_ecfg.county_weights)
COUNTY_CENTROIDS = _ecfg.county_centroids

# OED mapping tables from config
_OMED = _ecfg.oed_mapping
CONST_TO_OED   = _OMED.construction_to_oed   # {"Wood Frame": 5050, ...}
OCC_TO_OED     = _OMED.occupancy_to_oed      # {"Single Family": 1051, ...}
PERIL          = _OMED.peril                 # "WTC"
CURRENCY       = _OMED.currency              # "USD"
COUNTRY_CODE   = _OMED.country_code          # "US"
AREA_CODE      = _OMED.area_code             # "FL"
AREA_NAME      = _OMED.area_name             # "Florida"
COUNTY_SCHEME  = _OMED.county_geog_scheme    # "CNTY"
DED_CODE       = int(_OMED.ded_code)         # 0
DED_TYPE       = int(_OMED.ded_type)         # 0
LIMIT_TYPE     = int(_OMED.limit_type)       # 0
ORG_CONST_SCH  = _OMED.org_construction_scheme  # "MODEL"
ORG_OCC_SCH    = _OMED.org_occupancy_scheme     # "MODEL"

# ---------------------------------------------------------------------------
# Single RNG — every draw from here in sequence; order must not change.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

# --- location_id ---
location_ids = [f"LOC{i:05d}" for i in range(1, N_LOCATIONS + 1)]

# --- state ---
states = ["FL"] * N_LOCATIONS

# --- county ---
# rng.choice accepts a Python list and a probability vector that sums to 1.
county_arr = rng.choice(COUNTIES, size=N_LOCATIONS, p=COUNTY_WEIGHTS)

# --- lat / lon ---
# For each location look up the centroid of its county, then perturb by
# independent Normal(0, 0.1°) draws on each axis.
cent_lat = np.array([COUNTY_CENTROIDS[c][0] for c in county_arr])
cent_lon = np.array([COUNTY_CENTROIDS[c][1] for c in county_arr])
lat = np.round(cent_lat + rng.normal(0, 0.1, N_LOCATIONS), 4)
lon = np.round(cent_lon + rng.normal(0, 0.1, N_LOCATIONS), 4)

# --- tiv ---
# Lognormal with E[X] = TARGET_TIV/N_LOCATIONS and a configurable CV.
# sigma_log = sqrt(ln(1 + CV^2));  mu_log = ln(E[X]) - sigma_log^2 / 2.
TIV_CV        = _ecfg.tiv_cv
tiv_sigma_log = np.sqrt(np.log(1 + TIV_CV**2))             # CV=1 -> sqrt(ln(2)) ~= 0.8326
tiv_mu_log    = np.log(TARGET_TIV / N_LOCATIONS) - tiv_sigma_log**2 / 2
raw_tiv       = rng.lognormal(mean=tiv_mu_log, sigma=tiv_sigma_log, size=N_LOCATIONS)

# Scale the vector so its sum equals TARGET_TIV, then round to integers.
tiv = np.round(raw_tiv / raw_tiv.sum() * TARGET_TIV).astype(np.int64)

# Rounding leaves a small residual (at most N_LOCATIONS / 2 dollars).
# Absorb it into the largest location so the total is exact.
tiv[np.argmax(tiv)] += TARGET_TIV - tiv.sum()

# --- occupancy ---
OCC_CHOICES = _ecfg.occupancy.choices
OCC_WEIGHTS = np.array(_ecfg.occupancy.weights)
occupancy   = rng.choice(OCC_CHOICES, size=N_LOCATIONS, p=OCC_WEIGHTS)

# --- construction (must be consistent with occupancy) ---
# Rule: Mobile Home → always "Manufactured".
# All other occupancies draw from the remaining types; Masonry is the majority.
CONST_OTHERS  = _ecfg.construction.non_manufactured_types
CONST_WEIGHTS = np.array(_ecfg.construction.weights)

construction = np.empty(N_LOCATIONS, dtype=object)
mobile_mask  = occupancy == "Mobile Home"
construction[mobile_mask]  = "Manufactured"
n_other = int((~mobile_mask).sum())
construction[~mobile_mask] = rng.choice(CONST_OTHERS, size=n_other, p=CONST_WEIGHTS)

# --- deductible_pct ---
# Hurricane deductibles in Florida are set as a percentage of TIV.
# Common regulatory tiers: 2 %, 5 %, 10 %.
DED_CHOICES = np.array(_ecfg.deductibles.choices)
DED_WEIGHTS = np.array(_ecfg.deductibles.weights)
deductible_pct = rng.choice(DED_CHOICES, size=N_LOCATIONS, p=DED_WEIGHTS)

# --- deductible (dollar amount) and limit ---
deductible = np.round(deductible_pct * tiv).astype(np.int64)
limit      = tiv.copy()

# ---------------------------------------------------------------------------
# Build OED Location DataFrame -- sorted by LocNumber (= location_id order)
# ---------------------------------------------------------------------------
loc_df = pd.DataFrame({
    "PortNumber":            [1]          * N_LOCATIONS,
    "AccNumber":             ["A0001"]    * N_LOCATIONS,
    "LocNumber":             location_ids,
    "CountryCode":           [COUNTRY_CODE] * N_LOCATIONS,
    "LocPerilsCovered":      [PERIL]      * N_LOCATIONS,
    "LocCurrency":           [CURRENCY]   * N_LOCATIONS,
    "AreaCode":              [AREA_CODE]  * N_LOCATIONS,
    "AreaName":              [AREA_NAME]  * N_LOCATIONS,
    "GeogScheme1":           [COUNTY_SCHEME] * N_LOCATIONS,
    "GeogName1":             county_arr,
    "Latitude":              lat,
    "Longitude":             lon,
    "BuildingTIV":           tiv,
    "ConstructionCode":      [CONST_TO_OED[c] for c in construction],
    "OccupancyCode":         [OCC_TO_OED[o] for o in occupancy],
    "LocDed1Building":       deductible,
    "DedCode1Building":      [DED_CODE]   * N_LOCATIONS,
    "DedType1Building":      [DED_TYPE]   * N_LOCATIONS,
    "LocLimit1Building":     limit,
    "LimitType1Building":    [LIMIT_TYPE] * N_LOCATIONS,
    "OrgConstructionScheme": [ORG_CONST_SCH] * N_LOCATIONS,
    "OrgConstructionCode":   construction,
    "OrgOccupancyScheme":    [ORG_OCC_SCH]   * N_LOCATIONS,
    "OrgOccupancyCode":      occupancy,
})
# LocNumber is already in ascending order; sort is a no-op but makes the
# contract explicit: output row order = LocNumber sort order.
loc_df = loc_df.sort_values("LocNumber").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Build OED Account DataFrame -- single row (one account, one policy)
# ---------------------------------------------------------------------------
acc_df = pd.DataFrame([{
    "PortNumber":        1,
    "AccNumber":         "A0001",
    "AccCurrency":       CURRENCY,
    "PolNumber":         "P0001",
    "PolPerilsCovered":  PERIL,
}])

# ---------------------------------------------------------------------------
# Save OED files
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oed")
os.makedirs(out_dir, exist_ok=True)

loc_path = os.path.join(out_dir, "location.csv")
acc_path = os.path.join(out_dir, "account.csv")

loc_df.to_csv(loc_path, index=False)
acc_df.to_csv(acc_path, index=False)

print(f"Saved {N_LOCATIONS} location rows -> {loc_path}")
print(f"Saved {len(acc_df)} account row   -> {acc_path}\n")

# ---------------------------------------------------------------------------
# Validation asserts
# ---------------------------------------------------------------------------
print("=== Validation ===")

tiv_sum = int(loc_df["BuildingTIV"].sum())
assert tiv_sum == TARGET_TIV, f"TIV sum {tiv_sum:,} != {TARGET_TIV:,}"
print(f"[OK] TIV sum == {tiv_sum:,}")

n_unique = loc_df["LocNumber"].nunique()
assert n_unique == N_LOCATIONS, f"Unique IDs {n_unique} != {N_LOCATIONS}"
print(f"[OK] LocNumber unique count == {n_unique}")

null_count = int(loc_df.isnull().sum().sum()) + int(acc_df.isnull().sum().sum())
assert null_count == 0, f"Nulls found:\n{loc_df.isnull().sum()}"
print("[OK] No nulls")

mfg_mask = loc_df["OrgConstructionCode"] == "Manufactured"
mh_mask  = loc_df["OrgOccupancyCode"]    == "Mobile Home"
assert (mfg_mask == mh_mask).all(), "Manufactured <=> Mobile Home mismatch"
print("[OK] Manufactured <=> Mobile Home (bijection holds)")

ded_check = (loc_df["LocDed1Building"].astype(np.int64) >= 0).all()
assert ded_check, "Negative deductible found"
print("[OK] All deductibles non-negative")

assert set(loc_df["LocPerilsCovered"].unique()) == {PERIL}, "Unexpected peril codes"
print(f"[OK] LocPerilsCovered == '{PERIL}' for all rows")

assert (loc_df["DedType1Building"] == DED_TYPE).all(), "DedType mismatch"
assert (loc_df["LimitType1Building"] == LIMIT_TYPE).all(), "LimitType mismatch"
print("[OK] DedType1Building == 0 (Amount); LimitType1Building == 0 (Amount)")

valid_const_codes = set(CONST_TO_OED.values())
assert set(loc_df["ConstructionCode"].unique()).issubset(valid_const_codes), \
    "Unknown ConstructionCode"
valid_occ_codes = set(OCC_TO_OED.values())
assert set(loc_df["OccupancyCode"].unique()).issubset(valid_occ_codes), \
    "Unknown OccupancyCode"
print("[OK] All ConstructionCode / OccupancyCode values in valid OED ranges")

assert len(acc_df) == 1, "Expected exactly 1 account row"
assert acc_df["PolPerilsCovered"].iloc[0] == PERIL, "PolPerilsCovered mismatch"
print("[OK] Single account row with PolPerilsCovered == 'WTC'")

print("\n=== Head ===")
print(loc_df.head().to_string())
print("\n=== Describe ===")
print(loc_df[["BuildingTIV", "Latitude", "Longitude",
              "LocDed1Building", "LocLimit1Building"]].describe().to_string())
