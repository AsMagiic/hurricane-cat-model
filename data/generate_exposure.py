"""
Generate a synthetic but realistic exposure file for a Florida coastal
homeowners portfolio and save it to data/exposure.csv.

All random draws come from a single Generator seeded at SEED so the output
is fully reproducible across machines and Python versions.
"""

import os
import sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
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
# Assemble DataFrame — column order is fixed per spec
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "location_id":    location_ids,
    "state":          states,
    "county":         county_arr,
    "lat":            lat,
    "lon":            lon,
    "tiv":            tiv,
    "construction":   construction,
    "occupancy":      occupancy,
    "deductible_pct": deductible_pct,
    "deductible":     deductible,
    "limit":          limit,
})

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "exposure.csv")
df.to_csv(out_path, index=False)
print(f"Saved {N_LOCATIONS} rows -> {out_path}\n")

# ---------------------------------------------------------------------------
# Validation asserts
# ---------------------------------------------------------------------------
print("=== Validation ===")

tiv_sum = int(df["tiv"].sum())
assert tiv_sum == TARGET_TIV, f"TIV sum {tiv_sum:,} != {TARGET_TIV:,}"
print(f"[OK] TIV sum == {tiv_sum:,}")

n_unique = df["location_id"].nunique()
assert n_unique == N_LOCATIONS, f"Unique IDs {n_unique} != {N_LOCATIONS}"
print(f"[OK] location_id unique count == {n_unique}")

null_count = int(df.isnull().sum().sum())
assert null_count == 0, f"Nulls found:\n{df.isnull().sum()}"
print("[OK] No nulls")

mfg_mask = df["construction"] == "Manufactured"
mh_mask  = df["occupancy"]    == "Mobile Home"
assert (mfg_mask == mh_mask).all(), "Manufactured <=> Mobile Home mismatch"
print("[OK] Manufactured <=> Mobile Home (bijection holds)")

expected_ded = (df["deductible_pct"] * df["tiv"]).round().astype(np.int64)
assert (df["deductible"].astype(np.int64) == expected_ded).all(), \
    "deductible != round(deductible_pct * tiv)"
print("[OK] deductible == round(deductible_pct * tiv)")

print("\n=== Head ===")
print(df.head().to_string())
print("\n=== Describe ===")
print(df.describe(include="all").to_string())
