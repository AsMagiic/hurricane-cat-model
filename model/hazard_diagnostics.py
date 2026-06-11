"""
Hazard diagnostics for the Florida hurricane cat model.

Imports sample_storm and wind_at_locations from hazard.py without modification.
Runs N_STORMS = 10,000 synthetic hurricanes and checks three properties:

  1. Landfall distribution along the FL coast
  2. TIV-weighted wind exposure by county  (SE corridor coverage test)
  3. Multi-county accumulation per storm   (spatial correlation test)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from model.hazard import sample_storm, wind_at_locations, COAST_POINTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_PATH = os.path.join(ROOT, "data",    "exposure.csv")
OUT_DIR  = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load exposure portfolio
# ---------------------------------------------------------------------------
exp      = pd.read_csv(EXP_PATH)
lats     = exp["lat"].to_numpy()
lons     = exp["lon"].to_numpy()
tivs     = exp["tiv"].to_numpy(dtype=float)
counties = exp["county"].to_numpy()

print(f"Portfolio: {len(exp):,} locations | TIV USD {tivs.sum()/1e6:.0f}M")

# ---------------------------------------------------------------------------
# Simulate N_STORMS storms and compute wind at every portfolio location
# ---------------------------------------------------------------------------
N_STORMS    = 10_000
SEED        = 42
CAT1_THRESH = 74.0   # mph -- minimum Saffir-Simpson Category 1 sustained wind

rng = np.random.default_rng(SEED)

# float32 keeps memory to ~40 MB for the full wind matrix
all_winds  = np.zeros((N_STORMS, len(lats)), dtype=np.float32)
lf_lats    = np.zeros(N_STORMS)
lf_lons    = np.zeros(N_STORMS)
storm_cats = np.zeros(N_STORMS, dtype=np.int32)

print(f"\nSimulating {N_STORMS:,} storms (seed={SEED}) ...")
for i in range(N_STORMS):
    if (i + 1) % 2_000 == 0:
        print(f"  {i + 1:,} / {N_STORMS:,}")
    track, meta   = sample_storm(rng)
    all_winds[i]  = wind_at_locations(track, meta["rmax"], lats, lons)
    lf_lats[i]    = meta["landfall_lat"]
    lf_lons[i]    = meta["landfall_lon"]
    storm_cats[i] = meta["category"]

print(f"Simulation complete.  Wind matrix: {all_winds.shape}  "
      f"(dtype={all_winds.dtype})\n")

# ===========================================================================
# DIAGNOSTIC 1 -- Landfall distribution
# ===========================================================================
print("=" * 62)
print("DIAGNOSTIC 1: Landfall distribution")
print("=" * 62)

fig1, ax1 = plt.subplots(figsize=(8, 9))
ax1.plot(COAST_POINTS[:, 1], COAST_POINTS[:, 0],
         "k--", linewidth=1.2, alpha=0.7, zorder=3,
         label="Coast polyline")
ax1.scatter(lf_lons, lf_lats,
            s=3, alpha=0.15, color="#1f77b4", zorder=2,
            label=f"Landfalls (n={N_STORMS:,})")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title(
    f"Landfall Distribution -- {N_STORMS:,} storms, seed={SEED}",
    fontsize=12,
)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.set_aspect("equal")

p1 = os.path.join(OUT_DIR, "landfall_distribution.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Plot saved -> {p1}")

# SE Atlantic corridor filter: lat [25.4, 27.0], lon > -80.5
# Covers COAST_POINTS segments 1-3 fully (weights 1.5+2.5+3.0 = 7.0)
# plus partial contributions from segments 0 (N tail) and 4 (Keys transition).
# Expected fraction: segs 2+3 fully (5.5/18.5=29.7%) + partial segs 1,4
# -> roughly 35-40% of all landfalls.
se_mask = (lf_lats >= 25.4) & (lf_lats <= 27.0) & (lf_lons > -80.5)
se_n    = int(se_mask.sum())
se_pct  = se_n / N_STORMS * 100

print(f"Landfalls in SE Atlantic (lat 25.4-27.0, lon > -80.5): "
      f"{se_n:,} / {N_STORMS:,} = {se_pct:.1f}%")
print("  Segments 2+3 alone: weights 5.5/18.5 = 29.7%")
print("  Partial segs 1 and 4 push the expected total to ~35-40%")
if se_pct >= 30:
    print(f"  [OK] SE corridor share {se_pct:.1f}% is within expected range")
else:
    print(f"  [WARN] SE corridor share {se_pct:.1f}% is lower than expected (~35%)")
print()

# ===========================================================================
# DIAGNOSTIC 2 -- TIV-weighted wind by county
# ===========================================================================
print("=" * 62)
print("DIAGNOSTIC 2: TIV-weighted wind by county")
print("=" * 62)
print("(avg wind = TIV-weighted mean across all 10k storms per county)")
print()

county_names = sorted(exp["county"].unique())
rows = []
for cty in county_names:
    mask    = counties == cty
    c_tivs  = tivs[mask]
    c_winds = all_winds[:, mask].astype(float)   # (N_STORMS, n_locs_in_county)

    # TIV-weighted mean wind per storm, then mean over all storms
    w              = c_tivs / c_tivs.sum()
    mean_per_storm = c_winds @ w                 # (N_STORMS,)
    avg_wind       = float(mean_per_storm.mean())
    max_wind       = float(c_winds.max())

    rows.append({
        "county":       cty,
        "tiv_M":        float(c_tivs.sum() / 1e6),
        "n_locs":       int(mask.sum()),
        "avg_wind_mph": avg_wind,
        "max_wind_mph": max_wind,
    })

county_df = (pd.DataFrame(rows)
             .sort_values("tiv_M", ascending=False)
             .reset_index(drop=True))

print(f"{'County':<16} {'TIV ($M)':>9} {'Locs':>5} "
      f"{'Avg wind (mph)':>15} {'Max wind (mph)':>15}")
print("-" * 64)
for _, r in county_df.iterrows():
    print(f"{r['county']:<16} {r['tiv_M']:>9.1f} {r['n_locs']:>5} "
          f"{r['avg_wind_mph']:>15.1f} {r['max_wind_mph']:>15.1f}")

# Key check: Broward and Palm Beach should stay close to Miami-Dade.
# If the track heading range only sweeps storms northward far enough to reach
# Miami-Dade but not Broward/Palm Beach, the ratios will be well below 1.
d2_verdict = "N/A"
r_bro = r_pab = 0.0
se3 = ["Miami-Dade", "Broward", "Palm Beach"]
cdf = county_df.set_index("county")

if all(c in cdf.index for c in se3):
    mia   = cdf.loc["Miami-Dade",  "avg_wind_mph"]
    bro   = cdf.loc["Broward",     "avg_wind_mph"]
    pab   = cdf.loc["Palm Beach",  "avg_wind_mph"]
    r_bro = bro / mia if mia > 0 else 0.0
    r_pab = pab / mia if mia > 0 else 0.0

    print(f"\nSE corridor ratios vs Miami-Dade (TIV-weighted avg wind):")
    print(f"  Broward    : {r_bro:.2f}x  "
          f"(Broward {bro:.1f} mph vs Miami-Dade {mia:.1f} mph)")
    print(f"  Palm Beach : {r_pab:.2f}x  "
          f"(Palm Beach {pab:.1f} mph vs Miami-Dade {mia:.1f} mph)")

    if r_bro >= 0.70 and r_pab >= 0.60:
        d2_verdict = ("WELL COVERED -- track heading sweeps Miami-Dade "
                      "through Palm Beach comparably")
    else:
        d2_verdict = ("BIASED -- heading range under-samples the N-S sweep; "
                      "consider widening to +/-60 deg")
    print(f"  Result: {d2_verdict}")

print()

# ===========================================================================
# DIAGNOSTIC 3 -- Multi-county accumulation per storm
# ===========================================================================
print("=" * 62)
print("DIAGNOSTIC 3: Multi-county accumulation per storm")
print("=" * 62)

unique_counties    = sorted(exp["county"].unique())
n_total_counties   = len(unique_counties)

# For each storm, count how many counties had at least one location with
# maximum sustained wind >= CAT1_THRESH (74 mph, Category 1 minimum).
counties_hit = np.zeros(N_STORMS, dtype=np.int32)
for cty in unique_counties:
    mask = counties == cty
    # any(axis=1): True if any location in this county exceeded the threshold
    hit  = (all_winds[:, mask] >= CAT1_THRESH).any(axis=1)
    counties_hit += hit.astype(np.int32)

# Histogram of how many counties each storm strikes above Cat 1
fig3, ax3 = plt.subplots(figsize=(8, 5))
bins = np.arange(-0.5, n_total_counties + 1.5, 1.0)
ax3.hist(counties_hit, bins=bins, color="#1f77b4",
         edgecolor="white", linewidth=0.5)
ax3.set_xlabel(f"Counties with >= 1 location above Cat 1 ({CAT1_THRESH:.0f} mph)",
               fontsize=11)
ax3.set_ylabel("Number of storms", fontsize=11)
ax3.set_title(
    f"Multi-county accumulation per storm  ({N_STORMS:,} storms)\n"
    f"Portfolio counties: {n_total_counties}  |  "
    f"Cat 1 threshold: {CAT1_THRESH:.0f} mph",
    fontsize=12,
)
ax3.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax3.grid(True, axis="y", linestyle=":", alpha=0.5)

p3 = os.path.join(OUT_DIR, "counties_hit_per_event.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Plot saved -> {p3}")

n0   = int((counties_hit == 0).sum())
n1   = int((counties_hit == 1).sum())
n2   = int((counties_hit == 2).sum())
n3p  = int((counties_hit >= 3).sum())
multi_pct = (n2 + n3p) / N_STORMS * 100

print(f"  0 counties hit : {n0:>6,}  ({n0 / N_STORMS * 100:.1f}%)")
print(f"  1 county  hit  : {n1:>6,}  ({n1 / N_STORMS * 100:.1f}%)")
print(f"  2 counties hit : {n2:>6,}  ({n2 / N_STORMS * 100:.1f}%)")
print(f"  3+ counties hit: {n3p:>6,}  ({n3p / N_STORMS * 100:.1f}%)")
print(f"  Multi-county (2+ counties): {multi_pct:.1f}% of all storms")

if multi_pct >= 10:
    d3_verdict = ("MULTI-COUNTY ACCUMULATION PRESENT -- "
                  "spatial correlation is realistic")
else:
    d3_verdict = ("ACCUMULATION ABSENT -- spread too local; "
                  "portfolio PMLs may be understated")
print(f"  Result: {d3_verdict}")
print()

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("=" * 62)
print("FINAL SUMMARY")
print("=" * 62)

# 1. SE coast coverage
if se_pct >= 30:
    c1 = f"SE Atlantic coast WELL COVERED  ({se_pct:.1f}% of landfalls)"
else:
    c1 = f"SE Atlantic coast UNDER-SAMPLED  ({se_pct:.1f}% -- expected ~35%)"
print(f"1. Landfall  : {c1}")

# 2. SE corridor wind sweep
print(f"2. SE sweep  : {d2_verdict}")
if r_bro > 0:
    print(f"             Broward {r_bro:.2f}x | Palm Beach {r_pab:.2f}x of Miami-Dade")

# 3. Multi-county accumulation
print(f"3. Accum     : {d3_verdict}")
print(f"             {multi_pct:.1f}% of storms affect 2+ counties above Cat 1")
