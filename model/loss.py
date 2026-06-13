"""
Loss integration engine for the Florida hurricane cat model (Step 4).

Wires together:
  hazard.py        -> stochastic moving-track wind fields (sustained mph)
  vulnerability.py -> HAZUS-anchored damage ratio curves (3-s gust)
  data/exposure.csv -> TIV, deductible, policy limit per location

Three-level loss hierarchy applied per location, per event (in this order):
  gust      = sustained_wind * GUST_FACTOR          (Exposure C, open coastal)
  ground_up = damage_ratio(gust, construction) * tiv
  gross     = clip(ground_up - deductible, 0, policy_limit)

Deductible is applied per-occurrence, per-location, BEFORE portfolio aggregation.
'Gross' = net of policy deductible only.  Reinsurance retention is Step 5.

Outputs -> results/  (gitignored; fully reproducible by seed=42):
  events.csv        one row per storm event
  annual_losses.csv exactly N_YEARS rows, including 0-loss years
"""

import os
import time
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr as _ndtr, betaincinv as _betaincinv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from model_config import load_model_cfg
_mcfg = load_model_cfg()

from model.hazard        import simulate_year, LAMBDA
from model.wind_field    import wind_at_locations, StormParams
from model.vulnerability import GUST_FACTOR, GUST_THRESHOLD, CONSTRUCTION_PARAMS, build_event_kernel
from model.ep_utils      import oep_pml, ep_curve, pml_rank_diagnostic
from model.exposure_io   import load_portfolio

# ---------------------------------------------------------------------------
# Configuration -- loaded from config/model_v3.yaml
# ---------------------------------------------------------------------------
SEED    = _mcfg.simulation.seed
N_YEARS = _mcfg.simulation.n_years

# Damage uncertainty switches (Step 3.1)
_DAMAGE_UNCERTAINTY = _mcfg.vulnerability.damage_uncertainty  # "off" | "on"
_DAMAGE_CV          = float(_mcfg.vulnerability.damage_cv)    # pointwise DR coeff of variation
_DAMAGE_RHO         = float(_mcfg.vulnerability.damage_rho)   # common-shock correlation in [0,1]

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load exposure (OED Location + Account via compatibility adapter)
# ---------------------------------------------------------------------------
_exp = load_portfolio()

lats          = _exp["lat"].to_numpy(dtype=float)
lons          = _exp["lon"].to_numpy(dtype=float)
tivs          = _exp["tiv"].to_numpy(dtype=float)
deductibles   = _exp["deductible"].to_numpy(dtype=float)
pol_limits    = _exp["limit"].to_numpy(dtype=float)   # pol_limits: avoids shadowing built-in
constructions = _exp["construction"].to_numpy()
counties      = _exp["county"].to_numpy()

n_loc     = len(_exp)
TOTAL_TIV = float(tivs.sum())

print(f"Exposure: {n_loc:,} locations | TIV USD {TOTAL_TIV/1e6:.0f}M")
print(f"Results  -> {RESULTS_DIR}")

# ---------------------------------------------------------------------------
# Precompute per-location vulnerability kernel (one-time, outside the loop)
#
# build_event_kernel() captures per-location parameter arrays in a closure so
# _event_loss() remains a pure numpy kernel with no Python-level iteration.
# The kernel switches on vulnerability.method from config; logistic mode is
# bit-identical to the old inlined midpoints/caps/ks expression.
# ---------------------------------------------------------------------------
_vuln_kernel = build_event_kernel(constructions)

# Integer index arrays for fast group aggregation via np.bincount
unique_constructions = list(CONSTRUCTION_PARAMS.keys())   # insertion order: Mfg, WF, Mas, RC
unique_counties      = sorted(set(counties))              # alphabetical

n_con = len(unique_constructions)
n_cty = len(unique_counties)

con_idx = np.array([unique_constructions.index(c) for c in constructions], dtype=np.int32)
cty_idx = np.array([unique_counties.index(c)      for c in counties],      dtype=np.int32)

tiv_per_con = np.array([tivs[con_idx == i].sum() for i in range(n_con)])
tiv_per_cty = np.array([tivs[cty_idx == i].sum() for i in range(n_cty)])


# ---------------------------------------------------------------------------
# Damage uncertainty helpers (Step 3.1)
# ---------------------------------------------------------------------------

def _beta_params(m, cv):
    """
    (mean array, scalar cv) -> (alpha, beta_param) arrays for Beta distribution.

    Extension point: v4 replaces constant cv with an MDR-dependent cv function here,
    capturing the empirical mean-variance relationship from per-event damage data.

    Guards:
      - v is capped below m*(1-m) to ensure alpha, beta > 0 (prevents degenerate Beta).
      - cv > 0 is guaranteed by range validation in model_config.py.
      - m=0 locations are handled upstream by the zeros mask in _damage_draw.
    """
    v     = (cv * m) ** 2
    max_v = m * (1.0 - m)
    v     = np.minimum(v, max_v * (1.0 - 1e-7))  # cap ensures conc > 0
    conc  = m * (1.0 - m) / v - 1.0
    return m * conc, (1.0 - m) * conc


def _damage_draw(dr_mean, rng):
    """
    Gaussian copula common-shock Beta draw for one event.

    Mechanism:
      z_event ~ N(0,1)          — one draw, shared across ALL locations (common shock)
      eps_i   ~ N(0,1)          — n_loc independent draws (idiosyncratic noise)
      U_i = Phi(sqrt(rho)*z + sqrt(1-rho)*eps_i)   in [0,1] per location
      dr_i = Beta.ppf(U_i ; alpha_i, beta_i)       realized damage ratio

    With rho=0: U_i are independent -> independent noise washes out over portfolio (LLN).
    With rho=1: U_i = Phi(z_event) identical for all i -> common shock does NOT wash out.

    Guards:
      m=0 locations return 0.0 (degenerate Beta; no uncertainty on zero damage).
      u is clipped to [1e-12, 1-1e-12] to prevent NaN from betaincinv when ndtr
      underflows to exactly 0.0 or 1.0 at extreme z values.
    """
    z_event = rng.standard_normal()           # 1 draw — common shock for this event
    eps     = rng.standard_normal(n_loc)      # n_loc draws — idiosyncratic noise
    u = _ndtr(np.sqrt(_DAMAGE_RHO) * z_event + np.sqrt(1.0 - _DAMAGE_RHO) * eps)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)       # tail-NaN guard

    zeros  = (dr_mean == 0.0)
    m_safe = np.where(zeros, 0.5, dr_mean)    # avoid degenerate Beta params for zero-damage locs
    alpha, beta_p = _beta_params(m_safe, _DAMAGE_CV)
    realized = _betaincinv(alpha, beta_p, u)
    realized = np.where(zeros, 0.0, realized)
    return np.clip(realized, 0.0, 1.0)        # safety clamp against float noise


# ---------------------------------------------------------------------------
# Vectorized per-event loss kernel
# ---------------------------------------------------------------------------
def _event_loss(wind_sustained, dmg_rng=None):
    """
    (n_loc,) sustained wind mph  ->  (ground_up, gross) both (n_loc,) float64.

    dmg_rng: when not None, realized damage ratios are drawn from a Beta
    distribution via _damage_draw (Gaussian copula common-shock). When None
    (default, damage_uncertainty=off), the deterministic mean curve is used —
    bit-identical to the pre-3.1 baseline.
    """
    gust      = wind_sustained * GUST_FACTOR
    dr        = _vuln_kernel(gust)
    if dmg_rng is not None:
        dr = _damage_draw(dr, dmg_rng)
    ground_up = dr * tivs
    gross     = np.clip(ground_up - deductibles, 0.0, pol_limits)
    return ground_up, gross


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_simulation(n_years, seed=SEED):
    """
    Simulate n_years of compound-Poisson hurricane losses.

    Returns
    -------
    events_df  : DataFrame  one row per event (storm years only)
    annual_df  : DataFrame  exactly n_years rows (incl. 0-loss years)
    aal_con_gu : ndarray (n_con,)  per-construction ground-up AAL, USD/year
    aal_con_gr : ndarray (n_con,)  per-construction gross AAL, USD/year
    aal_cty_gu : ndarray (n_cty,)  per-county ground-up AAL
    aal_cty_gr : ndarray (n_cty,)  per-county gross AAL
    """
    rng = np.random.default_rng(seed)
    # Damage substream: two-integer entropy [seed, 1] produces a SeedSequence hash
    # orthogonal to single-integer SeedSequence(seed) used by rng. Does NOT share
    # rng's spawn tree — rng's spawn counter stays at 0 before the year loop, so
    # storm 1 keeps slot 0 (unchanged from pre-3.1, bit-identical to 3.0c when off).
    damage_rng = np.random.default_rng([seed, 1])

    event_rows  = []
    annual_rows = []

    # Running sums for group AALs (divided by n_years at the end)
    sum_con_gu = np.zeros(n_con)
    sum_con_gr = np.zeros(n_con)
    sum_cty_gu = np.zeros(n_cty)
    sum_cty_gr = np.zeros(n_cty)

    t0 = time.perf_counter()

    for yr in range(1, n_years + 1):
        if yr % 10_000 == 0:
            el  = time.perf_counter() - t0
            pct = yr / n_years * 100
            print(f"  {yr:>7,} / {n_years:,}  ({pct:.0f}%)"
                  f"  {el:.0f}s  events so far: {len(event_rows):,}")

        year_events = simulate_year(rng)   # list of (track, meta)
        n_ev        = len(year_events)
        agg_gu = agg_gr = max_gr = 0.0

        for track, meta in year_events:
            winds   = wind_at_locations(
                track,
                StormParams(
                    rmax=meta["rmax"],
                    b=meta["b"],
                    dp_mb=meta["dp_mb"],
                    lat=meta["landfall_lat"],
                    heading_deg=meta["heading_deg"],
                    vt_kmh=meta["translation_speed_kmh"],
                ),
                lats, lons,
            )
            gu, gr  = _event_loss(winds, damage_rng if _DAMAGE_UNCERTAINTY == "on" else None)
            port_gu = float(gu.sum())
            port_gr = float(gr.sum())

            agg_gu += port_gu
            agg_gr += port_gr
            if port_gr > max_gr:
                max_gr = port_gr

            # Group AAL accumulators (C-level bincount; no Python loop over locations)
            sum_con_gu += np.bincount(con_idx, weights=gu, minlength=n_con)
            sum_con_gr += np.bincount(con_idx, weights=gr, minlength=n_con)
            sum_cty_gu += np.bincount(cty_idx, weights=gu, minlength=n_cty)
            sum_cty_gr += np.bincount(cty_idx, weights=gr, minlength=n_cty)

            event_rows.append({
                "year":                yr,
                "vmax":                float(meta["vmax_landfall"]),
                "category":            int(meta["category"]),
                "portfolio_ground_up": port_gu,
                "portfolio_gross":     port_gr,
            })

        # One row per year -- 0-loss years (n_ev == 0) produce all-zero losses.
        # These rows MUST be included; omitting them inflates EP probabilities.
        annual_rows.append({
            "year":                yr,
            "n_events":            n_ev,
            "aggregate_ground_up": agg_gu,
            "aggregate_gross":     agg_gr,
            "max_event_gross":     max_gr,
        })

    elapsed    = time.perf_counter() - t0
    n_ev_total = len(event_rows)
    print(f"\nDone: {n_years:,} years | {n_ev_total:,} events | "
          f"{elapsed:.1f}s  ({elapsed / n_years * 1000:.2f} ms/year)")

    return (pd.DataFrame(event_rows),
            pd.DataFrame(annual_rows),
            sum_con_gu / n_years,
            sum_con_gr / n_years,
            sum_cty_gu / n_years,
            sum_cty_gr / n_years)


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------
def validate_and_report(events_df, annual_df,
                        aal_con_gu, aal_con_gr,
                        aal_cty_gu, aal_cty_gr,
                        n_years):
    print()
    print("=" * 64)
    print("VALIDATION")
    print("=" * 64)

    # ---- Hard structural asserts -----------------------------------------
    if len(events_df) > 0:
        gu_arr = events_df["portfolio_ground_up"].to_numpy()
        gr_arr = events_df["portfolio_gross"].to_numpy()
        assert (gr_arr <= gu_arr + 1e-4).all(), \
            "FAIL: gross > ground_up for some events"
    print("[OK] gross <= ground_up for all events")

    assert len(annual_df) == n_years, \
        f"FAIL: annual_df has {len(annual_df)} rows, expected {n_years}"
    print(f"[OK] annual_losses has exactly {n_years:,} rows (incl. 0-loss years)")

    agg_arr = annual_df["aggregate_gross"].to_numpy()
    max_arr = annual_df["max_event_gross"].to_numpy()
    assert (max_arr <= agg_arr + 1e-4).all(), \
        "FAIL: max_event_gross > aggregate_gross in some year"
    print("[OK] max_event_gross <= aggregate_gross for all years")

    # ---- AAL eyeball ------------------------------------------------------
    print()
    aal_gu = float(annual_df["aggregate_ground_up"].mean())
    aal_gr = float(annual_df["aggregate_gross"].mean())
    print(f"AAL ground-up : USD {aal_gu:>14,.0f}  "
          f"({aal_gu / TOTAL_TIV * 100:.4f}% of TIV)")
    print(f"AAL gross     : USD {aal_gr:>14,.0f}  "
          f"({aal_gr / TOTAL_TIV * 100:.4f}% of TIV)")
    ded_rate = (aal_gu - aal_gr) / aal_gu * 100 if aal_gu > 0 else 0.0
    print(f"Deductible absorption: {ded_rate:.1f}% of ground-up AAL")
    assert aal_gr < aal_gu, "FAIL: AAL_gross should be < AAL_ground_up"
    print("[OK] AAL_gross < AAL_ground_up (deductible absorbing as expected)")

    # ---- Zero-loss fraction ----------------------------------------------
    zero_yrs  = int((annual_df["aggregate_gross"].to_numpy() == 0.0).sum())
    zero_pct  = zero_yrs / n_years * 100
    floor_pct = math.exp(-LAMBDA) * 100
    print(f"\nYears with zero gross loss: {zero_yrs:,} / {n_years:,} = {zero_pct:.2f}%")
    print(f"  (Poisson zero-event floor = {floor_pct:.2f}%  -- actual must be >= floor)")
    assert zero_pct >= floor_pct - 0.5, \
        f"FAIL: zero-loss {zero_pct:.2f}% < floor {floor_pct:.2f}%"
    print("[OK] Zero-loss fraction >= Poisson floor")

    # ---- PML eyeball -----------------------------------------------------
    print()
    oep_gross_arr = annual_df["max_event_gross"].to_numpy()
    agg_gross_arr = annual_df["aggregate_gross"].to_numpy()
    oep_sorted, _ = ep_curve(oep_gross_arr, n_years)   # sorted desc; used for diagnostics
    print(f"  {'Return period':<14} {'AEP gross (USD M)':>20} {'OEP gross (USD M)':>20}")
    print("  " + "-" * 56)
    for rp in [100, 250]:
        pml_aep = oep_pml(agg_gross_arr, rp, n_years)
        pml_oep = oep_pml(oep_gross_arr, rp, n_years)
        print(f"  1-in-{rp:<9}  {pml_aep / 1e6:>18.1f}  {pml_oep / 1e6:>18.1f}")
    print("  (v1 parametric reference: AEP 1-in-100 ~100M | 1-in-250 ~142M)")
    print()
    pml_rank_diagnostic(oep_sorted, n_years)   # raw rank sanity check

    # ---- AAL by construction type ----------------------------------------
    print()
    print("AAL by construction type (gross):")
    print(f"  {'Type':<22} {'TIV share%':>11} {'AAL share%':>11} {'Ratio':>7}")
    print("  " + "-" * 54)
    total_aal_gr = aal_con_gr.sum()
    for i, c in enumerate(unique_constructions):
        tiv_sh = tiv_per_con[i] / TOTAL_TIV * 100
        aal_sh = aal_con_gr[i] / total_aal_gr * 100 if total_aal_gr > 0 else 0.0
        ratio  = aal_sh / tiv_sh if tiv_sh > 0 else 0.0
        print(f"  {c:<22} {tiv_sh:>11.1f} {aal_sh:>11.1f} {ratio:>7.2f}x")
    print("  (ratio > 1.0 -> fragility exceeds its TIV share; expected for Manufactured)")

    # ---- AAL by county (sorted by gross AAL) -----------------------------
    print()
    print("AAL by county (gross):")
    print(f"  {'County':<16} {'TIV($M)':>9} {'AAL_gr($M)':>11} {'AAL/TIV%':>10}")
    print("  " + "-" * 48)
    for i in np.argsort(-aal_cty_gr):
        cty = unique_counties[i]
        print(f"  {cty:<16} {tiv_per_cty[i] / 1e6:>9.1f}"
              f" {aal_cty_gr[i] / 1e6:>11.3f}"
              f"  {aal_cty_gr[i] / tiv_per_cty[i] * 100:>9.3f}%")

    print()
    print("=" * 64)
    print("All structural asserts passed.")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Loss simulation (Step 4)")
    _parser.add_argument(
        "--quick", action="store_true",
        help="Smoke run only (1,000 years); skip the 100k full simulation"
    )
    _args = _parser.parse_args()

    # ---- Smoke test: 1,000 years (always runs) ---------------------------
    print()
    print("=" * 64)
    print("SMOKE TEST  (N_YEARS = 1,000)")
    print("=" * 64)
    smoke = run_simulation(n_years=1_000, seed=SEED)
    validate_and_report(*smoke, n_years=1_000)

    if _args.quick:
        print("\nSmoke test passed.  (--quick: skipping 100k full simulation)")
        ev_df = smoke[0]
        an_df = smoke[1]
    else:
        print("\nSmoke test passed.  Scaling to full simulation ...")

        # ---- Full run: 100,000 years -------------------------------------
        print()
        print("=" * 64)
        print(f"FULL SIMULATION  (N_YEARS = {N_YEARS:,})")
        print("=" * 64)
        full = run_simulation(n_years=N_YEARS, seed=SEED)
        validate_and_report(*full, n_years=N_YEARS)
        ev_df = full[0]
        an_df = full[1]

    ev_path = os.path.join(RESULTS_DIR, "events.csv")
    an_path = os.path.join(RESULTS_DIR, "annual_losses.csv")
    ev_df.to_csv(ev_path, index=False)
    an_df.to_csv(an_path, index=False)

    print(f"\nSaved: {ev_path}  ({len(ev_df):,} rows)")
    print(f"Saved: {an_path}  ({len(an_df):,} rows)")
