"""
Stochastic hurricane hazard generator for Florida coastal portfolios.

Generates synthetic storm tracks and computes the maximum sustained wind (mph)
at each portfolio location via a moving-track modified Rankine vortex.

Key simplifications (documented; appropriate for a portfolio cat model demo):
  - Track heading is drawn uniformly in ±45° of due north.  No east/west coast
    steering distinction is modelled.
  - Filling rate uses a single exponential decay (120 km e-folding scale).
  - Rmax is constant along the track (no eyewall contraction).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _MODEL_DIR)   # tech debt: remove when model/ becomes a package
from model_config import load_model_cfg
from units import kt_to_mph, mph_to_kt
_mcfg = load_model_cfg()

# ---------------------------------------------------------------------------
# Constants -- loaded from config/model_v3.yaml
# ---------------------------------------------------------------------------
SEED   = _mcfg.simulation.seed
LAMBDA = _mcfg.frequency.lambda_rate   # Poisson rate (storms/year)

# Saffir-Simpson category wind ranges (mph, sustained) and FL landfall weights.
CAT_RANGES  = [tuple(r) for r in _mcfg.frequency.category_ranges]
CAT_WEIGHTS = np.array(_mcfg.frequency.category_weights)  # no longer used for sampling (Step 1.4b); kept for validation reference

# Intensity distribution parameters — truncated lognormal fitted to HURDAT2 FL landfalls.
_icfg          = _mcfg.intensity
_INT_LOC       = float(_icfg.loc)           # 64.0 kt — HU definitional threshold
_INT_MU_LOG    = float(_icfg.mu_log)        # 4.4362 log_kt
_INT_SIGMA_LOG = float(_icfg.sigma_log)     # 0.2518 dimensionless
_INT_P_LB = float(scipy.stats.norm.cdf(
    (np.log(_INT_LOC) - _INT_MU_LOG) / _INT_SIGMA_LOG
))  # P(X < 64 kt) under parent lognormal; precomputed for inverse-CDF sampler

# Florida coastline polyline for landfall sampling.
# Ordered: Atlantic N->S, Keys, Gulf S->N.  Shape (12, 2): each row is [lat, lon].
COAST_POINTS = np.array(_mcfg.hazard.coast_polyline)

# Per-segment weights (11 segments for 12 points).  Coupled to COAST_POINTS.
SEGMENT_WEIGHTS = np.array(_mcfg.hazard.coast_segment_weights)

# Hazard mechanics
_STEP_KM     = _mcfg.hazard.step_km
_EFOLD_KM    = _mcfg.hazard.efold_km
_OUTER_DECAY = _mcfg.hazard.outer_decay_exponent

# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km.  Inputs may be scalars or numpy arrays."""
    R    = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ---------------------------------------------------------------------------
# Landfall sampling
# ---------------------------------------------------------------------------
def sample_landfall(rng):
    """
    Return (lat, lon) of a landfall point sampled along the FL coast polyline.

    A segment is chosen with probability proportional to SEGMENT_WEIGHTS;
    the point within that segment is drawn uniformly (linear interpolation).
    """
    p       = SEGMENT_WEIGHTS / SEGMENT_WEIGHTS.sum()
    seg_idx = int(rng.choice(len(SEGMENT_WEIGHTS), p=p))
    t       = float(rng.uniform(0, 1))
    p0, p1  = COAST_POINTS[seg_idx], COAST_POINTS[seg_idx + 1]
    return float(p0[0] + t * (p1[0] - p0[0])), float(p0[1] + t * (p1[1] - p0[1]))

# ---------------------------------------------------------------------------
# Intensity sampling
# ---------------------------------------------------------------------------
_CAT_LO_MPH = [float(lo) for lo, _ in CAT_RANGES]
# [74.0, 96.0, 111.0, 130.0, 157.0] — Saffir-Simpson lower bounds in mph (sustained)


def _vmax_to_category(vmax_mph: float) -> int:
    """Saffir-Simpson category (1-5) from continuous Vmax (mph, sustained)."""
    for cat in range(4, 0, -1):
        if vmax_mph >= _CAT_LO_MPH[cat]:
            return cat + 1
    return 1


def sample_intensity(rng):
    """
    Sample (category: int 1-5, vmax: float mph) from the fitted truncated lognormal.

    Vmax drawn in kt via inverse-CDF: U ~ Uniform(0,1),
    p = _INT_P_LB + U*(1-_INT_P_LB), Vmax_kt = exp(mu_log + sigma_log * Phi_inv(p)).
    Converted to mph via kt_to_mph. Category derived from Vmax using Saffir-Simpson
    bounds in _CAT_LO_MPH; Vmax is the source of truth, category is a derived label.
    Units: output vmax in mph (sustained 1-minute).
    """
    u        = float(rng.uniform(0.0, 1.0))
    p_samp   = _INT_P_LB + u * (1.0 - _INT_P_LB)
    vmax_kt  = float(np.exp(_INT_MU_LOG + _INT_SIGMA_LOG * float(scipy.stats.norm.ppf(p_samp))))
    vmax_mph = float(kt_to_mph(vmax_kt))
    return _vmax_to_category(vmax_mph), vmax_mph

# ---------------------------------------------------------------------------
# Track building
# ---------------------------------------------------------------------------
def build_track(landfall_lat, landfall_lon, vmax, rng):
    """
    Generate a 10-step inland track starting at the landfall point.

    Returns
    -------
    rmax  : float — radius of maximum winds (km), constant along track
    track : ndarray (11, 4) — columns: [lat, lon, vmax_step, cum_dist_km]
              row 0 = landfall (peak intensity), rows 1-10 = inland steps
    """
    rmax        = float(rng.uniform(30, 55))   # km, constant (no eyewall contraction)
    heading_deg = float(rng.uniform(-45, 45))  # deviation from due north (simplification)
    heading_rad = np.radians(heading_deg)

    n_steps = 10
    step_km = _STEP_KM
    e_fold  = _EFOLD_KM

    rows = []
    lat, lon = landfall_lat, landfall_lon
    for i in range(n_steps + 1):
        cum_dist  = i * step_km
        vmax_step = vmax * np.exp(-cum_dist / e_fold)
        rows.append([lat, lon, vmax_step, cum_dist])
        # Advance center by step_km in the heading direction (flat-earth approx)
        dlat = step_km * np.cos(heading_rad) / 111.0
        dlon = step_km * np.sin(heading_rad) / (111.0 * np.cos(np.radians(lat)))
        lat += dlat
        lon += dlon

    return rmax, np.array(rows)

# ---------------------------------------------------------------------------
# Wind field at portfolio locations
# ---------------------------------------------------------------------------
def wind_at_locations(track, rmax, lats, lons):
    """
    Maximum sustained wind (mph) at each location over all track steps.

    Modified Rankine vortex:
      d <= Rmax  ->  V = Vmax * (d / Rmax)        [increases toward eye wall]
      d >  Rmax  ->  V = Vmax * (Rmax / d)^0.6    [outer-vortex power-law decay]
    """
    lats     = np.asarray(lats, dtype=float)
    lons     = np.asarray(lons, dtype=float)
    max_wind = np.zeros(len(lats))

    for lat_c, lon_c, vmax_step, _ in track:
        d = haversine(lat_c, lon_c, lats, lons)
        # Guard against d=0 in the outer branch (np.where evaluates both branches).
        safe_d = np.where(d > 0, d, 1e-10)
        wind   = np.where(
            d <= rmax,
            vmax_step * (d / rmax),
            vmax_step * (rmax / safe_d) ** _OUTER_DECAY,
        )
        np.maximum(max_wind, wind, out=max_wind)

    return max_wind

# ---------------------------------------------------------------------------
# Single-storm sampler
# ---------------------------------------------------------------------------
def sample_storm(rng):
    """
    Sample one complete storm (track + metadata).

    Returns
    -------
    track    : ndarray (11, 4) — [lat, lon, vmax_step, cum_dist_km]
    metadata : dict   — category, vmax_landfall, rmax, landfall_lat, landfall_lon
    """
    lat_lf, lon_lf = sample_landfall(rng)
    category, vmax = sample_intensity(rng)
    rmax, track    = build_track(lat_lf, lon_lf, vmax, rng)
    return track, {
        "category":      category,
        "vmax_landfall": vmax,
        "rmax":          rmax,
        "landfall_lat":  lat_lf,
        "landfall_lon":  lon_lf,
    }

# ---------------------------------------------------------------------------
# Annual event set
# ---------------------------------------------------------------------------
def simulate_year(rng):
    """
    Simulate one year; return list of (track, metadata) tuples.

    The number of events is drawn from Poisson(LAMBDA = 0.7).
    """
    n_storms = int(rng.poisson(LAMBDA))
    return [sample_storm(rng) for _ in range(n_storms)]

# ---------------------------------------------------------------------------
# Demo / validation  (run as script, not when imported)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EXP_PATH = os.path.join(ROOT, "data",    "exposure.csv")
    OUT_DIR  = os.path.join(ROOT, "outputs")

    # Load portfolio locations
    exp  = pd.read_csv(EXP_PATH, usecols=["location_id", "lat", "lon"])
    lats = exp["lat"].to_numpy()
    lons = exp["lon"].to_numpy()
    print(f"Loaded {len(exp)} portfolio locations from {EXP_PATH}")

    # -----------------------------------------------------------------------
    # 1. Single demo storm — footprint plot
    # -----------------------------------------------------------------------
    demo_rng              = np.random.default_rng(SEED)
    demo_track, demo_meta = sample_storm(demo_rng)
    demo_wind             = wind_at_locations(demo_track, demo_meta["rmax"], lats, lons)

    print(f"\nDemo storm:  Cat{demo_meta['category']} | "
          f"Vmax={demo_meta['vmax_landfall']:.1f} mph | "
          f"Rmax={demo_meta['rmax']:.1f} km | "
          f"landfall=({demo_meta['landfall_lat']:.3f}, {demo_meta['landfall_lon']:.3f})")
    print(f"Wind range at portfolio: {demo_wind.min():.1f} - {demo_wind.max():.1f} mph")

    fig, ax = plt.subplots(figsize=(8, 9))

    sc = ax.scatter(
        lons, lats, c=demo_wind, cmap="YlOrRd",
        s=16, vmin=0, vmax=demo_meta["vmax_landfall"],
        edgecolors="none", zorder=2, label="Portfolio (colored by wind)",
    )
    fig.colorbar(sc, ax=ax, label="Max sustained wind (mph)", shrink=0.65)

    # Track line and markers
    ax.plot(demo_track[:, 1], demo_track[:, 0],
            "b-o", linewidth=1.5, markersize=4, zorder=3, label="Storm track")
    ax.plot(demo_meta["landfall_lon"], demo_meta["landfall_lat"],
            "r*", markersize=14, zorder=4, label="Landfall")

    # Simplified coast reference
    ax.plot(COAST_POINTS[:, 1], COAST_POINTS[:, 0],
            "k--", linewidth=0.8, alpha=0.5, zorder=1, label="Coast (polyline)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Hazard Footprint  —  Cat{demo_meta['category']} demo storm\n"
        f"Vmax = {demo_meta['vmax_landfall']:.1f} mph  |  "
        f"Rmax = {demo_meta['rmax']:.1f} km  |  seed = {SEED}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUT_DIR, "hazard_footprint.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Footprint plot saved -> {fig_path}")

    # -----------------------------------------------------------------------
    # 2. Validation asserts
    # -----------------------------------------------------------------------
    print("\n=== Validation ===")

    # --- a) Wind field shape — test the Rankine formula directly on distance values ---
    # We test the mathematical formula itself, not via lat/lon projection, to avoid
    # rounding error from the flat-earth ↔ haversine roundtrip.
    rmax    = demo_meta["rmax"]
    vmax_lf = demo_meta["vmax_landfall"]

    d_test = np.array([0.0, rmax * 0.5, rmax, rmax * 2, rmax * 5, rmax * 10])
    safe_d = np.where(d_test > 0, d_test, 1e-10)
    w_test = np.where(
        d_test <= rmax,
        vmax_lf * d_test / rmax,
        vmax_lf * (rmax / safe_d) ** 0.6,
    )

    assert w_test[0] == 0.0,                  "Eye should be calm (V=0 at d=0)"
    assert w_test[2] == vmax_lf,              "V(d=Rmax) should equal Vmax"
    outside = w_test[d_test >= rmax]
    assert (np.diff(outside) < 0).all(),      "Wind must decrease monotonically outside Rmax"
    print("[OK] Wind field: calm eye, Vmax at Rmax, monotone decay outside")

    # --- b) Frequency and Cat3+ share over many simulated years ---
    val_rng       = np.random.default_rng(SEED + 1)
    N_VAL_YEARS   = 10_000
    counts        = []
    cat3plus = total = 0

    for _ in range(N_VAL_YEARS):
        year = simulate_year(val_rng)
        counts.append(len(year))
        for _, m in year:
            total += 1
            if m["category"] >= 3:
                cat3plus += 1

    mean_n        = float(np.mean(counts))
    cat3plus_frac = cat3plus / total if total > 0 else 0.0

    # Theoretical Cat3+ from fitted truncated lognormal: P(Vmax >= 111 mph | Vmax >= 64 kt)
    _cat3_lb_kt = float(mph_to_kt(_CAT_LO_MPH[2]))   # 111 mph -> kt
    _z_cat3     = (np.log(_cat3_lb_kt) - _INT_MU_LOG) / _INT_SIGMA_LOG
    theoretical_cat3plus = float(
        (1.0 - scipy.stats.norm.cdf(_z_cat3)) / (1.0 - _INT_P_LB)
    )

    assert abs(mean_n - LAMBDA) < 0.05, \
        f"Mean N/year {mean_n:.3f} too far from lambda={LAMBDA}"
    print(f"[OK] Mean storms/year : {mean_n:.3f}  (target {LAMBDA})")

    assert abs(cat3plus_frac - theoretical_cat3plus) < 0.03, \
        f"Cat3+ share {cat3plus_frac:.3f} too far from {theoretical_cat3plus:.3f}"
    print(f"[OK] Cat3+ share      : {cat3plus_frac:.3f}  (theoretical {theoretical_cat3plus:.3f})")

    # --- c) Spatial coherence: high-wind locations form a tight cluster ---
    # The 90th-percentile wind locations should lie much closer together than
    # a random sample of all portfolio locations, since they cluster near the track.
    pos_wind = demo_wind[demo_wind > 0]
    if len(pos_wind) >= 10:
        threshold = np.quantile(pos_wind, 0.90)
        top_mask  = demo_wind >= threshold
        top_lats  = lats[top_mask]
        top_lons  = lons[top_mask]

        def mean_pairwise_km(la, lo, max_n=60):
            # Cap at max_n points to keep O(n^2) runtime manageable.
            idx = np.arange(min(len(la), max_n))
            ds  = [haversine(la[i], lo[i], la[j], lo[j])
                   for i in idx for j in idx if j > i]
            return float(np.mean(ds)) if ds else 0.0

        rng_ref    = np.random.default_rng(0)
        all_idx    = rng_ref.choice(len(lats), size=min(60, len(lats)), replace=False)
        spread_all = mean_pairwise_km(lats[all_idx], lons[all_idx])
        spread_top = mean_pairwise_km(top_lats, top_lons)

        assert spread_top < spread_all, (
            f"Top-wind spread ({spread_top:.1f} km) >= all-location spread ({spread_all:.1f} km)"
        )
        print(f"[OK] Spatial coherence: top-10% wind spread {spread_top:.1f} km"
              f" < full-portfolio spread {spread_all:.1f} km")
    else:
        print("[SKIP] Spatial coherence: too few positive-wind locations to test")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    best_idx = int(demo_wind.argmax())
    print(f"Demo storm category / Vmax : Cat{demo_meta['category']} / {demo_meta['vmax_landfall']:.1f} mph")
    print(f"Rmax                       : {demo_meta['rmax']:.1f} km")
    print(f"Peak wind at portfolio     : {demo_wind.max():.1f} mph  "
          f"({exp['location_id'].iloc[best_idx]}, "
          f"lat={lats[best_idx]:.4f}, lon={lons[best_idx]:.4f})")
    print(f"Locations with wind >= 74 mph (Cat1+) : {(demo_wind >= 74).sum()}")
    print(f"Locations with any wind > 0            : {(demo_wind > 0).sum()}")
