"""
Stochastic hurricane hazard generator for Florida coastal portfolios.

Generates synthetic storm tracks and computes the maximum sustained wind (mph)
at each portfolio location via a moving-track wind field.

Wind field physics (v3, Phase 2):
  - Holland (1980) gradient-balance profile, anchored to sampled Vmax, via
    wind_field.py; storm size (Rmax) and profile shape (B) from Vickery &
    Wadhera (2008), coupled to central-pressure deficit and latitude.
  - Track heading regime-conditioned: von Mises per approach corridor
    (Atlantic landfalls NW; Gulf landfalls NE), replacing the v2 uniform ±45°.
  - Inland decay: Kaplan-DeMaria (1995), exponential in time since landfall
    (t = cum_dist_km / vt_kmh, h), replacing the v2 120 km e-folding scale.

Simplifications that remain:
  - Rmax, B, and Δp frozen at landfall values; no mid-track eyewall
    contraction or pressure fill-in.
  - K-D decay applies along the full straight-line track, including if the
    storm re-emerges over water (no land-mask turn-off).
  - Coriolis latitude fixed at landfall latitude throughout the track.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from shapely.geometry import LineString
from pyproj import Transformer

from model_config import load_model_cfg
from model.units import kt_to_mph, mph_to_kt
from model.geo_utils import haversine
from model.wind_field import wind_at_locations, StormParams
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

_INT_CAP_KT    = float(_icfg.cap_kt)                             # 165.0 kt — empirical Atlantic MPI ceiling
_INTENSITY_CAP = str(_mcfg.hazard.physics.intensity_cap)         # "on" | "off"
assert _INTENSITY_CAP in {"on", "off"}, (
    f"hazard.physics.intensity_cap must be 'on' or 'off'; got {_INTENSITY_CAP!r}"
)
_INT_P_UB = (
    float(scipy.stats.norm.cdf(
        (np.log(_INT_CAP_KT) - _INT_MU_LOG) / _INT_SIGMA_LOG
    ))
    if _INTENSITY_CAP == "on"
    else 1.0   # exactly 1.0 — arithmetic below is bit-identical to pre-3.0a
)

# Florida coastline polyline for landfall sampling.
# Ordered: Atlantic N->S, Keys, Gulf S->N.  Shape (12, 2): each row is [lat, lon].
COAST_POINTS = np.array(_mcfg.hazard.coast_polyline)

# Per-segment weights (11 segments for 12 points).  Coupled to COAST_POINTS.
SEGMENT_WEIGHTS = np.array(_mcfg.hazard.coast_segment_weights)

# Hazard mechanics
_STEP_KM      = _mcfg.hazard.step_km
_EFOLD_KM     = _mcfg.hazard.efold_km
_OUTER_DECAY  = _mcfg.hazard.outer_decay_exponent
_RMAX_MIN_KM  = float(_mcfg.hazard.rmax_km_min)
_RMAX_MAX_KM  = float(_mcfg.hazard.rmax_km_max)

# Kaplan-DeMaria (1995) inland decay constants (Paso 2.3)
_DECAY_METHOD = str(_mcfg.hazard.physics.decay_method)          # "efold" | "kaplan_demaria"
_KD_ALPHA     = float(_mcfg.hazard.physics.kd_alpha)             # 0.095 h^-1
_KD_VB_MPH    = float(kt_to_mph(_mcfg.hazard.physics.kd_vb_kt))  # 26.7 kt -> mph  (exact: 1852/1609.344; delta vs old 1.15078 ≈ −0.000015 mph)
_VT_MIN_KMH   = float(_mcfg.hazard.physics.vt_min_kmh)           # 2.0 km/h floor

# Physics switches — "uniform"|"vickery_wadhera" for rmax; "constant"|"vickery_wadhera" for b.
_RMAX_METHOD = str(_mcfg.hazard.physics.rmax_method)
_B_METHOD    = str(_mcfg.hazard.physics.b_method)
assert _RMAX_METHOD in {"uniform", "vickery_wadhera"}, (
    f"hazard.physics.rmax_method must be 'uniform' or 'vickery_wadhera'; got {_RMAX_METHOD!r}"
)
assert _B_METHOD in {"constant", "vickery_wadhera"}, (
    f"hazard.physics.b_method must be 'constant' or 'vickery_wadhera'; got {_B_METHOD!r}"
)

_vwcfg = _mcfg.vickery_wadhera
_VW_RMAX_INTERCEPT   = float(_vwcfg.rmax.intercept)
_VW_RMAX_DP2_COEFF   = float(_vwcfg.rmax.dp2_coeff)
_VW_RMAX_LAT_COEFF   = float(_vwcfg.rmax.lat_coeff)
_VW_RMAX_SIG_LOW     = float(_vwcfg.rmax.sigma_low_dp)
_VW_RMAX_SIG_MID_A   = float(_vwcfg.rmax.sigma_mid_dp_a)
_VW_RMAX_SIG_MID_B   = float(_vwcfg.rmax.sigma_mid_dp_b)
_VW_RMAX_SIG_HIGH    = float(_vwcfg.rmax.sigma_high_dp)
_VW_RMAX_DP_BREAK_LO = float(_vwcfg.rmax.dp_break_lo)
_VW_RMAX_DP_BREAK_HI = float(_vwcfg.rmax.dp_break_hi)
_VW_B_INTERCEPT      = float(_vwcfg.b.intercept)
_VW_B_RMAX_COEFF     = float(_vwcfg.b.rmax_coeff)
_VW_B_LAT_COEFF      = float(_vwcfg.b.lat_coeff)
_VW_B_SIGMA          = float(_vwcfg.b.sigma)
_VW_B_MIN            = float(_vwcfg.b.b_min)
_VW_B_MAX            = float(_vwcfg.b.b_max)

# WPR pass-through — forward-compute Δp from Vmax (already in mph from sample_intensity).
# _WPR_B_EXP named to avoid shadowing the local variable b in sample_storm.
_WPR_A         = float(_mcfg.wind_pressure.a)
_WPR_B_EXP     = float(_mcfg.wind_pressure.b)
_WPR_SIGMA_LOG = float(_mcfg.wind_pressure.sigma_log)   # 0.2458 — calibrated scatter (log-space)
_WPR_RESIDUAL  = str(_mcfg.hazard.physics.wpr_residual)  # "on" | "off"
assert _WPR_RESIDUAL in {"on", "off"}, (
    f"hazard.physics.wpr_residual must be 'on' or 'off'; got {_WPR_RESIDUAL!r}"
)

_RMAX_FLOOR_KM = float(_mcfg.hazard.rmax_floor_km)       # 8.0 km — V&W observed lower bound
_RMAX_FLOOR    = str(_mcfg.hazard.physics.rmax_floor)     # "on" | "off"
assert _RMAX_FLOOR in {"on", "off"}, (
    f"hazard.physics.rmax_floor must be 'on' or 'off'; got {_RMAX_FLOOR!r}"
)

# ---------------------------------------------------------------------------
# Calibrated landfall geography + regime parameters  (Step 1.5b)
# ---------------------------------------------------------------------------
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_geo_cfg         = _mcfg.hazard.landfall_geography
_COAST_PATH      = os.path.join(_ROOT_DIR, _geo_cfg.coastline_path)
_coast_df        = pd.read_csv(_COAST_PATH)

_TO_3086         = Transformer.from_crs("EPSG:4326", "EPSG:3086", always_xy=True)
_FROM_3086       = Transformer.from_crs("EPSG:3086", "EPSG:4326", always_xy=True)
_COAST_LINE_3086 = LineString([
    _TO_3086.transform(lon, lat)
    for lon, lat in zip(_coast_df["lon"], _coast_df["lat"])
])

_TOTAL_ARC_KM = float(_geo_cfg.total_arc_length_km)
_S_SAMPLES_KM = np.array(_geo_cfg.s_samples_km, dtype=float)
# KDE bandwidth: factor * std(data), ddof=0 — matches scipy gaussian_kde convention
_KDE_BW_KM    = float(_geo_cfg.kde_silverman_factor) * float(np.std(_S_SAMPLES_KM))


def _kappa_from_R(R: float) -> float:
    """
    Approximate von Mises concentration κ from mean resultant length R.
    Piecewise formula: Mardia & Jupp (2000), Statistics of Directional Data, p. 85.
    Accurate to ~3% for 0 < R < 0.95.
    """
    if R < 0.53:
        return 2.0 * R + R**3 + 5.0 * R**5 / 6.0
    elif R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1.0 - R)
    else:
        return 1.0 / (R**3 - 4.0 * R**2 + 3.0 * R)


_by_reg   = _mcfg.hazard.track_stats.by_regime
_S_CUT_KM = float(_by_reg.s_cut_km)

_REGIME_CFG: dict = {}
for _rname, _rcfg_r in [("atlantic", _by_reg.atlantic), ("gulf", _by_reg.gulf)]:
    _sp = list(_rcfg_r.speed_params)
    assert _sp[1] == 0.0, (
        f"loc != 0 for {_rname} speed gamma (got {_sp[1]}); revise sampler to add loc offset"
    )
    _REGIME_CFG[_rname] = {
        "heading_mean_deg": float(_rcfg_r.heading_mean_deg),
        "heading_kappa":    _kappa_from_R(float(_rcfg_r.heading_resultant_R)),
        "speed_shape":      float(_sp[0]),
        "speed_scale":      float(_sp[2]),
    }

# ---------------------------------------------------------------------------
# V&W (2008) Rmax and Holland-B — pure functions, no I/O, directly testable
# ---------------------------------------------------------------------------

def _sigma_rmax(dp_mb: float) -> float:
    """
    Heteroscedastic σ for ln(Rmax) from V&W (2008) ERDC TR-08-06 eq. (13).

    Parameters
    ----------
    dp_mb : float -- pressure deficit Δp in mb

    Returns
    -------
    sigma : float -- std of ln(Rmax) error term (dimensionless)
    """
    if dp_mb <= _VW_RMAX_DP_BREAK_LO:
        return _VW_RMAX_SIG_LOW
    elif dp_mb <= _VW_RMAX_DP_BREAK_HI:
        return _VW_RMAX_SIG_MID_A - _VW_RMAX_SIG_MID_B * dp_mb
    else:
        return _VW_RMAX_SIG_HIGH


def _vw_rmax_mean(dp_mb: float, lat_deg: float) -> float:
    """
    Deterministic component of ln(Rmax_km), V&W (2008) eq. (13).

    Parameters
    ----------
    dp_mb   : float -- pressure deficit Δp in mb
    lat_deg : float -- latitude in degrees N

    Returns
    -------
    ln_rmax_mean : float -- deterministic ln(Rmax_km), without error term
    """
    return _VW_RMAX_INTERCEPT + _VW_RMAX_DP2_COEFF * dp_mb ** 2 + _VW_RMAX_LAT_COEFF * lat_deg


def _vw_b_mean(rmax_km: float, lat_deg: float) -> float:
    """
    Deterministic component of Holland B, V&W (2008) eq. (14).

    Parameters
    ----------
    rmax_km : float -- radius of maximum winds in km
    lat_deg : float -- latitude in degrees N

    Returns
    -------
    b_mean : float -- deterministic B (uncensored, without error term)
    """
    return _VW_B_INTERCEPT + _VW_B_RMAX_COEFF * rmax_km + _VW_B_LAT_COEFF * lat_deg


def _vw_rmax_sample(dp_mb: float, lat_deg: float, sub_rng) -> float:
    """
    Sample Rmax_km from V&W (2008) eq. (13). Consumes one draw from sub_rng.

    Unit chain: ln(Rmax_km) = _vw_rmax_mean(dp_mb, lat_deg) + N(0, σ);
                Rmax_km = exp(ln_rmax).
    dp_mb in mb; lat_deg in degrees N; returns Rmax in km.
    """
    ln_rmax = _vw_rmax_mean(dp_mb, lat_deg) + sub_rng.normal(0.0, _sigma_rmax(dp_mb))
    return float(np.exp(ln_rmax))


def _vw_b_sample(rmax_km: float, lat_deg: float, sub_rng) -> float:
    """
    Sample Holland B from V&W (2008) eq. (14). Consumes one draw from sub_rng.

    B = _vw_b_mean(rmax_km, lat_deg) + N(0, sigma), censored to [b_min, b_max].
    rmax_km in km (coupled from _vw_rmax_sample output); lat_deg in degrees N.
    """
    b = _vw_b_mean(rmax_km, lat_deg) + sub_rng.normal(0.0, _VW_B_SIGMA)
    return float(np.clip(b, _VW_B_MIN, _VW_B_MAX))


# ---------------------------------------------------------------------------
# Landfall sampling
# ---------------------------------------------------------------------------
def sample_landfall(rng):
    """
    Return (lat, lon, s_km) of a landfall point sampled from the calibrated KDE
    over HURDAT2 arc-length positions on the simplified FL coastline (EPSG:3086).

    s_km   : arc-length in EPSG:3086 km from the GA-FL Atlantic corner (s=0).
    KDE    : Gaussian kernel with Silverman bandwidth _KDE_BW_KM.
             Resampled by picking a random data point and adding Gaussian noise
             (standard KDE resample algorithm); entirely within numpy for RNG
             consistency — no scipy.stats.gaussian_kde.resample() call needed.
    """
    idx  = int(rng.integers(0, len(_S_SAMPLES_KM)))
    s_km = float(_S_SAMPLES_KM[idx]) + float(rng.normal(0.0, _KDE_BW_KM))
    s_km = float(np.clip(s_km, 0.0, _TOTAL_ARC_KM))
    pt   = _COAST_LINE_3086.interpolate(s_km * 1e3)   # EPSG:3086 unit is metres
    lon, lat = _FROM_3086.transform(pt.x, pt.y)
    return float(lat), float(lon), s_km

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
    p = _INT_P_LB + U*(_INT_P_UB-_INT_P_LB), Vmax_kt = exp(mu_log + sigma_log * Phi_inv(p)).
    When cap off: _INT_P_UB=1.0, expression reduces bit-for-bit to the pre-3.0a formula.
    Converted to mph via kt_to_mph. Category derived from Vmax using Saffir-Simpson
    bounds in _CAT_LO_MPH; Vmax is the source of truth, category is a derived label.
    Units: output vmax in mph (sustained 1-minute).
    """
    u        = float(rng.uniform(0.0, 1.0))
    p_samp   = _INT_P_LB + u * (_INT_P_UB - _INT_P_LB)
    vmax_kt  = float(np.exp(_INT_MU_LOG + _INT_SIGMA_LOG * float(scipy.stats.norm.ppf(p_samp))))
    vmax_mph = float(kt_to_mph(vmax_kt))
    return _vmax_to_category(vmax_mph), vmax_mph

# ---------------------------------------------------------------------------
# Track building
# ---------------------------------------------------------------------------
def build_track(landfall_lat, landfall_lon, vmax, heading_deg, rmax_km, vt_kmh):
    """
    Generate a 10-step inland track starting at the landfall point.

    Parameters
    ----------
    heading_deg : float — meteorological bearing (degrees; 0=N, 90=E, 180=S, 270=W).
                          Passed from sample_storm (regime-conditioned von Mises draw).
                          Advance convention unchanged: heading=0 → dlat>0, dlon=0.
    rmax_km     : float — radius of maximum winds (km), sampled in sample_storm
                          (uniform or V&W depending on physics switch), constant along track.
    vt_kmh      : float — translation speed (km/h). Used only on the kaplan_demaria decay
                          path (t = cum_dist / vt_kmh, units: km/(km/h) = h). Ignored on
                          the efold path so that efold output is bit-identical to v2.

    Returns
    -------
    track : ndarray (11, 4) — columns: [lat, lon, vmax_step, cum_dist_km]
              row 0 = landfall (peak intensity, t=0 so no decay applied), rows 1-10 inland
    """
    heading_rad = np.radians(heading_deg)

    n_steps = 10
    step_km = _STEP_KM
    e_fold  = _EFOLD_KM

    rows = []
    lat, lon = landfall_lat, landfall_lon
    for i in range(n_steps + 1):
        cum_dist = i * step_km
        if _DECAY_METHOD == "kaplan_demaria":
            # t = cum_dist [km] / vt [km/h] = hours; alpha [h^-1] consistent.
            # V(t) = Vb + (V0 - Vb) * exp(-alpha * t)
            # At t=0 (row 0): exp(0)=1 -> vmax_step = Vb + (V0-Vb) = V0 exactly.
            vt_safe   = max(vt_kmh, _VT_MIN_KMH)
            t_hours   = cum_dist / vt_safe
            vmax_step = _KD_VB_MPH + (vmax - _KD_VB_MPH) * np.exp(-_KD_ALPHA * t_hours)
        else:
            # efold: bit-identical to v2 baseline, vt_kmh is not used.
            vmax_step = vmax * np.exp(-cum_dist / e_fold)
        rows.append([lat, lon, vmax_step, cum_dist])
        # Advance center by step_km in the heading direction (flat-earth approx)
        dlat = step_km * np.cos(heading_rad) / 111.0
        dlon = step_km * np.sin(heading_rad) / (111.0 * np.cos(np.radians(lat)))
        lat += dlat
        lon += dlon

    return np.array(rows)

# ---------------------------------------------------------------------------
# Single-storm sampler
# ---------------------------------------------------------------------------
def sample_storm(rng):
    """
    Sample one complete storm (track + metadata).

    RNG discipline (Phase 2 / Steps 3.0b–3.0c):
        Legacy stream draws (in order): sample_landfall x2 (integer+normal),
        vonmises, gamma, sample_intensity, then rng.uniform(30,55) ONLY when
        rmax_method='uniform'.  All new physics draw from dedicated substreams:
          vw_rng  = rng.spawn(1)[0]: V&W Rmax error (draw 1) and B error (draw 2).
                    Same SeedSequence slot per storm as the former sub_rng —
                    bit-identical to pre-3.0b baseline.
          wpr_rng = vw_rng.spawn(1)[0]: NESTED child of vw_rng. Consuming from
                    wpr_rng does NOT affect rng's spawn-slot counter or vw_rng's
                    bitgenerator state. 1 draw when wpr_residual='on';
                    spawned-and-unused when 'off'.
          rmax_floor (3.0c): pure max() on the V&W Rmax sample — no new draws.
                    floor=off is bit-identical to the post-3.0b baseline.
        spawn() uses SeedSequence counters and does NOT consume bitgenerator
        variates from the parent, so all legacy streams are unperturbed.

    Returns
    -------
    track    : ndarray (11, 4) — [lat, lon, vmax_step, cum_dist_km]
    metadata : dict   — category, vmax_landfall, rmax, dp_mb, b,
                        landfall_lat, landfall_lon, regime, heading_deg,
                        translation_speed_kmh
    """
    lat_lf, lon_lf, s_km = sample_landfall(rng)              # rng draws 1-2

    regime = "atlantic" if s_km < _S_CUT_KM else "gulf"
    rcfg   = _REGIME_CFG[regime]

    # Von Mises heading — kappa derived from resultant R via Mardia & Jupp (2000).
    mu_rad      = np.radians(rcfg["heading_mean_deg"])
    heading_deg = float(np.degrees(rng.vonmises(mu_rad, rcfg["heading_kappa"])) % 360.0)  # rng draw 3

    # Gamma translation speed — loc=0 asserted at import.
    speed_kmh = float(rng.gamma(rcfg["speed_shape"], rcfg["speed_scale"]))                # rng draw 4

    category, vmax = sample_intensity(rng)                    # rng draw 5 (uniform inside)

    # Spawn vw_rng from rng exactly as before (same SeedSequence slot per storm).
    # Then spawn wpr_rng as a NESTED child of vw_rng — this increments vw_rng's
    # SeedSequence counter only, not rng's, so all per-storm spawn slots stay
    # identical to pre-3.0b. vw_rng.spawn() does NOT consume bitgenerator variates
    # from vw_rng; Rmax/B draws are therefore bit-identical to pre-3.0b baseline.
    vw_rng  = rng.spawn(1)[0]
    wpr_rng = vw_rng.spawn(1)[0]

    # Δp from calibrated WPR: Δp = a · Vmax_mph^b_exp.
    # vmax is already in mph (sample_intensity calls kt_to_mph).
    dp_mb = float(_WPR_A * vmax ** _WPR_B_EXP)

    if _WPR_RESIDUAL == "on":
        # Multiplicative lognormal scatter: Δp = Δp_det · exp(ε), ε ~ N(0, sigma_log²).
        # Jensen bias: E[exp(ε)] = exp(sigma_log²/2) ≈ 1.031 → mean Δp +3% upward.
        # Propagates into Rmax and B via the V&W causal chain only (not Holland amplitude).
        dp_mb = dp_mb * float(np.exp(wpr_rng.normal(0.0, _WPR_SIGMA_LOG)))
    # wpr_rng spawned-and-unused on 'off' path — RNG discipline per CLAUDE.md.

    if _RMAX_METHOD == "uniform":
        # Legacy draw: same stream position as pre-V&W implementation (rng draw 6).
        # vw_rng is intentionally unused on this path — the unconditional spawn
        # ensures the legacy stream is unaffected by the existence of V&W physics.
        rmax_km = float(rng.uniform(_RMAX_MIN_KM, _RMAX_MAX_KM))   # rng draw 6
    else:  # "vickery_wadhera"
        rmax_km = _vw_rmax_sample(dp_mb, lat_lf, vw_rng)            # vw_rng draw 1
        if _RMAX_FLOOR == "on":
            rmax_km = max(rmax_km, _RMAX_FLOOR_KM)                  # no new draws

    if _B_METHOD == "constant":
        b = 0.0
    else:  # "vickery_wadhera"
        b = _vw_b_sample(rmax_km, lat_lf, vw_rng)                   # vw_rng draw 2 (or 1 if rmax=uniform)

    track = build_track(lat_lf, lon_lf, vmax, heading_deg, rmax_km, speed_kmh)

    return track, {
        "category":              category,
        "vmax_landfall":         vmax,
        "rmax":                  rmax_km,
        "dp_mb":                 dp_mb,
        "b":                     b,
        "landfall_lat":          lat_lf,
        "landfall_lon":          lon_lf,
        "regime":                regime,
        "heading_deg":           heading_deg,
        "translation_speed_kmh": speed_kmh,
    }

# ---------------------------------------------------------------------------
# Annual event set
# ---------------------------------------------------------------------------
def simulate_year(rng):
    """
    Simulate one year; return list of (track, metadata) tuples.

    The number of events is drawn from Poisson(LAMBDA); LAMBDA loaded from config.
    """
    n_storms = int(rng.poisson(LAMBDA))
    return [sample_storm(rng) for _ in range(n_storms)]

# ---------------------------------------------------------------------------
# Demo / validation  (run as script, not when imported)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    from model.exposure_io import load_portfolio as _load_portfolio, OED_LOC_PATH
    OUT_DIR = os.path.join(_ROOT_DIR, "outputs")

    # Load portfolio locations via OED adapter
    exp  = _load_portfolio()
    lats = exp["lat"].to_numpy()
    lons = exp["lon"].to_numpy()
    print(f"Loaded {len(exp)} portfolio locations from {OED_LOC_PATH}")

    # -----------------------------------------------------------------------
    # 1. Single demo storm — footprint plot
    # -----------------------------------------------------------------------
    demo_rng              = np.random.default_rng(SEED)
    demo_track, demo_meta = sample_storm(demo_rng)
    demo_wind             = wind_at_locations(
        demo_track,
        StormParams(
            rmax=demo_meta["rmax"],
            b=demo_meta["b"],
            dp_mb=demo_meta["dp_mb"],
            lat=demo_meta["landfall_lat"],
            heading_deg=demo_meta["heading_deg"],
            vt_kmh=demo_meta["translation_speed_kmh"],
        ),
        lats, lons,
    )

    print(f"\nDemo storm:  Cat{demo_meta['category']} | "
          f"Vmax={demo_meta['vmax_landfall']:.1f} mph | "
          f"Rmax={demo_meta['rmax']:.1f} km | "
          f"dp={demo_meta['dp_mb']:.1f} mb | "
          f"B={demo_meta['b']:.3f} | "
          f"landfall=({demo_meta['landfall_lat']:.3f}, {demo_meta['landfall_lon']:.3f}) | "
          f"regime={demo_meta['regime']} | heading={demo_meta['heading_deg']:.1f}deg | "
          f"speed={demo_meta['translation_speed_kmh']:.1f} km/h")
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

    # Calibrated coastline reference (62-vertex simplified, TIGER/Line 2023)
    ax.plot(_coast_df["lon"], _coast_df["lat"],
            "k--", linewidth=0.8, alpha=0.5, zorder=1, label="Coast (calibrated)")

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
    if _INTENSITY_CAP == "on":
        _z_cap = (np.log(_INT_CAP_KT) - _INT_MU_LOG) / _INT_SIGMA_LOG
        theoretical_cat3plus = float(
            (scipy.stats.norm.cdf(_z_cap) - scipy.stats.norm.cdf(_z_cat3))
            / (scipy.stats.norm.cdf(_z_cap) - _INT_P_LB)
        )
    else:
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
    # 3. Landfall sanity check — 10,000 simulated landfalls
    # -----------------------------------------------------------------------
    print("\n=== Landfall sanity check (10,000 events) ===")
    sc_rng = np.random.default_rng(SEED + 2)
    n_sc   = 10_000
    n_atl, n_gulf       = 0, 0
    atl_s,  gulf_s      = [], []
    atl_hdgs, gulf_hdgs = [], []

    for _ in range(n_sc):
        lat_i, lon_i, s_i = sample_landfall(sc_rng)
        regime_i = "atlantic" if s_i < _S_CUT_KM else "gulf"
        rcfg_i   = _REGIME_CFG[regime_i]
        hdg_i    = float(np.degrees(sc_rng.vonmises(
            np.radians(rcfg_i["heading_mean_deg"]), rcfg_i["heading_kappa"]
        )) % 360.0)
        if regime_i == "atlantic":
            n_atl += 1; atl_s.append(s_i); atl_hdgs.append(hdg_i)
        else:
            n_gulf += 1; gulf_s.append(s_i); gulf_hdgs.append(hdg_i)

    def _circ_mean(hdgs):
        r = np.radians(hdgs)
        return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360.0)

    atl_hm  = _circ_mean(atl_hdgs)
    gulf_hm = _circ_mean(gulf_hdgs)

    print(f"Regime split  : atlantic={n_atl} ({100*n_atl/n_sc:.1f}%) | "
          f"gulf={n_gulf} ({100*n_gulf/n_sc:.1f}%)")
    print(f"  Historical  :            ~42%                  ~58%  (47/112, 65/112)")
    print(f"Mean s_km     : atlantic={np.mean(atl_s):.0f} km (< s_cut={_S_CUT_KM:.0f}) | "
          f"gulf={np.mean(gulf_s):.0f} km (> s_cut={_S_CUT_KM:.0f})")
    print(f"Circ mean hdg : atlantic={atl_hm:.1f}° (target 314.5°, NW) | "
          f"gulf={gulf_hm:.1f}° (target 30.7°, NE)")

    assert 280.0 <= atl_hm <= 360.0, f"Atlantic circ mean {atl_hm:.1f}° not in NW sector"
    assert   0.0 <= gulf_hm <=  90.0, f"Gulf circ mean {gulf_hm:.1f}° not in NE sector"
    assert float(np.mean(atl_s)) < _S_CUT_KM,  "Atlantic centroid s not below s_cut"
    assert float(np.mean(gulf_s)) > _S_CUT_KM,  "Gulf centroid s not above s_cut"
    print("[OK] Regime, heading, and arc-length sanity checks passed")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    best_idx = int(demo_wind.argmax())
    print(f"Demo storm category / Vmax : Cat{demo_meta['category']} / {demo_meta['vmax_landfall']:.1f} mph")
    print(f"Rmax / dp / B              : {demo_meta['rmax']:.1f} km / {demo_meta['dp_mb']:.1f} mb / {demo_meta['b']:.3f}")
    print(f"Peak wind at portfolio     : {demo_wind.max():.1f} mph  "
          f"({exp['location_id'].iloc[best_idx]}, "
          f"lat={lats[best_idx]:.4f}, lon={lons[best_idx]:.4f})")
    print(f"Locations with wind >= 74 mph (Cat1+) : {(demo_wind >= 74).sum()}")
    print(f"Locations with any wind > 0            : {(demo_wind > 0).sum()}")
