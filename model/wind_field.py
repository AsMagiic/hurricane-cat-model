"""
Wind field module for the Florida hurricane cat model.

Provides the radial wind profile and the per-location maximum-over-track computation.
All wind speeds in mph (sustained 1-minute).  Distances in km.
"""

from dataclasses import dataclass

import numpy as np

from model_config import load_model_cfg
from model.geo_utils import haversine, bearing

_mcfg         = load_model_cfg()
_OUTER_DECAY  = float(_mcfg.hazard.outer_decay_exponent)
_WIND_PROFILE = str(_mcfg.hazard.physics.wind_profile)
_B_METHOD     = str(_mcfg.hazard.physics.b_method)
_RHO          = float(_mcfg.hazard.physics.air_density_kg_m3)
_ASYMMETRY_ON = str(_mcfg.hazard.physics.translation_asymmetry) == "on"
_ASYM_FRAC    = float(_mcfg.hazard.physics.asymmetry_fraction)

assert _WIND_PROFILE in {"rankine", "holland"}, (
    f"hazard.physics.wind_profile must be 'rankine' or 'holland'; got {_WIND_PROFILE!r}"
)


def _validate_physics_config(wind_profile: str, b_method: str) -> None:
    """
    Raise ValueError for physics switch combinations that produce meaningless results.

    Specifically: wind_profile='holland' with b_method='constant' (b=0.0) makes
    (Rmax/r)^B = 1 everywhere, so V(r) = Vmax for ALL distances — every location
    sees full landfall intensity regardless of range. This catastrophically inflates AAL.
    """
    if wind_profile == "holland" and b_method == "constant":
        raise ValueError(
            "wind_profile=holland requires a positive B parameter. "
            "b_method='constant' sets b=0.0, which makes (Rmax/r)^B=1 everywhere and "
            "V(r)=Vmax for all distances — every location sees full landfall Vmax "
            "regardless of distance. Set b_method='vickery_wadhera', or use "
            "wind_profile='rankine'."
        )


_validate_physics_config(_WIND_PROFILE, _B_METHOD)


@dataclass
class StormParams:
    """
    Per-storm wind-field parameters.

    Fields active (Paso 2.1b):
        rmax        : float -- radius of maximum winds (km)
        b           : float -- Holland B parameter (dimensionless); required > 0 for Holland
        dp_mb       : float -- central pressure deficit (mb); required for Holland gradient formula
        lat         : float -- storm latitude (degrees); Coriolis parameter for Holland

    Placeholder fields for upcoming steps (default to 0.0; callers do not set them yet):
        heading_deg : float -- meteorological bearing (deg CW from N); Paso 2.2
        vt_kmh      : float -- translation speed (km/h);               Paso 2.2
    """
    rmax:        float
    heading_deg: float = 0.0
    vt_kmh:      float = 0.0
    b:           float = 0.0
    dp_mb:       float = 0.0
    lat:         float = 25.0


def _holland(d: np.ndarray, rmax: float, vmax_step: float, b: float,
             dp_mb: float, lat: float, rho: float) -> np.ndarray:
    """
    Holland (1980) full gradient-balance profile, anchored to vmax_step at r=Rmax.
    Wind speed in mph; distances in km; all internal computation in SI (m, Pa, kg/m³, s).

    Gradient-balance wind (Holland 1980, Eq. 4):
      f      = 2 × 7.292e-5 × sin(lat × π/180)          [Coriolis, s⁻¹]
      dp_pa  = dp_mb × 100                               [Pa]
      x      = (Rmax_m / r_m)^B
      Vg(r)  = sqrt( max( (B/rho)×x×dp_pa×exp(-x) + (r_m×f/2)², 0 ) ) - r_m×f/2

    Anchoring: V(r) = vmax_step × Vg(r) / Vg(Rmax)
      - Vg(Rmax) is evaluated analytically at x=1 (exact, no grid search).
      - Makes V(Rmax) = vmax_step × 1.0 = vmax_step exactly.
      - vmax_step (HURDAT2-calibrated, n=112) provides the amplitude;
        Holland (1980) provides the radial shape.

    d=0 eye guard: safe_d substitutes r=∞ where d=0 in the pressure term only.
      - Pressure term at r=∞: x=0 → 0·exp(0)=0 ✓
      - Coriolis at d=0: uses raw r_actual=0 → (0·f/2)²=0, subtract 0·f/2=0 ✓
      - Vg(0) = sqrt(0+0) − 0 = 0.0 exactly — calm eye, no nan ✓

    Parameters
    ----------
    d         : ndarray, km   — distances from storm centre to locations
    rmax      : float,   km   — radius of maximum winds
    vmax_step : float,   mph  — per-step decayed Vmax from track
    b         : float,   —    — Holland B (> 0; guaranteed by _validate_physics_config)
    dp_mb     : float,   mb   — central pressure deficit (always > 0 for real storms)
    lat       : float,   deg  — storm latitude (for Coriolis parameter)
    rho       : float,   kg/m³— air density (from config)

    Returns
    -------
    wind : ndarray, mph
    """
    dp_pa  = dp_mb * 100.0                                      # mb -> Pa
    rmax_m = rmax * 1000.0                                      # km -> m
    f      = 2.0 * 7.292e-5 * np.sin(np.radians(lat))          # Coriolis, s^-1

    # Pressure term: use safe_d (d=0 -> r=inf -> x=0 -> pressure_term=0)
    safe_d    = np.where(d > 0, d, np.inf)
    r_m       = safe_d * 1000.0                                  # km -> m (pressure)
    x         = (rmax_m / r_m) ** b
    p_term    = (b / rho) * x * dp_pa * np.exp(-x)

    # Coriolis term: raw d (r_actual=0 at eye gives zero Coriolis, no guard needed)
    r_act     = d * 1000.0                                       # km -> m (Coriolis)
    c_term_sq = (r_act * f / 2.0) ** 2

    vg = np.sqrt(np.maximum(p_term + c_term_sq, 0.0)) - r_act * f / 2.0

    # Anchor: evaluate Vg analytically at r=Rmax (x0=1 by definition)
    p0   = (b / rho) * np.exp(-1.0) * dp_pa                    # pressure at Rmax
    c0sq = (rmax_m * f / 2.0) ** 2                              # Coriolis² at Rmax
    vg0  = np.sqrt(p0 + c0sq) - rmax_m * f / 2.0               # Vg(Rmax), m/s

    return vmax_step * (vg / vg0)


def _rankine(d: np.ndarray, rmax: float, vmax_step: float,
             outer_decay_exponent: float) -> np.ndarray:
    """
    Modified Rankine vortex profile.  Wind speed in mph; distances in km.

    d <= rmax  ->  V = vmax_step * (d / rmax)               [linear ramp to eye wall]
    d >  rmax  ->  V = vmax_step * (rmax / d)^exponent      [outer power-law decay]

    safe_d guards the outer-branch division when d=0 (np.where evaluates both
    branches eagerly); d=0 always falls in the inner branch and returns V=0.
    """
    safe_d = np.where(d > 0, d, 1e-10)
    return np.where(
        d <= rmax,
        vmax_step * (d / rmax),
        vmax_step * (rmax / safe_d) ** outer_decay_exponent,
    )


def _apply_asymmetry(wind_sym: np.ndarray, bearing_deg: np.ndarray,
                     heading_deg: float, vt_mph: float, a: float) -> np.ndarray:
    """
    Schwerdt-Ho-Watkins (1979) / HAZUS-MH forward-motion asymmetry correction.

    V_total = max(0,  V_sym  +  a · Vt_mph · sin(bearing_deg − heading_deg))

    Angle convention — both in meteorological degrees (0=N, 90=E, clockwise):
      sin(β_loc − β_heading) = +1  when location is 90° CW from heading (RIGHT side)
                              = −1  when 90° CCW from heading (LEFT side)
                              =  0  directly ahead or behind

    The max(0, …) clip zeroes out the weak left-flank periphery where V_sym is already
    below the loss-relevant damage threshold. See test_clip_only_below_damage_threshold.

    Parameters
    ----------
    wind_sym    : ndarray, mph — symmetric radial profile from _holland or _rankine
    bearing_deg : ndarray, deg — meteorological bearing from storm centre to each location
    heading_deg : float,   deg — storm heading (direction of motion), meteorological
    vt_mph      : float,   mph — translation speed; caller converts from km/h
    a           : float,   —   — asymmetry fraction (config: asymmetry_fraction)

    Returns
    -------
    wind_asym : ndarray, mph
    """
    delta = np.radians(bearing_deg - heading_deg)
    return np.maximum(0.0, wind_sym + a * vt_mph * np.sin(delta))


def wind_at_locations(track: np.ndarray, storm_params: StormParams,
                      lats, lons) -> np.ndarray:
    """
    Maximum sustained wind (mph) at each location over all track steps.

    Parameters
    ----------
    track        : ndarray (N, 4) -- columns [lat, lon, vmax_step_mph, cum_dist_km]
    storm_params : StormParams    -- at minimum, rmax (km) must be set
    lats, lons   : array-like     -- portfolio location coordinates (degrees)

    Returns
    -------
    max_wind : ndarray (len(lats),) -- maximum sustained wind (mph) per location
    """
    lats     = np.asarray(lats, dtype=float)
    lons     = np.asarray(lons, dtype=float)
    max_wind = np.zeros(len(lats))

    for lat_c, lon_c, vmax_step, _ in track:
        d = haversine(lat_c, lon_c, lats, lons)
        if _WIND_PROFILE == "holland":
            wind = _holland(d, storm_params.rmax, vmax_step, storm_params.b,
                            storm_params.dp_mb, storm_params.lat, _RHO)
        else:
            wind = _rankine(d, storm_params.rmax, vmax_step, _OUTER_DECAY)
        if _ASYMMETRY_ON:
            brg    = bearing(lat_c, lon_c, lats, lons)
            vt_mph = storm_params.vt_kmh * 0.621371   # km/h -> mph
            wind   = _apply_asymmetry(wind, brg, storm_params.heading_deg, vt_mph, _ASYM_FRAC)
        np.maximum(max_wind, wind, out=max_wind)

    return max_wind


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    VMAX      = 120.0   # mph — Cat 3 mid-range
    RMAX      = 40.0    # km
    B         = 1.2     # typical V&W value
    DP_MB     = 60.0    # mb — representative Cat 3 pressure deficit
    LAT       = 25.0    # degrees — typical FL landfall latitude
    R_MAX_PLOT = 300.0  # km

    r = np.linspace(0.1, R_MAX_PLOT, 1000)

    v_holland = _holland(r, RMAX, VMAX, B, DP_MB, LAT, _RHO)
    v_rankine = _rankine(r, RMAX, VMAX, _OUTER_DECAY)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, v_holland, "b-",  linewidth=2.0, label=f"Holland (1980)  B={B}")
    ax.plot(r, v_rankine, "r--", linewidth=2.0,
            label=f"Rankine  exponent={_OUTER_DECAY}")
    ax.axvline(RMAX, color="gray", linestyle=":", linewidth=1.2, label=f"Rmax = {RMAX} km")
    ax.axhline(VMAX, color="gray", linestyle="-.", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Distance from storm centre (km)", fontsize=11)
    ax.set_ylabel("Wind speed (mph, 1-min sustained)", fontsize=11)
    ax.set_title(
        f"Rankine vs Holland radial profiles  |  Cat 3  Vmax={VMAX} mph  Rmax={RMAX} km",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, R_MAX_PLOT)
    ax.set_ylim(0, VMAX * 1.05)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wind_profile_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
