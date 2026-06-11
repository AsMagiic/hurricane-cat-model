"""
Wind field module for the Florida hurricane cat model.

Provides the radial wind profile and the per-location maximum-over-track computation.
All wind speeds in mph (sustained 1-minute).  Distances in km.
"""

from dataclasses import dataclass

import numpy as np

from model_config import load_model_cfg
from model.geo_utils import haversine

_mcfg        = load_model_cfg()
_OUTER_DECAY = float(_mcfg.hazard.outer_decay_exponent)


@dataclass
class StormParams:
    """
    Per-storm wind-field parameters.

    Fields active now (Step 2.1a):
        rmax        : float -- radius of maximum winds (km)

    Placeholder fields for upcoming steps (default to 0.0; callers do not set them yet):
        heading_deg : float -- meteorological bearing (deg CW from N); Paso 2.2
        vt_kmh      : float -- translation speed (km/h);               Paso 2.2
        b           : float -- Holland B parameter (dimensionless);     Paso 2.1b
    """
    rmax:        float
    heading_deg: float = 0.0
    vt_kmh:      float = 0.0
    b:           float = 0.0


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
        d    = haversine(lat_c, lon_c, lats, lons)
        wind = _rankine(d, storm_params.rmax, vmax_step, _OUTER_DECAY)
        np.maximum(max_wind, wind, out=max_wind)

    return max_wind
