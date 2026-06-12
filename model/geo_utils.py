"""
Geographic utility functions shared across model modules.
"""

import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km.  Inputs may be scalars or numpy arrays."""
    R    = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    """
    Forward bearing from (lat1, lon1) to (lat2, lon2) in meteorological degrees
    (0 = N, 90 = E, 180 = S, 270 = W, clockwise, range [0, 360)).

    Uses the standard forward-azimuth formula. Inputs may be scalars or numpy
    arrays; scalar centre (lat1, lon1) broadcast against array locations works
    identically to haversine().

    At coincident points (lat1==lat2, lon1==lon2): arctan2(0, 0) returns 0°
    (north) — harmless since haversine gives d=0 and wind is 0 at the eye.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    y    = np.sin(dlam) * np.cos(phi2)
    x    = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return np.degrees(np.arctan2(y, x)) % 360
