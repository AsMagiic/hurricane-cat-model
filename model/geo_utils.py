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
