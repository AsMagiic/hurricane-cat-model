"""
Unit conversion functions for wind speeds.

Conversion factors derived from exact SI definitions:
  1 nautical mile = 1852 m      (exact, SI)
  1 statute mile  = 1609.344 m  (exact, SI)
  1 hour          = 3600 s      (exact)

Derived (computed once at import):
  1 kt  = 1852 / 3600 m/s        = 0.51444... m/s
  1 mph = 1609.344 / 3600 m/s    = 0.44704 m/s  (exact)
  1 kt  = 1852 / 1609.344 mph    = 1.15078... mph

Do NOT hardcode conversion factors elsewhere in the codebase.
Import and use these functions instead.
"""

import numpy as np

# Single source of truth for all conversion factors.
_KT_TO_MS   = 1852.0 / 3600.0          # m/s per kt   (exact by SI definition)
_MS_PER_MPH = 1609.344 / 3600.0        # m/s per mph  (exact by SI definition)
_KT_TO_MPH  = _KT_TO_MS / _MS_PER_MPH  # mph per kt   = 1852 / 1609.344 ≈ 1.15078


def kt_to_mph(v_kt):
    """
    Convert wind speed from knots (kt) to miles per hour (mph).

    Parameters
    ----------
    v_kt : float or array-like  -- wind speed in knots (kt)

    Returns
    -------
    v_mph : ndarray or scalar   -- wind speed in miles per hour (mph)
    """
    return np.asarray(v_kt, dtype=float) * _KT_TO_MPH


def kt_to_ms(v_kt):
    """
    Convert wind speed from knots (kt) to metres per second (m/s).

    Parameters
    ----------
    v_kt : float or array-like  -- wind speed in knots (kt)

    Returns
    -------
    v_ms : ndarray or scalar    -- wind speed in metres per second (m/s)
    """
    return np.asarray(v_kt, dtype=float) * _KT_TO_MS


def ms_to_kt(v_ms):
    """
    Convert wind speed from metres per second (m/s) to knots (kt).

    Parameters
    ----------
    v_ms : float or array-like  -- wind speed in metres per second (m/s)

    Returns
    -------
    v_kt : ndarray or scalar    -- wind speed in knots (kt)
    """
    return np.asarray(v_ms, dtype=float) / _KT_TO_MS


def mph_to_kt(v_mph):
    """
    Convert wind speed from miles per hour (mph) to knots (kt).

    Parameters
    ----------
    v_mph : float or array-like -- wind speed in miles per hour (mph)

    Returns
    -------
    v_kt : ndarray or scalar    -- wind speed in knots (kt)
    """
    return np.asarray(v_mph, dtype=float) / _KT_TO_MPH


_KMH_TO_MPH = 1000.0 / 1609.344  # exact: 1 km/h = 1000 m / 1609.344 m/mile


def kmh_to_mph(v_kmh):
    """
    Convert speed from kilometres per hour (km/h) to miles per hour (mph).

    Parameters
    ----------
    v_kmh : float or array-like  -- speed in km/h

    Returns
    -------
    v_mph : ndarray or scalar    -- speed in mph
    """
    return np.asarray(v_kmh, dtype=float) * _KMH_TO_MPH
