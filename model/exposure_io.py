"""
OED exposure read adapter (Step 4.1).

Single entry point: load_oed_exposure(loc_path, acc_path) -> pd.DataFrame

Returns a compatibility view with the exact legacy columns that the model uses:
    location_id, state, county, lat, lon, tiv, construction, occupancy,
    deductible, limit

Construction and occupancy strings are recovered from OrgConstructionCode /
OrgOccupancyCode (lossless provenance fields).  If those fields are absent
(forward-compatibility path), the function falls back to reversing the
ConstructionCode integer via the OED mapping table loaded from config.

OccupancyCode is NOT invertible alone (Single Family and Mobile Home both
map to 1051); the provenance field is required for a lossless occupancy
round-trip.  The fallback for occupancy in the absence of provenance fields
returns the OED code as a string, which is an acceptable degraded mode for
v4 terrain work that doesn't need the original label.

Terrain invariant (Phase 3 closure)
------------------------------------
This module does NOT wire any OED terrain / roughness / secondary-modifier
field into the return value.  Exposure C uniform open-terrain is a model
assumption captured in config/model_v3.yaml (GUST_FACTOR = 1.3).
Per-site terrain is v4 scope.
"""

import os
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reverse map: ConstructionCode int -> string label (for fallback only)
# Built once at import time from config so it stays in sync.
_REVERSE_CONST: dict[int, str] = {}

try:
    import sys as _sys
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from model_config import load_exposure_cfg as _load_exp
    _ecfg = _load_exp()
    _REVERSE_CONST = {v: k for k, v in _ecfg.oed_mapping.construction_to_oed.items()}
except Exception:
    # If config is unavailable, fallback produces OED codes as strings
    pass


def load_oed_exposure(loc_path: str, acc_path: str) -> pd.DataFrame:
    """
    Load OED Location + Account CSVs and return a legacy-compatible DataFrame.

    Parameters
    ----------
    loc_path : str
        Path to OED Location CSV (data/oed/location.csv).
    acc_path : str
        Path to OED Account CSV (data/oed/account.csv).
        Validated for required fields; account-level data is not merged into
        the per-location rows (the model is single-account).

    Returns
    -------
    pd.DataFrame
        Columns (in order): location_id, state, county, lat, lon, tiv,
        construction, occupancy, deductible, limit.

        Dtypes mirror the legacy exposure.csv:
          location_id  object (str)
          state        object (str)
          county       object (str)
          lat          float64
          lon          float64
          tiv          int64
          construction object (str)
          occupancy    object (str)
          deductible   int64
          limit        int64
    """
    loc = pd.read_csv(loc_path)
    acc = pd.read_csv(acc_path)

    # ---- basic OED validity checks -----------------------------------------
    for col in ("LocNumber", "CountryCode", "LocPerilsCovered", "LocCurrency"):
        if col not in loc.columns:
            raise ValueError(f"OED Location file missing required column: {col}")
        if loc[col].isnull().any():
            raise ValueError(f"OED Location column {col} contains nulls")

    for col in ("AccNumber", "AccCurrency", "PolNumber", "PolPerilsCovered"):
        if col not in acc.columns:
            raise ValueError(f"OED Account file missing required column: {col}")

    # ---- recover construction / occupancy strings --------------------------
    if "OrgConstructionCode" in loc.columns:
        construction = loc["OrgConstructionCode"].to_numpy(dtype=object)
    else:
        # Fallback: reverse-map ConstructionCode int -> string
        construction = np.array(
            [_REVERSE_CONST.get(int(c), str(c)) for c in loc["ConstructionCode"]],
            dtype=object,
        )

    if "OrgOccupancyCode" in loc.columns:
        occupancy = loc["OrgOccupancyCode"].to_numpy(dtype=object)
    else:
        # OccupancyCode is not uniquely invertible; return code as string
        occupancy = loc["OccupancyCode"].astype(str).to_numpy(dtype=object)

    # ---- recover state from AreaCode (portfolio is all FL) -----------------
    state = loc["AreaCode"].to_numpy(dtype=object)

    # ---- assemble compatibility view ---------------------------------------
    result = pd.DataFrame({
        "location_id":  loc["LocNumber"].to_numpy(dtype=object),
        "state":        state,
        "county":       loc["GeogName1"].to_numpy(dtype=object),
        "lat":          loc["Latitude"].to_numpy(dtype=float),
        "lon":          loc["Longitude"].to_numpy(dtype=float),
        "tiv":          loc["BuildingTIV"].to_numpy(dtype=np.int64),
        "construction": construction,
        "occupancy":    occupancy,
        "deductible":   loc["LocDed1Building"].to_numpy(dtype=np.int64),
        "limit":        loc["LocLimit1Building"].to_numpy(dtype=np.int64),
    })
    return result


# ---------------------------------------------------------------------------
# Convenience: canonical paths used by all model readers
# ---------------------------------------------------------------------------
OED_LOC_PATH = os.path.join(_ROOT, "data", "oed", "location.csv")
OED_ACC_PATH = os.path.join(_ROOT, "data", "oed", "account.csv")


def load_portfolio() -> pd.DataFrame:
    """Load the canonical portfolio from data/oed/ using the module-level paths."""
    return load_oed_exposure(OED_LOC_PATH, OED_ACC_PATH)
