"""
Calibrate the wind-pressure relationship (WPR): Δp = a · Vmax^b.

Fits log-linear OLS on CONUS satellite-era HU landfalls from HURDAT2,
bridging the Vmax-primary catalogue to the Δp-parameterised Vickery-Wadhera
(2008) Rmax and B equations.

Unit chain (explicit — silent conversion bugs are the #1 wind-model failure mode):
  vmax_kt (CSV) × KT_TO_MPH (1.15078) → vmax_mph
  Δp = p_env_mb − pmin_mb              (mb)
  fit: Δp (mb) = a · vmax_mph^b

V&W (2008) consume Vmax in mph and Δp in mb — this output matches that interface.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from calibration.intensity import SAT_START
from model_config import load_calibration_cfg, load_model_cfg

_ccfg      = load_calibration_cfg()
_CONUS_CSV = os.path.join(_ROOT, _ccfg.conus_landfalls.processed_path)
_FIG       = os.path.join(_ROOT, _ccfg.wind_pressure_calibration.figure_path)
_MODEL_CFG = os.path.join(_ROOT, "config", "model_v3.yaml")

KT_TO_MPH = 1.15078  # exact NHC conversion factor; vmax_kt × KT_TO_MPH = vmax_mph

# Saffir-Simpson lower bounds in mph (1-min sustained wind)
_SS_MPH    = [74, 96, 111, 130, 157, 9999]
_SS_LABELS = ["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]
_SS_COLORS = ["#4dac26", "#f4a582", "#d7191c", "#762a83", "#1a1a1a"]

_MISSING_P_ENV_MSG = (
    "seed wind_pressure.p_env_mb in config/model_v3.yaml before running.\n"
    "Required block:\n"
    "wind_pressure:\n"
    "  p_env_mb:\n"
    "    value: 1013.0\n"
    '    units: "mb"\n'
    '    source: "standard atmosphere ambient pressure; static input"'
)

_SOURCE_TMPL = (
    "Log-linear OLS (np.polyfit, multiplicative-error model) on CONUS "
    "satellite-era HU landfalls (HURDAT2 {sat_start}-2024, n={n} records "
    "with non-null pmin_mb). WPR is basin physics, not portfolio-specific — "
    "restricting to Florida would be a specification error. Satellite-era "
    "cut (>={sat_start}) breaks Dvorak circularity: pre-satellite Vmax was "
    "partly derived from pressure via the Dvorak technique, making a "
    "Vmax-pressure regression on that data circular. "
    "Unit chain: vmax_kt * {kt_to_mph} = vmax_mph; "
    "delta_p = p_env_mb - pmin_mb (mb). "
    "See calibration/wind_pressure.py."
)


# ---------------------------------------------------------------------------
# Pure functions — no I/O, directly testable
# ---------------------------------------------------------------------------

def predict_dp(
    vmax_mph: "float | np.ndarray",
    a: float,
    b: float,
) -> np.ndarray:
    """
    Return Δp (mb) = a · vmax_mph^b.

    Parameters
    ----------
    vmax_mph : array_like, mph (1-min sustained wind)
    a        : float, coefficient  (mb / mph^b)
    b        : float, exponent     (dimensionless)

    Returns
    -------
    dp_mb : np.ndarray, mb
    """
    return a * np.asarray(vmax_mph, dtype=float) ** b


def fit_wpr(
    vmax_mph: np.ndarray,
    dp_mb: np.ndarray,
) -> "tuple[float, float, float, float]":
    """
    Log-linear OLS fit: ln(Δp) = ln(a) + b · ln(Vmax_mph).

    Multiplicative-error model — appropriate for a power law where relative
    scatter in Δp is approximately constant (not absolute scatter).

    Parameters
    ----------
    vmax_mph : array, mph (1-min sustained wind at landfall)
    dp_mb    : array, mb  (p_env_mb − pmin_mb; must be > 0)

    Returns
    -------
    a         : float, mb / mph^b
    b         : float, dimensionless
    sigma_log : float, residual std in log space (ddof=2)
    r_squared : float, R² in log space
    """
    ln_v  = np.log(vmax_mph)
    ln_dp = np.log(dp_mb)
    n     = len(ln_v)

    b_fit, ln_a = np.polyfit(ln_v, ln_dp, 1)
    a_fit       = float(np.exp(ln_a))
    b_fit       = float(b_fit)

    resid     = ln_dp - (ln_a + b_fit * ln_v)
    ss_res    = float(np.dot(resid, resid))
    ss_tot    = float(np.sum((ln_dp - ln_dp.mean()) ** 2))
    sigma_log = float(np.sqrt(ss_res / (n - 2)))
    r_squared = float(1.0 - ss_res / ss_tot)

    return a_fit, b_fit, sigma_log, r_squared


# ---------------------------------------------------------------------------
# Config write — targeted text replacement (no yaml.dump)
# ---------------------------------------------------------------------------

def _write_wpr_config(
    path: str,
    a: float,
    b: float,
    sigma_log: float,
    sample_n: int,
    source: str,
) -> None:
    """
    Update fitted leaves (a, b, sigma_log, sample_n) in the wind_pressure:
    block of config/model_v3.yaml.

    p_env_mb is NEVER touched — it is a static input owned by the operator.
    Raises RuntimeError if wind_pressure: or p_env_mb: is absent.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # Locate wind_pressure: block start
    start_i = None
    for i, line in enumerate(lines):
        if line.rstrip() == "wind_pressure:":
            start_i = i
            break
    if start_i is None:
        raise RuntimeError(
            "wind_pressure: block not found in config/model_v3.yaml\n"
            + _MISSING_P_ENV_MSG
        )

    # Locate block end — first top-level non-blank non-comment line after start_i
    end_i = len(lines)
    for j in range(start_i + 1, len(lines)):
        s = lines[j].rstrip()
        if s and not s.startswith(" ") and not s.startswith("#"):
            end_i = j
            break

    # Within block, find end of p_env_mb sub-block (stop at next indent-2 key)
    penv_start = None
    penv_end   = end_i
    for j in range(start_i + 1, end_i):
        bl = lines[j]
        if bl.rstrip() == "  p_env_mb:":
            penv_start = j
        elif penv_start is not None:
            # An indent-2 key: 2 spaces then a non-space, non-comment character
            if len(bl) > 2 and bl[0] == " " and bl[1] == " " and bl[2] != " " and bl.strip() and not bl.strip().startswith("#"):
                penv_end = j
                break

    if penv_start is None:
        raise RuntimeError(
            "wind_pressure.p_env_mb not found in config/model_v3.yaml\n"
            + _MISSING_P_ENV_MSG
        )

    # Keep only p_env_mb sub-block lines (strip trailing blank lines)
    penv_content = list(lines[start_i + 1 : penv_end])
    while penv_content and not penv_content[-1].strip():
        penv_content.pop()

    fitted = (
        "  a:\n"
        f"    value: {round(a, 4)}\n"
        '    units: "mb / mph^b"\n'
        f'    source: "{source}"\n'
        "  b:\n"
        f"    value: {round(b, 4)}\n"
        '    units: "dimensionless"\n'
        f'    source: "{source}"\n'
        "  sigma_log:\n"
        f"    value: {round(sigma_log, 4)}\n"
        '    units: "dimensionless"\n'
        f'    source: "{source}"\n'
        "  sample_n:\n"
        f"    value: {sample_n}\n"
        '    units: "count"\n'
        f'    source: "{source}"\n'
    )

    new_block = lines[start_i] + "".join(penv_content) + fitted + "\n"
    lines[start_i : end_i] = [new_block]

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _make_figure(
    vmax_mph: np.ndarray,
    dp_mb: np.ndarray,
    a: float,
    b: float,
    r2: float,
    n: int,
    path: str,
) -> None:
    """
    Scatter of observed Δp vs Vmax (satellite-era CONUS HU landfalls),
    colored by SS category, with fitted curve overlaid.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (lo, hi) in enumerate(zip(_SS_MPH[:-1], _SS_MPH[1:])):
        mask = (vmax_mph >= lo) & (vmax_mph < hi)
        if mask.any():
            ax.scatter(
                vmax_mph[mask], dp_mb[mask],
                color=_SS_COLORS[i], label=_SS_LABELS[i],
                s=45, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3,
            )

    v_grid  = np.linspace(vmax_mph.min() * 0.92, vmax_mph.max() * 1.08, 300)
    dp_grid = predict_dp(v_grid, a, b)
    ax.plot(
        v_grid, dp_grid, "k-", linewidth=2.0,
        label=f"Fit: Δp = {a:.3f}·V^{b:.3f}", zorder=4,
    )

    textstr = f"a = {a:.4f}\nb = {b:.4f}\nR² = {r2:.4f}\nn = {n}"
    ax.text(
        0.05, 0.95, textstr, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_xlabel("Vmax at landfall (mph, 1-min sustained)", fontsize=11)
    ax.set_ylabel("Δp = p_env − pmin (mb)", fontsize=11)
    ax.set_title(
        f"CONUS Satellite-era HU WPR  |  HURDAT2 {SAT_START}–2024  |  n={n} landfalls",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_calibration() -> None:
    """
    Load CONUS landfalls → filter → fit WPR → write config → save figure.

    Unit chain: vmax_kt × KT_TO_MPH (1.15078) → vmax_mph;
                Δp = p_env_mb − pmin_mb (mb).
    p_env_mb is read from config — this function raises if absent.
    """
    # --- Read p_env_mb from config (no hardcoded fallback) -----------------
    mcfg = load_model_cfg()
    try:
        wp = mcfg.wind_pressure
    except AttributeError:
        raise RuntimeError(_MISSING_P_ENV_MSG) from None
    try:
        p_env_mb = float(wp.p_env_mb)
    except AttributeError:
        raise RuntimeError(_MISSING_P_ENV_MSG) from None

    print("=== WPR Calibration (CONUS satellite-era HU landfalls) ===\n")
    print(f"  p_env_mb = {p_env_mb:.1f} mb  (read from config/model_v3.yaml)")

    # --- Filter chain -------------------------------------------------------
    df = pd.read_csv(_CONUS_CSV)
    n0 = len(df)
    df = df[df["status"] == "HU"]
    n1 = len(df)
    df = df[df["year"] >= SAT_START]
    n2 = len(df)
    df = df.dropna(subset=["pmin_mb"]).reset_index(drop=True)
    n  = len(df)

    print(f"\n  Filter chain:")
    print(f"    {n0:>4}  all CONUS crossings")
    print(f"    {n1:>4}  status == HU")
    print(f"    {n2:>4}  year >= {SAT_START}  (satellite era)")
    print(f"    {n:>4}  non-null pmin_mb  [sample used for fit]")

    vmax_kt  = df["vmax_kt"].values.astype(float)
    pmin_mb  = df["pmin_mb"].values.astype(float)
    vmax_mph = vmax_kt * KT_TO_MPH
    dp_mb    = p_env_mb - pmin_mb

    bad = (dp_mb <= 0)
    assert not bad.any(), (
        f"Non-positive Δp in {bad.sum()} row(s): min Δp = {dp_mb.min():.1f} mb. "
        "Check pmin_mb values or p_env_mb in config."
    )

    # --- Fit ----------------------------------------------------------------
    a, b, sigma_log, r2 = fit_wpr(vmax_mph, dp_mb)

    print(f"\n  Fit results:")
    print(f"    a         = {a:.4f}  mb / mph^b")
    print(f"    b         = {b:.4f}  (dimensionless)")
    print(f"    sigma_log = {sigma_log:.4f}  (residual std in log space, ddof=2)")
    print(f"    R²        = {r2:.4f}  (in log space)")

    # Gradient-wind sanity check — the central physical check for this whole chain
    print(f"\n  *** b_fit = {b:.4f}  |  gradient-wind expectation = 2.0  |  deviation = {b - 2.0:+.4f} ***")
    if 1.5 <= b <= 2.5:
        print("  [OK] b within gradient-wind band [1.5, 2.5]")
    else:
        print("  [WARN] b outside gradient-wind band [1.5, 2.5] — review data/methodology")

    # --- Source string ------------------------------------------------------
    source = _SOURCE_TMPL.format(sat_start=SAT_START, n=n, kt_to_mph=KT_TO_MPH)

    # --- Config write -------------------------------------------------------
    _write_wpr_config(_MODEL_CFG, a, b, sigma_log, n, source)
    print(f"\n  wind_pressure: a/b/sigma_log/sample_n written to config/model_v3.yaml")

    # --- Figure -------------------------------------------------------------
    _make_figure(vmax_mph, dp_mb, a, b, r2, n, _FIG)
    print(f"  Figure saved -> {_FIG}")


if __name__ == "__main__":
    run_calibration()
