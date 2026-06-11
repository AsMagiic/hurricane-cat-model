"""
Fit a climate-conditioned Poisson GLM for Florida annual HU landfall frequency.

Model
-----
  log(lambda_t) = b0 + b1*TNA_t + b2*AMO_t

where TNA and AMO are standardized (zero mean, unit variance) over the fit
period. lambda_current is evaluated at the mean climate of the last
`current_climate_years` years and written to config/model_v3.yaml as
lambda_hu_fl, replacing the Step 1.3 MLE value.

Data sources (downloaded at runtime)
-------------------------------------
TNA : NOAA PSL Tropical North Atlantic SST index (5.5-23.5N, 15-57.5W)
      — nearest available proxy for MDR SST (ersst.long.data is defunct at PSL)
AMO : NOAA PSL AMO unsmoothed monthly index
      — data available through 2023; 2024 is NaN-dropped, fit period = 1966-2023

Physical sign expectations
--------------------------
b1 > 0 : warmer Tropical N. Atlantic → higher HU frequency
b2 > 0 : positive AMO                → higher HU frequency

If AMO sign is wrong in the selected model, the YAML is NOT updated.

Model selection
---------------
When |r(TNA, AMO)| > 0.8, both the full (TNA+AMO) and reduced (AMO-only)
models are always fitted and compared by AIC. The lower-AIC model is selected.
The AIC comparison is printed to stdout before any config write.
"""

import argparse
import os
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from model_config import load_calibration_cfg

_ccfg      = load_calibration_cfg()
_FL_CSV    = os.path.join(_ROOT, _ccfg.fl_landfalls.processed_path)
_FIG       = os.path.join(_ROOT, _ccfg.frequency_glm.figure_path)
_MODEL_CFG = os.path.join(_ROOT, "config", "model_v3.yaml")
_TNA_URL   = _ccfg.frequency_glm.tna_url
_AMO_URL   = _ccfg.frequency_glm.amo_url
_CW_YEARS  = int(_ccfg.frequency_glm.current_climate_years)

SAT_START, SAT_END = 1966, 2024
_MIN_MONTHS = 10   # min valid months required to compute an annual mean

_GLM_SOURCE = (
    "Poisson GLM (log link), covariates: standardized TNA SST + AMO, "
    "{start}-{end} satellite era; current-climate value = "
    "exp(b0 + b1*TNA_curr + b2*AMO_curr) evaluated at {cw_start}-{cw_end} "
    "mean climate. HURDAT2 + NOAA PSL TNA + AMO unsmoothed. "
    "See calibration/frequency_glm.py."
)


# ---------------------------------------------------------------------------
# HU count series  (standalone — no import from frequency.py)
# ---------------------------------------------------------------------------

def _annual_hu_counts(fl_csv: str, start: int, end: int) -> pd.Series:
    """
    Return annual HU landfall counts for [start, end] with explicit zeros.

    Parameters
    ----------
    fl_csv : str -- path to fl_landfalls.csv
    start  : int -- first year
    end    : int -- last year (inclusive)

    Returns
    -------
    pd.Series, int dtype, index = year
    """
    df = pd.read_csv(fl_csv)
    hu = df.loc[df["status"] == "HU", "year"].value_counts()
    counts = pd.Series(
        [hu.get(y, 0) for y in range(start, end + 1)],
        index=range(start, end + 1),
        dtype=int,
        name="count",
    )
    assert len(counts) == end - start + 1, (
        f"_annual_hu_counts: expected {end - start + 1} entries, got {len(counts)}"
    )
    return counts


# ---------------------------------------------------------------------------
# PSL file download + parse
# ---------------------------------------------------------------------------

def _parse_psl_file(url: str, label: str) -> pd.Series:
    """
    Download and parse a NOAA PSL fixed-width correlation data file.

    Format
    ------
    Line 1 : header (year range — skip)
    Lines 2+: YYYY  M01  M02  ... M12  (whitespace-separated floats)
    Missing : -99.99

    Returns annual means as pd.Series indexed by year.
    Years with fewer than _MIN_MONTHS valid months are excluded.

    Parameters
    ----------
    url   : str -- PSL data URL
    label : str -- short name for logging
    """
    print(f"  Downloading {label} from {url} ...")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    records = {}
    for i, line in enumerate(raw.splitlines()):
        if i == 0:
            continue                     # skip header (year-range line)
        tokens = line.split()
        if len(tokens) < 13:
            continue                     # skip malformed or footer lines
        try:
            year = int(tokens[0])
        except ValueError:
            continue
        monthly = []
        for t in tokens[1:13]:
            try:
                v = float(t)
            except ValueError:
                v = np.nan
            monthly.append(np.nan if v == -99.99 else v)
        valid = [v for v in monthly if not np.isnan(v)]
        if len(valid) >= _MIN_MONTHS:
            records[year] = float(np.mean(valid))

    series = pd.Series(records, name=label, dtype=float)
    print(f"    {label}: {series.index.min()}–{series.index.max()}, "
          f"{len(series)} annual means")
    return series


# ---------------------------------------------------------------------------
# GLM config write — targeted text insertion, preserves comments
# ---------------------------------------------------------------------------

def _write_lambda_hu_fl(path: str, value: float, source: str) -> None:
    """
    Replace the lambda_hu_fl leaf in config/model_v3.yaml as plain text.
    Preserves all existing comments and formatting.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    block = (
        f"  lambda_hu_fl:\n"
        f"    value: {round(value, 4)}\n"
        f'    units: "events/year"\n'
        f'    source: "{source}"\n'
    )

    # Replace existing block.
    for i, line in enumerate(lines):
        if line.rstrip().startswith("  lambda_hu_fl:"):
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                j += 1
            lines[i:j] = [block]
            break
    else:
        # Insert after lambda_rate source line (first-run fallback).
        insert_after = None
        in_lambda_rate = False
        for i, line in enumerate(lines):
            if line.rstrip() == "  lambda_rate:":
                in_lambda_rate = True
            if in_lambda_rate and line.strip().startswith("source:"):
                insert_after = i
                in_lambda_rate = False
                break
        if insert_after is None:
            raise RuntimeError(
                "_write_lambda_hu_fl: cannot find anchor in " + path
            )
        lines.insert(insert_after + 1, block)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_plot(df: pd.DataFrame, fitted: np.ndarray, path: str) -> None:
    """
    Two-panel figure:
      Left  — observed counts vs GLM fitted values (time series)
      Right — GLM fitted lambda vs standardized TNA, colour-coded by AMO sign
    """
    years  = df.index.values
    counts = df["count"].values
    amo_z  = df["amo_z"].values
    tna_z  = df["tna_z"].values
    b0     = df.attrs["b0"]
    b1     = df.attrs["b1"]
    b2     = df.attrs["b2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: time series ---
    ax1.bar(years, counts, color="#4c72b0", width=0.8, alpha=0.7,
            label="Observed (HU landfalls)")
    ax1.plot(years, fitted, color="#d62728", linewidth=1.8,
             label="GLM fitted λ_t")
    ci_lo = np.maximum(0.0, fitted - 1.96 * np.sqrt(fitted))
    ci_hi = fitted + 1.96 * np.sqrt(fitted)
    ax1.fill_between(years, ci_lo, ci_hi, alpha=0.15, color="#d62728",
                     label="Poisson 95% band")
    ax1.set_xlabel("Year", fontsize=10)
    ax1.set_ylabel("Annual HU landfall count", fontsize=10)
    ax1.set_title(f"Observed vs GLM-Fitted Counts\n"
                  f"{years.min()}–{years.max()}", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.set_xlim(years.min() - 1, years.max() + 1)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # --- Right: fitted lambda vs TNA_z, colour by AMO sign ---
    pos_amo = amo_z >= 0
    neg_amo = ~pos_amo
    ax2.scatter(tna_z[pos_amo], fitted[pos_amo],
                c="#d62728", alpha=0.8, s=40, zorder=4,
                label="AMO ≥ 0 (warm phase)")
    ax2.scatter(tna_z[neg_amo], fitted[neg_amo],
                c="#4c72b0", alpha=0.8, s=40, zorder=4,
                label="AMO < 0 (cool phase)")

    # Marginal TNA effect at neutral AMO (AMO_z = 0).
    tna_grid = np.linspace(tna_z.min() - 0.2, tna_z.max() + 0.2, 120)
    lambda_neutral = np.exp(b0 + b1 * tna_grid)
    ax2.plot(tna_grid, lambda_neutral, "k--", linewidth=1.2,
             label="GLM marginal (AMO = 0)", zorder=3)

    ax2.set_xlabel("Standardized TNA SST (σ)", fontsize=10)
    ax2.set_ylabel("GLM fitted λ", fontsize=10)
    ax2.set_title("Fitted λ vs TNA SST\n(colour = AMO sign)", fontsize=11)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Florida HU Landfall Frequency — Poisson GLM  |  HURDAT2 + NOAA PSL",
        fontsize=12, y=1.01,
    )
    fig.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fit Poisson GLM for FL HU landfall frequency."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print results but do NOT write lambda_hu_fl to config/model_v3.yaml.",
    )
    args = ap.parse_args()
    dry_run = args.dry_run

    # ---- Step 1: HU count series ----------------------------------------
    counts = _annual_hu_counts(_FL_CSV, SAT_START, SAT_END)

    # ---- Step 2: Download covariates ------------------------------------
    print("Downloading climate indices ...")
    tna_raw = _parse_psl_file(_TNA_URL, "TNA")
    amo_raw = _parse_psl_file(_AMO_URL, "AMO")

    # ---- Step 3: Align and build DataFrame ------------------------------
    df = pd.DataFrame({
        "count": counts,
        "tna":   tna_raw,
        "amo":   amo_raw,
    }).loc[SAT_START:SAT_END]

    n_missing = df[["tna", "amo"]].isna().sum()
    if n_missing.sum() > 0:
        print(f"\nNaN in covariates: TNA={n_missing['tna']}, "
              f"AMO={n_missing['amo']} — dropping affected rows.")
        df = df.dropna(subset=["tna", "amo"])

    fit_start = int(df.index.min())
    fit_end   = int(df.index.max())
    n_fit     = len(df)
    assert n_fit >= 40, f"Too few complete rows after NaN drop: {n_fit}"

    print(f"\nFit period: {fit_start}–{fit_end}  n={n_fit} yr")

    # ---- Step 4: Standardize covariates ---------------------------------
    tna_mean, tna_std = df["tna"].mean(), df["tna"].std(ddof=1)
    amo_mean, amo_std = df["amo"].mean(), df["amo"].std(ddof=1)
    df = df.copy()
    df["tna_z"] = (df["tna"] - tna_mean) / tna_std
    df["amo_z"] = (df["amo"] - amo_mean) / amo_std

    # ---- Step 5: Fit both GLMs (two-covariate and AMO-only) -------------
    X_full = sm.add_constant(df[["tna_z", "amo_z"]])
    result_full = sm.GLM(df["count"], X_full, family=sm.families.Poisson()).fit()

    X_amo_only = sm.add_constant(df[["amo_z"]])
    result_amo_only = sm.GLM(
        df["count"], X_amo_only, family=sm.families.Poisson()
    ).fit()

    b0 = float(result_full.params["const"])
    b1 = float(result_full.params["tna_z"])
    b2 = float(result_full.params["amo_z"])
    df.attrs["b0"] = b0
    df.attrs["b1"] = b1
    df.attrs["b2"] = b2

    null_dev  = float(result_full.null_deviance)
    resid_dev = float(result_full.deviance)
    dev_expl  = (null_dev - resid_dev) / null_dev
    aic_full  = float(result_full.aic)
    pvals     = result_full.pvalues
    bses      = result_full.bse
    zstats    = result_full.tvalues

    b0_amo_only       = float(result_amo_only.params["const"])
    b2_amo_only       = float(result_amo_only.params["amo_z"])
    aic_amo_only      = float(result_amo_only.aic)
    dev_expl_amo_only = (
        (float(result_amo_only.null_deviance) - float(result_amo_only.deviance))
        / float(result_amo_only.null_deviance)
    )

    r_tna_amo, _ = pearsonr(df["tna"], df["amo"])

    # ---- Step 6: Report full model --------------------------------------
    print("\n=== Poisson GLM — Florida HU landfall frequency ===\n")
    print(f"Covariates: TNA SST (standardized), AMO (standardized)"
          f"  |  {fit_start}–{fit_end}  n={n_fit}\n")
    header = f"{'':>14}  {'coef':>8}  {'SE':>8}  {'z-stat':>7}  {'p-value':>8}  sign"
    print(header)
    print("-" * len(header))
    for name, b, se, z, p, expected in [
        ("intercept", b0, bses["const"],  zstats["const"],  pvals["const"],  None),
        ("tna_z",     b1, bses["tna_z"],  zstats["tna_z"],  pvals["tna_z"],  ">0"),
        ("amo_z",     b2, bses["amo_z"],  zstats["amo_z"],  pvals["amo_z"],  ">0"),
    ]:
        sign_flag = ""
        if expected == ">0":
            sign_flag = "OK" if b > 0 else "*** WRONG SIGN ***"
        print(f"  {name:<14}  {b:>8.4f}  {se:>8.4f}  {z:>7.3f}  {p:>8.4f}  {sign_flag}")

    print()
    print(f"  AIC (TNA+AMO)     : {aic_full:.2f}")
    print(f"  Deviance explained: {dev_expl * 100:.1f}%")
    print(f"  Null deviance     : {null_dev:.4f}")
    print(f"  Residual deviance : {resid_dev:.4f}")
    print()
    print(f"  Pearson r(TNA, AMO): {r_tna_amo:.3f}", end="")
    if abs(r_tna_amo) > 0.7:
        print("  [NOTE: |r| > 0.7 — moderate multicollinearity]", end="")
    print()

    # ---- Step 7: Model selection — always compare AICs when |r| > 0.8 ---
    # Even when both coefficients have correct signs, high collinearity inflates
    # standard errors and p-values, making the extra TNA parameter unreliable.
    # AIC comparison is the correct selection criterion here.
    active_model   = "two-covariate (TNA + AMO)"
    active_result  = result_full
    active_b0      = b0
    fitted         = result_full.fittedvalues.values

    if abs(r_tna_amo) > 0.8:
        print(f"\n  |r(TNA, AMO)| = {abs(r_tna_amo):.3f} > 0.8 "
              "— AIC model comparison:")
        print(f"    AIC  TNA+AMO  : {aic_full:.2f}")
        print(f"    AIC  AMO-only : {aic_amo_only:.2f}")
        print()
        print("  AMO-only model coefficients:")
        print(f"    intercept : {b0_amo_only:.4f}  "
              f"(SE {result_amo_only.bse['const']:.4f}, "
              f"p {result_amo_only.pvalues['const']:.4f})")
        print(f"    amo_z     : {b2_amo_only:.4f}  "
              f"(SE {result_amo_only.bse['amo_z']:.4f}, "
              f"p {result_amo_only.pvalues['amo_z']:.4f})  "
              f"{'OK' if b2_amo_only > 0 else '*** WRONG SIGN ***'}")
        print(f"    AIC       : {aic_amo_only:.2f}")
        print(f"    Dev. expl.: {dev_expl_amo_only * 100:.1f}%")

        if aic_amo_only <= aic_full:
            delta = aic_full - aic_amo_only
            print(f"\n  -> AMO-only SELECTED  "
                  f"(dAIC = {delta:.2f} in favour of reduced model)")
            active_model  = (
                f"AMO-only (TNA dropped: dAIC = {delta:.2f}, "
                f"|r| = {abs(r_tna_amo):.3f})"
            )
            active_result = result_amo_only
            active_b0     = b0_amo_only
            fitted        = result_amo_only.fittedvalues.values
            df.attrs["b0"] = b0_amo_only
            df.attrs["b1"] = 0.0
            df.attrs["b2"] = b2_amo_only
        else:
            delta = aic_amo_only - aic_full
            print(f"\n  -> Two-covariate SELECTED  "
                  f"(dAIC = {delta:.2f} in favour of full model)")

    # Sign check on selected model's AMO coefficient.
    b2_active = float(active_result.params["amo_z"])
    signs_ok  = b2_active > 0
    if not signs_ok:
        print(f"\n  WARNING: selected model AMO coefficient {b2_active:.4f} <= 0 "
              "(expected > 0) — config NOT updated.")

    # ---- Step 8: lambda_current -----------------------------------------
    window_end   = fit_end
    window_start = window_end - _CW_YEARS + 1

    window_mask = (df.index >= window_start) & (df.index <= window_end)
    n_window    = int(window_mask.sum())
    if n_window < _CW_YEARS // 2:
        print(f"WARNING: only {n_window} of {_CW_YEARS} window years in data.")

    amo_curr_z = (df.loc[window_mask, "amo"].mean() - amo_mean) / amo_std

    if "tna_z" in active_result.params.index:
        tna_curr_z     = (df.loc[window_mask, "tna"].mean() - tna_mean) / tna_std
        lambda_current = float(np.exp(
            active_b0
            + float(active_result.params["tna_z"]) * tna_curr_z
            + b2_active * amo_curr_z
        ))
    else:
        lambda_current = float(np.exp(active_b0 + b2_active * amo_curr_z))

    # Recomputed from catalogue — same value frequency.py writes to config.
    lambda_constant = float(counts.mean())
    print(f"\n  Model used                              : {active_model}")
    print(f"  lambda_constant (Step 1.3 MLE, 1966-{fit_end}): "
          f"{lambda_constant:.4f} events/yr")
    print(f"  lambda_current  (GLM, {window_start}-{window_end}, "
          f"n={n_window} yr)       : {lambda_current:.4f} events/yr")
    print(f"  Ratio (current / constant)              : "
          f"{lambda_current / lambda_constant:.3f}")

    # ---- Step 9: Write to config ----------------------------------------
    if signs_ok:
        source = _GLM_SOURCE.format(
            start=fit_start, end=fit_end,
            cw_start=window_start, cw_end=window_end,
        )
        if dry_run:
            print(f"\n  DRY RUN — lambda_hu_fl = {lambda_current:.4f} "
                  "NOT written to config/model_v3.yaml "
                  "(re-run without --dry-run to persist).")
        else:
            _write_lambda_hu_fl(_MODEL_CFG, lambda_current, source)
            print(f"\n  lambda_hu_fl = {lambda_current:.4f} written to "
                  "config/model_v3.yaml")
    else:
        print("\n  config/model_v3.yaml not modified.")

    # ---- Step 10: Plot --------------------------------------------------
    _make_plot(df, fitted, _FIG)
    print(f"  Plot saved -> {_FIG}")
