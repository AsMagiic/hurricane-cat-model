"""
Calibrate annual hurricane (HU) landfall frequency for Florida.

Methodology
-----------
1. Filter fl_landfalls.csv to status == 'HU'.
2. Build annual count series (explicit zeros for landfall-free years) for:
     Full record  : 1851–2024  (174 yr)
     Satellite era: 1966–2024  ( 59 yr)
3. MLE: lambda_hat = mean(annual counts).
4. Gamma conjugate posterior: Gamma(1,1) prior + Poisson likelihood.
5. Chi-squared goodness-of-fit vs Poisson(lambda_hat) — satellite era.
6. Write satellite-era lambda_hat to config/model_v3.yaml (targeted text
   insertion — preserves all existing comments and formatting).
7. Save two-panel figure to outputs/frequency_calibration.png.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from model_config import load_calibration_cfg

_ccfg      = load_calibration_cfg()
_IN        = os.path.join(_ROOT, _ccfg.fl_landfalls.processed_path)
_FIG       = os.path.join(_ROOT, _ccfg.frequency_calibration.figure_path)
_MODEL_CFG = os.path.join(_ROOT, "config", "model_v3.yaml")

_SOURCE_STR = (
    "Poisson MLE, FL HU landfalls 1966-2024, HURDAT2; "
    "satellite era chosen for record completeness. "
    "Full record (1851-2024) estimate also reported in "
    "calibration/frequency.py output. See Step 1.3b for "
    "climate-conditioned extension."
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_hu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[df["status"] == "HU"].copy()


def annual_counts(hu_df: pd.DataFrame, start: int, end: int = 2024) -> np.ndarray:
    """
    Return annual HU landfall count array for years [start, end] inclusive.

    Parameters
    ----------
    hu_df : pd.DataFrame  -- FL HU landfall rows (one row per storm)
    start : int           -- first year of period
    end   : int           -- last year of period (default 2024; 2025 excluded)

    Returns
    -------
    np.ndarray of int, length == end - start + 1, with explicit zeros for
    years that had no landfalls.
    """
    n_years = end - start + 1
    counts_by_year = hu_df["year"].value_counts()
    counts = np.array([counts_by_year.get(y, 0) for y in range(start, end + 1)],
                      dtype=int)
    # Guard against missing-zero bug before any downstream calculation.
    assert len(counts) == n_years, (
        f"annual_counts: expected {n_years} entries for {start}–{end}, "
        f"got {len(counts)}"
    )
    return counts


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mle_lambda(counts: np.ndarray) -> float:
    """Poisson MLE: sample mean of annual counts."""
    return float(counts.mean())


def gamma_posterior(counts: np.ndarray,
                    alpha_prior: float = 1.0,
                    beta_prior: float = 1.0):
    """
    Gamma conjugate posterior for Poisson rate.

    Prior  : Gamma(alpha_prior, rate=beta_prior)
    Update : alpha_post = alpha_prior + sum(k),  beta_post = beta_prior + n
    Returns: (posterior_mean, 90% CI array [lo, hi], alpha_post, beta_post)
    """
    alpha_post = alpha_prior + float(counts.sum())
    beta_post  = beta_prior  + float(len(counts))
    post_mean  = alpha_post / beta_post
    ci_90 = scipy.stats.gamma.ppf([0.05, 0.95], a=alpha_post,
                                   scale=1.0 / beta_post)
    return post_mean, ci_90, alpha_post, beta_post


def chi_squared_gof(counts: np.ndarray, lambda_hat: float):
    """
    Chi-squared goodness-of-fit of Poisson(lambda_hat) to observed counts.

    Tail bins are merged right-to-left until all expected values >= 5,
    satisfying the standard chi-squared validity condition.  ddof=1 because
    lambda was estimated from the same data.

    Returns: (stat, p_val, n_bins, obs_array, exp_array, last_k, tail_merged)
      last_k      -- index of final bin (used for "≥k" axis label)
      tail_merged -- True if any tail merging occurred
    """
    n = len(counts)
    # Include bins up to a few beyond the observed maximum.
    max_k = max(int(counts.max()), int(lambda_hat) + 4)

    obs = np.array([int((counts == k).sum()) for k in range(max_k + 1)])
    exp = np.array([scipy.stats.poisson.pmf(k, lambda_hat) * n
                    for k in range(max_k + 1)])

    original_max_k = max_k
    while len(exp) > 2 and exp[-1] < 5.0:
        exp[-2] += exp[-1];  exp = exp[:-1]
        obs[-2] += obs[-1];  obs = obs[:-1]

    last_k      = len(exp) - 1
    tail_merged = last_k < original_max_k

    # Truncated PMF sums to slightly less than 1; rescale so sum(exp) == sum(obs)
    # exactly (standard practice; distortion is < 1e-4 relative).
    exp = exp * (float(obs.sum()) / float(exp.sum()))

    n_bins = len(exp)
    df = n_bins - 1 - 1  # ddof=1 (one estimated parameter: lambda)
    if df <= 0:
        # Degenerate: lambda too small relative to sample size — merging consumed
        # all degrees of freedom.  Return NaN so the caller can report clearly.
        return float("nan"), float("nan"), n_bins, obs, exp, last_k, tail_merged

    stat, p_val = scipy.stats.chisquare(obs, f_exp=exp, ddof=1)
    return stat, p_val, n_bins, obs, exp, last_k, tail_merged


# ---------------------------------------------------------------------------
# Config write — targeted text insertion, preserves comments
# ---------------------------------------------------------------------------

def write_lambda_hu_fl(path: str, value: float) -> None:
    """
    Insert or replace the lambda_hu_fl leaf in config/model_v3.yaml.

    Uses plain-text line insertion rather than yaml.safe_load + yaml.dump
    so that all existing comments and formatting are preserved.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    block = (
        f"  lambda_hu_fl:\n"
        f"    value: {round(value, 4)}\n"
        f'    units: "events/year"\n'
        f'    source: "{_SOURCE_STR}"\n'
    )

    # Case 1: lambda_hu_fl already present — replace it.
    for i, line in enumerate(lines):
        if line.rstrip().startswith("  lambda_hu_fl:"):
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                j += 1
            lines[i:j] = [block]
            break
    else:
        # Case 2: first run — insert after the lambda_rate source line.
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
                "write_lambda_hu_fl: cannot find 'lambda_rate:' block in "
                + path
            )
        lines.insert(insert_after + 1, block)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_plot(sat_start: int, sat_end: int, sat_counts: np.ndarray,
               lambda_hat_sat: float,
               chi_obs: np.ndarray, chi_exp: np.ndarray,
               last_k: int, tail_merged: bool,
               p_val: float, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: annual count time series ---
    years = np.arange(sat_start, sat_end + 1)
    ax1.bar(years, sat_counts, color="#4c72b0", width=0.8, alpha=0.75,
            label="Annual HU landfalls")
    ax1.axhline(lambda_hat_sat, color="#d62728", linewidth=1.6,
                linestyle="--", label=f"λ = {lambda_hat_sat:.4f} events/yr")
    ax1.set_xlabel("Year", fontsize=10)
    ax1.set_ylabel("HU landfalls (count)", fontsize=10)
    ax1.set_title("Florida HU Landfall Counts\n"
                  f"{sat_start}–{sat_end} (satellite era)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_xlim(sat_start - 1, sat_end + 1)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # --- Right: observed vs Poisson-expected distribution ---
    n_bins = len(chi_obs)
    k_positions = np.arange(n_bins)
    n_years = len(sat_counts)
    obs_frac = chi_obs / n_years
    exp_frac = chi_exp / n_years

    width = 0.38
    ax2.bar(k_positions - width / 2, obs_frac, width=width,
            color="#4c72b0", alpha=0.8, label="Observed")
    ax2.bar(k_positions + width / 2, exp_frac, width=width,
            color="#dd8452", alpha=0.8,
            label=f"Poisson(λ={lambda_hat_sat:.3f})")

    labels = [str(k) for k in range(n_bins)]
    if tail_merged:
        labels[-1] = f"≥{last_k}"
    ax2.set_xticks(k_positions)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("Annual HU landfall count (k)", fontsize=10)
    ax2.set_ylabel("Fraction of years", fontsize=10)
    ax2.set_title("Observed vs Poisson-Expected\nCount Distribution",
                  fontsize=11)
    ax2.legend(fontsize=9)
    chi_label = (
        f"χ² GoF  p = {p_val:.3f}"
        if not np.isnan(p_val)
        else "χ² GoF: degenerate\n(df = 0 after tail merge)"
    )
    ax2.annotate(
        chi_label,
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.9),
    )

    fig.suptitle("Florida HU Landfall Frequency Calibration  |  HURDAT2",
                 fontsize=12, y=1.01)
    fig.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FULL_START, FULL_END = 1851, 2024
    SAT_START,  SAT_END  = 1966, 2024

    hu = _load_hu(_IN)

    full_counts = annual_counts(hu, FULL_START, FULL_END)
    sat_counts  = annual_counts(hu, SAT_START,  SAT_END)

    lambda_full = mle_lambda(full_counts)
    lambda_sat  = mle_lambda(sat_counts)

    post_mean_full, ci_full, *_ = gamma_posterior(full_counts)
    post_mean_sat,  ci_sat,  *_ = gamma_posterior(sat_counts)

    chi_stat, p_val, n_bins, chi_obs, chi_exp, last_k, tail_merged = \
        chi_squared_gof(sat_counts, lambda_sat)

    print("=== Annual HU landfall frequency — Florida ===\n")
    print(f"Full record  ({FULL_START}–{FULL_END}, n={FULL_END - FULL_START + 1} yr):")
    print(f"  MLE lambda      : {lambda_full:.4f} events/yr")
    print(f"  Posterior mean  : {post_mean_full:.4f}  |  "
          f"90% CI [{ci_full[0]:.4f}, {ci_full[1]:.4f}]")
    print()
    print(f"Satellite era ({SAT_START}–{SAT_END}, n={SAT_END - SAT_START + 1} yr):")
    print(f"  MLE lambda      : {lambda_sat:.4f} events/yr")
    print(f"  Posterior mean  : {post_mean_sat:.4f}  |  "
          f"90% CI [{ci_sat[0]:.4f}, {ci_sat[1]:.4f}]")
    if np.isnan(chi_stat):
        print(f"  Chi-sq GoF      : degenerate (df=0 after tail merge to {n_bins} bins;"
              f" lambda too small for {SAT_END - SAT_START + 1}-yr sample)")
    else:
        print(f"  Chi-sq GoF      : stat={chi_stat:.4f}, p={p_val:.4f}  "
              f"({n_bins} bins after tail merge)")
    print()

    write_lambda_hu_fl(_MODEL_CFG, lambda_sat)
    print(f"lambda_hu_fl = {lambda_sat:.4f} written to config/model_v3.yaml")

    _make_plot(SAT_START, SAT_END, sat_counts, lambda_sat,
               chi_obs, chi_exp, last_k, tail_merged, p_val, _FIG)
    print(f"Plot saved   -> {_FIG}")
