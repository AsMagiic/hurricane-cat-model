"""
EP curve utilities -- single source of truth for exceedance-probability
interpolation used throughout the Florida hurricane cat model.

Convention (applies to both OEP and AEP)
-----------------------------------------
1. Sort the N-year annual loss series DESCENDING: L_1 >= L_2 >= ... >= L_N.
2. Assign exceedance probability  p_k = k / N  for  k = 1 .. N.
     p_1 = 1/N  -- probability of exceeding the largest simulated loss.
     p_N = 1.0  -- probability of exceeding 0 (the smallest value).
3. ALL N years must be included.  0-loss years sit at the tail and correctly
   reduce exceedance probabilities for small loss thresholds.
4. PML at return period T:
       pml = np.interp(1/T, [1/N, 2/N, ..., 1], [L_1, L_2, ..., L_N])
   For round T that divides N exactly:
       pml = L_{N/T}  (the N/T-th largest annual loss, 1-indexed)
             = losses_sorted_desc[ N//T - 1 ]  (0-indexed in Python)
   Example: N=100,000, T=100  ->  pml = losses_sorted_desc[999]
            N=100,000, T=250  ->  pml = losses_sorted_desc[399]

Usage
-----
   from ep_utils import oep_pml, ep_curve

   pml_100 = oep_pml(annual_df["max_event_gross"], 100, n_years=N_YEARS)
   pml_250 = oep_pml(annual_df["max_event_gross"], 250, n_years=N_YEARS)

   losses_desc, ep = ep_curve(annual_df["max_event_gross"], n_years=N_YEARS)
   # losses_desc and ep are ready for matplotlib plotting
"""

import numpy as np


def oep_pml(annual_losses, return_period, n_years=None):
    """
    Probable Maximum Loss at a given return period.

    Works for both OEP (pass annual max-occurrence losses) and AEP
    (pass annual aggregate losses) -- the interpolation is identical.

    Parameters
    ----------
    annual_losses : array-like, length n_years
        Annual loss series.  MUST include ALL years (0-loss years included).
    return_period : float
        Return period in years (e.g. 100 for 1-in-100).
    n_years : int, optional
        Expected length; defaults to len(annual_losses).  Passing it
        explicitly adds a length-consistency check.

    Returns
    -------
    pml : float  USD loss at the requested exceedance probability 1/return_period.
    """
    arr = np.asarray(annual_losses, dtype=float)
    n   = len(arr) if n_years is None else int(n_years)
    if len(arr) != n:
        raise ValueError(
            f"ep_utils.oep_pml: length mismatch -- got {len(arr)}, expected {n}"
        )
    losses_desc = np.sort(arr)[::-1]
    ep          = np.arange(1, n + 1) / n    # [1/N, 2/N, ..., 1.0]
    return float(np.interp(1.0 / return_period, ep, losses_desc))


def ep_curve(annual_losses, n_years=None):
    """
    Full exceedance-probability curve for plotting.

    Parameters
    ----------
    annual_losses : array-like, length n_years
        Annual loss series (OEP: max-occurrence; AEP: aggregate).
    n_years : int, optional
        Expected length; defaults to len(annual_losses).

    Returns
    -------
    losses_desc : ndarray  losses sorted descending (X axis on an EP plot)
    ep          : ndarray  matching exceedance probabilities (Y axis)
                           both arrays have length n_years
    """
    arr = np.asarray(annual_losses, dtype=float)
    n   = len(arr) if n_years is None else int(n_years)
    if len(arr) != n:
        raise ValueError(
            f"ep_utils.ep_curve: length mismatch -- got {len(arr)}, expected {n}"
        )
    losses_desc = np.sort(arr)[::-1]
    ep          = np.arange(1, n + 1) / n
    return losses_desc, ep


def pml_rank_diagnostic(losses_desc, n_years, return_periods=(100, 250)):
    """
    Print the raw array-rank values that oep_pml() returns for round RPs.

    Useful for cross-script reconciliation: for a round RP T that divides
    n_years, pml = losses_desc[n_years//T - 1] with no interpolation.

    Parameters
    ----------
    losses_desc    : ndarray  sorted descending (output of ep_curve)
    n_years        : int
    return_periods : tuple of int
    """
    print(f"  Raw rank check (N={n_years:,}, 0-indexed rank = n_years//T - 1):")
    for rp in return_periods:
        rank_1idx = n_years // rp          # 1-indexed rank
        rank_0idx = rank_1idx - 1          # 0-indexed
        val = losses_desc[rank_0idx]
        print(f"    Rank {rank_1idx:>6} (1-in-{rp}): USD {val / 1e6:.2f}M")
