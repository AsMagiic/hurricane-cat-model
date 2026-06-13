"""
Standard output tables for the Florida hurricane cat model (Steps 4.2 / 4.3).

Year Loss Table (YLT) — results/ylt.csv
-----------------------------------------
One row per simulated year.  Columns: Year, NumEvents, AggGroundUp, AggGross,
AggNet, MaxOccGross, MaxOccNet.  All four EP views (AEP/OEP × gross/net) flow
from the YLT columns via ep_utils — the sole EP/PML kernel (p_k = k/N).

AggNet = AggGross − sum(recovery_total per year)  [identical to summary.py's
         former _load() path — byte-for-byte identity guaranteed by design]
MaxOccNet = max(portfolio_net per year); 0 for no-event years.

Sampled Event Loss Table (SELT) — results/elt.csv
---------------------------------------------------
One row per simulated event.  This is a SAMPLED ELT, not a rated catalog:
AnnualRate = 1/N_years for every event (uniform).  EP metrics continue to
flow from the YLT; the SELT is a parallel standard representation.

  EventId            stable monotonic 1-based int (primary key from loss.py)
  AnnualRate         1.0 / N_years (uniform; sampled-event-set convention)
  MeanLossGross      portfolio_gross (deterministic production run → IS E[L|event])
  MeanLossNet        portfolio_net   (gross − XoL recoveries)
  StdDevIndependent  NaN  (v4: requires calibrated CV and moment treatment)
  StdDevCorrelated   NaN  (v4: requires calibrated ρ and moment treatment)
  ExposureValue      reference portfolio TIV (constant; per-event affected TIV is v4)

AAL reconciliation identity:
  sum(AnnualRate × MeanLossGross) = (1/N) × Σ_e portfolio_gross_e
                                  = (1/N) × Σ_y AggGross_y   [events partition years]
                                  = mean(AggGross)            = YLT AEP-gross AAL
Same identity holds for net.
"""

import os
import numpy as np
import pandas as pd

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")


def build_ylt(annual_df: pd.DataFrame, events_net_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Year Loss Table from annual_losses and events_net DataFrames.

    Arithmetic matches summary.py's former _load() net reconstruction exactly —
    byte-for-byte identity with the previous net series is guaranteed by using the
    same groupby-sum / groupby-max / fillna(0.0) operations.

    Parameters
    ----------
    annual_df      : from annual_losses.csv (loss.py output)
    events_net_df  : from events_net.csv (reinsurance.py output)

    Returns
    -------
    DataFrame with columns: Year, NumEvents, AggGroundUp, AggGross, AggNet,
                            MaxOccGross, MaxOccNet
    """
    if len(events_net_df) > 0:
        yr_rec = events_net_df.groupby("year")["recovery_total"].sum()
        yr_net = events_net_df.groupby("year")["portfolio_net"].max()
    else:
        yr_rec = pd.Series(dtype=float)
        yr_net = pd.Series(dtype=float)

    agg_net     = annual_df["aggregate_gross"] - annual_df["year"].map(yr_rec).fillna(0.0)
    max_occ_net = annual_df["year"].map(yr_net).fillna(0.0)

    return pd.DataFrame({
        "Year":        annual_df["year"].to_numpy(),
        "NumEvents":   annual_df["n_events"].to_numpy(),
        "AggGroundUp": annual_df["aggregate_ground_up"].to_numpy(),
        "AggGross":    annual_df["aggregate_gross"].to_numpy(),
        "AggNet":      agg_net.to_numpy(),
        "MaxOccGross": annual_df["max_event_gross"].to_numpy(),
        "MaxOccNet":   max_occ_net.to_numpy(),
    })


def build_elt(
    events_net_df: pd.DataFrame,
    n_years: int,
    total_tiv: float,
) -> pd.DataFrame:
    """
    Build the Sampled Event Loss Table (SELT) from the events_net DataFrame.

    Parameters
    ----------
    events_net_df : from events_net.csv; must contain EventId, portfolio_gross,
                    portfolio_net.  No RNG draws — pure transformation.
    n_years       : total simulated years; AnnualRate = 1/n_years for every event.
    total_tiv     : reference portfolio TIV (USD); written to every row as
                    ExposureValue.  Source: load_portfolio()["tiv"].sum().

    Returns
    -------
    DataFrame with columns: EventId, AnnualRate, MeanLossGross, MeanLossNet,
                            StdDevIndependent, StdDevCorrelated, ExposureValue
    """
    n_events    = len(events_net_df)
    annual_rate = 1.0 / n_years
    return pd.DataFrame({
        "EventId":           events_net_df["EventId"].to_numpy(),
        "AnnualRate":        np.full(n_events, annual_rate),
        "MeanLossGross":     events_net_df["portfolio_gross"].to_numpy(),
        "MeanLossNet":       events_net_df["portfolio_net"].to_numpy(),
        "StdDevIndependent": np.full(n_events, np.nan),
        "StdDevCorrelated":  np.full(n_events, np.nan),
        "ExposureValue":     np.full(n_events, float(total_tiv)),
    })


def main():
    ann_path = os.path.join(RESULTS_DIR, "annual_losses.csv")
    evn_path = os.path.join(RESULTS_DIR, "events_net.csv")

    for p in (ann_path, evn_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- run model/loss.py then model/reinsurance.py first."
            )

    annual_df     = pd.read_csv(ann_path)
    events_net_df = pd.read_csv(evn_path)
    N_YEARS       = len(annual_df)
    N_EVENTS      = len(events_net_df)

    print(f"Building YLT + SELT: {N_YEARS:,} years | {N_EVENTS:,} events")

    # ---- Year Loss Table -------------------------------------------------------
    ylt = build_ylt(annual_df, events_net_df)

    assert len(ylt) == N_YEARS, f"FAIL: YLT row count {len(ylt)} != {N_YEARS}"
    assert (ylt["AggNet"] <= ylt["AggGross"] + 1e-4).all(), \
        "FAIL: AggNet > AggGross in some year"
    assert (ylt["MaxOccNet"] <= ylt["MaxOccGross"] + 1e-4).all(), \
        "FAIL: MaxOccNet > MaxOccGross in some year"
    assert (ylt["MaxOccGross"] <= ylt["AggGross"] + 1e-4).all(), \
        "FAIL: MaxOccGross > AggGross in some year"
    assert (ylt["AggNet"] >= -1e-4).all(), "FAIL: negative AggNet"
    assert (ylt["MaxOccNet"] >= -1e-4).all(), "FAIL: negative MaxOccNet"

    zero_years = int((ylt["AggGross"] == 0.0).sum())
    print(f"[OK] YLT: {zero_years:,} zero-loss years")

    ylt_path = os.path.join(RESULTS_DIR, "ylt.csv")
    ylt.to_csv(ylt_path, index=False)
    print(f"Saved: {ylt_path}  ({len(ylt):,} rows x {len(ylt.columns)} columns)")

    # ---- Sampled Event Loss Table --------------------------------------------
    import sys as _sys
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from model.exposure_io import load_portfolio

    total_tiv = float(load_portfolio()["tiv"].sum())
    elt = build_elt(events_net_df, N_YEARS, total_tiv)

    assert len(elt) == N_EVENTS, f"FAIL: ELT row count {len(elt)} != {N_EVENTS}"
    assert (elt["MeanLossNet"] <= elt["MeanLossGross"] + 1e-4).all(), \
        "FAIL: MeanLossNet > MeanLossGross for some event"
    assert elt["StdDevIndependent"].isna().all(), "FAIL: StdDevIndependent not null"
    assert elt["StdDevCorrelated"].isna().all(),  "FAIL: StdDevCorrelated not null"

    aal_gross = float((elt["AnnualRate"] * elt["MeanLossGross"]).sum())
    aal_ylt   = float(ylt["AggGross"].mean())
    rel_err   = abs(aal_gross - aal_ylt) / max(aal_ylt, 1.0)
    assert rel_err < 1e-5, (
        f"FAIL: ELT gross AAL {aal_gross/1e6:.6f}M != YLT AAL {aal_ylt/1e6:.6f}M "
        f"(rel err {rel_err:.2e})"
    )
    print(f"[OK] ELT gross AAL reconciles to YLT: {aal_gross/1e6:.4f} M")

    elt_path = os.path.join(RESULTS_DIR, "elt.csv")
    elt.to_csv(elt_path, index=False)
    print(f"Saved: {elt_path}  ({len(elt):,} rows x {len(elt.columns)} columns)")

    print("[OK] All integrity checks passed")


if __name__ == "__main__":
    main()
