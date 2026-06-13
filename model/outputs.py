"""
Standard Year Loss Table (YLT) builder for the Florida hurricane cat model (Step 4.2).

Reads Step 4 (annual_losses.csv) and Step 5 (events_net.csv) outputs.
Produces results/ylt.csv — the single source of truth for all EP metrics.

YLT schema (one row per simulated year, full float precision, no rounding):
  Year         int    simulated calendar year (1..N_YEARS)
  NumEvents    int    storm count in this year (0 for no-storm years)
  AggGroundUp  float  annual aggregate ground-up loss (USD)
  AggGross     float  annual aggregate gross loss (USD, net of per-policy deductible)
  AggNet       float  annual aggregate net loss (USD, net of deductible + XoL recovery)
  MaxOccGross  float  largest per-occurrence gross loss in the year (0 for no-storm years)
  MaxOccNet    float  largest per-occurrence net loss in the year (0 for no-storm years)

All four EP views (AEP/OEP x gross/net) flow from these columns via ep_utils.
ep_utils remains the sole EP/PML kernel (convention p_k = k/N).

AggNet and MaxOccNet arithmetic is identical to summary.py's former _load() path:
  AggNet     = AggGross - sum(recovery_total for year)
  MaxOccNet  = max(portfolio_net for year); 0 for no-event years
"""

import os
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

    print(f"Building YLT: {N_YEARS:,} years | {N_EVENTS:,} events")

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
    print(f"[OK] {zero_years:,} zero-loss years present")
    print("[OK] All integrity checks passed")

    out_path = os.path.join(RESULTS_DIR, "ylt.csv")
    ylt.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(ylt):,} rows x {len(ylt.columns)} columns)")


if __name__ == "__main__":
    main()
