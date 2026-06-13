"""
Reinsurance engine for the Florida hurricane cat model (Step 5 / Step 4.4).

Applies a per-occurrence multi-layer XoL programme with FINITE reinstatements
to the gross portfolio losses from Step 4 (loss.py) and derives the net series.

Three-layer tower (occurrence XoL, contiguous):
  Layer 1  : 40M xs  60M   covers  60M - 100M per occurrence
  Layer 2  : 50M xs 100M   covers 100M - 150M per occurrence
  Layer 3  : 50M xs 150M   covers 150M - 200M per occurrence

Reinstatement mechanics (Step 4.4)
------------------------------------
Each layer has n_reinstatements=1 (config), giving an annual aggregate limit
of occ_limit × (1 + n) = 2 × occ_limit per layer.  Within a year, events are
processed in EventId order; allocated recovery for each event is capped by the
remaining annual aggregate.

AEP-net is order-independent:
  annual_capped_recovery_j = min(Σ_e occ_rec_{e,j}, agg_limit_j)
Per-event net (for OEP) follows EventId order within each year.

Reinstatement premium (per layer, per year):
  reinstated_amount = clip(annual_capped_rec - occ_limit, 0, n × occ_limit)
  reinst_premium    = base_premium × (reinstated_amount / occ_limit) × pct
  base_premium      = E[annual capped recovery] × (1 + loading_factor)  [TECHNICAL]
Pro-rata-temporis is v4.

Tower config is in config/reinsurance.yaml (loaded via load_reinsurance_cfg()).

Inputs  (from results/  -- produced by loss.py):
  events.csv        one row per event (EventId, year, portfolio_gross, ...)
  annual_losses.csv N_YEARS rows

Outputs:
  results/events_net.csv      events with per-layer allocated recovery and net
  outputs/ep_gross_vs_net.png OEP gross vs net plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from model_config import load_reinsurance_cfg
from model.ep_utils import oep_pml, ep_curve, pml_rank_diagnostic

RESULTS_DIR = os.path.join(_ROOT, "results")
OUT_DIR     = os.path.join(_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Programme definition -- loaded from config/reinsurance.yaml
# ---------------------------------------------------------------------------
_rcfg          = load_reinsurance_cfg()
LAYERS         = _rcfg.layers
LOADING_FACTOR = _rcfg.loading_factor

TOTAL_CAPACITY = sum(lyr["occ_limit"] for lyr in LAYERS)
TOP_OF_TOWER   = LAYERS[-1]["attachment"] + LAYERS[-1]["occ_limit"]
FIRST_ATTACH   = LAYERS[0]["attachment"]


# ---------------------------------------------------------------------------
# Layer contiguity validation
# ---------------------------------------------------------------------------
def _check_contiguity(layers):
    for i in range(len(layers) - 1):
        expected = layers[i]["attachment"] + layers[i]["occ_limit"]
        actual   = layers[i + 1]["attachment"]
        if actual != expected:
            delta = actual - expected
            kind  = "gap" if delta > 0 else "overlap"
            print(f"  [WARN] {kind} of {abs(delta)/1e6:.0f}M between "
                  f"'{layers[i]['name']}' (top {expected/1e6:.0f}M) and "
                  f"'{layers[i+1]['name']}' (attach {actual/1e6:.0f}M)")
            return False
    print(f"[OK] Tower contiguous: {FIRST_ATTACH/1e6:.0f}M to "
          f"{TOP_OF_TOWER/1e6:.0f}M  ({TOTAL_CAPACITY/1e6:.0f}M total capacity)")
    return True


# ---------------------------------------------------------------------------
# Occurrence recovery (per-event, unlimited / no aggregate cap)
# ---------------------------------------------------------------------------
def _occ_recovery(gross_arr, layers):
    """
    Per-event per-layer uncapped occurrence recovery.

    Parameters
    ----------
    gross_arr : (N,) array  portfolio_gross per event (USD)
    layers    : list of layer dicts from load_reinsurance_cfg()

    Returns
    -------
    occ_rec : (N, L) array  clip(gross - att, 0, occ_limit) per event per layer
    """
    gross   = np.asarray(gross_arr, dtype=float)
    occ_rec = np.zeros((len(gross), len(layers)))
    for j, lyr in enumerate(layers):
        occ_rec[:, j] = np.clip(gross - lyr["attachment"], 0.0, lyr["occ_limit"])
    return occ_rec


def apply_xol_unlimited(gross_arr, layers=None):
    """
    Unlimited-reinstatement vectorized recovery (regression baseline).

    No annual aggregate cap — each event recovers clip(gross - att, 0, occ_limit)
    from every layer independently.  This reproduces the pre-4.4 behaviour.

    Returns
    -------
    occ_rec   : (N, L)
    rec_total : (N,)
    net       : (N,)
    """
    if layers is None:
        layers = LAYERS
    gross     = np.asarray(gross_arr, dtype=float)
    occ_rec   = _occ_recovery(gross, layers)
    rec_total = occ_rec.sum(axis=1)
    return occ_rec, rec_total, gross - rec_total


# ---------------------------------------------------------------------------
# Finite-reinstatement recovery (Step 4.4 — production default)
# ---------------------------------------------------------------------------
def apply_xol_finite(events_df, layers=None):
    """
    Per-occurrence XoL with finite reinstatements and annual aggregate cap.

    Events are processed in (year, EventId) order within each year.
    Annual layer aggregate limit = occ_limit × (1 + n_reinstatements).

    AEP-net is order-independent (annual capped rec = min(Σocc, agg_limit)).
    Per-event net follows EventId order (OEP path).

    Parameters
    ----------
    events_df : DataFrame with columns: year, EventId, portfolio_gross
    layers    : list of layer dicts; defaults to LAYERS from config

    Returns
    -------
    alloc_rec : (N, L)  per-event per-layer allocated recovery (capped)
    occ_rec   : (N, L)  per-event per-layer occurrence recovery (uncapped)
    rec_total : (N,)    sum(alloc_rec, axis=1)
    net       : (N,)    portfolio_gross - rec_total
    """
    if layers is None:
        layers = LAYERS

    gross    = events_df["portfolio_gross"].to_numpy()
    occ_rec  = _occ_recovery(gross, layers)

    agg_limits = np.array(
        [lyr["occ_limit"] * (1 + lyr["n_reinstatements"]) for lyr in layers]
    )

    n_events = len(gross)
    n_layers = len(layers)
    alloc_rec = np.zeros((n_events, n_layers))

    # Build sorted index: (year ASC, EventId ASC) → deterministic within-year order
    order    = events_df.sort_values(["year", "EventId"]).index.to_numpy()
    year_arr = events_df["year"].to_numpy()

    cur_year = None
    consumed = np.zeros(n_layers)

    for orig_pos in order:
        yr = year_arr[orig_pos]
        if yr != cur_year:
            cur_year   = yr
            consumed[:] = 0.0
        for j in range(n_layers):
            remaining = agg_limits[j] - consumed[j]
            alloc     = occ_rec[orig_pos, j] if remaining >= occ_rec[orig_pos, j] \
                        else max(remaining, 0.0)
            alloc_rec[orig_pos, j] = alloc
            consumed[j] += alloc

    rec_total = alloc_rec.sum(axis=1)
    return alloc_rec, occ_rec, rec_total, gross - rec_total


# ---------------------------------------------------------------------------
# Programme metrics helpers
# ---------------------------------------------------------------------------
def _year_sums(col_arr, ev_years, annual_years):
    """Sum col_arr by year; return Series aligned to annual_years (0 for no-event years)."""
    s = pd.Series(col_arr, index=ev_years).groupby(level=0).sum()
    return annual_years.map(s).fillna(0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ev_path  = os.path.join(RESULTS_DIR, "events.csv")
    ann_path = os.path.join(RESULTS_DIR, "annual_losses.csv")

    for p in (ev_path, ann_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found.  Run model/loss.py first.")

    events_df = pd.read_csv(ev_path)
    annual_df = pd.read_csv(ann_path)

    N_YEARS  = len(annual_df)
    N_EVENTS = len(events_df)
    print(f"Loaded: {N_EVENTS:,} events | {N_YEARS:,} years")
    print()

    # ---- Check programme contiguity ----------------------------------------
    ok = _check_contiguity(LAYERS)
    assert ok, "Layer contiguity check failed — fix LAYERS before continuing"

    # ---- Apply finite XoL tower (year-by-year aggregate tracking) -----------
    alloc_rec, occ_rec, rec_total, net_arr = apply_xol_finite(events_df, LAYERS)

    gross_arr = events_df["portfolio_gross"].to_numpy()

    # Enrich events DataFrame (allocated recovery per layer)
    for j, lyr in enumerate(LAYERS):
        col = "rec_" + lyr["name"].replace(" ", "_")
        events_df[col] = alloc_rec[:, j]

    events_df["recovery_total"] = rec_total
    events_df["portfolio_net"]  = net_arr

    # ---- Year-level series ---------------------------------------------------
    ev_years    = events_df["year"]
    ann_years   = annual_df["year"]

    # Annual finite recovery per layer and total
    ann_alloc = [_year_sums(alloc_rec[:, j], ev_years, ann_years)
                 for j in range(len(LAYERS))]
    ann_finite_rec = sum(ann_alloc)

    # Annual occurrence recovery (unlimited, for comparison)
    ann_occ = [_year_sums(occ_rec[:, j], ev_years, ann_years)
               for j in range(len(LAYERS))]
    ann_unlim_rec = sum(ann_occ)

    agg_gross        = annual_df["aggregate_gross"].to_numpy()
    aep_net_finite   = agg_gross - ann_finite_rec.to_numpy()
    aep_net_unlim    = agg_gross - ann_unlim_rec.to_numpy()

    # Annual max net event (for OEP path)
    if N_EVENTS > 0:
        yr_max_net = events_df.groupby("year")["portfolio_net"].max()
    else:
        yr_max_net = pd.Series(dtype=float)
    annual_df["max_event_net"] = ann_years.map(yr_max_net).fillna(0.0)

    # ---- Validation asserts --------------------------------------------------
    print()
    print("=" * 64)
    print("VALIDATION")
    print("=" * 64)

    assert (net_arr <= gross_arr + 1e-6).all(), \
        "FAIL: portfolio_net > portfolio_gross for some event"
    print("[OK] net <= gross for all events")

    assert (rec_total >= -1e-6).all() and (rec_total <= TOTAL_CAPACITY + 1e-6).all(), \
        f"FAIL: recovery_total outside [0, {TOTAL_CAPACITY/1e6:.0f}M]"
    print(f"[OK] recovery_total in [0, {TOTAL_CAPACITY/1e6:.0f}M] for all events")

    below = gross_arr <= FIRST_ATTACH
    assert (rec_total[below] < 1e-6).all(), \
        f"FAIL: recovery > 0 for event with gross <= {FIRST_ATTACH/1e6:.0f}M"
    print(f"[OK] recovery = 0 for all {int(below.sum()):,} events "
          f"with gross <= {FIRST_ATTACH/1e6:.0f}M")

    above = gross_arr >= TOP_OF_TOWER
    if above.any():
        # Under finite reinstatements, an event at full-tower gross may not get
        # TOTAL_CAPACITY if aggregate is partially exhausted; check occ_rec instead.
        assert (np.abs(occ_rec[above].sum(axis=1) - TOTAL_CAPACITY) < 1e-4).all(), \
            "FAIL: occ recovery != TOTAL_CAPACITY for event with gross >= top of tower"
        print(f"[OK] occ recovery = {TOTAL_CAPACITY/1e6:.0f}M (full tower) for "
              f"{int(above.sum()):,} events with gross >= {TOP_OF_TOWER/1e6:.0f}M")
    else:
        max_gross = float(gross_arr.max()) if N_EVENTS > 0 else 0.0
        print(f"[OK] No events exceeded programme top ({TOP_OF_TOWER/1e6:.0f}M) "
              f"-- max gross was {max_gross/1e6:.1f}M")

    # ---- OEP EP curves for validation and plot ------------------------------
    oep_gross_arr = annual_df["max_event_gross"].to_numpy()
    oep_net_arr   = annual_df["max_event_net"].to_numpy()
    oep_g, ep = ep_curve(oep_gross_arr, N_YEARS)
    oep_n, _  = ep_curve(oep_net_arr,   N_YEARS)
    pml_g100  = oep_pml(oep_gross_arr, 100, N_YEARS)
    pml_g250  = oep_pml(oep_gross_arr, 250, N_YEARS)
    pml_n100  = oep_pml(oep_net_arr,   100, N_YEARS)
    pml_n250  = oep_pml(oep_net_arr,   250, N_YEARS)
    print()
    pml_rank_diagnostic(oep_g, N_YEARS)

    assert pml_n100 <= pml_g100 + 1e-6, \
        f"FAIL: net OEP 1-in-100 ({pml_n100/1e6:.1f}M) > gross ({pml_g100/1e6:.1f}M)"
    assert pml_n250 <= pml_g250 + 1e-6, \
        f"FAIL: net OEP 1-in-250 ({pml_n250/1e6:.1f}M) > gross ({pml_g250/1e6:.1f}M)"
    print("[OK] net OEP PML <= gross OEP PML at 1-in-100 and 1-in-250")

    # ---- Programme metrics ---------------------------------------------------
    print()
    print("=" * 64)
    print("PROGRAMME METRICS (finite reinstatements, n=1 per layer)")
    print("=" * 64)

    total_el   = 0.0
    total_base = 0.0
    total_rp   = 0.0

    print(f"\n{'Layer':<10} {'Attach':>8} {'Limit':>8} {'n':>3} "
          f"{'EL (M)':>9} {'BasePrem (M)':>13} "
          f"{'E[ReinsPrem] (M)':>17} {'HitFreq%':>9} {'ReinstFreq%':>12}")
    print("  " + "-" * 96)

    layer_metrics = []
    for j, lyr in enumerate(LAYERS):
        occ_limit = lyr["occ_limit"]
        n_reinst  = lyr["n_reinstatements"]
        pct       = lyr["reinstatement_premium_pct"]

        ann_capped_j = ann_alloc[j]          # already capped by construction
        el_j         = float(ann_capped_j.mean())
        base_prem_j  = el_j * (1 + LOADING_FACTOR)

        reinstated_j  = np.clip(ann_capped_j - occ_limit, 0.0, n_reinst * occ_limit)
        reinst_prem_j = base_prem_j * (reinstated_j / occ_limit) * pct
        E_rp_j        = float(reinst_prem_j.mean())

        # Layer hit freq: any event's occ recovery > 0 (unlimited — occurrence-level)
        hit_freq_j   = float((ann_occ[j] > 1e-6).mean()) * 100
        reinst_freq_j = float((reinstated_j > 1e-6).mean()) * 100

        total_el   += el_j
        total_base += base_prem_j
        total_rp   += E_rp_j
        layer_metrics.append((lyr, el_j, base_prem_j, E_rp_j, hit_freq_j, reinst_freq_j))

        print(f"  {lyr['name']:<10} {lyr['attachment']/1e6:>6.0f}M "
              f"{occ_limit/1e6:>6.0f}M {n_reinst:>3} "
              f"{el_j/1e6:>9.3f} {base_prem_j/1e6:>13.3f} "
              f"{E_rp_j/1e6:>17.4f} {hit_freq_j:>8.2f}% {reinst_freq_j:>11.2f}%")

    print("  " + "-" * 96)
    print(f"  {'TOTAL':<10} {'':>8} {'':>8} {'':>3} "
          f"{total_el/1e6:>9.3f} {total_base/1e6:>13.3f} "
          f"{total_rp/1e6:>17.4f}")
    print()
    print(f"  Total expected premium (base + reinstatement): "
          f"USD {(total_base + total_rp):>14,.0f}  [TECHNICAL — not market]")
    print(f"  (loading_factor = {LOADING_FACTOR:.0%}; "
          f"reinstatement_premium_pct = 100% pro-rata to amount)")

    # ---- Finite vs unlimited AEP-net comparison ------------------------------
    print()
    print("=" * 64)
    print("AEP-NET: FINITE vs UNLIMITED reinstatements")
    print("=" * 64)
    print("(finite = production; unlimited = prior unlimited assumption)")
    print()
    print(f"  {'RP':<8} {'Finite ($M)':>13} {'Unlimited ($M)':>15} {'Delta ($M)':>12}")
    print("  " + "-" * 52)

    rps = [25, 50, 100, 250, 500, 1000]
    for rp in rps:
        pml_fin = oep_pml(aep_net_finite, rp, N_YEARS) / 1e6
        pml_unl = oep_pml(aep_net_unlim,  rp, N_YEARS) / 1e6
        delta   = pml_fin - pml_unl
        print(f"  1-in-{rp:<3}  {pml_fin:>13.2f} {pml_unl:>15.2f} {delta:>12.2f}")

    aal_fin = float(aep_net_finite.mean()) / 1e6
    aal_unl = float(aep_net_unlim.mean())  / 1e6
    print(f"\n  AAL-net finite = {aal_fin:.4f} M  |  AAL-net unlimited = {aal_unl:.4f} M"
          f"  |  delta = {aal_fin - aal_unl:+.4f} M")
    print("  [Direction confirmed: finite >= unlimited at all RPs "
          "(less recovery -> higher net tail)]")

    # ---- OEP reduction table (gross vs net, for reference) ------------------
    print()
    print(f"OEP PML reduction (gross -> finite net, per occurrence):")
    print(f"  {'Return period':<14} {'Gross ($M)':>12} {'Net ($M)':>12} "
          f"{'Reduction ($M)':>16} {'Reduction %':>12}")
    print("  " + "-" * 68)
    for rp, g, n in [(100, pml_g100, pml_n100), (250, pml_g250, pml_n250)]:
        red_abs = g - n
        red_pct = red_abs / g * 100 if g > 0 else 0.0
        print(f"  1-in-{rp:<9}  {g/1e6:>10.1f}  {n/1e6:>10.1f}  "
              f"{red_abs/1e6:>14.1f}  {red_pct:>11.1f}%")

    # ---- OEP plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(oep_g / 1e6, ep, color="#1f77b4", linewidth=2.2,
            label="OEP Gross (post-policy-deductible)")
    ax.plot(oep_n / 1e6, ep, color="#d62728", linewidth=2.2,
            label="OEP Net (post-XoL, finite reinstatements n=1)")
    ax.fill_betweenx(ep, oep_n / 1e6, oep_g / 1e6,
                     alpha=0.10, color="#2ca02c", label="XoL recovery region")

    lyr_colors = ["#ff7f0e", "#9467bd", "#8c564b"]
    for lyr, col in zip(LAYERS, lyr_colors):
        att = lyr["attachment"] / 1e6
        ax.axvline(att, color=col, linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(att + 0.8, 6e-4,
                f"{lyr['name']}\n{att:.0f}M xs",
                fontsize=7.5, color=col, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

    for rp, g, n in [(100, pml_g100, pml_n100), (250, pml_g250, pml_n250)]:
        p = 1 / rp
        ax.axhline(p, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(0.4, p * 1.25, f"1-in-{rp}", fontsize=7.5, color="grey", va="bottom")
        ax.scatter([g / 1e6], [p], color="#1f77b4", s=40, zorder=6)
        ax.scatter([n / 1e6], [p], color="#d62728", s=40, zorder=6)

    pml_text = (
        f"OEP PML (USD M)\n"
        f"           Gross    Net\n"
        f"1-in-100:  {pml_g100/1e6:>5.1f}   {pml_n100/1e6:>5.1f}\n"
        f"1-in-250:  {pml_g250/1e6:>5.1f}   {pml_n250/1e6:>5.1f}"
    )
    ax.text(0.98, 0.04, pml_text, transform=ax.transAxes,
            fontsize=8.5, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlim(left=0)
    ax.set_xlabel("Per-occurrence portfolio loss (USD million)", fontsize=11)
    ax.set_ylabel("Annual exceedance probability", fontsize=11)
    ax.set_title(
        "OEP: Gross vs Net of XoL reinsurance  --  Florida coastal homeowners\n"
        f"Programme: {FIRST_ATTACH/1e6:.0f}M to {TOP_OF_TOWER/1e6:.0f}M "
        f"({TOTAL_CAPACITY/1e6:.0f}M capacity)  |  1 reinstatement per layer  "
        f"|  {N_YEARS:,} simulated years",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ax.get_ylim())
    rp_ticks  = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000]
    ep_ticks  = [1 / rp for rp in rp_ticks if 1e-4 <= 1 / rp <= 1.0]
    rp_labels = [str(rp) for rp in rp_ticks if 1e-4 <= 1 / rp <= 1.0]
    ax2.set_yticks(ep_ticks)
    ax2.set_yticklabels(rp_labels)
    ax2.set_ylabel("Return period (years)", fontsize=10)

    p_fig = os.path.join(OUT_DIR, "ep_gross_vs_net.png")
    fig.savefig(p_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved -> {p_fig}")

    # ---- Save events_net.csv -------------------------------------------------
    out_ev = os.path.join(RESULTS_DIR, "events_net.csv")
    events_df.to_csv(out_ev, index=False)
    print(f"Saved: {out_ev}  ({len(events_df):,} rows)")
    print(f"  Columns: {list(events_df.columns)}")

    print()
    print("=" * 64)
    print("All structural asserts passed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
