"""
Tests for Step 4.4: finite reinstatements with annual aggregate tracking.

TestHandExample
    Encodes the worked case from CLAUDE.md Step 4.4 spec:
    Events A=120M, B=110M, C=90M gross in the same year; Layer 1 (40M xs 60M, n=1).
    - Annual capped recovery for Layer 1 = 80M
    - Event C's Layer 1 allocated recovery = 0 (aggregate exhausted)
    - reinstated_amount = 40M (one full reinstatement used)

TestGrossUnchanged
    The occ_rec matrix (per-event occurrence recovery) from apply_xol_finite equals
    the output of apply_xol_unlimited row-by-row — gross metrics are untouched because
    the occurrence recovery is purely a function of event gross and layer parameters,
    not of within-year ordering or aggregate caps.

TestUnlimitedRegression
    n_reinstatements=10^9 reproduces the unlimited baseline exactly: allocated recovery
    per event equals uncapped occurrence recovery for every event in every year.

TestAggregateCapEnforced
    Per layer per year: sum of allocated recovery never exceeds occ_limit × (1 + n).

TestOrderIndependence
    Permuting events within a year leaves the annual capped layer recovery invariant
    (i.e., AEP-net is order-independent as stated in the spec).

TestDirection
    Finite allocated recovery per year <= unlimited occurrence recovery per year for
    every layer every year — finite net >= unlimited net at the annual level.

TestReinstatementPremium
    Reinstatement premium is non-negative, and equals zero for years where the annual
    capped recovery does not exceed the first occ_limit bucket.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.reinsurance import apply_xol_finite, apply_xol_unlimited, _occ_recovery


# ---------------------------------------------------------------------------
# Shared layer fixtures
# ---------------------------------------------------------------------------

_LAYER1_ONLY = [
    {
        "name":                     "Layer 1",
        "attachment":               60_000_000.0,
        "occ_limit":                40_000_000.0,
        "n_reinstatements":         1,
        "reinstatement_premium_pct": 1.0,
    }
]

_THREE_LAYERS = [
    {
        "name":                     "Layer 1",
        "attachment":               60_000_000.0,
        "occ_limit":                40_000_000.0,
        "n_reinstatements":         1,
        "reinstatement_premium_pct": 1.0,
    },
    {
        "name":                     "Layer 2",
        "attachment":               100_000_000.0,
        "occ_limit":                50_000_000.0,
        "n_reinstatements":         1,
        "reinstatement_premium_pct": 1.0,
    },
    {
        "name":                     "Layer 3",
        "attachment":               150_000_000.0,
        "occ_limit":                50_000_000.0,
        "n_reinstatements":         1,
        "reinstatement_premium_pct": 1.0,
    },
]

_LAYERS_UNLIM = [
    {
        "name":                     "Layer 1",
        "attachment":               60_000_000.0,
        "occ_limit":                40_000_000.0,
        "n_reinstatements":         10 ** 9,
        "reinstatement_premium_pct": 1.0,
    },
    {
        "name":                     "Layer 2",
        "attachment":               100_000_000.0,
        "occ_limit":                50_000_000.0,
        "n_reinstatements":         10 ** 9,
        "reinstatement_premium_pct": 1.0,
    },
    {
        "name":                     "Layer 3",
        "attachment":               150_000_000.0,
        "occ_limit":                50_000_000.0,
        "n_reinstatements":         10 ** 9,
        "reinstatement_premium_pct": 1.0,
    },
]


def _make_events(gross_list, year_list):
    return pd.DataFrame({
        "EventId":         list(range(1, len(gross_list) + 1)),
        "year":            year_list,
        "portfolio_gross": gross_list,
    })


# ---------------------------------------------------------------------------
# TestHandExample
# ---------------------------------------------------------------------------

class TestHandExample:
    """
    Hand example from CLAUDE.md Step 4.4 spec:
    Year 1: A=120M, B=110M, C=90M gross (EventId 1,2,3).
    Layer 1: 40M xs 60M, n=1 → agg_limit = 80M.

    A: occ=40M, consumed=40M, alloc=40M.
    B: occ=40M, consumed=80M, alloc=40M.
    C: occ=30M, consumed=80M, alloc=0M.  (aggregate exhausted)

    Annual capped rec = 80M.
    reinstated_amount = clip(80M - 40M, 0, 40M) = 40M.
    """

    @pytest.fixture(scope="class")
    def result(self):
        events = _make_events(
            [120e6, 110e6, 90e6],
            [1, 1, 1],
        )
        alloc_rec, occ_rec, rec_total, net = apply_xol_finite(events, _LAYER1_ONLY)
        return alloc_rec, occ_rec, rec_total, net

    def test_annual_capped_recovery_layer1(self, result):
        alloc_rec, _, _, _ = result
        annual_capped = alloc_rec[:, 0].sum()  # all three events in year 1
        assert abs(annual_capped - 80e6) < 1.0, (
            f"Layer 1 annual capped recovery {annual_capped/1e6:.1f}M != 80M"
        )

    def test_event_c_gets_zero(self, result):
        alloc_rec, _, _, _ = result
        # Event C is the third event (index 2); EventId=3
        alloc_c = alloc_rec[2, 0]  # Layer 1 allocated recovery for event C
        assert abs(alloc_c) < 1.0, (
            f"Event C Layer 1 allocated recovery {alloc_c/1e6:.3f}M != 0"
        )

    def test_reinstated_amount(self, result):
        alloc_rec, _, _, _ = result
        occ_limit      = _LAYER1_ONLY[0]["occ_limit"]   # 40M
        n_reinst       = _LAYER1_ONLY[0]["n_reinstatements"]  # 1
        annual_capped  = alloc_rec[:, 0].sum()           # 80M
        reinstated     = min(max(annual_capped - occ_limit, 0.0),
                             n_reinst * occ_limit)
        assert abs(reinstated - 40e6) < 1.0, (
            f"reinstated_amount {reinstated/1e6:.1f}M != 40M"
        )


# ---------------------------------------------------------------------------
# TestGrossUnchanged
# ---------------------------------------------------------------------------

class TestGrossUnchanged:
    """
    The per-event occurrence recovery (occ_rec) from apply_xol_finite equals
    apply_xol_unlimited row-by-row — gross is a function of event loss and layer
    parameters only, not of within-year ordering or aggregate caps.
    """

    def test_occ_rec_matches_unlimited(self):
        rng = np.random.default_rng(99)
        gross = rng.uniform(0, 250e6, size=100)
        year  = rng.integers(1, 11, size=100)

        events = pd.DataFrame({
            "EventId":         np.arange(1, 101),
            "year":            year,
            "portfolio_gross": gross,
        })

        alloc_rec, occ_rec_fin, _, _ = apply_xol_finite(events, _THREE_LAYERS)
        occ_rec_unl, _, _            = apply_xol_unlimited(gross, _THREE_LAYERS)

        np.testing.assert_array_almost_equal(
            occ_rec_fin, occ_rec_unl, decimal=2,
            err_msg="occ_rec from apply_xol_finite != apply_xol_unlimited"
        )


# ---------------------------------------------------------------------------
# TestUnlimitedRegression
# ---------------------------------------------------------------------------

class TestUnlimitedRegression:
    """
    n_reinstatements=10^9 → allocated recovery equals uncapped occ recovery exactly.
    """

    def test_large_n_equals_unlimited(self):
        # Construct multi-event years where finite capping would differ for n=1
        events = _make_events(
            [180e6, 170e6, 160e6, 80e6, 200e6, 100e6],
            [1,     1,     1,     2,    2,     3],
        )
        gross = events["portfolio_gross"].to_numpy()

        alloc_rec, occ_rec_fin, _, _ = apply_xol_finite(events, _LAYERS_UNLIM)
        occ_rec_unl, _, _            = apply_xol_unlimited(gross, _LAYERS_UNLIM)

        np.testing.assert_array_almost_equal(
            alloc_rec, occ_rec_unl, decimal=1,
            err_msg="Large-n finite != unlimited baseline"
        )


# ---------------------------------------------------------------------------
# TestAggregateCapEnforced
# ---------------------------------------------------------------------------

class TestAggregateCapEnforced:
    """
    Annual capped recovery per layer never exceeds occ_limit * (1 + n_reinstatements).
    """

    def test_agg_cap_never_exceeded(self):
        rng    = np.random.default_rng(42)
        gross  = rng.uniform(40e6, 300e6, size=300)
        year   = np.repeat(np.arange(1, 31), 10)   # 30 years, 10 events each

        events = pd.DataFrame({
            "EventId":         np.arange(1, 301),
            "year":            year,
            "portfolio_gross": gross,
        })

        alloc_rec, _, _, _ = apply_xol_finite(events, _THREE_LAYERS)

        # Build annual sum per layer
        ev_yrs = pd.Series(year)
        for j, lyr in enumerate(_THREE_LAYERS):
            agg_limit   = lyr["occ_limit"] * (1 + lyr["n_reinstatements"])
            annual_alloc = pd.Series(alloc_rec[:, j], index=ev_yrs).groupby(level=0).sum()
            over_cap = annual_alloc[annual_alloc > agg_limit + 1.0]
            assert len(over_cap) == 0, (
                f"{lyr['name']}: {len(over_cap)} years exceed agg_limit "
                f"{agg_limit/1e6:.0f}M. Max: {annual_alloc.max()/1e6:.2f}M"
            )


# ---------------------------------------------------------------------------
# TestOrderIndependence
# ---------------------------------------------------------------------------

class TestOrderIndependence:
    """
    Permuting events within a year leaves the annual capped recovery invariant.
    AEP-net is order-independent.
    """

    def test_annual_rec_order_independent(self):
        # Year 1: three large events that will trigger the aggregate cap
        gross_orig = [180e6, 160e6, 140e6]
        events_abc = _make_events(gross_orig, [1, 1, 1])
        events_cba = _make_events(list(reversed(gross_orig)), [1, 1, 1])
        # Re-assign EventId so sort order differs
        events_cba["EventId"] = [1, 2, 3]

        alloc_abc, _, _, _ = apply_xol_finite(events_abc, _THREE_LAYERS)
        alloc_cba, _, _, _ = apply_xol_finite(events_cba, _THREE_LAYERS)

        # Annual capped recovery per layer must be equal regardless of per-event order
        for j, lyr in enumerate(_THREE_LAYERS):
            ann_abc = alloc_abc[:, j].sum()
            ann_cba = alloc_cba[:, j].sum()
            assert abs(ann_abc - ann_cba) < 1.0, (
                f"{lyr['name']}: ABC annual={ann_abc/1e6:.3f}M vs "
                f"CBA annual={ann_cba/1e6:.3f}M (should be equal)"
            )


# ---------------------------------------------------------------------------
# TestDirection
# ---------------------------------------------------------------------------

class TestDirection:
    """
    Finite allocated recovery <= unlimited occurrence recovery per layer per year.
    Therefore finite net >= unlimited net.
    """

    def test_finite_rec_le_unlimited_per_year(self):
        rng   = np.random.default_rng(7)
        gross = rng.uniform(50e6, 250e6, size=200)
        year  = np.repeat(np.arange(1, 41), 5)  # 40 years × 5 events

        events = pd.DataFrame({
            "EventId":         np.arange(1, 201),
            "year":            year,
            "portfolio_gross": gross,
        })

        alloc_rec, _, _, net_fin = apply_xol_finite(events, _THREE_LAYERS)
        occ_rec, _, net_unl     = apply_xol_unlimited(gross, _THREE_LAYERS)

        ev_yrs = pd.Series(year)
        for j, lyr in enumerate(_THREE_LAYERS):
            ann_alloc = pd.Series(alloc_rec[:, j], index=ev_yrs).groupby(level=0).sum()
            ann_occ   = pd.Series(occ_rec[:, j],   index=ev_yrs).groupby(level=0).sum()
            over = (ann_alloc - ann_occ) > 1.0
            assert not over.any(), (
                f"{lyr['name']}: annual allocated > occurrence in {over.sum()} years"
            )


# ---------------------------------------------------------------------------
# TestReinstatementPremium
# ---------------------------------------------------------------------------

class TestReinstatementPremium:
    """
    Reinstatement premium is non-negative and zero when no reinstatement is consumed.
    """

    def _year_alloc_series(self, alloc_rec_j, events_df):
        return pd.Series(
            alloc_rec_j, index=events_df["year"]
        ).groupby(level=0).sum()

    def test_premium_nonneg(self):
        rng   = np.random.default_rng(13)
        gross = rng.uniform(0, 300e6, size=100)
        year  = np.repeat(np.arange(1, 21), 5)

        events = pd.DataFrame({
            "EventId":         np.arange(1, 101),
            "year":            year,
            "portfolio_gross": gross,
        })

        alloc_rec, _, _, _ = apply_xol_finite(events, _THREE_LAYERS)

        for j, lyr in enumerate(_THREE_LAYERS):
            occ_limit     = lyr["occ_limit"]
            n_reinst      = lyr["n_reinstatements"]
            pct           = lyr["reinstatement_premium_pct"]
            ann_alloc_j   = self._year_alloc_series(alloc_rec[:, j], events)
            reinstated_j  = np.clip(ann_alloc_j - occ_limit, 0.0, n_reinst * occ_limit)
            base_prem_j   = float(ann_alloc_j.mean()) * 1.15
            reinst_prem_j = base_prem_j * (reinstated_j / occ_limit) * pct
            assert (reinst_prem_j >= -1e-6).all(), (
                f"{lyr['name']}: negative reinstatement premium found"
            )

    def test_premium_zero_when_no_reinstatement(self):
        # Single event per year, large enough to reach Layer 1 but small enough
        # that annual capped rec stays within first occ_limit (no reinstatement needed)
        # Each year has ONE event at 80M gross → Layer 1 occ_rec = 20M < occ_limit=40M
        events = pd.DataFrame({
            "EventId":         np.arange(1, 11),
            "year":            np.arange(1, 11),
            "portfolio_gross": [80e6] * 10,
        })

        alloc_rec, _, _, _ = apply_xol_finite(events, _LAYER1_ONLY)

        occ_limit    = _LAYER1_ONLY[0]["occ_limit"]   # 40M
        n_reinst     = _LAYER1_ONLY[0]["n_reinstatements"]
        pct          = _LAYER1_ONLY[0]["reinstatement_premium_pct"]

        ann_alloc_j = self._year_alloc_series(alloc_rec[:, 0], events)
        reinstated  = np.clip(ann_alloc_j - occ_limit, 0.0, n_reinst * occ_limit)
        base_prem   = float(ann_alloc_j.mean()) * 1.15
        reinst_prem = base_prem * (reinstated / occ_limit) * pct

        assert (reinst_prem.abs() < 1e-6).all(), (
            f"Expected zero reinstatement premium (no reinstatement used); "
            f"got max {reinst_prem.max():.6f}"
        )
