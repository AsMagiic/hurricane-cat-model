"""
Tests for Step 4.2: standard YLT, EventId, and EP equivalence.

TestEquivalence  (requires results/ylt.csv from a real run — skipped otherwise)
    PML at 1-in-100 and 1-in-250 for all four EP views computed from ylt.csv
    via ep_utils must equal the committed baseline values (to the precision stored
    in summary_metrics.csv: AAL to 3 dp, PML to 2 dp in USD millions).
    AAL of each series is also checked. This is the load-bearing regression guard
    that proves the YLT refactor is behaviour-preserving.

TestYltIntegrity
    build_ylt() on synthetic data satisfies structural invariants:
    - exactly N_YEARS rows
    - AggNet <= AggGross for every year
    - MaxOccNet <= MaxOccGross for every year
    - MaxOccGross <= AggGross for every year
    - AggNet >= 0 and MaxOccNet >= 0 (no negative net loss)
    - Zero-loss years present (no-event years produce all-zero rows)
    - Column order matches spec

TestEventId  (requires results/events.csv + results/events_net.csv — skipped otherwise)
    EventId is contiguous 1..N_events, unique, and total count equals len(events.csv).
    The EventId set in events_net.csv equals that in events.csv exactly.
"""

import os
import numpy as np
import pandas as pd
import pytest

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.outputs import build_ylt, build_elt
from model.ep_utils import oep_pml

_RES = os.path.join(_ROOT, "results")

# ---------------------------------------------------------------------------
# Committed baseline (seed=42, 100k years, v3+4.1.1)
# Precision matches summary_metrics.csv: AAL 3 dp, PML 2 dp (USD M)
# ---------------------------------------------------------------------------
_N_YEARS_FULL = 100_000

_AAL_BASELINE = {
    "aep_g": 9.151,
    "aep_n": 7.709,
    "oep_g": 8.471,
    "oep_n": 7.040,
}

_PML100_BASELINE = {
    "aep_g": 122.39,
    "aep_n":  70.32,
    "oep_g": 113.23,
    "oep_n":  60.00,
}

_PML250_BASELINE = {
    "aep_g": 158.69,
    "aep_n":  90.61,
    "oep_g": 146.88,
    "oep_n":  60.00,
}

_YLT_PATH  = os.path.join(_RES, "ylt.csv")
_EVTS_PATH = os.path.join(_RES, "events.csv")
_EVTN_PATH = os.path.join(_RES, "events_net.csv")

_SKIP_YLT  = pytest.mark.skipif(
    not os.path.exists(_YLT_PATH),
    reason="results/ylt.csv not found — run `python run_all.py` first",
)
def _events_have_eventid():
    if not (os.path.exists(_EVTS_PATH) and os.path.exists(_EVTN_PATH)):
        return False
    try:
        import csv
        with open(_EVTS_PATH, newline="") as f:
            header = next(csv.reader(f))
        return "EventId" in header
    except Exception:
        return False

_SKIP_EVTS = pytest.mark.skipif(
    not _events_have_eventid(),
    reason="results/events.csv missing or lacks EventId — run `python run_all.py` first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic(n_years=500, seed=7):
    """Build minimal annual_df + events_net_df for unit testing build_ylt()."""
    rng = np.random.default_rng(seed)

    # ~40% of years have at least one event
    has_event = rng.random(n_years) < 0.4

    annual_rows = []
    event_rows  = []

    for yr in range(1, n_years + 1):
        if has_event[yr - 1]:
            n_ev   = int(rng.integers(1, 4))
            agg_gu = float(rng.uniform(1e6, 80e6))
            agg_gr = agg_gu * float(rng.uniform(0.7, 0.95))
            max_gr = agg_gr / n_ev  # simplified: equal events
            for _ in range(n_ev):
                recovery = float(rng.uniform(0, max_gr * 0.3))
                net_ev   = max_gr - recovery
                event_rows.append({
                    "year":            yr,
                    "recovery_total":  recovery,
                    "portfolio_net":   net_ev,
                })
            annual_rows.append({
                "year":                yr,
                "n_events":            n_ev,
                "aggregate_ground_up": agg_gu,
                "aggregate_gross":     agg_gr,
                "max_event_gross":     max_gr,
            })
        else:
            annual_rows.append({
                "year":                yr,
                "n_events":            0,
                "aggregate_ground_up": 0.0,
                "aggregate_gross":     0.0,
                "max_event_gross":     0.0,
            })

    return pd.DataFrame(annual_rows), pd.DataFrame(event_rows)


# ---------------------------------------------------------------------------
# TestEquivalence
# ---------------------------------------------------------------------------

@_SKIP_YLT
class TestEquivalence:
    """EP metrics from ylt.csv match committed baseline to committed precision."""

    @pytest.fixture(scope="class")
    def ylt(self):
        return pd.read_csv(_YLT_PATH)

    @pytest.fixture(scope="class")
    def series(self, ylt):
        return {
            "aep_g": ylt["AggGross"].to_numpy(),
            "aep_n": ylt["AggNet"].to_numpy(),
            "oep_g": ylt["MaxOccGross"].to_numpy(),
            "oep_n": ylt["MaxOccNet"].to_numpy(),
        }

    @pytest.fixture(scope="class")
    def n_years(self, ylt):
        return len(ylt)

    # ---- AAL ----------------------------------------------------------------
    def test_aal_aep_gross(self, series):
        got = round(float(series["aep_g"].mean()) / 1e6, 3)
        assert got == _AAL_BASELINE["aep_g"], \
            f"AAL aep_g: got {got} expected {_AAL_BASELINE['aep_g']}"

    def test_aal_aep_net(self, series):
        got = round(float(series["aep_n"].mean()) / 1e6, 3)
        assert got == _AAL_BASELINE["aep_n"], \
            f"AAL aep_n: got {got} expected {_AAL_BASELINE['aep_n']}"

    def test_aal_oep_gross(self, series):
        got = round(float(series["oep_g"].mean()) / 1e6, 3)
        assert got == _AAL_BASELINE["oep_g"], \
            f"AAL oep_g: got {got} expected {_AAL_BASELINE['oep_g']}"

    def test_aal_oep_net(self, series):
        got = round(float(series["oep_n"].mean()) / 1e6, 3)
        assert got == _AAL_BASELINE["oep_n"], \
            f"AAL oep_n: got {got} expected {_AAL_BASELINE['oep_n']}"

    # ---- PML 1-in-100 -------------------------------------------------------
    def test_pml100_aep_gross(self, series, n_years):
        got = round(oep_pml(series["aep_g"], 100, n_years) / 1e6, 2)
        assert got == _PML100_BASELINE["aep_g"], \
            f"PML-100 aep_g: got {got} expected {_PML100_BASELINE['aep_g']}"

    def test_pml100_aep_net(self, series, n_years):
        got = round(oep_pml(series["aep_n"], 100, n_years) / 1e6, 2)
        assert got == _PML100_BASELINE["aep_n"], \
            f"PML-100 aep_n: got {got} expected {_PML100_BASELINE['aep_n']}"

    def test_pml100_oep_gross(self, series, n_years):
        got = round(oep_pml(series["oep_g"], 100, n_years) / 1e6, 2)
        assert got == _PML100_BASELINE["oep_g"], \
            f"PML-100 oep_g: got {got} expected {_PML100_BASELINE['oep_g']}"

    def test_pml100_oep_net(self, series, n_years):
        got = round(oep_pml(series["oep_n"], 100, n_years) / 1e6, 2)
        assert got == _PML100_BASELINE["oep_n"], \
            f"PML-100 oep_n: got {got} expected {_PML100_BASELINE['oep_n']}"

    # ---- PML 1-in-250 -------------------------------------------------------
    def test_pml250_aep_gross(self, series, n_years):
        got = round(oep_pml(series["aep_g"], 250, n_years) / 1e6, 2)
        assert got == _PML250_BASELINE["aep_g"], \
            f"PML-250 aep_g: got {got} expected {_PML250_BASELINE['aep_g']}"

    def test_pml250_aep_net(self, series, n_years):
        got = round(oep_pml(series["aep_n"], 250, n_years) / 1e6, 2)
        assert got == _PML250_BASELINE["aep_n"], \
            f"PML-250 aep_n: got {got} expected {_PML250_BASELINE['aep_n']}"

    def test_pml250_oep_gross(self, series, n_years):
        got = round(oep_pml(series["oep_g"], 250, n_years) / 1e6, 2)
        assert got == _PML250_BASELINE["oep_g"], \
            f"PML-250 oep_g: got {got} expected {_PML250_BASELINE['oep_g']}"

    def test_pml250_oep_net(self, series, n_years):
        got = round(oep_pml(series["oep_n"], 250, n_years) / 1e6, 2)
        assert got == _PML250_BASELINE["oep_n"], \
            f"PML-250 oep_n: got {got} expected {_PML250_BASELINE['oep_n']}"


# ---------------------------------------------------------------------------
# TestYltIntegrity
# ---------------------------------------------------------------------------

class TestYltIntegrity:
    """build_ylt() satisfies structural invariants on synthetic data."""

    @pytest.fixture(scope="class")
    def ylt(self):
        annual_df, events_net_df = _make_synthetic(n_years=500)
        return build_ylt(annual_df, events_net_df)

    def test_row_count(self, ylt):
        assert len(ylt) == 500

    def test_column_order(self, ylt):
        expected = ["Year", "NumEvents", "AggGroundUp", "AggGross",
                    "AggNet", "MaxOccGross", "MaxOccNet"]
        assert list(ylt.columns) == expected

    def test_agg_net_le_agg_gross(self, ylt):
        assert (ylt["AggNet"] <= ylt["AggGross"] + 1e-6).all(), \
            "AggNet > AggGross in some year"

    def test_max_occ_net_le_max_occ_gross(self, ylt):
        assert (ylt["MaxOccNet"] <= ylt["MaxOccGross"] + 1e-6).all(), \
            "MaxOccNet > MaxOccGross in some year"

    def test_max_occ_gross_le_agg_gross(self, ylt):
        assert (ylt["MaxOccGross"] <= ylt["AggGross"] + 1e-6).all(), \
            "MaxOccGross > AggGross in some year"

    def test_agg_net_non_negative(self, ylt):
        assert (ylt["AggNet"] >= -1e-6).all(), "Negative AggNet found"

    def test_max_occ_net_non_negative(self, ylt):
        assert (ylt["MaxOccNet"] >= -1e-6).all(), "Negative MaxOccNet found"

    def test_zero_loss_years_present(self, ylt):
        zero = (ylt["AggGross"] == 0.0).sum()
        assert zero > 0, "No zero-loss years found — all years have events (unexpected)"

    def test_event_years_have_positive_loss(self, ylt):
        event_yrs = ylt[ylt["NumEvents"] > 0]
        assert (event_yrs["AggGross"] > 0).all(), \
            "Event year with zero AggGross found"

    def test_no_event_years_zero_losses(self, ylt):
        no_event_yrs = ylt[ylt["NumEvents"] == 0]
        assert (no_event_yrs["AggGross"] == 0.0).all(), \
            "No-event year with non-zero AggGross found"
        assert (no_event_yrs["MaxOccGross"] == 0.0).all(), \
            "No-event year with non-zero MaxOccGross found"
        assert (no_event_yrs["MaxOccNet"] == 0.0).all(), \
            "No-event year with non-zero MaxOccNet found"

    def test_empty_events_net(self):
        n_years = 50
        annual = pd.DataFrame({
            "year":                range(1, n_years + 1),
            "n_events":            [0] * n_years,
            "aggregate_ground_up": [0.0] * n_years,
            "aggregate_gross":     [0.0] * n_years,
            "max_event_gross":     [0.0] * n_years,
        })
        empty_evts = pd.DataFrame(columns=["year", "recovery_total", "portfolio_net"])
        ylt = build_ylt(annual, empty_evts)
        assert len(ylt) == n_years
        assert (ylt["AggNet"] == 0.0).all()
        assert (ylt["MaxOccNet"] == 0.0).all()


# ---------------------------------------------------------------------------
# TestEventId
# ---------------------------------------------------------------------------

@_SKIP_EVTS
class TestEventId:
    """EventId in events.csv is a stable monotonic index; carried to events_net.csv."""

    @pytest.fixture(scope="class")
    def events(self):
        return pd.read_csv(_EVTS_PATH)

    @pytest.fixture(scope="class")
    def events_net(self):
        return pd.read_csv(_EVTN_PATH)

    def test_eventid_is_first_column_events(self, events):
        assert events.columns[0] == "EventId", \
            f"First column of events.csv is {events.columns[0]!r}, expected 'EventId'"

    def test_eventid_is_first_column_events_net(self, events_net):
        assert events_net.columns[0] == "EventId", \
            f"First column of events_net.csv is {events_net.columns[0]!r}, expected 'EventId'"

    def test_eventid_unique(self, events):
        assert events["EventId"].nunique() == len(events), \
            "EventId values are not unique in events.csv"

    def test_eventid_contiguous(self, events):
        ids = events["EventId"].to_numpy()
        assert ids[0] == 1, f"EventId starts at {ids[0]}, expected 1"
        assert ids[-1] == len(events), \
            f"EventId ends at {ids[-1]}, expected {len(events)}"
        assert np.all(np.diff(ids) == 1), "EventId is not contiguous (gaps or repeats)"

    def test_eventid_count_matches_rows(self, events):
        assert len(events["EventId"]) == len(events)

    def test_events_net_eventid_matches_events(self, events, events_net):
        ids_ev  = set(events["EventId"].tolist())
        ids_evn = set(events_net["EventId"].tolist())
        assert ids_ev == ids_evn, (
            f"EventId sets differ between events.csv and events_net.csv. "
            f"Only in events: {ids_ev - ids_evn}. "
            f"Only in events_net: {ids_evn - ids_ev}."
        )


# ---------------------------------------------------------------------------
# ELT helpers
# ---------------------------------------------------------------------------

def _make_elt_events(n_events=200, seed=13):
    """Minimal events_net_df for unit-testing build_elt()."""
    rng   = np.random.default_rng(seed)
    gross = rng.uniform(1e5, 50e6, size=n_events)
    net   = gross * rng.uniform(0.3, 1.0, size=n_events)
    return pd.DataFrame({
        "EventId":        np.arange(1, n_events + 1, dtype=int),
        "portfolio_gross": gross,
        "portfolio_net":   net,
    })

_ELT_PATH  = os.path.join(_RES, "elt.csv")
_SKIP_ELT  = pytest.mark.skipif(
    not os.path.exists(_ELT_PATH),
    reason="results/elt.csv not found — run `python run_all.py` first",
)
_SKIP_ELT_EVTN = pytest.mark.skipif(
    not (os.path.exists(_ELT_PATH) and os.path.exists(_EVTN_PATH)),
    reason="results/elt.csv or events_net.csv not found — run `python run_all.py` first",
)

# Reference TIV (sum of synthetic portfolio; used in reconciliation tolerance)
_TOTAL_TIV_REF = 500_000_000.0


# ---------------------------------------------------------------------------
# TestEltReconciliation
# ---------------------------------------------------------------------------

@_SKIP_ELT
class TestEltReconciliation:
    """
    Load-bearing guard: sum(AnnualRate × MeanLoss) from elt.csv equals the
    AEP AAL from ylt.csv, for both gross and net series.

    Identity: (1/N) × Σ_e portfolio_gross_e = (1/N) × Σ_y AggGross_y = YLT AAL_aep_gross.
    Events partition years (each event belongs to exactly one year), so the sums are
    identical by construction; floating-point order of addition may introduce < 1e-6
    relative error.
    """

    @pytest.fixture(scope="class")
    def elt(self):
        return pd.read_csv(_ELT_PATH)

    @pytest.fixture(scope="class")
    def ylt(self):
        if not os.path.exists(_YLT_PATH):
            pytest.skip("results/ylt.csv not found — run `python run_all.py` first")
        return pd.read_csv(_YLT_PATH)

    def test_aal_gross_reconciles(self, elt, ylt):
        elt_aal = float((elt["AnnualRate"] * elt["MeanLossGross"]).sum()) / 1e6
        ylt_aal = float(ylt["AggGross"].mean()) / 1e6  # full precision — no rounding
        rel_err = abs(elt_aal - ylt_aal) / max(ylt_aal, 1.0)
        assert rel_err < 1e-6, (
            f"ELT gross AAL {elt_aal:.8f} M vs YLT {ylt_aal:.8f} M "
            f"(rel err {rel_err:.2e})"
        )

    def test_aal_net_reconciles(self, elt, ylt):
        elt_aal = float((elt["AnnualRate"] * elt["MeanLossNet"]).sum()) / 1e6
        ylt_aal = float(ylt["AggNet"].mean()) / 1e6    # full precision — no rounding
        rel_err = abs(elt_aal - ylt_aal) / max(ylt_aal, 1.0)
        assert rel_err < 1e-6, (
            f"ELT net AAL {elt_aal:.8f} M vs YLT {ylt_aal:.8f} M "
            f"(rel err {rel_err:.2e})"
        )

    def test_rate_uniform(self, elt):
        n_years = round(1.0 / float(elt["AnnualRate"].iloc[0]))
        expected = 1.0 / n_years
        assert (elt["AnnualRate"] == expected).all(), \
            "AnnualRate is not uniform 1/n_years across all events"

    def test_rate_sum_equals_realized_frequency(self, elt):
        n_years  = round(1.0 / float(elt["AnnualRate"].iloc[0]))
        n_events = len(elt)
        expected = n_events / n_years
        got = float(elt["AnnualRate"].sum())
        assert abs(got - expected) < 1e-9, (
            f"sum(AnnualRate) {got:.8f} != n_events/n_years {expected:.8f}"
        )


# ---------------------------------------------------------------------------
# TestEltIntegrity
# ---------------------------------------------------------------------------

class TestEltIntegrity:
    """build_elt() satisfies structural invariants on synthetic data."""

    @pytest.fixture(scope="class")
    def elt(self):
        evn_df = _make_elt_events(n_events=200, seed=13)
        return build_elt(evn_df, n_years=500, total_tiv=_TOTAL_TIV_REF)

    def test_row_count(self, elt):
        assert len(elt) == 200

    def test_column_order(self, elt):
        expected = ["EventId", "AnnualRate", "MeanLossGross", "MeanLossNet",
                    "StdDevIndependent", "StdDevCorrelated", "ExposureValue"]
        assert list(elt.columns) == expected

    def test_annual_rate_uniform(self, elt):
        assert (elt["AnnualRate"] == 1.0 / 500).all(), \
            "AnnualRate must be exactly 1/n_years for every row"

    def test_rate_sum(self, elt):
        got      = float(elt["AnnualRate"].sum())
        expected = 200 / 500
        assert abs(got - expected) < 1e-10, \
            f"sum(AnnualRate) {got:.10f} != n_events/n_years {expected:.10f}"

    def test_stddev_independent_null(self, elt):
        assert elt["StdDevIndependent"].isna().all(), \
            "StdDevIndependent must be entirely null (v4 calibration)"

    def test_stddev_correlated_null(self, elt):
        assert elt["StdDevCorrelated"].isna().all(), \
            "StdDevCorrelated must be entirely null (v4 calibration)"

    def test_exposure_value_constant(self, elt):
        assert (elt["ExposureValue"] == _TOTAL_TIV_REF).all(), \
            f"ExposureValue must equal reference TIV {_TOTAL_TIV_REF:.0f} for every row"

    def test_net_le_gross(self, elt):
        assert (elt["MeanLossNet"] <= elt["MeanLossGross"] + 1e-6).all(), \
            "MeanLossNet > MeanLossGross for some event"


# ---------------------------------------------------------------------------
# TestEltEventIdSchema
# ---------------------------------------------------------------------------

@_SKIP_ELT_EVTN
class TestEltEventIdSchema:
    """ELT EventId set matches events_net.csv EventId set (uniqueness + coverage)."""

    @pytest.fixture(scope="class")
    def elt(self):
        return pd.read_csv(_ELT_PATH)

    @pytest.fixture(scope="class")
    def events_net(self):
        return pd.read_csv(_EVTN_PATH)

    def test_elt_eventid_unique(self, elt):
        assert elt["EventId"].nunique() == len(elt), \
            "EventId values are not unique in elt.csv"

    def test_elt_eventid_matches_events_net(self, elt, events_net):
        ids_elt = set(elt["EventId"].tolist())
        ids_evn = set(events_net["EventId"].tolist())
        assert ids_elt == ids_evn, (
            f"EventId sets differ between elt.csv and events_net.csv. "
            f"Only in elt: {ids_elt - ids_evn}. "
            f"Only in events_net: {ids_evn - ids_elt}."
        )
