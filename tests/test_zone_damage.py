"""
Tests for validation/zone_damage.py.

All scenario runs are deterministic (zero RNG). Module-scope fixtures run each storm
once for the entire session; individual tests do no heavy I/O.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import validation.zone_damage as zd


# ---------------------------------------------------------------------------
# Module-scope scenario fixtures — run once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def andrew_result():
    return zd.run_zone_damage("ANDREW", 1992)


@pytest.fixture(scope="module")
def ian_result():
    return zd.run_zone_damage("IAN", 2022)


# ---------------------------------------------------------------------------
# TestCountyAggregation — pure function, no I/O
# ---------------------------------------------------------------------------

class TestCountyAggregation:
    """Verify the DR formula on synthetic data without any scenario run."""

    def test_dr_formula(self):
        """DR = sum(ground_up) / sum(tiv) exactly for each county."""
        exp_df = pd.DataFrame({
            "county": ["A", "A", "B", "B", "C"],
            "tiv":    [100.0, 200.0, 300.0, 100.0, 500.0],
        })
        ground_up = np.array([10.0, 20.0, 60.0, 10.0, 100.0])
        dr = zd.compute_county_dr(ground_up, exp_df)

        assert abs(dr["A"] - 30.0 / 300.0) < 1e-12
        assert abs(dr["B"] - 70.0 / 400.0) < 1e-12
        assert abs(dr["C"] - 100.0 / 500.0) < 1e-12

    def test_all_zero_ground_up(self):
        """All ground_up=0 -> all DR=0; no division errors."""
        exp_df = pd.DataFrame({
            "county": ["X", "X", "Y"],
            "tiv":    [1000.0, 2000.0, 500.0],
        })
        ground_up = np.zeros(3)
        dr = zd.compute_county_dr(ground_up, exp_df)

        assert float(dr["X"]) == 0.0
        assert float(dr["Y"]) == 0.0

    def test_sorted_descending(self):
        """Output Series must be sorted descending by DR."""
        exp_df = pd.DataFrame({
            "county": ["Low", "High", "Mid"],
            "tiv":    [100.0, 100.0, 100.0],
        })
        ground_up = np.array([10.0, 80.0, 40.0])
        dr = zd.compute_county_dr(ground_up, exp_df)

        vals = dr.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), (
            f"DR Series not sorted descending: {vals}"
        )

    def test_misalignment_raises(self):
        """Length mismatch between ground_up and exp_df must raise AssertionError."""
        exp_df = pd.DataFrame({
            "county": ["A", "B"],
            "tiv":    [100.0, 200.0],
        })
        ground_up = np.array([10.0, 20.0, 30.0])  # length 3 vs 2
        with pytest.raises(AssertionError):
            zd.compute_county_dr(ground_up, exp_df)


# ---------------------------------------------------------------------------
# TestMiamiDadePresent
# ---------------------------------------------------------------------------

class TestMiamiDadePresent:
    def test_andrew_miami_dade_present(self, andrew_result):
        """Miami-Dade must appear in Andrew's county DR index."""
        assert "Miami-Dade" in andrew_result["county_dr"].index, (
            f"Miami-Dade not found; counties: {list(andrew_result['county_dr'].index)}"
        )


# ---------------------------------------------------------------------------
# TestCharlotteAbsent
# ---------------------------------------------------------------------------

class TestCharlotteAbsent:
    def test_ian_charlotte_not_in_dr(self, ian_result):
        """Charlotte must not appear in Ian's county DR — absent from portfolio."""
        assert "Charlotte" not in ian_result["county_dr"].index

    def test_ian_charlotte_in_absent_list(self, ian_result):
        """Charlotte must appear in the absent_counties list."""
        assert "Charlotte" in ian_result["absent_counties"], (
            f"absent_counties: {ian_result['absent_counties']}"
        )

    def test_ian_run_completes(self, ian_result):
        """run_zone_damage('IAN', 2022) must complete without error."""
        assert ian_result["county_dr"] is not None
        assert len(ian_result["county_dr"]) > 0


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_andrew_deterministic(self):
        """Two calls to run_zone_damage('ANDREW', 1992) -> identical county_dr."""
        r1 = zd.run_zone_damage("ANDREW", 1992)
        r2 = zd.run_zone_damage("ANDREW", 1992)
        pd.testing.assert_series_equal(r1["county_dr"], r2["county_dr"])

    def test_ian_deterministic(self):
        """Two calls to run_zone_damage('IAN', 2022) -> identical county_dr."""
        r1 = zd.run_zone_damage("IAN", 2022)
        r2 = zd.run_zone_damage("IAN", 2022)
        pd.testing.assert_series_equal(r1["county_dr"], r2["county_dr"])


# ---------------------------------------------------------------------------
# TestRankingSign
# ---------------------------------------------------------------------------

class TestRankingSign:
    def test_andrew_miami_dade_exceeds_ian_lee(self, andrew_result, ian_result):
        """Andrew Miami-Dade DR (83%) > Ian Lee DR (80%) — regression guard."""
        andrew_md = float(andrew_result["county_dr"]["Miami-Dade"])
        ian_lee   = float(ian_result["county_dr"]["Lee"])
        assert andrew_md > ian_lee, (
            f"Expected Andrew Miami-Dade ({andrew_md:.4f}) > Ian Lee ({ian_lee:.4f})"
        )

    def test_andrew_miami_dade_exceeds_pinellas(self, andrew_result):
        """Andrew Miami-Dade DR >> Pinellas — sharpest gradient, most robust check."""
        dr = andrew_result["county_dr"]
        miami_dade = float(dr["Miami-Dade"])
        pinellas   = float(dr["Pinellas"])
        assert miami_dade > pinellas, (
            f"Expected Miami-Dade ({miami_dade:.4f}) > Pinellas ({pinellas:.4f})"
        )
        # Gradient must be substantial (not just noise)
        assert miami_dade > 10 * pinellas, (
            f"Gradient too small: Miami-Dade/Pinellas = {miami_dade/pinellas:.1f}x"
        )

    def test_ian_lee_exceeds_peripherals(self, ian_result):
        """Ian Lee DR >> Pinellas and Hillsborough — impact vs peripheral gradient holds."""
        dr = ian_result["county_dr"]
        lee        = float(dr["Lee"])
        pinellas   = float(dr["Pinellas"])
        hillsboro  = float(dr["Hillsborough"])
        assert lee > pinellas, (
            f"Expected Ian Lee ({lee:.4f}) > Pinellas ({pinellas:.4f})"
        )
        assert lee > hillsboro, (
            f"Expected Ian Lee ({lee:.4f}) > Hillsborough ({hillsboro:.4f})"
        )
