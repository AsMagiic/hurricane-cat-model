"""
Tests for validation/footprint_metrics.py — Step 3 Level 1a.

All tests use the vendored HURDAT2 fixture (tests/fixtures/hurdat2_fixture.txt)
so no real HURDAT2 file is required.  Grid resolution is coarsened to 0.5°
for speed; the tests verify correctness of algorithm, not absolute values.

Golden vectors for the prepare_storm / run_scenario refactor guard are stored
in tests/fixtures/golden_scenario_footprints.npz (generated before the refactor).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model.scenario as sc
from model.units import kt_to_mph
import validation.footprint_metrics as fm

FIXTURE   = os.path.join(os.path.dirname(__file__), "fixtures", "hurdat2_fixture.txt")
GOLDEN    = os.path.join(os.path.dirname(__file__), "fixtures", "golden_scenario_footprints.npz")
STEP_FAST = 0.5   # coarse grid step (degrees) for test speed


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def andrew_result():
    return fm.run_footprint_validation(
        "ANDREW", 1992, hurdat2_path=FIXTURE, grid_step_deg=STEP_FAST
    )


@pytest.fixture(scope="module")
def ian_result():
    return fm.run_footprint_validation(
        "IAN", 2022, hurdat2_path=FIXTURE, grid_step_deg=STEP_FAST
    )


# ---------------------------------------------------------------------------
# TestGridShape
# ---------------------------------------------------------------------------

class TestGridShape:
    def test_grid_has_many_points(self, andrew_result):
        """Grid must have at least 100 points even at the coarse test step."""
        assert len(andrew_result["grid_lats"]) >= 100

    def test_grid_covers_landfall(self, ian_result):
        """Landfall lat/lon falls inside the grid bounding box."""
        lf   = ian_result["landfall_fix"]
        lats = ian_result["grid_lats"]
        lons = ian_result["grid_lons"]
        assert lats.min() <= float(lf["lat"]) <= lats.max()
        assert lons.min() <= float(lf["lon"]) <= lons.max()


# ---------------------------------------------------------------------------
# TestRadiiMonotone
# ---------------------------------------------------------------------------

class TestRadiiMonotone:
    """R34 >= R50 >= R64 in every non-NaN quadrant (instantaneous field)."""

    def _check_monotone_pair(self, radii, label_outer, label_inner, name):
        for q in ["NE", "SE", "SW", "NW"]:
            r_outer = radii[label_outer][q]
            r_inner = radii[label_inner][q]
            if np.isnan(r_outer) or np.isnan(r_inner):
                continue
            assert r_outer >= r_inner - 1e-9, (
                f"{name}: {label_outer}[{q}]={r_outer:.1f} nm < "
                f"{label_inner}[{q}]={r_inner:.1f} nm — monotone violated"
            )

    def test_r34_ge_r50_andrew(self, andrew_result):
        self._check_monotone_pair(
            andrew_result["radii_instant"], "R34", "R50", "ANDREW"
        )

    def test_r50_ge_r64_ian(self, ian_result):
        self._check_monotone_pair(
            ian_result["radii_instant"], "R50", "R64", "IAN"
        )

    def test_r34_ge_r64_ian(self, ian_result):
        self._check_monotone_pair(
            ian_result["radii_instant"], "R34", "R64", "IAN"
        )


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_instantaneous_wind_deterministic(self):
        """Two identical calls produce bit-identical instantaneous wind arrays."""
        r1 = fm.run_footprint_validation(
            "ANDREW", 1992, hurdat2_path=FIXTURE, grid_step_deg=STEP_FAST
        )
        r2 = fm.run_footprint_validation(
            "ANDREW", 1992, hurdat2_path=FIXTURE, grid_step_deg=STEP_FAST
        )
        np.testing.assert_array_equal(
            r1["instantaneous_wind_mph"], r2["instantaneous_wind_mph"]
        )
        np.testing.assert_array_equal(
            r1["envelope_wind_mph"], r2["envelope_wind_mph"]
        )


# ---------------------------------------------------------------------------
# TestIanR64Sanity
# ---------------------------------------------------------------------------

class TestIanR64Sanity:
    """
    The fixture Ian Cayo Costa observed R64: NE=30, SE=40, SW=30, NW=45 nm.
    Modelled values should be in a physically sane range — these bounds are
    intentionally loose to avoid overfitting to one fixture configuration.
    """

    def test_ian_r64_all_quadrants_sane(self, ian_result):
        """All non-NaN Ian R64 quadrant radii in [1, 150] nm."""
        r64 = ian_result["radii_instant"]["R64"]
        for q in ["NE", "SE", "SW", "NW"]:
            val = r64[q]
            if np.isnan(val):
                continue
            assert 1.0 <= val <= 150.0, (
                f"Ian R64 {q} = {val:.1f} nm — outside sane [1, 150] nm band"
            )

    def test_ian_r64_ne_in_range(self, ian_result):
        """Ian R64 NE modelled in [1, 100] nm (observed = 30 nm)."""
        val = ian_result["radii_instant"]["R64"]["NE"]
        if not np.isnan(val):
            assert 1.0 <= val <= 100.0, (
                f"Ian R64 NE = {val:.1f} nm — too far from observed 30 nm"
            )


# ---------------------------------------------------------------------------
# TestAndrewQualitativePath
# ---------------------------------------------------------------------------

class TestAndrewQualitativePath:
    def test_andrew_no_observed_radii(self, andrew_result):
        """Andrew 1992 best-track has only -999 radii (pre-2004) → has_observed=False."""
        assert andrew_result["has_observed"] is False, (
            "Andrew 1992 should have has_observed=False (pre-2004 HURDAT2 radii)"
        )


# ---------------------------------------------------------------------------
# TestPrepareStormGolden
# ---------------------------------------------------------------------------

class TestPrepareStormGolden:
    """
    Guard: prepare_storm refactor must not change run_scenario output.
    Compares current run_scenario results against the golden vectors captured
    before the refactor in tests/fixtures/golden_scenario_footprints.npz.
    """

    def test_prepare_storm_returns_track(self):
        """prepare_storm returns an (N, 4) track array with N > 1."""
        track, sp, lf = sc.prepare_storm("ANDREW", 1992, hurdat2_path=FIXTURE)
        assert track.ndim == 2
        assert track.shape[1] == 4
        assert track.shape[0] > 1

    def test_prepare_storm_returns_valid_storm_params(self):
        """StormParams from prepare_storm has physically sensible values."""
        from model import hazard as _hazard
        track, sp, lf = sc.prepare_storm("IAN", 2022, hurdat2_path=FIXTURE)
        assert sp.rmax >= float(_hazard._RMAX_FLOOR_KM)
        assert sp.b > 0.0
        assert sp.dp_mb > 0.0
        assert 0.0 <= sp.heading_deg < 360.0

    def test_run_scenario_unchanged_after_refactor(self):
        """
        run_scenario footprint is bit-identical to the golden vectors captured
        before prepare_storm was introduced (verifies DRY refactor correctness).
        """
        if not os.path.exists(GOLDEN):
            pytest.skip("golden_scenario_footprints.npz not found — regenerate")
        golden = np.load(GOLDEN)

        fp_andrew, _, _, _ = sc.run_scenario("ANDREW", 1992, hurdat2_path=FIXTURE)
        fp_ian,    _, _, _ = sc.run_scenario("IAN",    2022, hurdat2_path=FIXTURE)

        np.testing.assert_array_equal(fp_andrew, golden["andrew"],
            err_msg="ANDREW footprint changed after prepare_storm refactor")
        np.testing.assert_array_equal(fp_ian,    golden["ian"],
            err_msg="IAN footprint changed after prepare_storm refactor")
